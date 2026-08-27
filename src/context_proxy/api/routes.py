from __future__ import annotations

import asyncio
import json
import logging

import asyncpg
from fastapi import APIRouter, Request

from context_proxy.api.responses import (
    error_body_response,
    openai_error,
    parse_json_body,
    streaming_response,
    upstream_response,
)
from context_proxy.api.validation import validate_chat_payload
from context_proxy.capture import PersistingLLMStream
from context_proxy.context.engine import ContextOverflowError as EngineOverflowError
from context_proxy.context.engine import separate_current_request
from context_proxy.context.planner import ContextOverflowError, plan_context
from context_proxy.context.query import extract_retrieval_query
from context_proxy.conversation.identity import (
    RESPONSE_CONVERSATION_HEADER,
    InvalidConversationId,
    resolve_conversation_id,
)
from context_proxy.conversation.store import HistoryDivergenceError
from context_proxy.memory.errors import PersistenceInfrastructureError, RetrievalError
from context_proxy.observability.metrics import record_tokens
from context_proxy.observability.middleware import record_stage
from context_proxy.providers.errors import ContextProxyError

router = APIRouter()
logger = logging.getLogger("context_proxy.request")

_PERSISTENCE_INFRA_ERRORS = (
    asyncpg.PostgresError,
    asyncpg.InterfaceError,
    asyncio.TimeoutError,
    OSError,
    PersistenceInfrastructureError,
)


def _conversation_headers(conversation_id: str | None) -> dict[str, str]:
    if conversation_id is None:
        return {}
    return {RESPONSE_CONVERSATION_HEADER: conversation_id}


def _now() -> float:
    import time

    return time.monotonic()


@router.get("/v1/models")
async def list_models(request: Request):
    llm = request.app.state.llm
    try:
        status_code, headers, body = await llm.list_models()
    except ContextProxyError as exc:
        return await error_body_response(exc)
    return upstream_response(status_code, headers, body)


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    app_state = request.app.state
    llm = app_state.llm
    store = getattr(app_state, "store", None)
    settings = app_state.settings

    try:
        payload = await parse_json_body(request)
        validate_chat_payload(payload)
    except ValueError as exc:
        return openai_error(
            str(exc),
            err_type="invalid_request_error",
            code="invalid_request_body",
            status_code=400,
        )

    try:
        conversation_id, payload = resolve_conversation_id(request, payload, settings)
    except InvalidConversationId as exc:
        return openai_error(
            str(exc),
            err_type="invalid_request_error",
            code="invalid_conversation_id",
            status_code=400,
        )
    extra_headers = _conversation_headers(conversation_id)

    if store is not None:
        stage_start = _now()
        try:
            await store.ensure_conversation(conversation_id)
            await store.reconcile_history(conversation_id, payload.get("messages") or [])
            record_stage(request, "inbound_persistence", stage_start)
        except HistoryDivergenceError as exc:
            logger.warning(
                "history_reconciliation_conflict",
                extra={
                    "conversation_id": exc.conversation_id,
                    "message_index": exc.index,
                    "persisted_sha256": exc.persisted_hash,
                    "incoming_sha256": exc.incoming_hash,
                    "different_fields": exc.different_fields,
                },
            )
            return openai_error(
                str(exc),
                err_type="invalid_request_error",
                code="history_conflict",
                status_code=409,
                headers=extra_headers,
            )
        except _PERSISTENCE_INFRA_ERRORS as exc:
            logger.warning("inbound_persistence_failed", extra={"error": str(exc)})
            store = None

    messages = payload.get("messages") or []
    tools = payload.get("tools")
    engine = getattr(app_state, "context_engine", None)
    stage_start = _now()
    try:
        if engine is not None:
            retrieved: list = []
            memory = getattr(app_state, "memory", None)
            query = extract_retrieval_query(messages)
            if memory is not None and query:
                try:
                    retrieved = await memory.retrieve(query, conversation_id)
                except RetrievalError as exc:
                    logger.warning(
                        "context_retrieval_failed",
                        extra={"conversation_id": conversation_id, "error": str(exc)},
                    )
            history, current_request = separate_current_request(messages)
            plan = engine.build(
                history=history,
                current_request=current_request,
                tools=tools,
                retrieved=retrieved,
                conversation_id=conversation_id,
            )
        else:
            history_msgs, current_req = separate_current_request(messages)
            plan = plan_context(
                history=history_msgs,
                current_request=current_req,
                tools=tools,
                usable_budget=settings.context.usable_budget_tokens,
                reserved_tokens=settings.context.pinned_budget_tokens,
            )
        out_messages = plan.messages
    except (ContextOverflowError, EngineOverflowError) as exc:
        return openai_error(
            str(exc),
            err_type="invalid_request_error",
            code="context_length_exceeded",
            param="messages",
            status_code=400,
            headers=extra_headers,
        )
    record_stage(request, "context_assembly", stage_start)
    out_payload = {**payload, "messages": out_messages}

    async def persist_assistant(message: dict | None, metadata: dict | None = None) -> None:
        record_tokens(metadata.get("usage") if metadata else None)
        if store is None or message is None:
            return
        try:
            await store.reconcile_history(
                conversation_id,
                [*messages, message],
                metadata=metadata,
            )
        except HistoryDivergenceError as exc:
            logger.warning(
                "assistant_persistence_conflict",
                extra={
                    "conversation_id": conversation_id,
                    "index": exc.index,
                    "persisted_sha256": exc.persisted_hash,
                    "incoming_sha256": exc.incoming_hash,
                    "different_fields": exc.different_fields,
                },
            )
            return
        except _PERSISTENCE_INFRA_ERRORS as exc:
            logger.warning("assistant_persistence_failed", extra={"error": str(exc)})
            return

    try:
        if payload.get("stream"):
            response = await llm.stream(out_payload)
            return streaming_response(
                PersistingLLMStream(
                    response,
                    persist_assistant,
                    conversation_id=conversation_id,
                    request=request,
                ),
                headers=extra_headers,
            )
        response = await llm.complete(out_payload)
        await persist_assistant(response.message, response.metadata)
        return upstream_response(response.status_code, response.headers, response.body)
    except ContextProxyError as exc:
        return await error_body_response(exc, headers=extra_headers)
