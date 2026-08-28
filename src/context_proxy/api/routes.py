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
from context_proxy.conversation.identity import RESPONSE_CONVERSATION_HEADER, InvalidConversationId, resolve_conversation_id
from context_proxy.conversation.projection import is_auxiliary_projection
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
    logger.debug("models_request", extra={"route": "/v1/models"})
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
    logger.debug(
        "chat_request_received",
        extra={
            "method": request.method,
            "path": request.url.path,
            "headers": dict(request.headers),
        },
    )

    try:
        payload = await parse_json_body(request)
        validate_chat_payload(payload)
    except ValueError as exc:
        logger.debug("chat_request_invalid", extra={"error": str(exc)})
        return openai_error(str(exc), err_type="invalid_request_error", code="invalid_request_body", status_code=400)

    try:
        conversation_id, payload = resolve_conversation_id(request, payload, settings)
    except InvalidConversationId as exc:
        logger.debug("conversation_id_invalid", extra={"error": str(exc)})
        return openai_error(str(exc), err_type="invalid_request_error", code="invalid_conversation_id", status_code=400)
    extra_headers = _conversation_headers(conversation_id)
    messages = payload.get("messages") or []
    auxiliary_projection = is_auxiliary_projection(messages)
    logger.debug(
        "chat_request_payload",
        extra={
            "conversation_id": conversation_id,
            "auxiliary_projection": auxiliary_projection,
            "payload": payload,
            "message_count": len(messages),
        },
    )

    if auxiliary_projection:
        logger.info("history_projection_skipped", extra={"conversation_id": conversation_id})

    if store is not None and not auxiliary_projection:
        stage_start = _now()
        try:
            await store.ensure_conversation(conversation_id)
            logger.debug("history_reconciliation_start", extra={"conversation_id": conversation_id, "incoming_messages": messages})
            await store.reconcile_history(conversation_id, messages)
            record_stage(request, "inbound_persistence", stage_start)
        except HistoryDivergenceError as exc:
            logger.warning(
                "history_reconciliation_conflict conversation_id=%s index=%s persisted_sha256=%s incoming_sha256=%s different_fields=%s persisted_len=%s incoming_len=%s prefix_len=%s; continuing with client projection",
                exc.conversation_id, exc.index, exc.persisted_hash, exc.incoming_hash, exc.different_fields,
                exc.persisted_len, exc.incoming_len, exc.prefix_len,
            )
            logger.debug("history_reconciliation_exception", extra={"conversation_id": conversation_id, "error": str(exc)})
            # Conversation persistence is observational. A history projection
            # conflict must never reject an otherwise valid inference request.
        except _PERSISTENCE_INFRA_ERRORS as exc:
            logger.warning("inbound_persistence_failed", extra={"error": str(exc)})
            store = None

    tools = payload.get("tools")
    engine = getattr(app_state, "context_engine", None)
    stage_start = _now()
    try:
        if engine is not None:
            retrieved: list = []
            memory = getattr(app_state, "memory", None)
            query = extract_retrieval_query(messages)
            logger.debug("context_retrieval_query", extra={"conversation_id": conversation_id, "query": query})
            if memory is not None and query:
                try:
                    retrieved = await memory.retrieve(query, conversation_id)
                    logger.debug("context_retrieval_result", extra={"conversation_id": conversation_id, "retrieved": retrieved})
                except RetrievalError as exc:
                    logger.warning("context_retrieval_failed", extra={"conversation_id": conversation_id, "error": str(exc)})
            history, current_request = separate_current_request(messages)
            logger.debug("context_engine_input", extra={"conversation_id": conversation_id, "history": history, "current_request": current_request, "tools": tools})
            plan = engine.build(history=history, current_request=current_request, tools=tools, retrieved=retrieved, conversation_id=conversation_id)
        else:
            history_msgs, current_req = separate_current_request(messages)
            logger.debug("planner_input", extra={"conversation_id": conversation_id, "history": history_msgs, "current_request": current_req, "tools": tools})
            plan = plan_context(history=history_msgs, current_request=current_req, tools=tools, usable_budget=settings.context.usable_budget_tokens, reserved_tokens=settings.context.pinned_budget_tokens)
        out_messages = plan.messages
        logger.debug("context_assembly_result", extra={"conversation_id": conversation_id, "output_messages": out_messages})
    except (ContextOverflowError, EngineOverflowError) as exc:
        return openai_error(str(exc), err_type="invalid_request_error", code="context_length_exceeded", param="messages", status_code=400, headers=extra_headers)
    record_stage(request, "context_assembly", stage_start)
    out_payload = {**payload, "messages": out_messages}
    logger.debug("upstream_request_payload", extra={"conversation_id": conversation_id, "payload": out_payload})

    async def persist_assistant(message: dict | None, metadata: dict | None = None) -> None:
        record_tokens(metadata.get("usage") if metadata else None)
        logger.debug("assistant_message_received", extra={"conversation_id": conversation_id, "assistant_message": message, "metadata": metadata})
        if store is None or message is None or auxiliary_projection:
            return
        try:
            logger.debug("assistant_persistence_start", extra={"conversation_id": conversation_id, "messages": [*messages, message], "metadata": metadata})
            await store.reconcile_history(conversation_id, [*messages, message], metadata=metadata)
        except HistoryDivergenceError as exc:
            logger.warning(
                "assistant_persistence_conflict conversation_id=%s index=%s persisted_sha256=%s incoming_sha256=%s different_fields=%s persisted_len=%s incoming_len=%s prefix_len=%s; continuing without persistence",
                exc.conversation_id, exc.index, exc.persisted_hash, exc.incoming_hash, exc.different_fields,
                exc.persisted_len, exc.incoming_len, exc.prefix_len,
            )
            return
        except _PERSISTENCE_INFRA_ERRORS as exc:
            logger.warning("assistant_persistence_failed", extra={"error": str(exc)})
            return
        await index_memory(conversation_id)

    async def index_memory(conversation_id: str) -> None:
        memory = getattr(app_state, "memory", None)
        if memory is None or not settings.memory.auto_index:
            return
        try:
            created = await asyncio.wait_for(memory.index_completed_turns(conversation_id), timeout=settings.memory.index_timeout_seconds)
            if created:
                logger.info("turns_indexed", extra={"conversation_id": conversation_id, "chunks": created})
        except TimeoutError:
            logger.warning("memory_index_timeout", extra={"conversation_id": conversation_id, "timeout_seconds": settings.memory.index_timeout_seconds})
        except Exception as exc:
            logger.warning("memory_index_failed", extra={"conversation_id": conversation_id, "error": str(exc)})

    if payload.get("stream") is True:
        try:
            logger.debug("upstream_stream_open_start", extra={"conversation_id": conversation_id})
            stream = await llm.open_stream(out_payload)
            logger.debug("upstream_stream_opened", extra={"conversation_id": conversation_id, "status_code": stream.status_code, "media_type": stream.media_type})
        except ContextProxyError as exc:
            return await error_body_response(exc)
        stream = PersistingLLMStream(stream, persist_assistant, max_capture_bytes=settings.server.max_capture_bytes)
        response = streaming_response(stream)
        for name, value in extra_headers.items():
            response.headers[name] = value
        return response

    try:
        logger.debug("upstream_complete_start", extra={"conversation_id": conversation_id})
        status_code, headers, body = await llm.complete(out_payload)
        logger.debug("upstream_complete_response", extra={"conversation_id": conversation_id, "status_code": status_code, "headers": headers, "body": body})
    except ContextProxyError as exc:
        return await error_body_response(exc)

    if 200 <= status_code < 300:
        try:
            parsed = json.loads(body)
            message = parsed["choices"][0]["message"]
            metadata = {
                key: value for key, value in (
                    ("finish_reason", parsed["choices"][0].get("finish_reason")),
                    ("usage", parsed.get("usage")),
                    ("model", parsed.get("model")),
                ) if value is not None
            }
        except Exception as exc:
            logger.warning("assistant_persistence_failed", extra={"conversation_id": conversation_id, "error": str(exc)})
        else:
            await persist_assistant(message, metadata or None)

    response = upstream_response(status_code, headers, body)
    for name, value in extra_headers.items():
        response.headers[name] = value
    return response
