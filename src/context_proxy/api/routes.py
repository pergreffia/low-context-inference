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
from context_proxy.context.engine import (
    ContextOverflowError as EngineOverflowError,
)
from context_proxy.context.engine import separate_current_request
from context_proxy.context.planner import ContextOverflowError, plan_context
from context_proxy.context.query import extract_retrieval_query
from context_proxy.conversation.identity import (
    RESPONSE_CONVERSATION_HEADER,
    InvalidConversationId,
    resolve_conversation_id,
)
from context_proxy.conversation.store import HistoryDivergenceError
from context_proxy.memory.errors import (
    PersistenceInfrastructureError,
    RetrievalError,
)
from context_proxy.observability.metrics import record_tokens
from context_proxy.observability.middleware import record_stage
from context_proxy.providers.errors import ContextProxyError

router = APIRouter()
logger = logging.getLogger("context_proxy.request")

# Expected infrastructure failures that legitimately degrade persistence
# (M6-final review §3). Anything else — TypeError, KeyError, assertion
# failures, reconciliation bugs — propagates as a real application error.
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

    # 1. Persist raw inbound messages (source of truth, §29 step 3). Full-history
    #    clients resend the whole conversation: only the new suffix is appended;
    #    divergent histories are rejected before any inference call (M2.1 §1).
    #    Degraded mode (no store) skips persistence entirely.
    if store is not None:
        stage_start = _now()
        try:
            await store.ensure_conversation(conversation_id)
            await store.reconcile_history(conversation_id, payload.get("messages") or [])
            record_stage(request, "inbound_persistence", stage_start)
        except HistoryDivergenceError as exc:
            return openai_error(
                str(exc),
                err_type="invalid_request_error",
                code="history_conflict",
                status_code=409,
                headers=extra_headers,
            )
        except _PERSISTENCE_INFRA_ERRORS as exc:
            # Expected infrastructure failure: degrade to passthrough-only.
            logger.warning("inbound_persistence_failed", extra={"error": str(exc)})
            store = None

    # 2. Assemble the model context within the usable budget (M4 §11).
    #    The inbound payload is structurally split into history and the
    #    current request: the engine models the request exactly once, as a
    #    mandatory atomic candidate. When the Context Assembly Engine is
    #    available it fuses recent raw turns with best-effort retrieval
    #    (memories/chunks, conversation-scoped, active-only), deduplicates,
    #    applies MMR diversity, and packs a deterministic ContextPlan.
    #    Otherwise the M2 raw window planner is used. Both paths guarantee:
    #    never exceed budget, current request preserved, atomic units.
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
                    # Expected retrieval failure: degrade to raw/recent only.
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
        """Best-effort assistant persistence + memory indexing (M2.3/M3).

        Concurrent identical requests each produce a real inference response;
        only the FIRST continuation reconciles cleanly. A loser diverges at the
        assistant index: the committed history stays source of truth, nothing
        is appended, and the already-generated upstream response still reaches
        its client untouched. Expected conflicts and unexpected failures get
        distinct structured events; neither alters the HTTP response.
        Indexing completed turns runs after persistence and can never affect
        the response either.

        M5 review §2: token accounting happens here EXACTLY ONCE per upstream
        response, before any store interaction — accounting works with or
        without persistence and is never doubled.
        """
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
                },
            )
            return
        except _PERSISTENCE_INFRA_ERRORS as exc:
            # Expected infrastructure failure: response already safe upstream.
            logger.warning(
                "assistant_persistence_failed",
                extra={
                    "conversation_id": conversation_id,
                    "error": str(exc),
                },
            )
            return
        await index_memory(conversation_id)

    async def index_memory(conversation_id: str) -> None:
        """Chunk+index completed turns; best-effort, never breaks responses.

        Latency trade-off (M3 review §4): indexing is synchronous but bounded
        by MEMORY__INDEX_TIMEOUT_SECONDS so slow embedding/vector endpoints
        cannot stall the request indefinitely.
        """
        memory = getattr(app_state, "memory", None)
        if memory is None or not settings.memory.auto_index:
            return
        try:
            created = await asyncio.wait_for(
                memory.index_completed_turns(conversation_id),
                timeout=settings.memory.index_timeout_seconds,
            )
            if created:
                logger.info(
                    "turns_indexed",
                    extra={"conversation_id": conversation_id, "chunks": created},
                )
        except TimeoutError:
            logger.warning(
                "memory_index_timeout",
                extra={
                    "conversation_id": conversation_id,
                    "timeout_seconds": settings.memory.index_timeout_seconds,
                },
            )
        except Exception as exc:  # noqa: BLE001 - degradation by design
            logger.warning(
                "memory_index_failed",
                extra={"conversation_id": conversation_id, "error": str(exc)},
            )

    if payload.get("stream") is True:
        try:
            stream = await llm.open_stream(out_payload)
        except ContextProxyError as exc:
            return await error_body_response(exc)
        # Always wrap: token accounting must survive a degraded store (M5).
        stream = PersistingLLMStream(stream, persist_assistant)
        response = streaming_response(stream)
        for name, value in extra_headers.items():
            response.headers[name] = value
        return response

    try:
        status_code, headers, body = await llm.complete(out_payload)
    except ContextProxyError as exc:
        return await error_body_response(exc)

    if 200 <= status_code < 300:
        try:
            parsed = json.loads(body)
            message = parsed["choices"][0]["message"]
            metadata = {
                key: value
                for key, value in (
                    ("finish_reason", parsed["choices"][0].get("finish_reason")),
                    ("usage", parsed.get("usage")),
                    ("model", parsed.get("model")),
                )
                if value is not None
            }
        except Exception as exc:  # noqa: BLE001 - opaque passthrough first
            logger.warning(
                "assistant_persistence_failed",
                extra={"conversation_id": conversation_id, "error": str(exc)},
            )
        else:
            # persist_assistant performs token accounting EXACTLY ONCE before
            # any store interaction (M5 review §2): accounting works with or
            # without persistence and is never doubled.
            await persist_assistant(message, metadata or None)

    response = upstream_response(status_code, headers, body)
    for name, value in extra_headers.items():
        response.headers[name] = value
    return response
