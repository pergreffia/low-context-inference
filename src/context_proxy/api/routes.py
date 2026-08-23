from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Request

from context_proxy.api.responses import (
    error_body_response,
    openai_error,
    parse_json_body,
    streaming_response,
    upstream_response,
)
from context_proxy.capture import PersistingLLMStream
from context_proxy.context.planner import ContextOverflowError, plan_context
from context_proxy.conversation.identity import (
    RESPONSE_CONVERSATION_HEADER,
    InvalidConversationId,
    resolve_conversation_id,
)
from context_proxy.conversation.store import HistoryDivergenceError
from context_proxy.providers.errors import ContextProxyError

router = APIRouter()
logger = logging.getLogger("context_proxy.request")


def _conversation_headers(conversation_id: str | None) -> dict[str, str]:
    if conversation_id is None:
        return {}
    return {RESPONSE_CONVERSATION_HEADER: conversation_id}


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
        try:
            await store.ensure_conversation(conversation_id)
            await store.reconcile_history(conversation_id, payload.get("messages") or [])
        except HistoryDivergenceError as exc:
            return openai_error(
                str(exc),
                err_type="invalid_request_error",
                code="history_conflict",
                status_code=409,
                headers=extra_headers,
            )
        except Exception as exc:  # noqa: BLE001 - degradation is intentional (§31)
            logger.warning("inbound_persistence_failed", extra={"error": str(exc)})
            store = None

    # 2. Budget the raw recent window; never exceed the usable budget (§15).
    # pinned_budget_tokens is reserved now so M3+ pinned injection cannot
    # silently push the final context over budget.
    messages = payload.get("messages") or []
    tools = payload.get("tools")
    try:
        plan = plan_context(
            messages,
            tools=tools,
            usable_budget=settings.context.usable_budget_tokens,
            reserved_tokens=settings.context.pinned_budget_tokens,
        )
    except ContextOverflowError as exc:
        return openai_error(
            str(exc),
            err_type="invalid_request_error",
            code="context_length_exceeded",
            param="messages",
            status_code=400,
            headers=extra_headers,
        )
    out_payload = {**payload, "messages": plan.messages}

    async def persist_assistant(message: dict | None, metadata: dict | None = None) -> None:
        """Best-effort assistant persistence (M2.3 §1–§3).

        Concurrent identical requests each produce a real inference response;
        only the FIRST continuation reconciles cleanly. A loser diverges at the
        assistant index: the committed history stays source of truth, nothing
        is appended, and the already-generated upstream response still reaches
        its client untouched. Expected conflicts and unexpected failures get
        distinct structured events; neither alters the HTTP response.
        """
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
        except Exception as exc:  # noqa: BLE001 - passthrough first, always
            logger.warning(
                "assistant_persistence_failed",
                extra={
                    "conversation_id": conversation_id,
                    "error": str(exc),
                },
            )

    if payload.get("stream") is True:
        try:
            stream = await llm.open_stream(out_payload)
        except ContextProxyError as exc:
            return await error_body_response(exc)
        if store is not None:
            stream = PersistingLLMStream(stream, persist_assistant)
        response = streaming_response(stream)
        for name, value in extra_headers.items():
            response.headers[name] = value
        return response

    try:
        status_code, headers, body = await llm.complete(out_payload)
    except ContextProxyError as exc:
        return await error_body_response(exc)

    if store is not None and 200 <= status_code < 300:
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
            await persist_assistant(message, metadata or None)

    response = upstream_response(status_code, headers, body)
    for name, value in extra_headers.items():
        response.headers[name] = value
    return response
