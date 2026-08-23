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
        conversation_id, payload = resolve_conversation_id(request, payload)
    except InvalidConversationId as exc:
        return openai_error(
            str(exc),
            err_type="invalid_request_error",
            code="invalid_conversation_id",
            status_code=400,
        )
    extra_headers = _conversation_headers(conversation_id)

    # 1. Persist raw inbound messages (source of truth). Degraded mode skips.
    if store is not None:
        try:
            await store.ensure_conversation(conversation_id)
            await store.append_messages(conversation_id, payload.get("messages") or [])
        except Exception as exc:  # noqa: BLE001 - degradation is intentional (§31)
            logger.warning("inbound_persistence_failed", extra={"error": str(exc)})
            store = None

    # 2. Budget the raw recent window; never exceed the usable budget (§15).
    messages = payload.get("messages") or []
    tools = payload.get("tools")
    try:
        plan = plan_context(
            messages,
            tools=tools,
            usable_budget=settings.context.usable_budget_tokens,
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

    async def persist_assistant(message: dict | None) -> None:
        if store is not None and message is not None:
            await store.append_messages(conversation_id, [message])

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
            message = json.loads(body)["choices"][0]["message"]
            await persist_assistant(message)
        except Exception as exc:  # noqa: BLE001 - opaque passthrough first
            logger.warning("assistant_persistence_failed", extra={"error": str(exc)})

    response = upstream_response(status_code, headers, body)
    for name, value in extra_headers.items():
        response.headers[name] = value
    return response
