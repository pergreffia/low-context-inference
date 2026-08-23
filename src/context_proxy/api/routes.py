from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from context_proxy.api.responses import (
    error_body_response,
    openai_error,
    parse_json_body,
    upstream_response,
)
from context_proxy.providers.errors import ContextProxyError

router = APIRouter()


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
    llm = request.app.state.llm

    try:
        payload = await parse_json_body(request)
    except ValueError as exc:
        return openai_error(
            str(exc),
            err_type="invalid_request_error",
            code="invalid_request_body",
            status_code=400,
        )

    if payload.get("stream") is True:
        try:
            stream = await llm.open_stream(payload)
        except ContextProxyError as exc:
            return await error_body_response(exc)
        return StreamingResponse(
            stream.iter_bytes(),
            status_code=stream.status_code,
            media_type=stream.media_type,
            headers=stream.passthrough_headers() or None,
        )

    try:
        status_code, headers, body = await llm.complete(payload)
    except ContextProxyError as exc:
        return await error_body_response(exc)
    return upstream_response(status_code, headers, body)
