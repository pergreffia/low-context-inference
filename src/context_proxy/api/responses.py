from __future__ import annotations

import json

from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from context_proxy.providers.headers import ensure_content_type


def openai_error(
    message: str,
    *,
    err_type: str = "api_error",
    code: str | None = None,
    param: str | None = None,
    status_code: int = 500,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": err_type,
                "param": param,
                "code": code,
            }
        },
    )


async def parse_json_body(request: Request) -> dict:
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001 - any decode failure is a client error
        raise ValueError(f"invalid JSON body: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    return payload


def upstream_response(status_code: int, headers: dict[str, str], body: bytes) -> Response:
    """Forward an upstream buffered response with its own headers intact.

    Content-Type comes from the forwarded headers (a default is applied when
    the upstream omitted it); bodies are never rewritten.
    """
    return Response(
        content=body,
        status_code=status_code,
        headers=ensure_content_type(headers or {}),
    )


def streaming_response(stream) -> StreamingResponse:
    """Forward an upstream stream incrementally, preserving its headers."""
    headers = stream.passthrough_headers()
    if not any(name.lower() == "content-type" for name in headers):
        headers["Content-Type"] = "text/event-stream"
    return StreamingResponse(
        stream.iter_bytes(),
        status_code=stream.status_code,
        headers=headers,
    )


async def error_body_response(exc: Exception) -> Response:
    from context_proxy.providers.errors import UpstreamHTTPError, UpstreamUnavailable

    if isinstance(exc, UpstreamHTTPError):
        headers = dict(exc.headers)
        if exc.content_type.startswith("application/json"):
            try:
                json.loads(exc.body)
                if not any(n.lower() == "content-type" for n in headers):
                    headers["Content-Type"] = exc.content_type
                return Response(
                    content=exc.body,
                    status_code=exc.status_code,
                    headers=headers,
                )
            except (ValueError, UnicodeDecodeError):
                pass
        return openai_error(
            "upstream inference endpoint returned an error",
            err_type="upstream_error",
            status_code=exc.status_code,
        )
    if isinstance(exc, UpstreamUnavailable):
        return openai_error(
            str(exc),
            err_type="api_error",
            code="upstream_unavailable",
            status_code=502,
        )
    return openai_error(str(exc), err_type="internal_error", status_code=500)

