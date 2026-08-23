"""HTTP observability middleware (M5, master prompt §12).

One ASGI middleware provides:

- request-id correlation (inbound X-Request-ID honored, else generated) —
  echoed back on the response and bound to every log record;
- end-to-end latency histogram + request counters with a low-cardinality
  route label (path templates only, never raw paths/ids);
- resource limits: oversized bodies rejected before route handling;
- structured completion log including stage breakdown when routes record
  named stages on request.state.

Streaming bodies are NEVER buffered or inspected: the middleware observes
only transport-level events, keeping SSE passthrough opaque (§8).
"""

from __future__ import annotations

import logging
import time
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from context_proxy.observability.metrics import (
    HTTP_REQUEST_DURATION,
    HTTP_REQUESTS_TOTAL,
    RATE_LIMIT_REJECTS_TOTAL,
)
from context_proxy.observability.ratelimit import RateLimiter

logger = logging.getLogger("context_proxy.request")

REQUEST_ID_CTX: ContextVar[str] = ContextVar("request_id", default="-")

ROUTE_ALIASES = (
    ("/v1/chat/completions", "/v1/chat/completions"),
    ("/v1/models", "/v1/models"),
    ("/metrics", "/metrics"),
    ("/healthz", "/healthz"),
    ("/readyz", "/readyz"),
)


def current_request_id() -> str:
    return REQUEST_ID_CTX.get()


def normalize_route(path: str) -> str:
    for prefix, template in ROUTE_ALIASES:
        if path == prefix or path.startswith(prefix + "/"):
            return template
    if path.startswith("/internal/v1/conversations/"):
        return "/internal/v1/conversations/{id}"
    if path.startswith("/internal/v1/memories/"):
        return "/internal/v1/memories/{id}"
    if path.startswith("/internal/v1"):
        return "/internal/v1/*"
    return "other"


class ObservabilityMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        *,
        max_body_bytes: int = 8 * 1024 * 1024,
        rate_limiter: RateLimiter | None = None,
        rate_limit_enabled: bool = False,
    ):
        super().__init__(app)
        self._max_body_bytes = max_body_bytes
        self._limiter = rate_limiter
        self._rate_limit_enabled = rate_limit_enabled and rate_limiter is not None

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start = time.monotonic()
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        token = REQUEST_ID_CTX.set(request_id)

        route = normalize_route(request.url.path)

        # Resource limit: reject before reading/parsing anything heavy.
        content_length = request.headers.get("content-length")
        if (
            content_length
            and content_length.isdigit()
            and int(content_length) > self._max_body_bytes
        ):
            HTTP_REQUESTS_TOTAL.labels(
                method=request.method, route=route, status="413"
            ).inc()
            response: Response = JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "message": f"request body exceeds {self._max_body_bytes} bytes",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "request_too_large",
                    }
                },
            )
            response.headers["X-Request-ID"] = request_id
            REQUEST_ID_CTX.reset(token)
            return response

        # Rate limiting keyed by conversation identity when present.
        if self._rate_limit_enabled:
            key = request.headers.get("x-conversation-id") or (
                request.client.host if request.client else "unknown"
            )
            if not self._limiter.allow(key):
                RATE_LIMIT_REJECTS_TOTAL.inc()
                retry_after = str(self._limiter.retry_after(key))
                HTTP_REQUESTS_TOTAL.labels(
                    method=request.method, route=route, status="429"
                ).inc()
                response = JSONResponse(
                    status_code=429,
                    content={
                        "error": {
                            "message": "rate limit exceeded",
                            "type": "rate_limit_error",
                            "param": None,
                            "code": "rate_limit_exceeded",
                        }
                    },
                    headers={"Retry-After": retry_after},
                )
                response.headers["X-Request-ID"] = request_id
                REQUEST_ID_CTX.reset(token)
                logger.warning(
                    "rate_limited",
                    extra={"route": route, "retry_after": retry_after},
                )
                return response

        try:
            response = await call_next(request)
        except Exception:
            duration = time.monotonic() - start
            HTTP_REQUESTS_TOTAL.labels(
                method=request.method, route=route, status="500"
            ).inc()
            HTTP_REQUEST_DURATION.labels(route=route).observe(duration)
            logger.exception(
                "request_failed",
                extra={"route": route, "duration_seconds": round(duration, 6)},
            )
            raise
        finally:
            pass

        duration = time.monotonic() - start
        status = str(response.status_code)
        HTTP_REQUESTS_TOTAL.labels(method=request.method, route=route, status=status).inc()
        HTTP_REQUEST_DURATION.labels(route=route).observe(duration)
        # Honor the M1.1 forwarding policy: an upstream-provided X-Request-ID
        # survives untouched; otherwise our correlation id is echoed back.
        if "x-request-id" not in {k.lower() for k in response.headers}:
            response.headers["X-Request-ID"] = request_id

        stages = getattr(request.state, "stages", None) or {}
        logger.info(
            "request_completed",
            extra={
                "route": route,
                "status": response.status_code,
                "duration_seconds": round(duration, 6),
                **{f"stage_{k}_seconds": round(v, 6) for k, v in sorted(stages.items())},
            },
        )
        REQUEST_ID_CTX.reset(token)
        return response


def record_stage(request: Request, name: str, started_monotonic: float) -> None:
    """Record one named pipeline stage; used for the latency breakdown."""
    stages = getattr(request.state, "stages", None)
    if stages is None:
        stages = {}
        request.state.stages = stages
    stages[name] = time.monotonic() - started_monotonic
