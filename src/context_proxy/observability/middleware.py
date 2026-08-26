"""HTTP observability + resource limits as a pure ASGI middleware (M5).

Provides:

- request-id correlation (inbound X-Request-ID honored, else generated) —
  bound to every log record via context var and echoed on the response
  unless the upstream already supplied one (M1.1 forwarding policy);
- end-to-end latency histogram + request counters with low-cardinality route
  templates (never raw paths/ids);
- resource limits enforced during body READ, not just Content-Length: a
  chunked body exceeding SERVER__MAX_BODY_BYTES is cut off mid-stream and
  answered 413 before the application ever runs;
- rate limiting keyed by identity policy (X-Conversation-ID header, else
  client host — see RateLimitSettings documentation);
- structured completion log including named stage breakdown recorded by
  routes on scope["state"].

Streaming bodies are never buffered or inspected beyond size accounting:
SSE passthrough stays opaque (master prompt §8).
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from contextvars import ContextVar

from starlette.responses import JSONResponse, Response

from context_proxy.observability.metrics import (
    CLIENT_DISCONNECTS_TOTAL,
    HTTP_REQUEST_DURATION,
    HTTP_REQUESTS_TOTAL,
    HTTP_STREAMS_ABORTED_TOTAL,
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

# Never throttled: liveness/readiness/telemetry must stay answerable even
# while an abusive client is being shed.
_RATE_LIMIT_EXEMPT_ROUTES = frozenset({"/healthz", "/readyz", "/metrics"})

# X-Request-ID policy (final hardening pass): client-controlled value, echoed
# into response headers and every log line. Accept only printable ASCII
# (visible characters, no space) up to 128 chars; anything else — oversized,
# control characters, CR/LF — falls back to a server-generated id instead of
# failing an otherwise valid request or amplifying logs.
_REQUEST_ID_MAX_CHARS = 128
_SAFE_REQUEST_ID = re.compile(r"^[\x21-\x7E]{1,128}$")


def _sanitize_request_id(raw: str | None) -> str | None:
    if raw and _SAFE_REQUEST_ID.match(raw):
        return raw
    return None


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


def _error_response(
    status_code: int,
    code: str,
    message: str,
    *,
    err_type: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    response: Response = JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": err_type
                or ("invalid_request_error" if status_code < 500 else "api_error"),
                "param": None,
                "code": code,
            }
        },
        headers=headers,
    )
    return response


class ObservabilityMiddleware:
    """Pure ASGI middleware; no BaseHTTPMiddleware buffering involved."""

    def __init__(
        self,
        app,
        *,
        max_body_bytes: int = 8 * 1024 * 1024,
        rate_limiter: RateLimiter | None = None,
        rate_limit_enabled: bool = False,
    ):
        self.app = app
        self._max_body_bytes = max_body_bytes
        self._limiter = rate_limiter
        self._rate_limit_enabled = rate_limit_enabled and rate_limiter is not None

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = time.monotonic()
        request_id = (
            _sanitize_request_id(_header(scope, "x-request-id"))
            or uuid.uuid4().hex[:16]
        )
        token = REQUEST_ID_CTX.set(request_id)
        state = scope.setdefault("state", {})
        state.setdefault("stages", {})
        route = normalize_route(scope.get("path", ""))
        # Single completion-accounting record per request (M0–M6 review P2):
        # every path — normal, 4xx, 413/429 rejects, exceptions — converges
        # here exactly once.
        accounted = {"done": False}

        def finish(status_code: int) -> None:
            if accounted["done"]:
                return
            accounted["done"] = True
            duration = time.monotonic() - started
            HTTP_REQUESTS_TOTAL.labels(
                method=scope["method"], route=route, status=str(status_code)
            ).inc()
            HTTP_REQUEST_DURATION.labels(route=route).observe(duration)
            stages = state.get("stages") or {}
            logger.log(
                logging.INFO,
                "request_failed" if status_code == 500 else "request_completed",
                extra={
                    "route": route,
                    "status": status_code,
                    "duration_seconds": round(duration, 6),
                    **{
                        f"stage_{k}_seconds": round(v, 6)
                        for k, v in sorted(stages.items())
                    },
                },
                exc_info=(status_code == 500),
            )

        async def send_wrapper(message) -> None:
            if message["type"] == "http.response.start":
                state["response_status"] = message["status"]
                state["response_started"] = True
                headers = list(message.get("headers") or [])
                names = {name.decode("latin-1").lower() for name, _ in headers}
                if "x-request-id" not in names:
                    # Upstream-provided X-Request-ID wins (M1.1 policy).
                    headers.append((b"x-request-id", request_id.encode("latin-1")))
                message = {**message, "headers": headers}
            await send(message)

        async def reject(
            status: int, code: str, message: str, err_type: str | None = None,
            **extra_headers,
        ) -> None:
            response = _error_response(
                status, code, message, err_type=err_type, headers=extra_headers or None
            )
            await response(scope, receive, send_wrapper)

        try:
            # --- pre-flight checks ------------------------------------------
            content_length = _header(scope, "content-length")
            if (
                content_length is not None
                and content_length.isdigit()
                and int(content_length) > self._max_body_bytes
            ):
                logger.warning("request_too_large", extra={"route": route})
                await reject(413, "request_too_large",
                             f"request body exceeds {self._max_body_bytes} bytes")
                return

            if self._rate_limit_enabled and route not in _RATE_LIMIT_EXEMPT_ROUTES:
                # Two-dimension admission (post-04592c0 review §2): the
                # client/IP bucket aggregates ALL of a host's API traffic,
                # so rotating X-Conversation-ID cannot mint fresh quota; the
                # conversation bucket (when the header is present) keeps its
                # per-conversation isolation. One rejection = one metric.
                # Operational endpoints (health/metrics) are exempt: liveness
                # must never be throttled by chat quota.
                conversation_header = _header(scope, "x-conversation-id")
                decision = self._limiter.admit(
                    client_key=_client_host(scope),
                    conversation_key=conversation_header,
                )
                if not decision.allowed:
                    RATE_LIMIT_REJECTS_TOTAL.inc()
                    retry_after = str(decision.retry_after)
                    logger.warning(
                        "rate_limited",
                        extra={
                            "route": route,
                            "scope": decision.scope,
                            "retry_after": retry_after,
                        },
                    )
                    await reject(429, "rate_limit_exceeded", "rate limit exceeded",
                                 err_type="rate_limit_error",
                                 **{"Retry-After": retry_after})
                    return

            # --- application -------------------------------------------------
            if content_length is None:
                # No Content-Length (chunked): pre-buffer up to the cap so
                # enforcement happens BEFORE the app reads anything.
                messages, outcome = await _drain_body(receive, self._max_body_bytes)
                if outcome == "oversized":
                    logger.warning("request_too_large", extra={"route": route})
                    await reject(413, "request_too_large",
                                 f"request body exceeds {self._max_body_bytes} bytes")
                    return
                if outcome == "disconnected":
                    # Client vanished mid-upload: never reach the application,
                    # never fabricate a normal end-of-body on its behalf.
                    # Tracked with a dedicated counter — 499 is not a real
                    # HTTP status and must not appear in request metrics.
                    CLIENT_DISCONNECTS_TOTAL.inc()
                    state["response_status"] = None  # no HTTP status to count
                    logger.info(
                        "client_disconnected_during_body", extra={"route": route}
                    )
                    return
                receive = _replaying_receive(messages, receive)
            try:
                await self.app(scope, receive, send_wrapper)
            except Exception:
                # Response already started (streaming): the client saw its
                # real status; track the abortion separately instead of
                # inventing a fake HTTP status (hardening P2.1).
                if state.get("response_started"):
                    HTTP_STREAMS_ABORTED_TOTAL.inc()
                else:
                    state["response_status"] = 500
                raise
        except Exception:
            raise
        finally:
            # Single completion-accounting record for EVERY path (review P2).
            # Client-disconnects carry no HTTP status: they are tracked by the
            # dedicated disconnect counter and excluded from request metrics.
            status = state.get("response_status")
            if status is not None:
                finish(int(status))
            REQUEST_ID_CTX.reset(token)


def record_stage(request, name: str, started_monotonic: float) -> None:
    """Record one named pipeline stage for the latency breakdown."""
    stages = getattr(getattr(request, "state", None), "stages", None)
    if stages is None:
        stages = {}
        request.state.stages = stages
    stages[name] = time.monotonic() - started_monotonic


async def _drain_body(receive, max_body_bytes: int) -> tuple[list[dict], str]:
    """Buffer request messages up to max_body_bytes.

    Returns (buffered_messages, outcome) where outcome is one of:

    - "ok":           body fully within the cap; messages are replayable;
    - "oversized":    cap violated -> caller answers 413 WITHOUT consuming any
                      further message (immediate stop, no arbitrary drain);
    - "disconnected": client went away mid-body -> caller must NOT invoke the
                      application and must NEVER synthesize a normal
                      end-of-body for it (disconnect != normal completion).

    The ASGI server owns the connection afterwards in every non-"ok" case.
    """
    buffered: list[dict] = []
    total = 0
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return [], "disconnected"
        if message["type"] != "http.request":
            continue
        total += len(message.get("body") or b"")
        if total > max_body_bytes:
            # Hard stop: no further receive() consumption of an oversized body.
            return [], "oversized"
        buffered.append(message)
        if not message.get("more_body"):
            return buffered, "ok"


def _replaying_receive(messages: list[dict], original_receive):
    """Replay pre-buffered request messages, then defer to the source."""
    queue = list(messages)

    async def wrapped_receive():
        if queue:
            return queue.pop(0)
        return await original_receive()

    return wrapped_receive


def _header(scope, name: str) -> str | None:
    target = name.encode("latin-1")
    for key, value in scope.get("headers") or []:
        if key == target:
            return value.decode("latin-1")
    return None


def _client_host(scope) -> str:
    client = scope.get("client") or ("unknown", 0)
    return str(client[0])
