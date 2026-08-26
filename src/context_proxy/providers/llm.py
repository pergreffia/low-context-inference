from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from context_proxy.config import EndpointSettings, ResilienceSettings
from context_proxy.observability.metrics import DEGRADATIONS_TOTAL, UPSTREAM_DURATION
from context_proxy.providers.base import LLMStream
from context_proxy.providers.errors import (
    UpstreamHTTPError,
    UpstreamUnavailable,
    map_upstream_error,
)
from context_proxy.providers.headers import (
    filter_response_headers,
    get_header,
)
from context_proxy.providers.resilience import CircuitBreaker, with_retries

logger = logging.getLogger(__name__)

# Retryable transport failures (post-0876b10 review §3): ONLY errors that are
# provably pre-send. A POST whose request may already have been delivered and
# accepted (ReadTimeout, WriteError, RemoteProtocolError, ...) must NEVER be
# retried — a second attempt could duplicate the inference.
RETRYABLE_TRANSPORT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout)


class OpenAICompatibleLLMProvider:
    """Thin passthrough client for any OpenAI-compatible endpoint.

    Responses are treated as opaque protocol data: bodies are never parsed or
    rewritten (master prompt §6, §30). Header forwarding follows the explicit
    policy in providers.headers.

    M5 resilience: transport failures before any response byte are retried a
    bounded number of times with full-jitter backoff, guarded by a circuit
    breaker that fails fast while the endpoint is down. Upstream HTTP error
    RESPONSES are answers and are neither retried nor counted as breaker
    failures. Streaming: only the pre-stream send is protected.
    """

    def __init__(
        self,
        settings: EndpointSettings,
        client: httpx.AsyncClient | None = None,
        *,
        resilience: ResilienceSettings | None = None,
        breaker: CircuitBreaker | None = None,
        route_label: str = "/v1/chat/completions",
    ):
        self._settings = settings
        self._client = client or httpx.AsyncClient(
            base_url=settings.base_url,
            timeout=httpx.Timeout(settings.timeout_seconds),
        )
        if settings.api_key:
            self._client.headers["Authorization"] = f"Bearer {settings.api_key}"
        self._resilience = resilience or ResilienceSettings()
        self._breaker = breaker or CircuitBreaker(
            failure_threshold=self._resilience.breaker_failure_threshold,
            reset_seconds=self._resilience.breaker_reset_seconds,
        )
        self._route_label = route_label

    @property
    def model(self) -> str | None:
        return self._settings.model

    async def aclose(self) -> None:
        await self._client.aclose()

    async def list_models(self) -> tuple[int, dict[str, str], bytes]:
        request = self._client.build_request("GET", "/models")
        response = await self._send(request)
        return self._pack(response)

    async def complete(self, payload: dict[str, Any]) -> tuple[int, dict[str, str], bytes]:
        request = self._client.build_request(
            "POST", "/chat/completions", json=self._prepared(payload)
        )
        response = await self._send(request)
        return self._pack(response)

    async def open_stream(self, payload: dict[str, Any]) -> LLMStream:
        request = self._client.build_request(
            "POST", "/chat/completions", json=self._prepared(payload)
        )
        response = await self._resilient_send(request, streaming=True)
        if response.status_code >= 400:
            body = await response.aread()
            raise UpstreamHTTPError(
                response.status_code,
                body,
                content_type=response.headers.get("content-type", "application/json"),
                headers=filter_response_headers(response.headers, keep_content_encoding=False),
            )
        return UpstreamLLMStream(response)

    async def _send(self, request: httpx.Request) -> httpx.Response:
        """Send a buffered request; upstream HTTP errors become UpstreamHTTPError.

        Shared by streaming and non-streaming paths so error handling stays
        consistent (M1.1 §1). Transport failures are retried/breakered here.
        """
        response = await self._resilient_send(request)
        if response.status_code >= 400:
            raise UpstreamHTTPError(
                response.status_code,
                response.content,
                content_type=response.headers.get("content-type", "application/json"),
                headers=filter_response_headers(response.headers, keep_content_encoding=False),
            )
        return response

    async def _resilient_send(
        self, first_request: httpx.Request, *, streaming: bool = False
    ) -> httpx.Response:
        """Send with breaker + bounded transport retries (M5).

        A fresh buffered request is built per attempt (httpx requests are not
        safely reusable after a send). The streaming path reuses the original
        request object — safe because `json=` payloads are encoded to immutable
        bytes at build time; retries there only cover the pre-open phase.

        Retry/breaker policy (post-0876b10 review §3): only provably pre-send
        failures (ConnectError/ConnectTimeout) are retried and counted as
        breaker failures. Any other transport error — ReadTimeout, WriteError,
        RemoteProtocolError, ... — may have reached the provider already, so
        it is NEVER retried and does NOT trip the breaker; it still surfaces
        to the client as the standard upstream-unavailable contract. An HTTP
        response — even 5xx — is an answer and closes the breaker.

        Probe lifecycle (final hardening pass): a HALF_OPEN probe reservation
        is ALWAYS released — classified outcomes via record_success/failure,
        everything else (cancellation on client disconnect, unexpected errors)
        via the idempotent `release_probe()` safety net in the finally block,
        so the breaker can never stay pinned in HALF_OPEN.
        """
        import time as _time

        probe_allowed = self._breaker.allow_attempt()
        if not probe_allowed:
            DEGRADATIONS_TOTAL.labels(component="upstream_breaker_open").inc()
            raise UpstreamUnavailable(
                f"inference endpoint circuit breaker is {self._breaker.state}"
            )

        started = _time.monotonic()
        settings = self._resilience

        def build() -> httpx.Request:
            if streaming:
                return first_request
            return self._client.build_request(
                first_request.method,
                first_request.url,
                headers=first_request.headers,
                content=first_request.content,
            )

        def send(req: httpx.Request):
            send_kwargs = {"stream": True} if streaming else {}
            return self._client.send(req, **send_kwargs)

        try:
            response = await with_retries(
                lambda: send(build()),
                max_retries=settings.max_retries,
                backoff_base_seconds=settings.backoff_base_seconds,
                backoff_max_seconds=settings.backoff_max_seconds,
                retry_on=RETRYABLE_TRANSPORT_ERRORS,
            )
        except RETRYABLE_TRANSPORT_ERRORS as exc:
            # Provably never delivered: safe to count against the breaker.
            self._breaker.record_failure()
            logger.warning(
                "upstream_connect_failed",
                extra={"error": str(exc), "attempts": settings.max_retries + 1},
            )
            raise map_upstream_error(exc) from exc
        except httpx.HTTPError as exc:
            # Post-send failure: a retry could duplicate the POST. No breaker
            # accounting either — the endpoint answered the dial, the problem
            # happened later on THIS request.
            logger.warning(
                "upstream_transport_failed_no_retry",
                extra={
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "attempts": 1,
                },
            )
            raise map_upstream_error(exc) from exc
        except BaseException:
            # Cancellation (client disconnects cancel handler tasks) or any
            # unclassified error: re-raise unchanged — the finally below
            # guarantees the HALF_OPEN probe reservation is released.
            raise
        else:
            UPSTREAM_DURATION.labels(route=self._route_label).observe(
                _time.monotonic() - started
            )
            # Any HTTP answer means the endpoint is reachable.
            self._breaker.record_success()
            return response
        finally:
            # Idempotent safety net guaranteeing no leaked HALF_OPEN probe on
            # ANY exit path (classified outcomes already released their own
            # reservation inside record_success/record_failure).
            self._breaker.release_probe()

    def _prepared(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Resolve the configured inference model without mutating the caller's payload.

        Policy: INFERENCE__MODEL set -> override; unset -> client value preserved.
        """
        if not self._settings.model:
            return payload
        prepared = dict(payload)
        prepared["model"] = self._settings.model
        return prepared

    @staticmethod
    def _pack(response: httpx.Response) -> tuple[int, dict[str, str], bytes]:
        headers = filter_response_headers(response.headers, keep_content_encoding=False)
        return response.status_code, headers, response.content



class UpstreamLLMStream(LLMStream):
    """Incremental raw passthrough over an upstream streaming response.

    The upstream connection is closed exactly once: when the downstream
    iteration finishes or fails (including client disconnects, which surface
    as generator cancellation).
    """

    def __init__(self, response: httpx.Response):
        self._response = response
        self._closed = False

    @property
    def status_code(self) -> int:
        return self._response.status_code

    @property
    def media_type(self) -> str:
        return get_header(self._response.headers, "content-type") or "text/event-stream"

    def passthrough_headers(self) -> dict[str, str]:
        return filter_response_headers(self._response.headers, keep_content_encoding=True)

    async def iter_bytes(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._response.aiter_raw():
                yield chunk
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            await self._response.aclose()
