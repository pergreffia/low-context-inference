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

        A fresh request is built per attempt: httpx requests are not safely
        reusable after a send. The breaker fails fast while OPEN. Only
        transport-level errors reach retry/breaker accounting; an HTTP
        response — even 5xx — proves the endpoint answered and closes the
        breaker.
        """
        import time as _time

        if not self._breaker.allow_attempt():
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
                retry_on=(httpx.HTTPError,),
            )
        except httpx.HTTPError as exc:
            self._breaker.record_failure()
            logger.warning(
                "upstream_transport_failed",
                extra={"error": str(exc), "attempts": settings.max_retries + 1},
            )
            raise map_upstream_error(exc) from exc

        UPSTREAM_DURATION.labels(route=self._route_label).observe(
            _time.monotonic() - started
        )
        # Any HTTP answer means the endpoint is reachable.
        self._breaker.record_success()
        return response

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
