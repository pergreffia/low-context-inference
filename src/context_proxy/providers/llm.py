from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from context_proxy.config import EndpointSettings, ResilienceSettings
from context_proxy.observability.metrics import DEGRADATIONS_TOTAL, UPSTREAM_DURATION
from context_proxy.providers.base import LLMStream
from context_proxy.providers.errors import UpstreamHTTPError, UpstreamUnavailable, map_upstream_error
from context_proxy.providers.headers import filter_response_headers, get_header
from context_proxy.providers.resilience import CircuitBreaker, with_retries

logger = logging.getLogger(__name__)
RETRYABLE_TRANSPORT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout)


class OpenAICompatibleLLMProvider:
    """Thin passthrough client for any OpenAI-compatible endpoint."""

    def __init__(self, settings: EndpointSettings, client: httpx.AsyncClient | None = None, *, resilience: ResilienceSettings | None = None, breaker: CircuitBreaker | None = None, route_label: str = "/v1/chat/completions"):
        self._settings = settings
        self._client = client or httpx.AsyncClient(base_url=settings.base_url, timeout=httpx.Timeout(settings.timeout_seconds))
        if settings.api_key:
            self._client.headers["Authorization"] = f"Bearer {settings.api_key}"
        self._resilience = resilience or ResilienceSettings()
        self._breaker = breaker or CircuitBreaker(failure_threshold=self._resilience.breaker_failure_threshold, reset_seconds=self._resilience.breaker_reset_seconds)
        self._route_label = route_label

    async def aclose(self) -> None:
        await self._client.aclose()

    async def list_models(self) -> tuple[int, dict[str, str], bytes]:
        request = self._client.build_request("GET", "/models")
        logger.debug("upstream_request", extra={"method": request.method, "url": str(request.url), "headers": dict(request.headers)})
        response = await self._send(request)
        return self._pack(response)

    async def complete(self, payload: dict[str, Any]) -> tuple[int, dict[str, str], bytes]:
        request = self._client.build_request("POST", "/chat/completions", json=payload)
        logger.debug("upstream_request_built", extra={"method": request.method, "url": str(request.url), "headers": dict(request.headers), "payload": payload, "request_body": request.content})
        response = await self._send(request)
        return self._pack(response)

    async def open_stream(self, payload: dict[str, Any]) -> LLMStream:
        request = self._client.build_request("POST", "/chat/completions", json=payload)
        logger.debug("upstream_stream_request_built", extra={"method": request.method, "url": str(request.url), "headers": dict(request.headers), "payload": payload, "request_body": request.content})
        response = await self._resilient_send(request, streaming=True)
        logger.debug("upstream_stream_response", extra={"status_code": response.status_code, "headers": dict(response.headers)})
        if response.status_code >= 400:
            body = await response.aread()
            logger.debug("upstream_stream_error_body", extra={"status_code": response.status_code, "body": body})
            raise UpstreamHTTPError(response.status_code, body, content_type=response.headers.get("content-type", "application/json"), headers=filter_response_headers(response.headers, keep_content_encoding=False))
        return UpstreamLLMStream(response)

    async def _send(self, request: httpx.Request) -> httpx.Response:
        response = await self._resilient_send(request)
        if response.status_code >= 400:
            raise UpstreamHTTPError(response.status_code, response.content, content_type=response.headers.get("content-type", "application/json"), headers=filter_response_headers(response.headers, keep_content_encoding=False))
        return response

    async def _resilient_send(self, first_request: httpx.Request, *, streaming: bool = False) -> httpx.Response:
        import time as _time
        probe_allowed = self._breaker.allow_attempt()
        if not probe_allowed:
            DEGRADATIONS_TOTAL.labels(component="upstream_breaker_open").inc()
            raise UpstreamUnavailable(f"inference endpoint circuit breaker is {self._breaker.state}")
        started = _time.monotonic()
        settings = self._resilience

        def build() -> httpx.Request:
            if streaming:
                return first_request
            return self._client.build_request(first_request.method, first_request.url, headers=first_request.headers, content=first_request.content)

        def send(req: httpx.Request):
            send_kwargs = {"stream": True} if streaming else {}
            logger.debug("upstream_send_attempt", extra={"method": req.method, "url": str(req.url), "headers": dict(req.headers), "body": req.content, "streaming": streaming})
            return self._client.send(req, **send_kwargs)

        try:
            response = await with_retries(lambda: send(build()), max_retries=settings.max_retries, backoff_base_seconds=settings.backoff_base_seconds, backoff_max_seconds=settings.backoff_max_seconds, retry_on=RETRYABLE_TRANSPORT_ERRORS)
        except RETRYABLE_TRANSPORT_ERRORS as exc:
            self._breaker.record_failure()
            logger.warning("upstream_connect_failed", extra={"error": str(exc), "attempts": settings.max_retries + 1})
            raise map_upstream_error(exc) from exc
        except httpx.HTTPError as exc:
            logger.warning("upstream_transport_failed_no_retry", extra={"error": str(exc), "error_type": type(exc).__name__, "attempts": 1})
            raise map_upstream_error(exc) from exc
        except BaseException:
            raise
        else:
            UPSTREAM_DURATION.labels(route=self._route_label).observe(_time.monotonic() - started)
            self._breaker.record_success()
            logger.debug("upstream_response_received", extra={"status_code": response.status_code, "headers": dict(response.headers), "streaming": streaming})
            return response
        finally:
            self._breaker.release_probe()

    @staticmethod
    def _pack(response: httpx.Response) -> tuple[int, dict[str, str], bytes]:
        headers = filter_response_headers(response.headers, keep_content_encoding=False)
        return response.status_code, headers, response.content


class UpstreamLLMStream(LLMStream):
    """Incremental raw passthrough over an upstream streaming response."""

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
                logger.debug("upstream_stream_chunk", extra={"size": len(chunk), "chunk": chunk})
                yield chunk
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            logger.debug("upstream_stream_closed")
            await self._response.aclose()
