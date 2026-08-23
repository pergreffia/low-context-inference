from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from context_proxy.config import EndpointSettings
from context_proxy.providers.errors import UpstreamHTTPError, map_upstream_error

logger = logging.getLogger(__name__)

PASSTHROUGH_RESPONSE_HEADERS = ("x-request-id", "openai-organization", "openai-processing-ms")


class OpenAICompatibleLLMProvider:
    """Thin passthrough client for any OpenAI-compatible endpoint.

    Responses are treated as opaque protocol data: bodies are never parsed or
    rewritten (master prompt §6, §30).
    """

    def __init__(self, settings: EndpointSettings, client: httpx.AsyncClient | None = None):
        self._settings = settings
        self._client = client or httpx.AsyncClient(
            base_url=settings.base_url,
            timeout=httpx.Timeout(settings.timeout_seconds),
        )
        if settings.api_key:
            self._client.headers["Authorization"] = f"Bearer {settings.api_key}"

    @property
    def model(self) -> str | None:
        return self._settings.model

    async def aclose(self) -> None:
        await self._client.aclose()

    async def list_models(self) -> tuple[int, dict[str, str], bytes]:
        request = self._client.build_request("GET", "/models")
        try:
            response = await self._client.send(request)
        except httpx.HTTPError as exc:
            raise map_upstream_error(exc) from exc
        return self._pack(response)

    async def complete(self, payload: dict[str, Any]) -> tuple[int, dict[str, str], bytes]:
        request = self._client.build_request("POST", "/chat/completions", json=payload)
        try:
            response = await self._client.send(request)
        except httpx.HTTPError as exc:
            raise map_upstream_error(exc) from exc
        return self._pack(response)

    async def open_stream(self, payload: dict[str, Any]) -> AsyncIterator[bytes]:
        request = self._client.build_request("POST", "/chat/completions", json=payload)
        try:
            response = await self._client.send(request, stream=True)
        except httpx.HTTPError as exc:
            raise map_upstream_error(exc) from exc
        if response.status_code >= 400:
            body = await response.aread()
            content_type = response.headers.get("content-type", "application/json")
            status_code = response.status_code
            await response.aclose()
            raise UpstreamHTTPError(status_code, body, content_type)
        return _StreamIterator(response)

    @staticmethod
    def _pack(response: httpx.Response) -> tuple[int, dict[str, str], bytes]:
        headers = {
            name: value
            for name, value in response.headers.items()
            if name.lower() in PASSTHROUGH_RESPONSE_HEADERS
        }
        return response.status_code, headers, response.content


class _StreamIterator:
    """Wraps a streaming upstream response for incremental passthrough."""

    def __init__(self, response: httpx.Response):
        self._response = response

    @property
    def status_code(self) -> int:
        return self._response.status_code

    @property
    def media_type(self) -> str:
        return self._response.headers.get("content-type", "text/event-stream")

    def passthrough_headers(self) -> dict[str, str]:
        return {
            name: value
            for name, value in self._response.headers.items()
            if name.lower() in PASSTHROUGH_RESPONSE_HEADERS
        }

    async def iter_bytes(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._response.aiter_raw():
                yield chunk
        finally:
            await self._response.aclose()
