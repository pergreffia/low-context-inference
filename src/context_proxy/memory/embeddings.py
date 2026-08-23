"""OpenAI-compatible embedding provider (master prompt §28).

Independent from inference/compact endpoints; replaceable via configuration.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import httpx

from context_proxy.config import EndpointSettings

logger = logging.getLogger(__name__)


class OpenAICompatibleEmbeddingProvider:
    def __init__(self, settings: EndpointSettings, client: httpx.AsyncClient | None = None):
        self._settings = settings
        self._client = client or httpx.AsyncClient(
            base_url=settings.base_url,
            timeout=httpx.Timeout(settings.timeout_seconds),
        )
        if settings.api_key:
            self._client.headers["Authorization"] = f"Bearer {settings.api_key}"

    async def aclose(self) -> None:
        await self._client.aclose()

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts; raises httpx/protocol errors to the caller."""
        payload: dict = {"input": list(texts)}
        if self._settings.model:
            payload["model"] = self._settings.model
        response = await self._client.post("/embeddings", json=payload)
        response.raise_for_status()
        data = response.json()["data"]
        return [item["embedding"] for item in data]
