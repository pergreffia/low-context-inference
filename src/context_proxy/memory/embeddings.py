"""OpenAI-compatible embedding provider (master prompt §28).

Independent from inference/compact endpoints; replaceable via configuration.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import httpx

from context_proxy.config import EndpointSettings
from context_proxy.memory.errors import EmbeddingProviderError

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
        """Embed texts; raises EmbeddingProviderError on expected provider
        failures (transport/timeout/HTTP status/malformed payload) so callers
        can degrade deliberately. Programming errors propagate untouched.
        """
        payload: dict = {"input": list(texts)}
        if self._settings.model:
            payload["model"] = self._settings.model
        try:
            response = await self._client.post("/embeddings", json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise EmbeddingProviderError(
                f"embedding endpoint failed: {exc}", cause=exc
            ) from exc
        except ValueError as exc:  # non-JSON body
            raise EmbeddingProviderError(
                f"embedding endpoint returned malformed body: {exc}", cause=exc
            ) from exc
        if not isinstance(data, dict) or not isinstance(data.get("data"), list):
            raise EmbeddingProviderError("embedding response missing 'data' array")
        embeddings: list[list[float]] = []
        for item in data["data"]:
            if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
                raise EmbeddingProviderError(
                    "embedding response items must carry an embedding array"
                )
            embeddings.append(item["embedding"])
        return embeddings
