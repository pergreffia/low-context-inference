"""Qdrant vector store (derived, rebuildable index — master prompt §32).

Minimal HTTP client: collection bootstrap, upsert, filtered search. All
payloads carry `conversation_id` so retrieval is conversation-scoped (§18).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    def __init__(
        self,
        base_url: str,
        collection: str = "context_proxy",
        client: httpx.AsyncClient | None = None,
    ):
        self._collection = collection
        self._client = client or httpx.AsyncClient(base_url=base_url)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def ping(self) -> None:
        response = await self._client.get("/collections")
        response.raise_for_status()

    async def ensure_collection(self, vector_size: int) -> None:
        """Idempotent bootstrap; Cosine distance for normalized semantic space."""
        response = await self._client.put(
            f"/collections/{self._collection}",
            json={"vectors": {"size": vector_size, "distance": "Cosine"}},
        )
        if response.status_code not in (200, 409):
            response.raise_for_status()

    async def upsert(
        self,
        points: Sequence[dict[str, Any]],
        vector_size: int,
    ) -> None:
        """points: [{id, vector, payload}]. Bootstraps the collection on demand."""
        if not points:
            return
        await self.ensure_collection(vector_size)
        response = await self._client.put(
            f"/collections/{self._collection}/points",
            json={"points": [
                {"id": p["id"], "vector": p["vector"], "payload": p["payload"]}
                for p in points
            ]},
        )
        response.raise_for_status()

    async def search(
        self,
        vector: Sequence[float],
        limit: int,
        conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return [{id, score, payload}] ordered by descending similarity."""
        body: dict[str, Any] = {"vector": list(vector), "limit": limit, "with_payload": True}
        if conversation_id is not None:
            body["filter"] = {
                "must": [
                    {
                        "key": "conversation_id",
                        "match": {"value": conversation_id},
                    }
                ]
            }
        response = await self._client.post(
            f"/collections/{self._collection}/points/search", json=body
        )
        response.raise_for_status()
        return response.json().get("result", [])
