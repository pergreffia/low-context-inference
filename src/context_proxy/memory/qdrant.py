"""Qdrant vector store (derived, rebuildable index — master prompt §32).

Minimal HTTP client: collection bootstrap, upsert, filtered search. All
payloads carry `conversation_id` so retrieval is conversation-scoped (§18).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

import httpx

from context_proxy.memory.errors import VectorStoreError

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
        # Process-local readiness cache (hardening P2.3): after one successful
        # ensure_collection the PUT is skipped until a collection-not-found
        # condition invalidates it. Lock serializes concurrent initialization.
        self._collection_ready = False
        self._ensure_lock = asyncio.Lock()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def ping(self) -> None:
        response = await self._client.get("/collections")
        response.raise_for_status()

    async def ensure_collection(self, vector_size: int) -> None:
        """Idempotent bootstrap; Cosine distance for normalized semantic space.

        Cached after first success; concurrent callers are serialized by an
        asyncio lock and reuse the result.
        """
        async with self._ensure_lock:
            if self._collection_ready:
                return
            response = await self._client.put(
                f"/collections/{self._collection}",
                json={"vectors": {"size": vector_size, "distance": "Cosine"}},
            )
            if response.status_code not in (200, 409):
                response.raise_for_status()
            self._collection_ready = True

    async def upsert(
        self,
        points: Sequence[dict[str, Any]],
        vector_size: int,
    ) -> None:
        """points: [{id, vector, payload}]. Bootstraps the collection on demand.

        Uses the readiness cache; a collection-not-found (404) invalidates the
        cache and retries the bootstrap exactly once.
        """
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
        if response.status_code == 404:
            # Collection disappeared underneath us: invalidate cache and
            # re-bootstrap once.
            self._collection_ready = False
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
        """Return [{id, score, payload}] ordered by descending similarity.

        Transport/timeouts/HTTP failures raise VectorStoreError so callers can
        degrade deliberately; unexpected errors (bad payloads, bugs) propagate
        untouched (M4 final review §1).
        """
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
        try:
            response = await self._client.post(
                f"/collections/{self._collection}/points/search", json=body
            )
            response.raise_for_status()
            result = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            # ValueError covers non-JSON bodies; anything else is a real bug.
            raise VectorStoreError(
                f"qdrant search failed: {exc}", cause=exc
            ) from exc
        if not isinstance(result, dict) or not isinstance(result.get("result", []), list):
            raise VectorStoreError("qdrant search returned malformed payload")
        return result.get("result", [])
