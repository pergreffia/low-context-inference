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

        Compatibility gate (post-0876b10 review §4): a 409 Conflict only
        proves SOME collection exists — not that it matches THIS embedding
        model. On 409 the live config is fetched and compared (vector size,
        distance). Incompatible collections raise an explicit error: never a
        silent delete, never an automatic recreate (the index is derived, but
        destroying data on config drift must stay a deliberate operator
        action).
        """
        expected = {"size": vector_size, "distance": "Cosine"}
        async with self._ensure_lock:
            if self._collection_ready:
                return
            response = await self._client.put(
                f"/collections/{self._collection}",
                json={"vectors": {"size": vector_size, "distance": "Cosine"}},
            )
            if response.status_code == 409:
                actual = await self._fetch_collection_params()
                if actual != expected:
                    raise VectorStoreError(
                        f"qdrant collection '{self._collection}' exists with "
                        f"incompatible configuration: expected "
                        f"size={expected['size']} distance={expected['distance']}, "
                        f"found size={actual.get('size')} distance={actual.get('distance')}; "
                        "refusing to overwrite — migrate or delete it explicitly"
                    )
                self._collection_ready = True
                return
            if response.status_code not in (200, 201):
                response.raise_for_status()
            self._collection_ready = True

    async def _fetch_collection_params(self) -> dict[str, Any]:
        """GET the existing collection's vector params; error is explicit."""
        try:
            response = await self._client.get(f"/collections/{self._collection}")
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise VectorStoreError(
                f"qdrant collection '{self._collection}' conflict check failed: {exc}",
                cause=exc,
            ) from exc
        result = body.get("result") if isinstance(body, dict) else None
        params = (
            result.get("config", {}).get("params", {}).get("vectors", {})
            if isinstance(result, dict)
            else {}
        )
        if isinstance(params, dict):
            size = params.get("size")
            distance = params.get("distance")
        else:
            # Named-vectors layouts are unsupported by this store; treat any
            # non-{size,distance} shape as incompatible rather than guessing.
            size, distance = None, None
        if not isinstance(size, int) or not isinstance(distance, str):
            raise VectorStoreError(
                f"qdrant collection '{self._collection}' reports no plain "
                f"single-vector configuration; refusing to guess compatibility"
            )
        return {"size": size, "distance": distance}

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
