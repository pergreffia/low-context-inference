"""Qdrant collection compatibility validation (post-0876b10 review §4).

A 409 on create only proves SOME collection exists. The store must verify
vector size + distance before declaring readiness; incompatible collections
raise an explicit error and are NEVER deleted or recreated.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from context_proxy.memory.errors import VectorStoreError
from context_proxy.memory.qdrant import QdrantVectorStore

COLLECTION_PATH = "/collections/context_proxy"


def make_store(handler) -> QdrantVectorStore:
    return QdrantVectorStore(
        "http://qdrant.test",
        client=httpx.AsyncClient(
            base_url="http://qdrant.test", transport=httpx.MockTransport(handler)
        ),
    )


def collection_body(size: int, distance: str = "Cosine") -> dict:
    return {
        "result": {
            "config": {"params": {"vectors": {"size": size, "distance": distance}}}
        }
    }


def put_conflicts_then_get(body: dict, calls: list[str]):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == COLLECTION_PATH:
            if request.method == "PUT":
                calls.append("PUT")
                return httpx.Response(409, json={"err": "exists"})
            calls.append("GET")
            return httpx.Response(200, json=body)
        if request.url.path.endswith("/points"):
            calls.append("points")
            return httpx.Response(200, json={})
        return httpx.Response(404)

    return handler


class TestCollectionCompatibilityGate:
    def test_absent_collection_created_and_ready(self):
        calls: list[str] = []

        def handler(request):
            if request.method == "PUT" and request.url.path == COLLECTION_PATH:
                calls.append("PUT-create")
                return httpx.Response(200, json={})
            calls.append(request.method + request.url.path)
            return httpx.Response(200, json={})

        store = make_store(handler)
        asyncio.run(store.ensure_collection(3))
        assert store._collection_ready is True
        assert calls == ["PUT-create"]

    def test_compatible_existing_collection_becomes_ready(self):
        calls: list[str] = []
        store = make_store(put_conflicts_then_get(collection_body(3), calls))
        asyncio.run(store.ensure_collection(3))
        assert store._collection_ready is True
        assert calls == ["PUT", "GET"]               # verified via GET

    def test_incompatible_size_raises_no_delete_no_ready(self):
        calls: list[str] = []
        store = make_store(put_conflicts_then_get(collection_body(1536), calls))
        with pytest.raises(VectorStoreError) as excinfo:
            asyncio.run(store.ensure_collection(3))
        message = str(excinfo.value)
        assert "incompatible" in message
        assert "size=3" in message and "size=1536" in message
        assert calls == ["PUT", "GET"]               # GET only; no DELETE anywhere
        assert store._collection_ready is False      # never marked ready
        assert not any("DELETE" in c for c in calls)

    def test_incompatible_distance_raises(self):
        calls: list[str] = []
        body = collection_body(3, distance="Euclid")
        store = make_store(put_conflicts_then_get(body, calls))
        with pytest.raises(VectorStoreError) as excinfo:
            asyncio.run(store.ensure_collection(3))
        assert "distance=Cosine" in str(excinfo.value)
        assert "Euclid" in str(excinfo.value)
        assert store._collection_ready is False

    def test_get_failure_is_explicit_error(self):
        calls: list[str] = []

        def handler(request):
            if request.method == "PUT":
                calls.append("PUT")
                return httpx.Response(409, json={})
            calls.append("GET")
            return httpx.Response(500, json={"err": "boom"})

        store = make_store(handler)
        with pytest.raises(VectorStoreError):
            asyncio.run(store.ensure_collection(7))
        assert calls == ["PUT", "GET"]
        assert store._collection_ready is False

    def test_409_with_nonstandard_vector_layout_rejected(self):
        """Named-vectors shapes are not guessed — explicit failure."""
        body = {
            "result": {
                "config": {"params": {"vectors": {"text": {"size": 3, "distance": "Cosine"}}}}
            }
        }
        calls: list[str] = []
        store = make_store(put_conflicts_then_get(body, calls))
        with pytest.raises(VectorStoreError):
            asyncio.run(store.ensure_collection(3))
        assert store._collection_ready is False

    def test_setup_idempotent_after_validation(self):
        calls: list[str] = []
        store = make_store(put_conflicts_then_get(collection_body(5), calls))
        asyncio.run(store.ensure_collection(5))
        asyncio.run(store.ensure_collection(5))      # cached -> no more calls
        assert calls == ["PUT", "GET"]
        asyncio.run(store.upsert([{"id": "a", "vector": [0.1] * 5, "payload": {}}], 5))
        assert calls[-1] == "points"

    def test_upsert_against_incompatible_collection_fails_loudly(self):
        """Incompatibility surfaces at first use instead of corrupting data."""
        calls: list[str] = []
        store = make_store(put_conflicts_then_get(collection_body(99), calls))
        with pytest.raises(VectorStoreError):
            asyncio.run(store.upsert([{"id": "a", "vector": [0.1] * 3, "payload": {}}], 3))

    def test_error_message_contains_no_secrets(self):
        body = collection_body(1536)
        calls: list[str] = []
        store = make_store(put_conflicts_then_get(body, calls))
        with pytest.raises(VectorStoreError) as excinfo:
            asyncio.run(store.ensure_collection(3))
        blob = str(excinfo.value)
        for leak in ("password", "secret", "postgresql://"):
            assert leak not in blob
