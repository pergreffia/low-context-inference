from __future__ import annotations

import httpx
import pytest

from context_proxy.config import ModelEndpointSettings
from context_proxy.memory.embeddings import OpenAICompatibleEmbeddingProvider
from context_proxy.memory.errors import EmbeddingProviderError
from context_proxy.memory.qdrant import QdrantVectorStore


@pytest.fixture
def embedder():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        requests.append(request)
        body = json.loads(request.content)
        inputs = body["input"]
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": i, "embedding": [float(len(t) % 7), 1.0, 0.5]}
                    for i, t in enumerate(inputs)
                ]
            },
        )

    provider = OpenAICompatibleEmbeddingProvider(
        ModelEndpointSettings(base_url="http://embed.test/v1", model="emb-1"),
        client=httpx.AsyncClient(
            base_url="http://embed.test/v1", transport=httpx.MockTransport(handler)
        ),
    )
    return provider, requests


async def test_embed_sends_model_and_returns_vectors(embedder):
    provider, requests = embedder
    vectors = await provider.embed(["hello", "world!"])
    assert len(vectors) == 2
    assert vectors[0] == [5.0, 1.0, 0.5]  # len('hello') % 7
    body = __import__("json").loads(requests[0].content)
    assert body["model"] == "emb-1"
    assert body["input"] == ["hello", "world!"]


async def test_embed_error_propagates_for_degradation_handling():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "overloaded"})

    provider = OpenAICompatibleEmbeddingProvider(
        ModelEndpointSettings(base_url="http://embed.test/v1", model="emb-1"),
        client=httpx.AsyncClient(
            base_url="http://embed.test/v1",
            transport=httpx.MockTransport(handler),
        ),
    )
    with pytest.raises(EmbeddingProviderError) as excinfo:
        await provider.embed(["x"])
    # expected provider failure keeps its cause chain for diagnostics
    assert isinstance(excinfo.value.__cause__, httpx.HTTPStatusError)


def test_qdrant_upsert_bootstraps_collection_and_filters_search():
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/collections/ctx":
            return httpx.Response(200, json={"result": "ok"})
        if request.url.path == "/collections/ctx/points":
            return httpx.Response(200, json={"result": {"status": "ok"}})
        if request.url.path == "/collections/ctx/points/search":
            import json

            body = json.loads(request.content)
            assert body["filter"]["must"][0]["key"] == "conversation_id"
            return httpx.Response(
                200,
                json={"result": [{"id": "p1", "score": 0.9, "payload": {"kind": "chunk"}}]},
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    store = QdrantVectorStore(
        "http://qdrant.test",
        collection="ctx",
        client=httpx.AsyncClient(base_url="http://qdrant.test", transport=transport),
    )

    import asyncio

    async def _run():
        await store.upsert([{"id": "p1", "vector": [0.1, 0.2], "payload": {"a": 1}}], vector_size=2)
        hits = await store.search([0.1, 0.2], limit=5, conversation_id="conv-1")
        return hits

    hits = asyncio.run(_run())
    assert ("PUT", "/collections/ctx") in calls
    assert ("PUT", "/collections/ctx/points") in calls
    assert ("POST", "/collections/ctx/points/search") in calls
    assert hits == [{"id": "p1", "score": 0.9, "payload": {"kind": "chunk"}}]
