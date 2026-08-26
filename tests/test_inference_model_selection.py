"""M8 regression tests for client-owned inference model selection."""

from __future__ import annotations

import json

import httpx
import pytest

from context_proxy.config import EndpointSettings, Settings
from context_proxy.providers.llm import OpenAICompatibleLLMProvider


class _TestAsyncStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b"data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_complete_forwards_client_model_without_server_override() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "x", "choices": []})

    client = httpx.AsyncClient(
        base_url="http://provider.test/v1",
        transport=httpx.MockTransport(handler),
    )
    provider = OpenAICompatibleLLMProvider(
        EndpointSettings(base_url="http://provider.test/v1"), client=client
    )
    try:
        await provider.complete(
            {
                "model": "model-a",
                "messages": [{"role": "user", "content": "hello"}],
            }
        )
        await provider.complete(
            {
                "model": "model-b",
                "messages": [{"role": "user", "content": "hello"}],
            }
        )
    finally:
        await provider.aclose()

    assert [payload["model"] for payload in seen] == ["model-a", "model-b"]


@pytest.mark.asyncio
async def test_stream_forwards_client_model_without_server_override() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_TestAsyncStream(),
        )

    client = httpx.AsyncClient(
        base_url="http://provider.test/v1",
        transport=httpx.MockTransport(handler),
    )
    provider = OpenAICompatibleLLMProvider(
        EndpointSettings(base_url="http://provider.test/v1"), client=client
    )
    try:
        stream = await provider.open_stream(
            {
                "model": "stream-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            }
        )
        chunks = [chunk async for chunk in stream.iter_bytes()]
        assert b"".join(chunks) == b"data: [DONE]\n\n"
    finally:
        await provider.aclose()

    assert seen[0]["model"] == "stream-model"


def test_inference_settings_have_no_model_configuration() -> None:
    settings = Settings()

    assert not hasattr(settings.inference, "model")
    assert settings.embeddings.model == "embedding-model"
    assert settings.compact.model == "compact-model"


def test_embedding_nested_env_override_preserves_other_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDINGS__API_KEY", "test-secret")

    settings = Settings()

    assert settings.embeddings.api_key == "test-secret"
    assert settings.embeddings.base_url == "http://localhost:8002/v1"
    assert settings.embeddings.model == "embedding-model"
    assert settings.embeddings.timeout_seconds == 600.0


def test_compact_nested_env_override_preserves_other_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMPACT__API_KEY", "test-secret")

    settings = Settings()

    assert settings.compact.api_key == "test-secret"
    assert settings.compact.base_url == "http://localhost:8001/v1"
    assert settings.compact.model == "compact-model"
    assert settings.compact.timeout_seconds == 600.0
