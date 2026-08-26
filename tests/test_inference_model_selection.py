"""M8 regression tests for client-owned inference model selection."""

from __future__ import annotations

import json

import httpx
import pytest

from context_proxy.config import EndpointSettings, Settings
from context_proxy.providers.llm import OpenAICompatibleLLMProvider


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
            content=b"data: [DONE]\n\n",
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
        assert b"".join([chunk async for chunk in stream.iter_bytes()]) == b"data: [DONE]\n\n"
    finally:
        await provider.aclose()

    assert seen[0]["model"] == "stream-model"


def test_inference_settings_have_no_model_configuration() -> None:
    settings = Settings()

    assert not hasattr(settings.inference, "model")
    assert settings.embeddings.model == "embedding-model"
    assert settings.compact.model == "compact-model"
