from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from context_proxy.config import DatabaseSettings, EndpointSettings, ServerSettings, Settings
from context_proxy.main import create_app

UPSTREAM = "http://upstream.test/v1"

MODELS_RESPONSE = {
    "object": "list",
    "data": [{"id": "test-model", "object": "model", "owned_by": "upstream"}],
}

CHAT_RESPONSE = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
}

SSE_BODY = (
    b'data: {"id":"chatcmpl-2","choices":[{"delta":{"content":"he"}}]}\n\n'
    b"data: [DONE]\n\n"
)


def upstream_handler(
    captured: list[httpx.Request],
    *,
    models_status: int = 200,
    chat_status: int = 200,
) -> httpx.MockTransport:
    async def sse_stream() -> Any:
        yield SSE_BODY[:20]
        yield SSE_BODY[20:]

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(models_status, json=MODELS_RESPONSE)
        if request.url.path == "/v1/chat/completions":
            payload = json.loads(request.content)
            if payload.get("stream") is True:
                return httpx.Response(
                    200,
                    content=sse_stream(),
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(chat_status, json=CHAT_RESPONSE)
        return httpx.Response(404, json={"error": {"message": "not found"}})

    return httpx.MockTransport(handler)


def make_settings(model: str | None = None) -> Settings:
    return Settings(
        _env_file=None,
        server=ServerSettings(port=8080),
        database=DatabaseSettings(url="postgresql://invalid:invalid@localhost:9/none"),
        inference=EndpointSettings(base_url=UPSTREAM, model=model),
    )


def client_for_handler(
    handler,
    *,
    model: str | None = None,
    raise_server_exceptions: bool = True,
) -> TestClient:
    settings = make_settings(model=model)
    app = create_app(
        settings,
        llm_client=httpx.AsyncClient(base_url=UPSTREAM, transport=httpx.MockTransport(handler)),
    )
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


def make_client(captured: list[httpx.Request], **kwargs: Any) -> TestClient:
    settings = make_settings()
    transport = upstream_handler(captured, **kwargs)
    app = create_app(
        settings,
        llm_client=httpx.AsyncClient(base_url=UPSTREAM, transport=transport),
    )
    return TestClient(app)


@pytest.fixture
def captured_requests() -> list[httpx.Request]:
    return []


@pytest.fixture
def client(captured_requests: list[httpx.Request]) -> TestClient:
    return make_client(captured_requests)
