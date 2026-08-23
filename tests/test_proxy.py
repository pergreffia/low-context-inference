from __future__ import annotations

import json

import httpx
import pytest
from conftest import CHAT_RESPONSE, MODELS_RESPONSE, SSE_BODY


def test_models_passthrough(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert response.json() == MODELS_RESPONSE
    assert response.headers["content-type"].startswith("application/json")


def test_chat_completions_non_streaming_passthrough(client, captured_requests):
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.2,
    }
    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert response.json() == CHAT_RESPONSE

    forwarded = json.loads(captured_requests[0].content)
    assert forwarded == payload
    assert captured_requests[0].url.path == "/v1/chat/completions"


def test_tool_call_payload_forwarded_unchanged(client, captured_requests):
    tool_call_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{\"path\": \"a.py\"}"},
            }
        ],
    }
    tool_result = {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "file contents",
    }
    payload = {
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "read a.py"},
            tool_call_message,
            tool_result,
        ],
        "tools": [{"type": "function", "function": {"name": "read_file"}}],
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert json.loads(captured_requests[0].content) == payload


def test_chat_completions_streaming_passthrough(client):
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = b"".join(response.iter_bytes())
    assert body == SSE_BODY


def test_invalid_json_body_returns_openai_error(client):
    response = client.post(
        "/v1/chat/completions",
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert set(error) == {"message", "type", "param", "code"}


def test_non_object_json_body_rejected(client):
    response = client.post("/v1/chat/completions", json=[1, 2, 3])
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_upstream_error_body_passed_through(captured_requests):
    import httpx
    from conftest import UPSTREAM
    from fastapi.testclient import TestClient

    from context_proxy.config import DatabaseSettings, EndpointSettings, Settings
    from context_proxy.main import create_app

    def failing_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key", "type": "auth"}})

    settings = Settings(
        _env_file=None,
        database=DatabaseSettings(url="postgresql://invalid:invalid@localhost:9/none"),
        inference=EndpointSettings(base_url=UPSTREAM),
    )
    app = create_app(
        settings,
        llm_client=httpx.AsyncClient(
            base_url=UPSTREAM,
            transport=httpx.MockTransport(failing_handler),
        ),
    )
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"messages": []})

    assert response.status_code == 401
    assert response.json() == {"error": {"message": "bad key", "type": "auth"}}


def test_upstream_unavailable_returns_openai_error():
    import httpx
    from fastapi.testclient import TestClient

    from context_proxy.config import DatabaseSettings, EndpointSettings, Settings
    from context_proxy.main import create_app

    def dead_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    settings = Settings(
        _env_file=None,
        database=DatabaseSettings(url="postgresql://invalid:invalid@localhost:9/none"),
        inference=EndpointSettings(base_url="http://dead.test/v1"),
    )
    app = create_app(
        settings,
        llm_client=httpx.AsyncClient(
            base_url="http://dead.test/v1",
            transport=httpx.MockTransport(dead_handler),
        ),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/v1/chat/completions", json={"messages": []})
        models_response = client.get("/v1/models")

    for r in (response, models_response):
        assert r.status_code == 502
        error = r.json()["error"]
        assert error["code"] == "upstream_unavailable"
        assert error["type"] == "api_error"


def test_healthz_reports_degraded_database_without_postgres(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "degraded"


@pytest.mark.parametrize(
    "field,expected",
    [
        ("INFERENCE__BASE_URL", "http://override.test/v1"),
        ("COMPACT__MODEL", "compact-x"),
        ("EMBEDDINGS__API_KEY", "secret"),
        ("SERVER__PORT", "9999"),
    ],
)
def test_env_nested_configuration(monkeypatch, field, expected):
    from context_proxy.config import Settings

    monkeypatch.setenv(field, expected)
    monkeypatch.setattr("context_proxy.config.load_settings.cache_clear", lambda: None)
    settings = Settings(_env_file=None)
    match field:
        case "INFERENCE__BASE_URL":
            assert settings.inference.base_url == expected
        case "COMPACT__MODEL":
            assert settings.compact.model == expected
        case "EMBEDDINGS__API_KEY":
            assert settings.embeddings.api_key == expected
        case "SERVER__PORT":
            assert settings.server.port == 9999
