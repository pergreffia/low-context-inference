from __future__ import annotations

import json
from typing import Any

import httpx
from conftest import CHAT_RESPONSE, SSE_BODY, client_for_handler, make_settings
from helpers import chat_payload


class FakeConversationStore:
    """In-memory ConversationStore for unit tests (mocks allowed in tests only)."""

    def __init__(self) -> None:
        self.conversations: dict[str, list[dict[str, Any]]] = {}

    async def ping(self) -> None:
        return None

    async def ensure_conversation(self, conversation_id: str) -> None:
        self.conversations.setdefault(conversation_id, [])

    async def append_messages(
        self, conversation_id: str, messages: list[dict[str, Any]]
    ) -> list[str]:
        bucket = self.conversations.setdefault(conversation_id, [])
        bucket.extend(messages)
        return [f"msg-{len(bucket)}" for _ in messages]

    async def get_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        return list(self.conversations.get(conversation_id, []))


def test_conversation_persisted_across_turns_non_streaming():
    store = FakeConversationStore()
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=CHAT_RESPONSE)

    client = client_for_handler(handler, store=store)

    conv = "44444444-4444-4444-4444-444444444444"
    with client:
        r1 = client.post(
            "/v1/chat/completions",
            json={**chat_payload(), "conversation_id": conv},
            headers={"X-Conversation-ID": conv},
        )
        r2 = client.post("/v1/chat/completions", json=chat_payload())

    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.headers["X-Conversation-ID"] == conv
    # turn 1: inbound user + persisted assistant response
    roles = [m["role"] for m in store.conversations[conv]]
    assert roles == ["user", "assistant"]
    # second request had no id -> separate conversation
    assert len(store.conversations) == 2
    # upstream received only OpenAI fields (conversation_id stripped)
    forwarded = json.loads(captured[0].content)
    assert "conversation_id" not in forwarded


def test_streaming_response_persisted_via_tee():
    store = FakeConversationStore()

    async def agen():
        yield SSE_BODY[:25]
        yield SSE_BODY[25:]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=agen(), headers={"content-type": "text/event-stream"})

    client = client_for_handler(handler, store=store)
    conv = "55555555-5555-5555-5555-555555555555"

    with client, client.stream(
        "POST",
        "/v1/chat/completions",
        json={**chat_payload(stream=True), "conversation_id": conv},
    ) as response:
        body = b"".join(response.iter_bytes())
        assert response.headers["X-Conversation-ID"] == conv

    assert body == SSE_BODY  # passthrough untouched
    messages = store.conversations[conv]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assistant = messages[1]
    assert assistant["content"] == "he"  # accumulated from delta chunks


def test_budget_overflow_returns_openai_error_without_forwarding():
    store = FakeConversationStore()
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=CHAT_RESPONSE)


    from context_proxy.config import ContextSettings

    settings = make_settings()
    settings.context = ContextSettings(model_limit_tokens=100, safety_margin_tokens=50)
    app = _app_with(settings, handler, store)
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        huge = "g" * 10_000
        r = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": huge}]},
        )

    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "context_length_exceeded"
    assert err["type"] == "invalid_request_error"
    assert not calls  # never sent over-budget upstream
    # raw inbound persisted BEFORE budget validation (§29 step order)
    assert list(store.conversations.values())[0][0]["content"] == huge


def test_over_budget_history_trimmed_before_upstream():
    store = FakeConversationStore()
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=CHAT_RESPONSE)

    from context_proxy.config import ContextSettings

    settings = make_settings()
    # usable = 300 - 50 = 250 tokens; each filler message ~204 tokens
    settings.context = ContextSettings(model_limit_tokens=300, safety_margin_tokens=50)
    app = _app_with(settings, handler, store)
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        fillers = [
            {"role": "user", "content": "w" * 800},
            {"role": "assistant", "content": "w" * 800},
        ]
        current = {"role": "user", "content": "current"}
        r = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [*fillers, current]},
        )

    assert r.status_code == 200
    forwarded_messages = json.loads(captured[0].content)["messages"]
    assert forwarded_messages[-1] == current  # current request preserved
    assert all(m != fillers[0] for m in forwarded_messages)  # oldest trimmed
    total_chars = sum(len(str(m)) for m in forwarded_messages)
    assert total_chars < sum(len(str(m)) for m in [*fillers, current])


def test_degraded_store_still_serves_passthrough():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=CHAT_RESPONSE)

    client = client_for_handler(handler)  # no store injected, db unreachable
    r = client.post("/v1/chat/completions", json=chat_payload())
    assert r.status_code == 200
    assert r.json() == CHAT_RESPONSE


def _app_with(settings, handler, store):
    import httpx as _httpx

    from context_proxy.main import create_app

    base_url = str(settings.inference.base_url)
    return create_app(
        settings,
        llm_client=_httpx.AsyncClient(base_url=base_url, transport=_httpx.MockTransport(handler)),
        store=store,
    )
