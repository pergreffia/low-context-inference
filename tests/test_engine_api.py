"""M4 API tests: engine integration keeps OpenAI compatibility + streaming."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx
from fastapi.testclient import TestClient

from context_proxy.config import AssemblySettings, ContextSettings
from context_proxy.main import create_app
from tests.conftest import UPSTREAM, make_settings, upstream_handler
from tests.test_context_engine import CONV_A, CONV_B, memory_item


class StubMemory:
    """Deterministic stand-in for MemoryService in the request path."""

    def __init__(self, items=None, fail=False):
        self.items = items or []
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    async def retrieve(self, query, conversation_id, limit=None):
        self.calls.append((query, str(conversation_id)))
        if self.fail:
            raise RuntimeError("retrieval backend down")
        return self.items

    async def index_completed_turns(self, conversation_id):
        return 0


def build_client(
    captured: list[httpx.Request],
    *,
    memory=None,
    assembly: AssemblySettings | None = None,
    usable_budget: int | None = None,
) -> TestClient:
    settings = make_settings()
    if assembly is not None or usable_budget is not None:
        overrides: dict[str, Any] = {"assembly": assembly or AssemblySettings()}
        if usable_budget is not None:
            overrides["context"] = ContextSettings(
                model_limit_tokens=usable_budget, safety_margin_tokens=0
            )
        settings = settings.model_copy(update=overrides)
    app = create_app(
        settings,
        llm_client=httpx.AsyncClient(
            base_url=UPSTREAM, transport=upstream_handler(captured)
        ),
        store=None,
        memory_service=memory,
    )
    return TestClient(app)


@contextmanager
def run_client(client: TestClient) -> Iterator[TestClient]:
    """Run lifespan so injected memory service lands on app.state."""
    with client as running:
        yield running


CHAT_BODY = {
    "model": "client-model",
    "messages": [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "what storage do we use?"},
    ],
}

AUTH = {"X-Conversation-ID": CONV_A}


def upstream_payload(captured: list[httpx.Request]) -> dict:
    return json.loads(captured[-1].content)


class TestEnginePath:
    def test_retrieved_memory_injected_as_system_block(self, captured_requests):
        memory = StubMemory([memory_item("mem-9", "we use PostgreSQL 16")])
        with run_client(build_client(captured_requests, memory=memory)) as client:
            response = client.post("/v1/chat/completions", json=CHAT_BODY, headers=AUTH)
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "hello"
        contents = [m.get("content") for m in upstream_payload(captured_requests)["messages"]]
        assert any("PostgreSQL 16" in c and "[memory:fact mem-9]" in c for c in contents)
        # system prompt stays first; request stays last
        assert contents[0] == "be terse"
        assert contents[-1] == "what storage do we use?"

    def test_retrieval_query_scoped_to_conversation_header(self, captured_requests):
        memory = StubMemory([])
        with run_client(build_client(captured_requests, memory=memory)) as client:
            response = client.post("/v1/chat/completions", json=CHAT_BODY, headers=AUTH)
        assert response.status_code == 200
        assert len(memory.calls) == 1
        assert memory.calls[0][1] == CONV_A

    def test_streaming_passthrough_with_injection(self, captured_requests):
        memory = StubMemory([memory_item("m", "injected fact")])
        body = {**CHAT_BODY, "stream": True}
        with run_client(build_client(captured_requests, memory=memory)) as client:
            with client.stream(
                "POST", "/v1/chat/completions", json=body, headers=AUTH
            ) as response:
                assert response.status_code == 200
                chunks = "".join(response.iter_text())
                assert chunks.endswith("data: [DONE]\n\n")
                assert response.headers["x-conversation-id"] == CONV_A
        payload = upstream_payload(captured_requests)
        assert payload["stream"] is True
        assert "[memory:fact m]" in json.dumps(payload["messages"])

    def test_retrieval_failure_degrades_to_raw_context(self, captured_requests):
        memory = StubMemory(fail=True)
        with run_client(build_client(captured_requests, memory=memory)) as client:
            response = client.post("/v1/chat/completions", json=CHAT_BODY, headers=AUTH)
        assert response.status_code == 200
        messages = upstream_payload(captured_requests)["messages"]
        assert [m["content"] for m in messages] == [
            "be terse",
            "what storage do we use?",
        ]

    def test_no_memory_service_still_completes(self, captured_requests):
        with run_client(build_client(captured_requests)) as client:
            response = client.post("/v1/chat/completions", json=CHAT_BODY, headers=AUTH)
        assert response.status_code == 200

    def test_overflow_returns_openai_error(self, captured_requests):
        huge_system = {"role": "system", "content": "s" * 400}
        huge_request = {"role": "user", "content": "u" * 400}
        body = {**CHAT_BODY, "messages": [huge_system, huge_request]}
        memory = StubMemory()
        with run_client(
            build_client(captured_requests, memory=memory, usable_budget=150)
        ) as client:
            response = client.post("/v1/chat/completions", json=body, headers=AUTH)
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["code"] == "context_length_exceeded"


class TestLegacyFallback:
    def test_disabled_engine_uses_m2_planner(self, captured_requests):
        memory = StubMemory([memory_item("m", "should not appear")])
        disabled = AssemblySettings(enabled=False)
        with run_client(
            build_client(captured_requests, memory=memory, assembly=disabled)
        ) as client:
            response = client.post("/v1/chat/completions", json=CHAT_BODY, headers=AUTH)
        assert response.status_code == 200
        payload = json.dumps(upstream_payload(captured_requests)["messages"])
        assert "should not appear" not in payload
        assert memory.calls == []  # legacy path never retrieves

    def test_preview_endpoint_503_without_engine(self, captured_requests):
        disabled = AssemblySettings(enabled=False)
        with run_client(build_client(captured_requests, assembly=disabled)) as client:
            response = client.post(
                f"/internal/v1/conversations/{CONV_A}/context/preview",
                json={"messages": CHAT_BODY["messages"]},
            )
        assert response.status_code == 503


class TestPreviewEndpoint:
    def test_preview_shape_and_isolation(self, captured_requests):
        foreign = memory_item("foreign-1", "other chat secret", conversation_id=CONV_B)
        own = memory_item("own-1", "own chat fact", conversation_id=CONV_A)
        memory = StubMemory([foreign, own])
        with run_client(build_client(captured_requests, memory=memory)) as client:
            response = client.post(
                f"/internal/v1/conversations/{CONV_A}/context/preview",
                json={
                    "messages": [
                        {"role": "user", "content": "old"},
                        {"role": "assistant", "content": "ok"},
                        {"role": "user", "content": "current"},
                    ]
                },
            )
        assert response.status_code == 200
        view = response.json()
        assert set(view) >= {
            "selected",
            "dropped",
            "token_estimate",
            "tools_tokens",
            "budget",
            "rationale",
            "diagnostics",
        }
        selected_keys = {s["key"] for s in view["selected"]}
        assert "own-1" in selected_keys
        assert "foreign-1" not in selected_keys
        dropped = {d["key"]: d["reason"] for d in view["dropped"]}
        assert dropped.get("foreign-1") == "foreign_conversation"
        blob = json.dumps(view)
        assert "secret" not in blob  # diagnostics carry ids/scores, no content

    def test_preview_does_not_persist_or_call_upstream(self, captured_requests):
        memory = StubMemory([])
        with run_client(build_client(captured_requests, memory=memory)) as client:
            before = len(captured_requests)
            response = client.post(
                f"/internal/v1/conversations/{CONV_A}/context/preview",
                json={"messages": [{"role": "user", "content": "q"}]},
            )
        assert response.status_code == 200
        assert len(captured_requests) == before
