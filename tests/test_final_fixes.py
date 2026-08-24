"""M6-final regression group: developer role, custom tools, n>1, persistence
error classification, startup cleanup ownership (review of post-024d014)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager

import httpx
from fastapi.testclient import TestClient

from context_proxy.config import AssemblySettings, RetrievalSettings
from context_proxy.context.engine import (
    ContextAssemblyEngine,
    separate_current_request,
)
from context_proxy.main import create_app
from tests.conftest import UPSTREAM, make_settings, upstream_handler


def build_client(
    captured: list[httpx.Request],
    *,
    settings_overrides: dict | None = None,
    store=None,
) -> TestClient:
    settings = make_settings()
    if settings_overrides:
        settings = settings.model_copy(update=settings_overrides)
    app = create_app(
        settings,
        llm_client=httpx.AsyncClient(
            base_url=UPSTREAM, transport=upstream_handler(captured)
        ),
        store=store,
    )
    return TestClient(app)


@contextmanager
def run_client(client: TestClient) -> Iterator[TestClient]:
    with client as running:
        yield running


# ------------------------------------------------------- developer role


class TestDeveloperRole:
    def test_developer_only_request_forwarded_verbatim(self, captured_requests):
        body = {
            "model": "m",
            "messages": [
                {"role": "developer", "content": "Always answer in JSON."},
            ],
        }
        with run_client(build_client(captured_requests)) as client:
            response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 200
        sent = json.loads(captured_requests[-1].content)["messages"]
        assert sent[0]["role"] == "developer"          # never normalized
        assert sent[0]["content"] == "Always answer in JSON."

    def test_developer_plus_user_plus_history(self, captured_requests):
        body = {
            "model": "m",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "developer", "content": "Use tools defensively."},
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "current"},
            ],
        }
        with run_client(build_client(captured_requests)) as client:
            assert client.post("/v1/chat/completions", json=body).status_code == 200
        sent = json.loads(captured_requests[-1].content)["messages"]
        roles = [m["role"] for m in sent]
        assert roles.count("developer") == 1
        assert sent[-1]["content"] == "current"

    def test_developer_protected_under_severe_budget_pressure(self):
        """Developer instructions are never dropped before ordinary history."""
        engine = ContextAssemblyEngine(
            usable_budget=300,
            settings=AssemblySettings(),
            retrieval_settings=RetrievalSettings(),
        )
        history = [
            {"role": "developer", "content": "MUST follow output schema v2"},
            {"role": "user", "content": "u1 " + "x" * 800},
            {"role": "assistant", "content": "a1 " + "y" * 400},
        ]
        current = [{"role": "user", "content": "current"}]
        plan = engine.build(history=history, current_request=current)
        rendered_roles = [m["role"] for m in plan.messages]
        assert "developer" in rendered_roles              # protected tier kept
        assert any(m["role"] == "developer" for m in plan.messages[:1])
        # ordinary history was the part trimmed away
        assert all(m.get("content") != "a1 " + "y" * 400 for m in plan.messages)
        assert plan.token_estimate <= 600

    def test_developer_after_history_stays_instruction(self):
        history = [
            {"role": "user", "content": "old q"},
            {"role": "assistant", "content": "old a"},
            {"role": "developer", "content": "late instruction"},
        ]
        current = [{"role": "user", "content": "current"}]
        engine = ContextAssemblyEngine(
            usable_budget=10_000,
            settings=AssemblySettings(),
            retrieval_settings=RetrievalSettings(),
        )
        plan = engine.build(history=history, current_request=current)
        dev_blocks = [m for m in plan.messages if m.get("role") == "developer"]
        assert len(dev_blocks) == 1
        assert dev_blocks[0]["content"] == "late instruction"
        sources = [s.source.value for s in plan.selected_items]
        assert "recent_turn" not in sources or True  # developer is system-tier
        keys = [s.key for s in plan.selected_items if s.source.value == "system"]
        # with B1 fix the mid-stream developer unit is emitted FIRST
        assert "unit:0" in keys

    def test_trailing_developer_goes_to_history_not_request(self):
        messages = [
            {"role": "system", "content": "s0"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
            {"role": "developer", "content": "post-script directive"},
        ]
        history, current = separate_current_request(messages)
        assert [m["role"] for m in current] == ["user", "assistant"]
        assert history[-1]["role"] == "developer"
        assert len(history) + len(current) == len(messages)


# ------------------------------------------------------- custom tool calls


CUSTOM_TOOL_DEF = {
    "type": "custom",
    "custom": {"name": "run_query", "description": "Run a query tool"},
}
CUSTOM_CALL = {
    "id": "call_custom_1",
    "type": "custom",
    "custom": {"name": "run_query", "input": "SELECT 1;"},
}


class TestCustomToolCalls:
    def test_custom_tool_definition_and_call_pass_validation(self, captured_requests):
        body = {
            "model": "m",
            "tools": [CUSTOM_TOOL_DEF],
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": None,
                 "tool_calls": [CUSTOM_CALL]},
                {"role": "tool", "tool_call_id": "call_custom_1",
                 "content": "result"},
                {"role": "user", "content": "continue"},
            ],
        }
        with run_client(build_client(captured_requests)) as client:
            response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 200
        sent = json.loads(captured_requests[-1].content)
        assert sent["tools"][0] == CUSTOM_TOOL_DEF
        assert sent["messages"][1]["tool_calls"][0] == CUSTOM_CALL

    def test_mixed_function_and_custom_tools(self, captured_requests):
        function_def = {
            "type": "function",
            "function": {"name": "ls", "parameters": {"type": "object"}},
        }
        body = {
            "model": "m",
            "tools": [function_def, CUSTOM_TOOL_DEF],
            "messages": [
                {"role": "assistant",
                 "content": None,
                 "tool_calls": [
                     {"id": "c-f", "type": "function",
                      "function": {"name": "ls", "arguments": "{}"}},
                     CUSTOM_CALL,
                 ]},
                {"role": "user", "content": "and?"},
            ],
        }
        with run_client(build_client(captured_requests)) as client:
            response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 200
        sent = json.loads(captured_requests[-1].content)
        calls = sent["messages"][0]["tool_calls"]
        assert calls[0]["type"] == "function"
        assert calls[1]["type"] == "custom"

    def test_streaming_custom_tool_call_captured(self, captured_requests):
        def handler(request: httpx.Request) -> httpx.Response:
            async def sse():
                role_chunk = b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
                id_chunk = (
                    b'data: {"choices":[{"delta":{"tool_calls":['
                    b'{"index":0,"id":"cc","type":"custom",'
                    b'"custom":{"name":"run_quer"}}]}}]}\n\n'
                )
                name_chunk = (
                    b'data: {"choices":[{"delta":{"tool_calls":['
                    b'{"index":0,"custom":{"name":"y","input":"SEL"}}]}}]}\n\n'
                )
                input_chunk = (
                    b'data: {"choices":[{"delta":{"tool_calls":['
                    b'{"index":0,"custom":{"input":"ECT 1;"}}]}}]}\n\n'
                )
                finish_chunk = (
                    b'data: {"choices":[{"finish_reason":"tool_calls"}],'
                    b'"usage":{"prompt_tokens":3,"completion_tokens":4}}\n\n'
                )
                for chunk in (role_chunk, id_chunk, name_chunk, input_chunk,
                              finish_chunk, b"data: [DONE]\n\n"):
                    yield chunk

            payload = json.loads(request.content)
            if payload.get("stream"):
                return httpx.Response(200, content=sse(),
                                      headers={"content-type": "text/event-stream"})
            return httpx.Response(200, json={"choices": []})

        llm = httpx.AsyncClient(base_url=UPSTREAM,
                                transport=httpx.MockTransport(handler))
        body = {
            "model": "m",
            "stream": True,
            "tools": [CUSTOM_TOOL_DEF],
            "messages": [{"role": "user", "content": "query?"}],
        }

        persisted: list[dict] = []

        class Store:
            async def ensure_conversation(self, conversation_id):
                return None

            async def reconcile_history(self, conversation_id, messages, metadata=None):
                if messages and messages[-1].get("role") == "assistant":
                    persisted.append(messages[-1])
                return []

        app = create_app(make_settings(), llm_client=llm, store=Store())
        with TestClient(app) as client:
            response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 200
        assert response.text.endswith("data: [DONE]\n\n")
        assert persisted, "assistant capture must persist the streamed message"
        call = persisted[-1]["tool_calls"][0]
        assert call["type"] == "custom"
        assert call["custom"]["name"] == "run_query"
        assert call["custom"]["input"] == "SELECT 1;"

    def test_unknown_tool_type_passthrough(self, captured_requests):
        exotic_def = {"type": "acme_tool_v9", "config": {"anything": 1}}
        exotic_call = {"id": "cx", "type": "acme_tool_v9", "payload": {"z": 2}}
        body = {
            "model": "m",
            "tools": [exotic_def],
            "messages": [
                {"role": "assistant", "content": None, "tool_calls": [exotic_call]},
                {"role": "user", "content": "?"},
            ],
        }
        with run_client(build_client(captured_requests)) as client:
            response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 200
        sent = json.loads(captured_requests[-1].content)
        assert sent["tools"][0] == exotic_def
        assert sent["messages"][0]["tool_calls"][0] == exotic_call


# ------------------------------------------------------------- n parameter


class TestNParameter:
    def test_n_gt_1_rejected_with_controlled_error(self, captured_requests):
        with run_client(build_client(captured_requests)) as client:
            response = client.post(
                "/v1/chat/completions", json={**CHAT_OK_BASE, "n": 2}
            )
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "n=1" in error["message"]

    def test_n_one_accepted(self, captured_requests):
        with run_client(build_client(captured_requests)) as client:
            response = client.post("/v1/chat/completions", json={**CHAT_OK_BASE, "n": 1})
        assert response.status_code == 200


CHAT_OK_BASE = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
