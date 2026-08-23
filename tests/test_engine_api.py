"""M4 API tests: engine integration keeps OpenAI compatibility + streaming.

Covers review-fix requirements: typed retrieval failures (RetrievalError
degrades, TypeError propagates), mandatory overflow never calls the LLM,
preview/production share query extraction and produce identical plans.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from context_proxy.config import AssemblySettings, ContextSettings
from context_proxy.main import create_app
from tests.conftest import UPSTREAM, make_settings, upstream_handler
from tests.test_context_engine import CONV_A, CONV_B, memory_item


class StubMemory:
    """Deterministic stand-in for MemoryService in the request path."""

    def __init__(self, items=None, fail_with: Exception | None = None):
        self.items = items or []
        self.fail_with = fail_with
        self.calls: list[tuple[str, str]] = []

    async def retrieve(self, query, conversation_id, limit=None):
        self.calls.append((query, str(conversation_id)))
        if self.fail_with is not None:
            raise self.fail_with
        return self.items

    async def index_completed_turns(self, conversation_id):
        return 0


def build_client(
    captured: list[httpx.Request],
    *,
    memory=None,
    assembly: AssemblySettings | None = None,
    usable_budget: int | None = None,
    raise_server_exceptions: bool = True,
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
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


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


def post_chat(client: TestClient, body: dict | None = None):
    return client.post("/v1/chat/completions", json=body or CHAT_BODY, headers=AUTH)


def post_preview(client: TestClient, body: dict):
    return client.post(
        f"/internal/v1/conversations/{CONV_A}/context/preview", json=body
    )


class TestEnginePath:
    def test_retrieved_memory_injected_as_system_block(self, captured_requests):
        memory = StubMemory([memory_item("mem-9", "we use PostgreSQL 16")])
        with run_client(build_client(captured_requests, memory=memory)) as client:
            response = post_chat(client)
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "hello"
        contents = [
            m.get("content") for m in upstream_payload(captured_requests)["messages"]
        ]
        assert any("PostgreSQL 16" in c and "[memory:fact mem-9]" in c for c in contents)
        assert contents[0] == "be terse"
        assert contents[-1] == "what storage do we use?"

    def test_retrieval_query_scoped_to_conversation_header(self, captured_requests):
        memory = StubMemory([])
        with run_client(build_client(captured_requests, memory=memory)) as client:
            assert post_chat(client).status_code == 200
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

    def test_no_memory_service_still_completes(self, captured_requests):
        with run_client(build_client(captured_requests)) as client:
            assert post_chat(client).status_code == 200

    def test_current_request_appears_exactly_once_upstream(self, captured_requests):
        memory = StubMemory([])
        body = {
            "model": "client-model",
            "messages": [
                {"role": "user", "content": "A"},
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": "B"},
            ],
        }
        with run_client(build_client(captured_requests, memory=memory)) as client:
            assert post_chat(client, body).status_code == 200
        contents = [
            m["content"] for m in upstream_payload(captured_requests)["messages"]
        ]
        assert contents == ["A", "a", "B"]

    def test_multimodal_text_parts_only_in_query(self, captured_requests):
        memory = StubMemory([])
        body = {
            "model": "client-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is wrong here?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AAAA"},
                        },
                    ],
                }
            ],
        }
        with run_client(build_client(captured_requests, memory=memory)) as client:
            assert post_chat(client, body).status_code == 200
        assert memory.calls[0][0] == "what is wrong here?"
        # multimodal content preserved verbatim upstream
        sent = upstream_payload(captured_requests)["messages"][0]["content"]
        assert sent[1]["type"] == "image_url"


class TestTypedRetrievalFailures:
    def test_retrieval_error_degrades_and_logs(self, captured_requests, caplog):
        from context_proxy.memory.errors import RetrievalError

        memory = StubMemory(fail_with=RetrievalError("lexical leg down"))
        with caplog.at_level(logging.WARNING, logger="context_proxy.request"):
            with run_client(build_client(captured_requests, memory=memory)) as client:
                response = post_chat(client)
        assert response.status_code == 200
        messages = upstream_payload(captured_requests)["messages"]
        assert [m["content"] for m in messages] == ["be terse", "what storage do we use?"]
        assert any(r.message == "context_retrieval_failed" for r in caplog.records)

    def test_unexpected_error_is_not_swallowed(self, captured_requests):
        memory = StubMemory(fail_with=TypeError("'NoneType' object is not subscriptable"))
        with pytest.raises(TypeError):
            with run_client(
                build_client(captured_requests, memory=memory)
            ) as client:
                post_chat(client)
        # nothing reached the LLM: programming errors must fail loudly
        assert captured_requests == []


class TestBudgetEnforcement:
    def test_mandatory_overflow_returns_400_without_llm_call(self, captured_requests):
        memory = StubMemory()
        huge_system = {"role": "system", "content": "s" * 400}
        huge_request = {"role": "user", "content": "u" * 400}
        body = {**CHAT_BODY, "messages": [huge_system, huge_request]}
        with run_client(
            build_client(captured_requests, memory=memory, usable_budget=150)
        ) as client:
            response = post_chat(client, body)
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["code"] == "context_length_exceeded"
        assert error["type"] == "invalid_request_error"
        assert captured_requests == []  # LLM never called

    def test_tool_definitions_consume_budget_upstream(self, captured_requests):
        big_tools = [
            {"type": "function", "function": {"name": f"t{i}", "parameters": {"x": "p" * 2000}}}
            for i in range(3)
        ]
        memory = StubMemory([])
        history = [{"role": "user", "content": "old " + "x" * 1500}]
        body = {
            "model": "client-model",
            "tools": big_tools,
            "messages": [*history, {"role": "user", "content": "current"}],
        }
        with run_client(
            build_client(captured_requests, memory=memory, usable_budget=1600)
        ) as client:
            response = post_chat(client, body)
        assert response.status_code == 200
        payload = upstream_payload(captured_requests)
        contents = [m["content"] for m in payload["messages"]]
        assert "current" in contents
        assert not any(c.startswith("old ") for c in contents)

    def test_retrieval_cap_never_exceeds_remaining_budget(self, captured_requests):
        assembly = AssemblySettings(retrieved_budget_tokens=4000, max_retrieved_items=10)
        retrieved = [
            memory_item(f"m{i}", f"distinct unique fact {i} " + "z" * 60, semantic=0.9 - i * 0.01)
            for i in range(6)
        ]
        memory = StubMemory(retrieved)
        history = [
            {"role": "user", "content": "h " + "x" * 2200},
            {"role": "assistant", "content": "a " + "y" * 2200},
        ]
        body = {
            "model": "client-model",
            "messages": [*history, {"role": "user", "content": "current"}],
        }
        with run_client(
            build_client(
                captured_requests, memory=memory, assembly=assembly, usable_budget=2600
            )
        ) as client:
            assert post_chat(client, body).status_code == 200
        messages = upstream_payload(captured_requests)["messages"]
        memory_blocks = [
            m for m in messages if m.get("role") == "system" and "[memory:" in m["content"]
        ]

        from context_proxy.context.tokens import TokenCounter

        counter = TokenCounter()
        used = sum(counter.message(m) for m in memory_blocks)
        assert used <= 1200


class TestPreviewEquivalence:
    MULTIMODAL_BODY = {
        "model": "client-model",
        "messages": [
            {"role": "user", "content": "design the retry policy"},
            {"role": "assistant", "content": "retry three times with backoff"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "summarize the retry policy"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,BBBB"},
                    },
                ],
            },
        ],
    }

    def test_preview_and_production_share_query_extraction(self, captured_requests):
        memory = StubMemory([])
        with run_client(build_client(captured_requests, memory=memory)) as client:
            assert post_chat(client, self.MULTIMODAL_BODY).status_code == 200
            preview_response = post_preview(client, self.MULTIMODAL_BODY)
        assert preview_response.status_code == 200
        prod_query = memory.calls[0][0]
        preview_query = memory.calls[1][0]
        assert prod_query == preview_query == "summarize the retry policy"

    def test_preview_matches_production_plan_exactly(self, captured_requests):
        retrieved = [
            memory_item("own-1", "retry policy uses backoff", semantic=0.8),
            memory_item("dup-1", "retry three times with backoff", semantic=0.7),
        ]
        memory = StubMemory(retrieved)
        with run_client(build_client(captured_requests, memory=memory)) as client:
            chat = post_chat(client, self.MULTIMODAL_BODY)
            upstream_messages = upstream_payload(captured_requests)["messages"]
            preview = post_preview(client, self.MULTIMODAL_BODY)

        assert chat.status_code == 200
        assert preview.status_code == 200
        view = preview.json()

        from context_proxy.config import RetrievalSettings
        from context_proxy.context.engine import ContextAssemblyEngine, separate_current_request
        from context_proxy.context.tokens import TokenCounter

        engine = ContextAssemblyEngine(
            usable_budget=make_settings().context.usable_budget_tokens,
            settings=AssemblySettings(),
            retrieval_settings=RetrievalSettings(),
            counter=TokenCounter(),
        )
        history, current = separate_current_request(self.MULTIMODAL_BODY["messages"])
        expected = engine.build(
            history=history,
            current_request=current,
            retrieved=retrieved,
            conversation_id=CONV_A,
        )

        assert view["selected"] == [
            {
                "key": s.key,
                "source": s.source.value,
                "tokens": s.tokens,
                "score": s.score,
            }
            for s in expected.selected_items
        ]
        assert view["dropped"] == [
            {"key": d.key, "source": d.source.value, "reason": d.reason}
            for d in expected.dropped_items
        ]
        assert view["token_estimate"] == expected.token_estimate
        assert view["budget"] == expected.budget
        # and production actually sent exactly those messages
        assert upstream_messages == expected.messages

    def test_preview_does_not_leak_raw_content(self, captured_requests):
        memory = StubMemory([memory_item("own-1", "own chat fact keyword orangutan")])
        with run_client(build_client(captured_requests, memory=memory)) as client:
            response = post_preview(client, self.MULTIMODAL_BODY)
        assert response.status_code == 200
        blob = json.dumps(response.json())
        assert "orangutan" not in blob
        assert "retry three times" not in blob

    def test_preview_isolation_foreign_candidates_never_shown(self, captured_requests):
        foreign = memory_item("foreign-1", "other chat secret", conversation_id=CONV_B)
        own = memory_item("own-1", "own chat fact", conversation_id=CONV_A)
        memory = StubMemory([foreign, own])
        with run_client(build_client(captured_requests, memory=memory)) as client:
            response = post_preview(client, self.MULTIMODAL_BODY)
        assert response.status_code == 200
        view = response.json()
        selected_keys = {s["key"] for s in view["selected"]}
        assert "foreign-1" not in selected_keys
        dropped = {d["key"]: d["reason"] for d in view["dropped"]}
        assert dropped.get("foreign-1") == "foreign_conversation"

    def test_preview_does_not_persist_or_call_upstream(self, captured_requests):
        memory = StubMemory([])
        with run_client(build_client(captured_requests, memory=memory)) as client:
            before = len(captured_requests)
            response = post_preview(
                client, {"messages": [{"role": "user", "content": "q"}]}
            )
        assert response.status_code == 200
        assert len(captured_requests) == before


class TestLegacyFallback:
    def test_disabled_engine_uses_m2_planner_and_skips_retrieval(self, captured_requests):
        memory = StubMemory([memory_item("m", "should not appear")])
        disabled = AssemblySettings(enabled=False)
        with run_client(
            build_client(captured_requests, memory=memory, assembly=disabled)
        ) as client:
            response = post_chat(client)
        assert response.status_code == 200
        payload = json.dumps(upstream_payload(captured_requests)["messages"])
        assert "should not appear" not in payload
        assert memory.calls == []  # M4 path never invoked

    def test_preview_endpoint_503_without_engine(self, captured_requests):
        disabled = AssemblySettings(enabled=False)
        with run_client(build_client(captured_requests, assembly=disabled)) as client:
            response = post_preview(client, {"messages": CHAT_BODY["messages"]})
        assert response.status_code == 503
