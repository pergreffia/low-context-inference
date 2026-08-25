"""Deterministic developer-role persistence/context contract (review §3B).

Real-provider E2E only proves request acceptance. This suite proves the full
deterministic contract with a fake provider + fake store:

    developer message -> persistence -> reload
                      -> context assembly -> provider-facing request

with `role == "developer"` preserved VERBATIM: never normalized to `system`,
never treated as ordinary user history.
"""

from __future__ import annotations

import json

import httpx
from fastapi.testclient import TestClient

from context_proxy.config import AssemblySettings, ContextSettings, RetrievalSettings
from context_proxy.context.engine import ContextAssemblyEngine
from context_proxy.context.planner import plan_context
from context_proxy.main import create_app
from tests.conftest import CHAT_RESPONSE, UPSTREAM, make_settings

DEV_DIRECTIVE = "Always answer with a YAML mapping under key `result`."
DEV_MESSAGE = {"role": "developer", "content": DEV_DIRECTIVE}


def app_with_store(store, captured):
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=CHAT_RESPONSE)

    return create_app(
        make_settings(),
        llm_client=httpx.AsyncClient(base_url=UPSTREAM, transport=httpx.MockTransport(handler)),
        store=store,
    )


def sent_messages(captured) -> list[dict]:
    return json.loads(captured[-1].content)["messages"]


class FakeStore:
    """Dict-backed ConversationStore mirroring positional reconcile."""

    def __init__(self) -> None:
        self.conversations: dict[str, list[dict]] = {}

    async def ping(self):
        return None

    async def ensure_conversation(self, conversation_id):
        self.conversations.setdefault(conversation_id, [])

    async def reconcile_history(self, conversation_id, messages, metadata=None):
        from context_proxy.conversation.store import HistoryDivergenceError

        bucket = self.conversations.setdefault(conversation_id, [])
        overlap = min(len(bucket), len(messages))
        for index in range(overlap):
            if bucket[index] != messages[index]:
                raise HistoryDivergenceError(conversation_id, index)
        bucket.extend(messages[len(bucket) :])
        return []

    async def get_messages(self, conversation_id):
        return list(self.conversations.get(conversation_id, []))


CONV = "33333333-3333-3333-3333-333333333333"


class TestDeveloperPersistenceContextContract:
    def test_persist_reload_reassemble_forward_keeps_developer(self):
        """The §3B pipeline end-to-end against deterministic doubles."""
        captured: list[httpx.Request] = []
        store = FakeStore()
        app = app_with_store(store, captured)

        with TestClient(app) as client:
            # -- turn 1: developer + user accepted, assistant persisted
            r1 = client.post(
                "/v1/chat/completions",
                json={
                    "model": "m",
                    "messages": [DEV_MESSAGE, {"role": "user", "content": "ping"}],
                    "conversation_id": CONV,
                },
            )
            assert r1.status_code == 200

            # provider-facing request #1 already carries developer verbatim
            first_sent = sent_messages(captured)
            assert first_sent[0]["role"] == "developer"
            assert first_sent[0]["content"] == DEV_DIRECTIVE

            # -- authoritative reload from the store
            import asyncio

            reloaded = asyncio.run(store.get_messages(CONV))
            assert [m["role"] for m in reloaded] == ["developer", "user", "assistant"]
            assert reloaded[0] == DEV_MESSAGE  # byte-identical inbound dict

            # -- turn 2: client replays reloaded history + new user turn
            replay = [*reloaded, {"role": "user", "content": "and now?"}]
            r2 = client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": replay, "conversation_id": CONV},
            )
            assert r2.status_code == 200

        final_sent = sent_messages(captured)
        devs = [m for m in final_sent if m["role"] == "developer"]
        assert len(devs) == 1                       # survives reload+assembly
        assert devs[0]["content"] == DEV_DIRECTIVE  # verbatim
        # never normalized: no system message carries the directive
        assert not [
            m for m in final_sent
            if m["role"] == "system" and m.get("content") == DEV_DIRECTIVE
        ]
        # never treated as ordinary history: emitted ahead of the turns
        assert final_sent[0]["role"] == "developer"
        # current request still last
        assert final_sent[-1]["content"] == "and now?"

    def test_developer_never_normalized_to_system_single_shot(self):
        captured: list[httpx.Request] = []
        store = FakeStore()
        app = app_with_store(store, captured)
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "m",
                    "messages": [DEV_MESSAGE, {"role": "user", "content": "go"}],
                },
            )
        assert response.status_code == 200
        sent = sent_messages(captured)
        assert sent[0]["role"] == "developer"
        assert all(m["role"] != "system" for m in sent)   # nothing invented

    def test_developer_survives_budget_pressure_at_route_level(self):
        captured: list[httpx.Request] = []
        store = FakeStore()
        settings = make_settings().model_copy(
            update={"context": ContextSettings(model_limit_tokens=300, safety_margin_tokens=50)}
        )

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json=CHAT_RESPONSE)

        app = create_app(
            settings,
            llm_client=httpx.AsyncClient(
                base_url=UPSTREAM, transport=httpx.MockTransport(handler)
            ),
            store=store,
        )
        big_user = "u" * 900
        big_assistant = "a" * 900
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "m",
                    "messages": [
                        DEV_MESSAGE,
                        {"role": "user", "content": big_user},
                        {"role": "assistant", "content": big_assistant},
                        {"role": "user", "content": "current"},
                    ],
                },
            )
        assert response.status_code == 200
        sent = sent_messages(captured)
        roles = [m["role"] for m in sent]
        assert roles.count("developer") == 1          # protected tier kept
        assert big_assistant not in [m.get("content") for m in sent]
        assert sent[-1]["content"] == "current"

    def test_developer_inserted_mid_interaction_stays_instruction(self):
        captured: list[httpx.Request] = []
        store = FakeStore()
        app = app_with_store(store, captured)
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "m",
                    "messages": [
                        {"role": "user", "content": "q1"},
                        {"role": "assistant", "content": "a1"},
                        {"role": "developer", "content": "late directive"},
                        {"role": "user", "content": "current"},
                    ],
                },
            )
        assert response.status_code == 200
        sent = sent_messages(captured)
        devs = [(m["role"], m["content"]) for m in sent if m["role"] == "developer"]
        assert devs == [("developer", "late directive")]      # verbatim, once
        assert sent[-1]["content"] == "current"

    def test_developer_with_multimodal_parts_forwarded_verbatim(self):
        captured: list[httpx.Request] = []
        store = FakeStore()
        app = app_with_store(store, captured)
        data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
        dev_multimodal = {
            "role": "developer",
            "content": [
                {"type": "text", "text": "Inspect attached reference layout."},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "m",
                    "messages": [
                        dev_multimodal,
                        {"role": "user", "content": "compare"},
                    ],
                },
            )
        assert response.status_code == 200
        sent = sent_messages(captured)
        assert sent[0] == dev_multimodal              # role AND parts untouched
        assert sent[0]["role"] == "developer"         # not rewritten for media


# ------------------------------------------- assembly-layer guarantees


class TestAssemblyLayerDeveloperProtection:
    def test_fallback_planner_protects_developer_under_pressure(self):
        history = [
            DEV_MESSAGE,
            {"role": "user", "content": "q " + "x" * 800},
            {"role": "assistant", "content": "a " + "y" * 600},
        ]
        plan = plan_context(
            history=history,
            current_request=[{"role": "user", "content": "current"}],
            tools=None,
            usable_budget=120,
        )
        roles = [m["role"] for m in plan.messages]
        assert roles.count("developer") == 1          # instruction tier survives
        assert roles.count("assistant") == 0          # ordinary turns dropped first
        assert plan.messages[-1]["content"] == "current"

    def test_engine_preserves_developer_position_and_content(self):
        engine = ContextAssemblyEngine(
            usable_budget=10_000,
            settings=AssemblySettings(),
            retrieval_settings=RetrievalSettings(),
        )
        history = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "older reply"},
            DEV_MESSAGE,
            {"role": "user", "content": "mid question"},
            {"role": "assistant", "content": "mid answer"},
        ]
        plan = engine.build(history=history, current_request=[{"role": "user", "content": "now"}])
        devs = [m for m in plan.messages if m["role"] == "developer"]
        assert devs == [DEV_MESSAGE]                  # exactly one, verbatim


# ------------------------------------------------- store-level contract


class TestStoreLevelDeveloperVerbatim:
    def test_fake_store_round_trip_is_verbatim(self):
        """Even the lightweight double must not rewrite roles (contract mirror)."""
        store = FakeStore()

        import asyncio

        inbound = [DEV_MESSAGE, {"role": "user", "content": "hi"}]

        async def run():
            await store.ensure_conversation(CONV)
            await store.reconcile_history(CONV, inbound)
            return await store.get_messages(CONV)

        assert asyncio.run(run()) == inbound
