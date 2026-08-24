"""M6 tests: multimodal transparency, persistence, interaction atomicity.

Master prompt §13.1–§13.3:
- content arrays pass through byte-semantically untouched (client -> upstream);
- unknown parts stay opaque;
- persistence stays verbatim/reconstructable, media registered per message;
- a multimodal interaction is one atomic unit in context selection.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from fastapi.testclient import TestClient

from context_proxy.context.tokens import (
    IMAGE_PART_TOKENS,
    UNKNOWN_PART_TOKENS,
    TokenCounter,
)
from context_proxy.main import create_app
from tests.conftest import UPSTREAM, make_settings, upstream_handler

DATA_URL = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAf"
    "FcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def multimodal_message(text: str, url: str = DATA_URL) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": url}},
        ],
    }


def build_client(
    captured: list[httpx.Request],
    *,
    store=None,
    llm_client: httpx.AsyncClient | None = None,
) -> TestClient:
    app = create_app(
        make_settings(),
        llm_client=llm_client
        or httpx.AsyncClient(base_url=UPSTREAM, transport=upstream_handler(captured)),
        store=store,
    )
    return TestClient(app)


# ------------------------------------------------------------ M6.1 tokens


class TestMultimodalTokenAccounting:
    def test_image_part_uses_flat_estimate_not_base64_inflation(self):
        counter = TokenCounter()
        huge_data_url = "data:image/png;base64," + "A" * 100_000
        message = multimodal_message("what is this?", huge_data_url)
        tokens = counter.message(message)
        # ~1024 for the image + small text + overhead; NOT ~25k from base64
        assert IMAGE_PART_TOKENS <= tokens < IMAGE_PART_TOKENS * 2

    def test_unknown_part_type_flat_estimate(self):
        counter = TokenCounter()
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "vendor_custom", "blob": "z" * 5000},
            ],
        }
        tokens = counter.message(message)
        assert UNKNOWN_PART_TOKENS <= tokens < UNKNOWN_PART_TOKENS * 4

    def test_estimate_is_deterministic_regardless_of_image_bytes(self):
        counter = TokenCounter()
        small = counter.message(multimodal_message("q", "data:image/png;base64,AA=="))
        large = counter.message(
            multimodal_message("q", "data:image/png;base64," + "B" * 80_000)
        )
        assert small == large


class TestMultimodalIdentity:
    def test_same_text_different_images_keep_distinct_identity(self):
        from context_proxy.context.candidates import content_texts

        one = content_texts(multimodal_message("describe", DATA_URL))
        other = content_texts(
            multimodal_message("describe", "data:image/png;base64,BBBBBB")
        )
        assert one != other
        assert any(t.startswith("[image:") for t in one)

    def test_same_text_same_image_same_identity(self):
        from context_proxy.context.candidates import content_texts

        assert content_texts(multimodal_message("d", DATA_URL)) == content_texts(
            multimodal_message("d", DATA_URL)
        )


# ------------------------------------------------- M6.1 API transparency


class TestTransparencyEndToEnd:
    def test_multimodal_content_reaches_upstream_verbatim(self, captured_requests):
        body = {
            "model": "client-model",
            "messages": [
                {"role": "system", "content": "be terse"},
                multimodal_message("what is wrong here?"),
            ],
        }
        client = build_client(captured_requests)
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 200
        sent = json.loads(captured_requests[-1].content)["messages"]
        # the array survives EXACTLY: order, parts, base64 untouched
        assert sent[1] == body["messages"][1]
        assert isinstance(sent[1]["content"], list)  # never degraded to string
        assert sent[1]["content"][1]["image_url"]["url"] == DATA_URL

    def test_unknown_part_types_stay_opaque_end_to_end(self, captured_requests):
        exotic_part = {"type": "acme_hologram", "payload": {"x": 1}, "extra": [1, 2]}
        body = {
            "model": "client-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        exotic_part,
                    ],
                }
            ],
        }
        client = build_client(captured_requests)
        assert client.post("/v1/chat/completions", json=body).status_code == 200
        sent_content = json.loads(captured_requests[-1].content)["messages"][0][
            "content"
        ]
        assert sent_content[1] == exotic_part  # forwarded untouched

    def test_streaming_passthrough_with_multimodal_history(
        self, captured_requests
    ):
        llm_client = httpx.AsyncClient(
            base_url=UPSTREAM, transport=upstream_handler(captured_requests)
        )
        body = {
            "model": "client-model",
            "stream": True,
            "messages": [multimodal_message("screenshot review")],
        }
        client = build_client(captured_requests, llm_client=llm_client)
        with client as running:
            with running.stream(
                "POST", "/v1/chat/completions", json=body
            ) as response:
                assert response.status_code == 200
                chunks = "".join(response.iter_text())
                assert chunks.endswith("data: [DONE]\n\n")
        sent = json.loads(captured_requests[-1].content)["messages"]
        assert sent[0]["content"][1]["image_url"]["url"] == DATA_URL

    def test_context_assembly_preserves_selected_interaction_intact(
        self, captured_requests
    ):
        """M6.3: when the multimodal unit is selected, its image survives."""
        body = {
            "model": "client-model",
            "messages": [
                multimodal_message("old screenshot question"),
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "current"},
            ],
        }
        client = build_client(captured_requests)
        with client:
            assert client.post("/v1/chat/completions", json=body).status_code == 200
        sent = json.loads(captured_requests[-1].content)["messages"]
        image_parts = [
            p
            for m in sent
            if isinstance(m.get("content"), list)
            for p in m["content"]
            if p.get("type") == "image_url"
        ]
        assert len(image_parts) == 1  # kept whole or dropped whole — never split

    def test_budget_pressure_drops_whole_unit_never_image_only(self):
        from context_proxy.config import AssemblySettings, RetrievalSettings
        from context_proxy.context.engine import (
            ContextAssemblyEngine,
        )

        big_data_url = "data:image/png;base64," + "A" * 4000
        multimodal_turn = [
            multimodal_message("look at this", big_data_url),
            {"role": "assistant", "content": "seen"},
        ]
        current = [{"role": "user", "content": "next"}]
        engine = ContextAssemblyEngine(
            usable_budget=900,  # forces the multimodal unit out entirely
            settings=AssemblySettings(),
            retrieval_settings=RetrievalSettings(),
        )
        plan = engine.build(history=multimodal_turn, current_request=current)
        contents = [m.get("content") for m in plan.messages]
        assert all(not isinstance(c, list) for c in contents)  # image unit gone
        assert plan.messages[-1]["content"] == "next"

    def test_image_estimate_keeps_large_screenshot_feasible(self):
        """Huge base64 does not starve the budget: request still fits."""
        from context_proxy.config import AssemblySettings, RetrievalSettings
        from context_proxy.context.engine import ContextAssemblyEngine

        giant = "data:image/png;base64," + "A" * 200_000  # would be ~50k chars/4
        current = [multimodal_message("review this screenshot", giant)]
        engine = ContextAssemblyEngine(
            usable_budget=3000,
            settings=AssemblySettings(),
            retrieval_settings=RetrievalSettings(),
        )
        plan = engine.build(history=[], current_request=current)
        assert plan.messages[-1]["content"][1]["image_url"]["url"] == giant
        assert plan.token_estimate <= 3000


class TestMultimodalReconciliationIdempotency:
    def test_identical_multimodal_replay_is_idempotent(self, captured_requests):
        calls: list[list[dict]] = []

        class RecordingStore:
            async def ensure_conversation(self, conversation_id):
                return None

            async def reconcile_history(self, conversation_id, messages, metadata=None):
                calls.append([json.loads(json.dumps(m)) for m in messages])
                return []

        store = RecordingStore()
        body = {
            "model": "client-model",
            "messages": [
                multimodal_message("shot"),
                {"role": "assistant", "content": "ok"},
            ],
        }
        client = build_client(captured_requests, store=store)
        with client as running:
            first = running.post("/v1/chat/completions", json=body)
            second = running.post("/v1/chat/completions", json=body)
        assert first.status_code == second.status_code == 200
        # two reconciles per request: inbound history + assistant capture
        assert len(calls) == 4
        # positional raw equality holds for content arrays across replays
        assert calls[0] == calls[2]
        assert calls[0] == body["messages"]
        assert calls[1] == calls[3]
        assert calls[1][-1]["content"] == "hello"


# --------------------------------------- M6 review: canonical identity fixes


def _tool_call(name: str, arguments: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "I'll inspect that",
        "tool_calls": [
            {"id": f"call_{name}", "type": "function",
             "function": {"name": name, "arguments": arguments}}
        ],
    }


class TestToolCallIdentity:
    """tool_calls always contribute to canonical identity (review §1)."""

    def test_string_content_plus_tool_call_is_fully_canonical(self):
        from context_proxy.context.candidates import message_texts

        text = message_texts([_tool_call("read_file", '{"path":"foo.py"}')])
        assert "read_file" in text
        assert "foo.py" in text
        assert "I'll inspect that" in text

    def test_multimodal_content_plus_tool_call_keeps_both(self):
        from context_proxy.context.candidates import message_texts

        message = _tool_call("read_file", "{}")
        message["content"] = [
            {"type": "text", "text": "I'll inspect that"},
            {"type": "image_url", "image_url": {"url": DATA_URL}},
        ]
        text = message_texts([message])
        assert "read_file" in text
        assert "[image:" in text
        assert "I'll inspect that" in text

    def test_different_arguments_different_identity(self):
        from context_proxy.context.candidates import (
            canonical_text,
            message_texts,
        )

        unit_a = [
            {"role": "user", "content": "go"},
            _tool_call("read_file", '{"path":"foo.py"}'),
            {"role": "tool", "tool_call_id": "call_read_file", "content": "data"},
        ]
        unit_b = [
            {"role": "user", "content": "go"},
            _tool_call("read_file", '{"path":"bar.py"}'),
            {"role": "tool", "tool_call_id": "call_read_file", "content": "data"},
        ]
        assert canonical_text(message_texts(unit_a)) != canonical_text(
            message_texts(unit_b)
        )

    def test_identical_units_same_identity(self):
        from context_proxy.context.candidates import (
            canonical_text,
            message_texts,
        )

        unit = [_tool_call("read_file", '{"path":"foo.py"}')]
        assert canonical_text(message_texts(unit)) == canonical_text(
            message_texts(list(unit))
        )


class TestImageFingerprintRobustness:
    """Full-length SHA-256 fingerprints (review §2)."""

    def test_fingerprint_length_is_64_hex(self):
        from context_proxy.context.candidates import content_texts

        texts = content_texts(multimodal_message("d", DATA_URL))
        image_token = next(t for t in texts if t.startswith("[image:"))
        fingerprint = image_token.removeprefix("[image:").removesuffix("]")
        assert len(fingerprint) == 64
        int(fingerprint, 16)  # hex

    def test_same_url_same_fingerprint(self):
        from context_proxy.context.candidates import content_texts

        assert content_texts(multimodal_message("d", DATA_URL)) == content_texts(
            multimodal_message("d", DATA_URL)
        )

    def test_different_url_different_fingerprint(self):
        from context_proxy.context.candidates import content_texts

        one = content_texts(multimodal_message("d", DATA_URL))
        two = content_texts(multimodal_message("d", DATA_URL + "x"))
        assert one != two


class TestUnknownPartIdentity:
    """Unknown parts carry payload-derived distinct identities (review §3)."""

    def test_same_payload_same_identity_regardless_of_key_order(self):
        from context_proxy.context.candidates import content_texts

        part_a = {"type": "custom_part", "payload": "A", "meta": 1}
        part_b = {"meta": 1, "payload": "A", "type": "custom_part"}
        one = {
            "role": "user",
            "content": [{"type": "text", "text": "t"}, part_a],
        }
        two = {
            "role": "user",
            "content": [{"type": "text", "text": "t"}, part_b],
        }
        assert content_texts(one) == content_texts(two)

    def test_different_payload_different_identity(self):
        from context_proxy.context.candidates import content_texts

        one = {
            "role": "user",
            "content": [
                {"type": "text", "text": "t"},
                {"type": "custom_part", "payload": "A"},
            ],
        }
        two = {
            "role": "user",
            "content": [
                {"type": "text", "text": "t"},
                {"type": "custom_part", "payload": "B"},
            ],
        }
        assert content_texts(one) != content_texts(two)

    def test_raw_payload_never_enters_canonical_text(self):
        from context_proxy.context.candidates import message_texts

        secret_payload = {"type": "custom_part", "payload": "SUPER-RAW-VALUE-123"}
        text = message_texts(
            [{"role": "user", "content": [{"type": "text", "text": "x"}, secret_payload]}]
        )
        assert "SUPER-RAW-VALUE-123" not in text


class TestMultimodalToolInteractionEndToEnd:
    """Combined M3/M4/M6 regression (review §4)."""

    @staticmethod
    def _unit(text: str, url: str, arguments: str) -> list[dict[str, Any]]:
        return [
            multimodal_message(text, url),
            _tool_call("inspect_screenshot", arguments),
            {"role": "tool", "tool_call_id": "call_inspect_screenshot",
             "content": "pixels analyzed"},
            {"role": "assistant", "content": "final answer"},
        ]

    URL_A = "data:image/png;base64,AAAAAAAA"
    URL_B = "data:image/png;base64,BBBBBBBB"

    def test_unit_stays_atomic_and_identity_components_contribute(self):
        from context_proxy.config import AssemblySettings, RetrievalSettings
        from context_proxy.context.engine import (
            ContextAssemblyEngine,
            separate_current_request,
        )

        unit = self._unit("look", self.URL_A, '{"zoom":2}')
        history = [*unit, {"role": "user", "content": "next"}]
        history_msgs, current = separate_current_request(history)

        plan = ContextAssemblyEngine(
            usable_budget=50_000,
            settings=AssemblySettings(),
            retrieval_settings=RetrievalSettings(),
        ).build(history=history_msgs, current_request=current)

        # 1. atomicity: all four roles present, image verbatim upstream
        roles = [m["role"] for m in plan.messages[:4]]
        assert roles == ["user", "assistant", "tool", "assistant"]
        assert plan.messages[0]["content"][1]["image_url"]["url"] == self.URL_A
        # 5. no erroneous dedup inside the assembled request
        blob = json.dumps(plan.messages)
        assert blob.count("pixels analyzed") == 1

    def test_image_change_changes_identity(self):
        from context_proxy.context.candidates import (
            canonical_text,
            message_texts,
        )

        a = canonical_text(message_texts(self._unit("look", self.URL_A, '{"zoom":2}')))
        b = canonical_text(message_texts(self._unit("look", self.URL_B, '{"zoom":2}')))
        assert a != b  # identity A != B

    def test_tool_argument_change_changes_identity(self):
        from context_proxy.context.candidates import (
            canonical_text,
            message_texts,
        )

        a = canonical_text(message_texts(self._unit("look", self.URL_A, '{"zoom":2}')))
        c = canonical_text(message_texts(self._unit("look", self.URL_A, '{"zoom":4}')))
        assert a != c  # identity A != C
