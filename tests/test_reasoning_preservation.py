"""Reasoning-field preservation regressions (OpenCode divergence fix).

Root cause this locks down: streaming capture reconstructed assistant
messages without provider reasoning fields (`reasoning` on Ollama/bonsai,
`reasoning_content` DeepSeek-style, `reasoning_text` elsewhere), so the
persisted history diverged from the exact response OpenCode replayed →
false `history_conflict`.

Canonicalization decision: reasoning is re-emitted under its ORIGINAL
provider key (no renaming) with values concatenated in arrival order — the
strict positional reconciliation requires byte-faithful shapes.
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from context_proxy.capture import AssistantCapture
from context_proxy.main import create_app
from tests.conftest import UPSTREAM, make_settings

CONV = "eeeeeeee-0000-0000-0000-000000000000"


def sse(delta: dict) -> bytes:
    return f"data: {json.dumps({'choices': [{'index': 0, 'delta': delta}]})}\n\n".encode()


DONE = b"data: [DONE]\n\n"


class RecordingStore:
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
                raise HistoryDivergenceError(
                    conversation_id,
                    index,
                    persisted=bucket[index],
                    incoming=messages[index],
                )
        bucket.extend(messages[len(bucket):])
        return []

    async def get_messages(self, conversation_id):
        return list(self.conversations.get(conversation_id, []))


def stream_app(chunks: list[bytes], store: RecordingStore):
    def handler(request: httpx.Request) -> httpx.Response:
        async def agen():
            for chunk in chunks:
                yield chunk

        return httpx.Response(
            200, content=agen(), headers={"content-type": "text/event-stream"}
        )

    app = create_app(
        make_settings(),
        llm_client=httpx.AsyncClient(
            base_url=UPSTREAM, transport=httpx.MockTransport(handler)
        ),
        store=store,
    )
    return app


def run_stream(client: TestClient) -> bytes:
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "m",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
            "conversation_id": CONV,
        },
    ) as response:
        assert response.status_code == 200
        return b"".join(response.iter_bytes())


# --------------------------------------------------- capture-layer units


class TestCaptureReasoningPreservation:
    def test_a_reasoning_content_multi_chunk_with_content(self):
        capture = AssistantCapture()
        for chunk in (
            {"role": "assistant"},
            {"reasoning_content": "step one. "},
            {"reasoning_content": "step two."},
            {"content": "final answer"},
        ):
            capture.feed(sse(chunk))
        message = capture.finalize()

        assert message["role"] == "assistant"
        assert message["content"] == "final answer"
        assert message["reasoning_content"] == "step one. step two."
        # no duplicate/renamed fields invented
        assert set(message) == {"role", "content", "reasoning_content"}

    @pytest.mark.parametrize("key", ["reasoning", "reasoning_text"])
    def test_b_alternate_reasoning_keys_preserved_as_emitted(self, key):
        """No renaming: whatever key the provider used comes back verbatim."""
        capture = AssistantCapture()
        for chunk in ({"role": "assistant"}, {key: "think"}, {key: "ing"}):
            capture.feed(sse(chunk))
        message = capture.finalize()
        assert message[key] == "thinking"
        assert not any(
            other in message for other in ("reasoning", "reasoning_content", "reasoning_text")
            if other != key
        )

    def test_c_reasoning_plus_tool_calls_intact(self):
        capture = AssistantCapture()
        chunks = [
            {"role": "assistant"},
            {"reasoning_content": "need file"},
            {"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                             "function": {"name": "read_file",
                                          "arguments": '{"path":"a.py"}'}}]},
            {"finish_reason": "tool_calls"},
        ]
        for chunk in chunks:
            capture.feed(sse(chunk))
        message = capture.finalize()

        assert message["reasoning_content"] == "need file"
        assert message["tool_calls"] == [
            {"id": "c1", "type": "function",
             "function": {"name": "read_file", "arguments": '{"path":"a.py"}'}}
        ]

    def test_d_multi_chunk_accumulation_is_exact_ordered_concatenation(self):
        pieces = ["alpha ", "beta ", "gamma"]
        first = AssistantCapture()
        for piece in pieces:
            first.feed(sse({"reasoning": piece}))
        second = AssistantCapture()
        for piece in reversed(pieces):
            second.feed(sse({"reasoning": piece}))

        assert first.finalize()["reasoning"] == "alpha beta gamma"
        assert second.finalize()["reasoning"] == "gammabeta alpha "  # order matters

    def test_no_reasoning_fields_when_upstream_sent_none(self):
        capture = AssistantCapture()
        for chunk in ({"role": "assistant"}, {"content": "plain"}):
            capture.feed(sse(chunk))
        message = capture.finalize()
        assert set(message) == {"role", "content"}   # nothing invented


# ------------------------------- route-level OpenCode replay regression


class TestOpenCodeReplayScenario:
    def test_streamed_reasoning_replay_produces_no_false_conflict(self):
        """The actual failure mode: streamed assistant (with reasoning) is
        persisted, then OpenCode replays it EXACTLY → must reconcile."""
        chunks = [
            sse({"role": "assistant"}),
            sse({"reasoning_content": "chain of thought "}),
            sse({"reasoning_content": "continues"}),
            sse({"content": "the answer"}),
            DONE,
        ]
        store = RecordingStore()
        app = stream_app(chunks, store)
        with TestClient(app) as client:
            body = run_stream(client)
            assert body.endswith(DONE)

            persisted = store.conversations[CONV]
            assert persisted[-1]["reasoning_content"] == "chain of thought continues"

            # OpenCode-style next turn: full history incl. the EXACT assistant msg
            replay = [*persisted, {"role": "user", "content": "next"}]
            response = client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": replay, "conversation_id": CONV},
            )
            assert response.status_code == 200       # NO false history_conflict

    def test_changed_reasoning_still_diverges(self):
        chunks = [sse({"role": "assistant"}),
                  sse({"reasoning_content": "A"}), sse({"content": "x"}), DONE]
        store = RecordingStore()
        app = stream_app(chunks, store)
        with TestClient(app) as client:
            run_stream(client)

            tampered = [dict(store.conversations[CONV][0]),
                        {"role": "assistant", "content": "x",
                         "reasoning_content": "B"},
                        {"role": "user", "content": "next"}]
            response = client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": tampered, "conversation_id": CONV},
            )
        assert response.status_code == 409           # real difference still caught
        error = response.json()["error"]
        assert "different_fields=['reasoning_content']" in error["message"]

    def test_idempotent_triple_replay_with_reasoning_and_tools(self):
        history = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "with thought",
             "reasoning_content": "because", 
             "tool_calls": [{"id": "t1", "type": "function",
                             "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "t1", "content": "res"},
            {"role": "user", "content": "again"},
        ]
        store = RecordingStore()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": "ok"},
                             "finish_reason": "stop"}]
            })

        app = create_app(
            make_settings(),
            llm_client=httpx.AsyncClient(
                base_url=UPSTREAM, transport=httpx.MockTransport(handler)
            ),
            store=store,
        )
        conv = "eeeeeeee-1111-1111-1111-111111111111"
        snapshots = []
        with TestClient(app) as client:
            for _turn in range(3):                  # persist + 2 identical replays
                response = client.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": history,
                          "conversation_id": conv},
                )
                assert response.status_code == 200
                snapshots.append([json.dumps(m, sort_keys=True)
                                  for m in store.conversations[conv]])
        # turn 1 persists history + its own assistant answer; replays are
        # idempotent (no duplicates, no divergence, identical canonical form)
        assert len(store.conversations[conv]) == len(history) + 1
        assert snapshots[1] == snapshots[2] == snapshots[0]


# ------------------------------------------- divergence diagnostics shape


class TestDivergenceDiagnostics:
    def test_error_carries_fingerprints_and_field_names_not_contents(self):
        from context_proxy.conversation.store import HistoryDivergenceError

        persisted = {"role": "assistant", "content": "SECRET-PROMPT-ALPHA",
                     "reasoning_content": "SECRET-THOUGHT"}
        incoming = {"role": "assistant", "content": "SECRET-BETA",
                    "reasoning_content": "SECRET-GAMMA"}
        error = HistoryDivergenceError("conv", 3, persisted=persisted,
                                       incoming=incoming)

        text = str(error)
        assert "persisted_sha256=" in text and "incoming_sha256=" in text
        assert "different_fields=['content', 'reasoning_content']" in text
        for secret in ("SECRET-PROMPT-ALPHA", "SECRET-THOUGHT",
                       "SECRET-BETA", "SECRET-GAMMA"):
            assert secret not in text

    def test_fingerprints_deterministic_and_discriminating(self):
        from context_proxy.conversation.store import _message_fingerprint

        a = {"role": "assistant", "content": "x"}
        b = {"content": "x", "role": "assistant"}          # same message, reordered keys
        c = {"role": "assistant", "content": "y"}
        assert _message_fingerprint(a) == _message_fingerprint(b)
        assert _message_fingerprint(a) != _message_fingerprint(c)
