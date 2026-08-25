"""Deterministic capture-overflow regressions (post-0d331f9 review §2).

No LLM involvement: every scenario drives a controlled upstream SSE stream
through the real proxy stack. Contract under test:

    MAX_CAPTURE_BYTES = N
    upstream sends > N bytes  ->  capture disabled
                              ->  ENTIRE stream still reaches the client
                              ->  no partial assistant persistence
                              ->  overflow metric increments

Boundary matrix: exactly N, N + 1, overflow across multiple SSE chunks,
very large stream. The real-provider E2E keeps only a best-effort variant;
this suite is the authoritative overflow coverage.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from fastapi.testclient import TestClient

from context_proxy.config import ServerSettings
from context_proxy.main import create_app
from context_proxy.observability.metrics import REGISTRY
from tests.conftest import UPSTREAM, make_settings

METRIC_LINE = "context_proxy_assistant_capture_overflow_total"

DONE = b"data: [DONE]\n\n"


def sse_event(piece: str) -> bytes:
    payload = json.dumps(
        {"choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
    )
    return f"data: {payload}\n\n".encode()


def overflow_metric() -> int | None:
    text = REGISTRY.render()
    line = next((ln for ln in text.splitlines() if ln.startswith(METRIC_LINE)), None)
    return int(line.rsplit(" ", 1)[1]) if line else None


class RecordingStore:
    """Minimal dict-backed store mirroring reconcile semantics."""

    def __init__(self) -> None:
        self.conversations: dict[str, list[dict[str, Any]]] = {}

    async def ping(self):
        return None

    async def ensure_conversation(self, conversation_id):
        self.conversations.setdefault(conversation_id, [])

    async def reconcile_history(self, conversation_id, messages, metadata=None):
        bucket = self.conversations.setdefault(conversation_id, [])
        overlap = min(len(bucket), len(messages))
        for index in range(overlap):
            if bucket[index] != messages[index]:
                from context_proxy.conversation.store import HistoryDivergenceError

                raise HistoryDivergenceError(conversation_id, index)
        bucket.extend(messages[len(bucket) :])
        return []

    async def get_messages(self, conversation_id):
        return list(self.conversations.get(conversation_id, []))


def streaming_client(chunks_source, *, max_capture_bytes: int, store: RecordingStore):
    def handler(request: httpx.Request) -> httpx.Response:
        async def agen():
            for chunk in chunks_source():
                yield chunk

        return httpx.Response(
            200,
            content=agen(),
            headers={"content-type": "text/event-stream"},
        )

    settings = make_settings().model_copy(
        update={"server": ServerSettings(port=8080, max_capture_bytes=max_capture_bytes)}
    )
    app = create_app(
        settings,
        llm_client=httpx.AsyncClient(base_url=UPSTREAM, transport=httpx.MockTransport(handler)),
        store=store,
    )
    return app


CONV = "77777777-7777-7777-7777-777777777777"


# ------------------------------------------------------- unit boundaries


class TestAssistantCaptureBoundaries:
    def _capture(self, max_bytes):
        from context_proxy.capture import AssistantCapture

        return AssistantCapture(max_bytes=max_bytes)

    def test_exactly_max_bytes_is_captured(self):
        body = b"x" * 500
        capture = self._capture(max_bytes=len(body))
        capture.feed(body)
        assert capture.overflowed is False
        assert bytes(capture._buffer) == body

    def test_limit_plus_one_overflows(self):
        body = b"y" * 501
        capture = self._capture(max_bytes=500)
        capture.feed(body)
        assert capture.overflowed is True
        assert capture.finalize() is None  # partial never persists

    def test_overflow_across_chunk_boundary(self):
        capture = self._capture(max_bytes=10)
        capture.feed(b"a" * 6)
        assert capture.overflowed is False
        capture.feed(b"b" * 5)  # 6 + 5 > 10: overflow lands mid-chunk-boundary
        assert capture.overflowed is True

    def test_capture_stays_disabled_after_overflow_single_increment(self):
        REGISTRY.reset()
        capture = self._capture(max_bytes=10)
        capture.feed(b"a" * 11)          # overflow here
        for _ in range(100):
            capture.feed(b"more data")   # passthrough continues, capture off
        assert capture.overflowed is True
        assert overflow_metric() == 1    # exactly one increment per response

    def test_finalize_after_disabled_capture_persists_nothing(self):
        capture = self._capture(max_bytes=10)
        capture.feed(b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n')
        capture.feed(b"data: [DONE]\n\n")
        assert capture.overflowed is True
        assert capture.finalize() is None


# ---------------------------------------------------- route-level contract


class TestStreamingCaptureOverflowRoute:
    def test_stream_total_exactly_at_limit_persists(self):
        REGISTRY.reset()
        first = sse_event("hello ")
        # cap sized to the exact byte total of the upstream stream
        cap = len(first) + len(DONE)

        store = RecordingStore()
        app = streaming_client(lambda: [first, DONE], max_capture_bytes=cap, store=store)
        with TestClient(app) as client:
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
                body = b"".join(response.iter_bytes())

        assert body == first + DONE                       # complete passthrough
        assert overflow_metric() is None                  # never tripped
        messages = store.conversations[CONV]
        assert [(m["role"], m["content"]) for m in messages] == [
            ("user", "hi"),
            ("assistant", "hello "),
        ]

    def test_one_byte_over_limit_overflows_but_passthrough_completes(self):
        REGISTRY.reset()
        first = sse_event("hello ")
        cap = len(first) + len(DONE) - 1                  # exactly one short

        store = RecordingStore()
        app = streaming_client(lambda: [first, DONE], max_capture_bytes=cap, store=store)
        with TestClient(app) as client:
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
                body = b"".join(response.iter_bytes())

        assert body == first + DONE                       # client stream COMPLETE
        assert body.endswith(b"data: [DONE]\n\n")
        assert overflow_metric() == 1
        assert [(m["role"], m["content"]) for m in store.conversations[CONV]] == [
            ("user", "hi")
        ]  # no assistant row

    def test_overflow_across_multiple_sse_chunks_mid_event(self):
        """Overflow fires while a single SSE event is split across fragments."""
        REGISTRY.reset()
        first = sse_event("abcdefgh")
        split = len(first) // 3                           # event spans 3 fragments
        fragments = [first[:split], first[split : 2 * split], first[2 * split:], DONE]
        cap = split                                       # second fragment overflows

        store = RecordingStore()
        app = streaming_client(lambda: fragments, max_capture_bytes=cap, store=store)
        with TestClient(app) as client:
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
                body = b"".join(response.iter_bytes())

        assert body == b"".join(fragments)                # every byte delivered
        assert overflow_metric() == 1
        assert [(m["role"], m["content"]) for m in store.conversations[CONV]] == [
            ("user", "hi")
        ]

    def test_very_large_stream_full_passthrough_no_partial_persistence(self):
        REGISTRY.reset()
        piece = sse_event("z" * 80)                       # ~120 bytes/event
        n_events = 200                                    # far beyond the bound
        chunks = [piece] * n_events + [DONE]
        cap = 500                                         # << total (~24 KB)

        store = RecordingStore()
        app = streaming_client(lambda: chunks, max_capture_bytes=cap, store=store)
        with TestClient(app) as client:
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
                body = b"".join(response.iter_bytes())

        assert body == piece * n_events + DONE            # byte-exact, complete
        assert overflow_metric() == 1                     # disabled once, not 200x
        assert [(m["role"], m["content"]) for m in store.conversations[CONV]] == [
            ("user", "hi")
        ]

    def test_within_limit_large_stream_persists_complete_content(self):
        REGISTRY.reset()
        piece = sse_event("ok ")
        chunks = [piece] * 20 + [DONE]
        cap = len(piece) * 20 + len(DONE) + 1             # headroom above total

        store = RecordingStore()
        app = streaming_client(lambda: chunks, max_capture_bytes=cap, store=store)
        with TestClient(app) as client:
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
                body = b"".join(response.iter_bytes())

        assert body == piece * 20 + DONE
        assert overflow_metric() is None
        messages = store.conversations[CONV]
        assert messages[-1]["content"] == "ok " * 20      # reconstructed intact


# --------------------------------------------- non-streaming unaffected


class TestBufferedResponsesIgnoreCaptureBound:
    def test_buffered_response_not_bounded_by_max_capture_bytes(self):
        REGISTRY.reset()

        big_content = "b" * 4096
        chat_response = {
            "id": "chatcmpl-big",
            "object": "chat.completion",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": big_content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1024, "total_tokens": 1029},
        }

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=chat_response)

        settings = make_settings().model_copy(
            update={"server": ServerSettings(port=8080, max_capture_bytes=64)}
        )
        store = RecordingStore()
        app = create_app(
            settings,
            llm_client=httpx.AsyncClient(
                base_url=UPSTREAM, transport=httpx.MockTransport(handler)
            ),
            store=store,
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                    "conversation_id": CONV,
                },
            )

        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == big_content
        assert overflow_metric() is None
        assert store.conversations[CONV][-1]["content"] == big_content
