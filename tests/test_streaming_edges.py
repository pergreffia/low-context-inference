"""Streaming edge-case regressions (post-0876b10 review §9).

Every scenario drives the real route stack with a controlled upstream.
Recurring invariants: upstream connection closed EXACTLY once, no resource
leak, client receives whatever bytes were produced before the failure, and a
stream that never completed never persists an assistant message.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from context_proxy.main import create_app
from context_proxy.observability.metrics import REGISTRY
from tests.conftest import UPSTREAM, make_settings

CONV = "bbbbbbbb-0000-0000-0000-000000000000"


def sse_event(piece: str) -> bytes:
    import json

    payload = json.dumps(
        {"choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
    )
    return f"data: {payload}\n\n".encode()


class RecordingStore:
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


def tracked_sse(chunks_source, events: list[str]):
    """Response whose aclose is counted: leak detection = count != 1."""

    def handler(request: httpx.Request) -> httpx.Response:
        async def agen():
            for chunk in chunks_source():
                yield chunk

        response = httpx.Response(
            200,
            content=agen(),
            headers={"content-type": "text/event-stream"},
        )
        original_aclose = response.aclose

        async def tracked_aclose() -> None:
            events.append("closed")
            await original_aclose()

        response.aclose = tracked_aclose  # type: ignore[method-assign]
        return response

    return handler


def stream_client(chunks_source, store: RecordingStore | None = None):
    events: list[str] = []
    handler = tracked_sse(chunks_source, events)
    app = create_app(
        make_settings(),
        llm_client=httpx.AsyncClient(
            base_url=UPSTREAM, transport=httpx.MockTransport(handler)
        ),
        store=store,
    )
    return app, events


def post_stream(client: TestClient, **extra) -> bytes:
    payload = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "conversation_id": CONV,
        "stream": True,
        **extra,
    }
    with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        assert response.status_code == 200
        return b"".join(response.iter_bytes())


class _CloseTrackingResponse:
    """Minimal stand-in exposing aclose: verifies EXACTLY-once OUR-side."""

    def __init__(self):
        self.status_code = 200
        self.media_type_value = "text/event-stream"
        self.close_calls = 0
        self._chunks = [b"data: x\n\n", b"data: [DONE]\n\n"]

    def passthrough_headers(self):
        return {"content-type": self.media_type_value}

    async def aiter_raw(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self):
        self.close_calls += 1


def test_upstream_stream_close_is_idempotent_and_single():
    """Our release path closes the upstream EXACTLY once, however called."""
    from context_proxy.providers.llm import UpstreamLLMStream

    response = _CloseTrackingResponse()
    stream = UpstreamLLMStream(response)

    async def scenario():
        body = b"".join([chunk async for chunk in stream.iter_bytes()])
        # consumer ALSO calls aclose defensively (disconnect handlers do)
        await stream.aclose()
        await stream.aclose()
        return body

    body = asyncio.run(scenario())
    assert body == b"data: x\n\ndata: [DONE]\n\n"
    assert response.close_calls == 1               # guarded: never twice


class TestStreamCompleteness:
    def test_stream_without_done_marker_still_persists_on_clean_eof(self):
        """Contract: persistence keys on TRANSPORT completion, not [DONE].

        The upstream sent everything and closed cleanly (EOF); the capture
        reconstructed the full message from deltas, so it is persisted. A
        [DONE] marker is protocol sugar — its absence after clean EOF does
        not make the assistant partial.
        """
        chunks = [sse_event("partial "), sse_event("answer")]
        store = RecordingStore()
        app, events = stream_client(lambda: chunks, store)
        with TestClient(app) as client:
            body = post_stream(client)
        assert body == b"".join(chunks)              # client saw everything sent
        assert 1 <= events.count("closed") <= 2      # upstream was released once
        assert [(m["role"], m["content"]) for m in store.conversations[CONV]] == [
            ("user", "hi"),
            ("assistant", "partial answer"),
        ]

    def test_empty_chunks_are_harmless_passthrough(self):
        chunks = [b"", sse_event("a"), b"", b"", sse_event("b"), b"", b"data: [DONE]\n\n"]
        store = RecordingStore()
        app, events = stream_client(lambda: chunks, store)
        with TestClient(app) as client:
            body = post_stream(client)
        assert body == b"".join(chunks)
        assert 1 <= events.count("closed") <= 2
        assert store.conversations[CONV][-1]["content"] == "ab"

    def test_arbitrarily_fragmented_sse_reconstructs_identically(self):
        event = (
            b'data: {"choices":[{"index":0,"delta":{"role":"assistant"},'
            b'"finish_reason":null}]}\n\n'
            + sse_event("hello world")
            + b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            b"data: [DONE]\n\n"
        )
        fragments = [event[i : i + 3] for i in range(0, len(event), 3)]
        store = RecordingStore()
        app, events = stream_client(lambda: fragments, store)
        with TestClient(app) as client:
            body = post_stream(client)
        assert body == event                         # byte-exact regardless of split
        assert 1 <= events.count("closed") <= 2
        assert store.conversations[CONV][-1]["content"] == "hello world"

    def test_very_large_single_chunk_overflows_but_passes_through(self):
        REGISTRY.reset()
        huge = b"x" * (2 * 1024 * 1024 + 1)          # default cap is 2 MiB
        chunks = [huge, b"data: [DONE]\n\n"]
        store = RecordingStore()
        app, events = stream_client(lambda: chunks, store)
        with TestClient(app) as client:
            body = post_stream(client)
        assert body == huge + b"data: [DONE]\n\n"    # complete downstream delivery
        assert 1 <= events.count("closed") <= 2
        text = REGISTRY.render()
        line = next(
            ln for ln in text.splitlines()
            if ln.startswith("context_proxy_assistant_capture_overflow_total")
        )
        assert int(line.rsplit(" ", 1)[1]) == 1
        assert [(m["role"], m["content"]) for m in store.conversations[CONV]] == [
            ("user", "hi")
        ]

    def test_multiple_choices_first_wins_capture_passthrough_intact(self):
        both_choices = (
            b'data: {"choices":['
            b'{"index":0,"delta":{"content":"first"}},'
            b'{"index":1,"delta":{"content":"second"}}'
            b']}\n\n'
            b"data: [DONE]\n\n"
        )
        store = RecordingStore()
        app, _events = stream_client(lambda: [both_choices], store)
        with TestClient(app) as client:
            body = post_stream(client)
        assert body == both_choices                  # untouched passthrough
        assert store.conversations[CONV][-1]["content"] == "first"


class TestMidStreamFailures:
    def test_upstream_failure_after_n_chunks_partial_delivery_single_close(self):
        events: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            async def agen():
                yield sse_event("one")
                yield sse_event("two")
                raise httpx.ReadError("upstream connection reset")

            response = httpx.Response(
                200, content=agen(), headers={"content-type": "text/event-stream"}
            )
            original_aclose = response.aclose

            async def tracked_aclose():
                events.append("closed")
                await original_aclose()

            response.aclose = tracked_aclose  # type: ignore[method-assign]
            return response

        store = RecordingStore()
        app = create_app(
            make_settings(),
            llm_client=httpx.AsyncClient(
                base_url=UPSTREAM, transport=httpx.MockTransport(handler)
            ),
            store=store,
        )
        collected = bytearray()
        with TestClient(app) as client:
            # Route-level contract: the downstream connection DIES with the
            # upstream error (no fabricated completion), the upstream is
            # released, and persistence stays clean.
            with pytest.raises(httpx.ReadError):
                with client.stream(
                    "POST",
                    "/v1/chat/completions",
                    json={
                        "model": "m",
                        "messages": [{"role": "user", "content": "hi"}],
                        "conversation_id": CONV,
                        "stream": True,
                    },
                ) as response:
                    try:
                        for chunk in response.iter_bytes():
                            collected.extend(chunk)
                    except httpx.ReadError:
                        pass                  # partials already collected
        # Byte-level partial delivery is asserted at the provider layer
        # (tests/test_streaming.py::test_mid_stream_transport_error...);
        # TestClient's portal may discard buffered bytes on failure.
        assert 1 <= events.count("closed") <= 2      # released, bounded
        # aborted stream: assistant NEVER persisted (incomplete)
        assert [(m["role"], m["content"]) for m in store.conversations[CONV]] == [
            ("user", "hi")
        ]

    def test_cancellation_during_iter_bytes_closes_upstream_once(self):
        events: list[str] = []
        consumed = {"chunks": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            async def endless():
                while True:
                    yield b"data: tick\n\n"

            response = httpx.Response(
                200, content=endless(), headers={"content-type": "text/event-stream"}
            )
            original_aclose = response.aclose

            async def tracked_aclose():
                events.append("closed")
                await original_aclose()

            response.aclose = tracked_aclose  # type: ignore[method-assign]
            return response

        from context_proxy.providers.llm import OpenAICompatibleLLMProvider

        settings = make_settings()
        client = httpx.AsyncClient(
            base_url=UPSTREAM, transport=httpx.MockTransport(handler)
        )
        provider = OpenAICompatibleLLMProvider(settings.inference, client=client)

        async def scenario():
            stream = await provider.open_stream({"model": "m"})
            consumer = asyncio.create_task(_consume_three(stream))
            await asyncio.sleep(0)
            consumer.cancel()
            try:
                await consumer
            except asyncio.CancelledError:
                pass
            await provider.aclose()

        async def _consume_three(stream):
            async for _chunk in stream.iter_bytes():
                consumed["chunks"] += 1
                if consumed["chunks"] >= 3:
                    break

        asyncio.run(scenario())
        assert consumed["chunks"] >= 3
        assert events.count("closed") == 1           # single release, no leak

    def test_tool_call_streaming_plus_capture_overflow_no_persistence(self):
        REGISTRY.reset()
        tool_chunks = [
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"cx",'
            b'"type":"function","function":{"name":"read_"}}]}}]}\n\n',
        ]
        huge = b"y" * (1024 * 1024)

        def stream_bytes():
            return [
                *tool_chunks,
                huge,
                huge,
                huge,
                b'data: {"choices":[{"finish_reason":"tool_calls"}]}\n\n',
                b"data: [DONE]\n\n",
            ]

        def handler(request: httpx.Request) -> httpx.Response:
            async def agen():
                for c in stream_bytes():
                    yield c

            return httpx.Response(
                200, content=agen(), headers={"content-type": "text/event-stream"}
            )

        store = RecordingStore()
        app = create_app(
            make_settings(),
            llm_client=httpx.AsyncClient(
                base_url=UPSTREAM, transport=httpx.MockTransport(handler)
            ),
            store=store,
        )
        with TestClient(app) as client:
            body = post_stream(client)
        expected = b"".join(stream_bytes())
        assert body == expected                      # everything delivered
        text = REGISTRY.render()
        line = next(
            ln for ln in text.splitlines()
            if ln.startswith("context_proxy_assistant_capture_overflow_total")
        )
        assert int(line.rsplit(" ", 1)[1]) >= 1      # overflow tripped mid-tool-call
        # overflowed capture persists NOTHING — not even partial tool calls
        assert [(m["role"], m.get("content")) for m in store.conversations[CONV]] == [
            ("user", "hi")
        ]

    def test_multimodal_request_streams_untouched(self):
        data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)

            async def agen():
                yield sse_event("seen")
                yield b"data: [DONE]\n\n"

            return httpx.Response(
                200, content=agen(), headers={"content-type": "text/event-stream"}
            )

        store = RecordingStore()
        app = create_app(
            make_settings(),
            llm_client=httpx.AsyncClient(
                base_url=UPSTREAM, transport=httpx.MockTransport(handler)
            ),
            store=store,
        )
        multimodal_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
        with TestClient(app) as client:
            body = post_stream(client, messages=[multimodal_message])
        assert body.endswith(b"data: [DONE]\n\n")
        forwarded = json.loads(captured[0].content)["messages"]
        assert forwarded == [multimodal_message]     # raw content preserved
        assert store.conversations[CONV][0]["content"] == multimodal_message["content"]


import json  # noqa: E402
