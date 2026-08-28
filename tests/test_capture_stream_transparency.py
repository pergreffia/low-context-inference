from __future__ import annotations

import httpx
import pytest

from context_proxy.capture import PersistingLLMStream
from context_proxy.conversation.store import HistoryDivergenceError
from context_proxy.memory.errors import PersistenceInfrastructureError


class FakeStream:
    status_code = 200
    media_type = "text/event-stream"

    def __init__(self, chunks: list[bytes], error: BaseException | None = None):
        self.chunks = chunks
        self.error = error
        self.closed = False

    def passthrough_headers(self) -> dict[str, str]:
        return {"content-type": self.media_type}

    async def iter_bytes(self):
        for chunk in self.chunks:
            yield chunk
        if self.error is not None:
            raise self.error

    async def aclose(self):
        self.closed = True


async def collect(stream: PersistingLLMStream) -> bytes:
    chunks: list[bytes] = []
    async for chunk in stream.iter_bytes():
        chunks.append(chunk)
    return b"".join(chunks)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        HistoryDivergenceError("conversation", 2),
        PersistenceInfrastructureError("postgres unavailable"),
        TypeError("persistence programming bug"),
    ],
)
async def test_persistence_failure_never_rewrites_completed_stream(failure):
    upstream = FakeStream([b"data: first\n\n", b"data: [DONE]\n\n"])
    calls: list[tuple[dict | None, dict]] = []

    async def persist(message, metadata):
        calls.append((message, metadata))
        raise failure

    wrapped = PersistingLLMStream(upstream, persist)

    body = await collect(wrapped)

    assert body == b"data: first\n\ndata: [DONE]\n\n"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_incomplete_upstream_stream_propagates_original_error_and_skips_persistence():
    failure = httpx.ReadError("upstream connection reset")
    upstream = FakeStream([b"data: partial\n\n"], error=failure)
    calls: list[tuple[dict | None, dict]] = []

    async def persist(message, metadata):
        calls.append((message, metadata))

    wrapped = PersistingLLMStream(upstream, persist)

    with pytest.raises(httpx.ReadError) as exc_info:
        await collect(wrapped)

    assert exc_info.value is failure
    assert calls == [(None, {})]


@pytest.mark.asyncio
async def test_downstream_cancellation_does_not_turn_into_persistence_error():
    upstream = FakeStream([b"data: partial\n\n"])
    calls: list[tuple[dict | None, dict]] = []

    async def persist(message, metadata):
        calls.append((message, metadata))
        raise RuntimeError("must not mask cancellation")

    wrapped = PersistingLLMStream(upstream, persist)
    iterator = wrapped.iter_bytes()
    assert await iterator.__anext__() == b"data: partial\n\n"
    await iterator.aclose()

    # Closing the consumer generator enters finally; persistence errors are
    # swallowed because the stream was already handed to the client.
    assert len(calls) == 1
