from __future__ import annotations

import asyncio

import httpx
import pytest
from conftest import SSE_BODY, client_for_handler
from helpers import chat_payload


def _tracked_sse_response(
    chunks_source,
    events: list[str] | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    response = httpx.Response(
        200,
        content=chunks_source(),
        headers={"content-type": "text/event-stream", **(headers or {})},
    )
    if events is not None:
        original_aclose = response.aclose

        async def tracked_aclose() -> None:
            events.append("closed")
            await original_aclose()

        response.aclose = tracked_aclose  # type: ignore[method-assign]
    return response


def _provider_with(handler):
    from context_proxy.config import EndpointSettings
    from context_proxy.providers.llm import OpenAICompatibleLLMProvider

    base_url = "http://upstream.test/v1"
    client = httpx.AsyncClient(base_url=base_url, transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleLLMProvider(EndpointSettings(base_url=base_url), client=client)
    return provider, client


def test_sse_streaming_incremental_with_headers():
    first, second = SSE_BODY[:20], SSE_BODY[20:]

    def handler(request: httpx.Request) -> httpx.Response:
        async def agen():
            yield first
            yield second

        return _tracked_sse_response(agen, headers={"x-request-id": "req-sse"})

    client = client_for_handler(handler)
    with client.stream(
        "POST", "/v1/chat/completions", json=chat_payload(stream=True)
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["x-request-id"] == "req-sse"
        body = b"".join(response.iter_bytes())
    assert body == SSE_BODY


def test_sse_content_type_preserved_when_nonstandard_suffix():
    def handler(request: httpx.Request) -> httpx.Response:
        async def agen():
            yield SSE_BODY

        return _tracked_sse_response(
            agen, headers={"content-type": "text/event-stream; charset=utf-8"}
        )

    client = client_for_handler(handler)
    with client.stream(
        "POST", "/v1/chat/completions", json=chat_payload(stream=True)
    ) as response:
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


def test_mid_stream_transport_error_closes_upstream_response():
    events: list[str] = []
    received = b""

    async def failing_agen():
        yield b'data: {"partial": true}\n\n'
        raise httpx.ReadError("upstream connection reset")

    def handler(request: httpx.Request) -> httpx.Response:
        return _tracked_sse_response(failing_agen, events)

    provider, _unused = _provider_with(handler)

    async def scenario() -> bytes:
        stream = await provider.open_stream(chat_payload())  # type: ignore[attr-defined]
        collected = b""
        with pytest.raises(httpx.ReadError):
            async for chunk in stream.iter_bytes():
                collected += chunk
        await provider.aclose()  # type: ignore[attr-defined]
        return collected

    received = asyncio.run(scenario())

    assert b"partial" in received
    assert "closed" in events


def test_downstream_disconnect_closes_upstream_response():
    """Client stops iterating mid-stream -> upstream connection must be released."""

    events: list[str] = []
    chunks_sent = 0

    async def endless_agen():
        nonlocal chunks_sent
        while True:
            chunks_sent += 1
            yield b"data: x\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return _tracked_sse_response(endless_agen, events)

    provider, _unused = _provider_with(handler)

    async def scenario() -> int:
        stream = await provider.open_stream(chat_payload())  # type: ignore[attr-defined]
        iterator = stream.iter_bytes()
        await iterator.__anext__()  # one chunk consumed, then "disconnect"
        await iterator.aclose()  # generator close == disconnect cancellation path
        await provider.aclose()  # type: ignore[attr-defined]
        return chunks_sent

    sent = asyncio.run(scenario())

    assert sent >= 1
    assert "closed" in events


def test_task_cancellation_mid_stream_closes_upstream_response():
    """Hard cancellation: consumer task killed mid-iteration -> upstream closed.

    Limitation: a true ASGI client disconnect cannot be simulated with
    TestClient; task cancellation exercises the same generator-cancellation
    path that Starlette's disconnect listener triggers.
    """

    events: list[str] = []

    async def endless_agen():
        while True:
            yield b"data: x\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return _tracked_sse_response(endless_agen, events)

    provider, _client = _provider_with(handler)

    async def scenario() -> None:
        stream = await provider.open_stream(chat_payload())  # type: ignore[attr-defined]
        consumer = asyncio.create_task(consume_few(stream))
        await asyncio.sleep(0)  # let it start consuming
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
        await provider.aclose()  # type: ignore[attr-defined]

    async def consume_few(stream) -> None:
        async for _chunk in stream.iter_bytes():
            await asyncio.sleep(0)  # simulate slow downstream

    asyncio.run(scenario())

    assert "closed" in events


def test_full_sse_body_forwarded_through_http_stack():
    def handler(request: httpx.Request) -> httpx.Response:
        async def agen():
            yield SSE_BODY

        return _tracked_sse_response(agen)

    client = client_for_handler(handler)
    with client.stream(
        "POST", "/v1/chat/completions", json=chat_payload(stream=True)
    ) as response:
        headers = dict(response.headers)
        body = b"".join(response.iter_bytes())
    assert body.endswith(b"data: [DONE]\n\n")
    assert "transfer-encoding" not in {k.lower() for k in headers}
