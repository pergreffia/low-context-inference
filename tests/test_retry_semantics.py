"""Upstream retry-semantics regressions (post-0876b10 review §3).

Contract: a POST that may already have reached the provider is NEVER retried.

    ConnectError / ConnectTimeout  -> provably pre-send  -> retried + breaker
    ReadTimeout / WriteError /
    RemoteProtocolError            -> possibly post-send -> single attempt,
                                      no breaker accounting
    HTTP 4xx/5xx                   -> answers           -> no retry

Regression anchor: provider receives the POST once, then the connection
times out reading the response — the proxy must NOT send a second POST.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from context_proxy.config import ResilienceSettings
from context_proxy.providers.errors import UpstreamUnavailable
from context_proxy.providers.llm import RETRYABLE_TRANSPORT_ERRORS, OpenAICompatibleLLMProvider
from context_proxy.providers.resilience import CircuitBreaker
from tests.conftest import make_settings


def _provider(attempts: list[int], fail_after_send: bool = False):
    """Provider counting POSTs; raises after acknowledging receipt."""
    settings = make_settings().model_copy(
        update={
            "resilience": ResilienceSettings(max_retries=2, backoff_base_seconds=0.0)
        }
    )
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        attempts.append(calls["n"])
        if not fail_after_send:
            raise httpx.ConnectError("connection refused")
        return _read_timeout_response()

    client = httpx.AsyncClient(
        base_url="http://upstream.test/v1", transport=httpx.MockTransport(handler)
    )
    provider = OpenAICompatibleLLMProvider(
        settings.inference, client=client, resilience=settings.resilience
    )
    return provider, calls


def _read_timeout_response():
    """A 200 response whose body read stalls: simulates post-send failure."""
    response = httpx.Response(200)

    async def timed_out_aread():
        raise httpx.ReadTimeout("timed out reading response body")

    response.aread = timed_out_aread  # type: ignore[method-assign]
    return response


def _provider_with_exc(exc_factory, *, status_first: dict | None = None):
    settings = make_settings().model_copy(
        update={"resilience": ResilienceSettings(max_retries=2, backoff_base_seconds=0.0)}
    )
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        exc = exc_factory(calls["n"])
        if exc is not None:
            raise exc
        if status_first is not None:
            return httpx.Response(status_first["status"], json=status_first.get("body", {}))
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant"}}]})

    client = httpx.AsyncClient(
        base_url="http://upstream.test/v1", transport=httpx.MockTransport(handler)
    )
    provider = OpenAICompatibleLLMProvider(
        settings.inference, client=client, resilience=settings.resilience
    )
    return provider, calls


@pytest.mark.parametrize("exc_cls", [httpx.ConnectError, httpx.ConnectTimeout])
def test_provably_pre_send_failures_are_retried(exc_cls):
    def factory(n):
        return exc_cls("pre-send failure") if n <= 2 else None


    provider, calls = _provider_with_exc(factory)   # max_retries=2 -> 3 attempts
    status, _headers, _body = asyncio.run(provider.complete({"model": "m", "messages": []}))
    assert status == 200
    assert calls["n"] == 3                          # exactly max_retries retries


@pytest.mark.parametrize(
    "exc_cls",
    [httpx.ReadTimeout, httpx.WriteError, httpx.RemoteProtocolError],
)
def test_post_send_failures_are_never_retried(exc_cls):
    provider, calls = _provider_with_exc(lambda n: exc_cls(f"post-send {exc_cls.__name__}"))
    from context_proxy.providers.errors import UpstreamUnavailable

    with pytest.raises(UpstreamUnavailable):
        asyncio.run(provider.complete({"model": "m", "messages": []}))
    assert calls["n"] == 1                          # ONE attempt, no duplication


@pytest.mark.parametrize("status", [400, 429, 500, 503])
def test_http_error_responses_are_answers_not_retried(status):
    provider, calls = _provider_with_exc(
        lambda n: None,
        status_first={"status": status, "body": {"error": {"message": "no"}}},
    )
    from context_proxy.providers.errors import UpstreamHTTPError

    with pytest.raises(UpstreamHTTPError) as excinfo:
        asyncio.run(provider.complete({"model": "m", "messages": []}))
    assert excinfo.value.status_code == status      # passthrough preserved
    assert calls["n"] == 1


def test_regression_provider_receives_post_once_then_read_timeout_no_second_post():
    """THE anchor regression: no duplicated inference after post-send stall."""
    settings = make_settings().model_copy(
        update={"resilience": ResilienceSettings(max_retries=3, backoff_base_seconds=0.0)}
    )
    posts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        posts["n"] += 1
        response = httpx.Response(200)

        async def stall():
            raise httpx.ReadTimeout("provider got it, response stalled")

        response.aread = stall  # type: ignore[method-assign]
        return response

    client = httpx.AsyncClient(
        base_url="http://upstream.test/v1", transport=httpx.MockTransport(handler)
    )
    provider = OpenAICompatibleLLMProvider(
        settings.inference, client=client, resilience=settings.resilience
    )
    with pytest.raises(UpstreamUnavailable):
        asyncio.run(provider.complete({"model": "m", "messages": [{"role": "user"}]}))
    assert posts["n"] == 1                          # NEVER a second POST


# ------------------------------------------------------------- streaming


class TestStreamingRetrySemantics:
    def test_pre_open_connect_failure_retries(self):
        settings = make_settings().model_copy(
            update={"resilience": ResilienceSettings(max_retries=2, backoff_base_seconds=0.0)}
        )
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 3:
                raise httpx.ConnectError("refused pre-stream")

            async def done_stream():
                yield b"data: [DONE]\n\n"

            return httpx.Response(200, content=done_stream(),
                                  headers={"content-type": "text/event-stream"})

        client = httpx.AsyncClient(
            base_url="http://upstream.test/v1",
            transport=httpx.MockTransport(handler),
        )
        provider = OpenAICompatibleLLMProvider(
            settings.inference, client=client, resilience=settings.resilience
        )

        async def scenario():
            stream = await provider.open_stream({"model": "m"})
            body = b"".join([c async for c in stream.iter_bytes()])
            await provider.aclose()
            return body

        body = asyncio.run(scenario())
        assert body == b"data: [DONE]\n\n"           # recovered before open
        assert calls["n"] == 3

    def test_post_open_failure_never_retries_and_closes_once(self):
        events: list[str] = []
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            tracked = httpx.Response(
                200,
                content=_fail_midway(),
                headers={"content-type": "text/event-stream"},
            )
            original_aclose = tracked.aclose

            async def tracked_aclose():
                events.append("closed")
                await original_aclose()

            tracked.aclose = tracked_aclose  # type: ignore[method-assign]
            return tracked

        async def _fail_midway():
            yield b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
            raise httpx.ReadError("upstream reset mid-stream")

        settings = make_settings().model_copy(
            update={"resilience": ResilienceSettings(max_retries=3, backoff_base_seconds=0.0)}
        )
        client = httpx.AsyncClient(
            base_url="http://upstream.test/v1",
            transport=httpx.MockTransport(handler),
        )
        provider = OpenAICompatibleLLMProvider(
            settings.inference, client=client, resilience=settings.resilience
        )

        async def scenario():
            stream = await provider.open_stream({"model": "m"})
            collected = b""
            with pytest.raises(httpx.ReadError):
                async for chunk in stream.iter_bytes():
                    collected += chunk
            await provider.aclose()

        asyncio.run(scenario())
        assert calls["n"] == 1                       # opened once, no resend


# ------------------------------------------------------ breaker accounting


class TestBreakerAccountingEligibility:
    def test_connect_errors_trip_breaker(self):
        breaker = CircuitBreaker(failure_threshold=2, reset_seconds=60.0)
        settings = make_settings().model_copy(
            update={"resilience": ResilienceSettings(max_retries=0)}
        )

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down")

        client = httpx.AsyncClient(
            base_url="http://upstream.test/v1", transport=httpx.MockTransport(handler)
        )
        provider = OpenAICompatibleLLMProvider(
            settings.inference, client=client, resilience=settings.resilience,
            breaker=breaker,
        )
        for _ in range(2):
            with pytest.raises(UpstreamUnavailable):
                asyncio.run(provider.complete({"model": "m"}))
        assert breaker.state == "open"

    def test_post_send_timeouts_never_trip_breaker(self):
        breaker = CircuitBreaker(failure_threshold=2, reset_seconds=60.0)
        settings = make_settings().model_copy(
            update={"resilience": ResilienceSettings(max_retries=0)}
        )

        def handler(request: httpx.Request) -> httpx.Response:
            response = httpx.Response(200)

            async def stall():
                raise httpx.ReadTimeout("slow generation")

            response.aread = stall  # type: ignore[method-assign]
            return response

        client = httpx.AsyncClient(
            base_url="http://upstream.test/v1", transport=httpx.MockTransport(handler)
        )
        provider = OpenAICompatibleLLMProvider(
            settings.inference, client=client, resilience=settings.resilience,
            breaker=breaker,
        )
        for _ in range(5):                           # far beyond threshold
            with pytest.raises(UpstreamUnavailable):
                asyncio.run(provider.complete({"model": "m"}))
        assert breaker.state == "closed"             # NOT tripped by post-send

    def test_retryable_tuple_is_narrow(self):
        assert RETRYABLE_TRANSPORT_ERRORS == (httpx.ConnectError, httpx.ConnectTimeout)


def test_request_body_not_mutated_across_attempts():
    """Fresh request built per attempt (unsafe reuse guard stays)."""
    seen_payloads: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_payloads.append(bytes(request.content))
        if len(seen_payloads) < 2:
            raise httpx.ConnectError("first attempt refused")
        return httpx.Response(200, json={"ok": True})

    settings = make_settings().model_copy(
        update={"resilience": ResilienceSettings(max_retries=1, backoff_base_seconds=0.0)}
    )
    client = httpx.AsyncClient(
        base_url="http://upstream.test/v1", transport=httpx.MockTransport(handler)
    )
    provider = OpenAICompatibleLLMProvider(
        settings.inference, client=client, resilience=settings.resilience
    )
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    status, _h, _b = asyncio.run(provider.complete(payload))
    assert status == 200
    assert seen_payloads[0] == seen_payloads[1]      # identical bodies resent
