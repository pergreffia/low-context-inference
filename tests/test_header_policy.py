from __future__ import annotations

import httpx
import pytest

from context_proxy.providers.headers import filter_response_headers


def test_connection_tokens_are_stripped():
    headers = {
        "Connection": "keep-alive, X-Custom-Hop",
        "X-Custom-Hop": "secret",
        "X-Keep": "forwarded",
    }
    filtered = filter_response_headers(headers, keep_content_encoding=True)
    assert filtered == {"X-Keep": "forwarded"}


def test_static_hop_by_hop_always_stripped():
    headers = {name: "v" for name in (
        "Connection", "Keep-Alive", "Proxy-Authenticate", "Proxy-Authorization",
        "TE", "Trailer", "Trailers", "Transfer-Encoding", "Upgrade",
    )}
    headers["X-Safe"] = "yes"
    filtered = filter_response_headers(headers, keep_content_encoding=True)
    assert list(filtered) == ["X-Safe"]


def test_content_length_and_set_cookie_stripped():
    headers = {"Content-Length": "10", "Set-Cookie": "a=b", "X-Request-ID": "r1"}
    filtered = filter_response_headers(headers, keep_content_encoding=True)
    assert "content-length" not in {k.lower() for k in filtered}
    assert "set-cookie" not in {k.lower() for k in filtered}
    assert filtered["X-Request-ID"] == "r1"


def test_buffered_mode_drops_content_encoding():
    headers = {"Content-Encoding": "gzip", "Content-Type": "application/json"}
    filtered = filter_response_headers(headers, keep_content_encoding=False)
    assert "Content-Encoding" not in filtered
    assert filtered["Content-Type"] == "application/json"


def test_streaming_mode_keeps_content_encoding():
    headers = {"Content-Encoding": "gzip", "Content-Type": "text/event-stream"}
    filtered = filter_response_headers(headers, keep_content_encoding=True)
    assert filtered["Content-Encoding"] == "gzip"
    assert filtered["Content-Type"] == "text/event-stream"


def test_rate_limit_and_retry_after_survive():
    headers = {
        "Retry-After": "30",
        "X-RateLimit-Remaining": "0",
        "X-RateLimit-Limit-Requests": "60",
        "X-Request-ID": "test-request",
    }
    filtered = filter_response_headers(headers, keep_content_encoding=False)
    assert filtered == headers


@pytest.mark.parametrize("conn_value", ["x-leak", " X-Leak , other", "keep-alive,x-leak"])
def test_connection_token_parsing_variants(conn_value):
    headers = {"Connection": conn_value, "X-Leak": "nope"}
    filtered = filter_response_headers(headers, keep_content_encoding=True)
    assert "X-Leak" not in filtered


def test_buffered_provider_response_drops_gzip_header():
    """httpx decompresses buffered bodies -> forwarded Content-Encoding would lie."""
    import gzip

    body = gzip.compress(b'{"ok": true}')

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "application/json", "content-encoding": "gzip"},
        )

    from conftest import client_for_handler
    from helpers import chat_payload

    client = client_for_handler(handler)
    r = client.post("/v1/chat/completions", json=chat_payload())
    assert r.status_code == 200
    assert r.json() == {"ok": True}
    assert "content-encoding" not in {k.lower() for k in r.headers}
