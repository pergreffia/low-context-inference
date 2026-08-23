from __future__ import annotations

import httpx
import pytest
from conftest import CHAT_RESPONSE, MODELS_RESPONSE, client_for_handler, make_client
from helpers import (
    assert_openai_error_shape,
    captured_json,
    chat_payload,
    error_body,
    json_response,
)


@pytest.fixture
def captured() -> list[httpx.Request]:
    return []


def _chat_client(captured: list[httpx.Request], status: int, body: dict | str):
    if isinstance(body, dict):
        response = json_response(status, body)
    else:
        response = httpx.Response(
            status,
            content=body.encode(),
            headers={"content-type": "text/html"},
        )

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return response

    return client_for_handler(handler)


def test_upstream_400_json_body_passthrough_verbatim(captured):
    body = {"error": {"message": "bad param", "type": "invalid_request_error"}}
    client = _chat_client(captured, 400, body)
    r = client.post("/v1/chat/completions", json=chat_payload())
    assert r.status_code == 400
    assert r.json() == body


def test_upstream_400_non_json_body_becomes_openai_error(captured):
    client = _chat_client(captured, 400, "<html>bad request</html>")
    r = client.post("/v1/chat/completions", json=chat_payload())
    assert r.status_code == 400
    assert_openai_error_shape(error_body(r))
    assert error_body(r)["type"] == "upstream_error"


@pytest.mark.parametrize("status", [401, 403])
def test_upstream_auth_errors_passthrough(captured, status):
    body = {"error": {"message": "bad key", "type": "auth"}}
    client = _chat_client(captured, status, body)
    r = client.post("/v1/chat/completions", json=chat_payload())
    assert r.status_code == status
    assert r.json() == body


def test_upstream_429_rate_limit_headers_preserved(captured):
    headers = {
        "retry-after": "17",
        "x-ratelimit-limit-requests": "60",
        "x-ratelimit-remaining-requests": "0",
        "x-request-id": "req-429",
    }
    response = httpx.Response(
        429,
        json={"error": {"message": "rate limited", "type": "rate_limit"}},
        headers=headers,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return response

    client = client_for_handler(handler)
    r = client.post("/v1/chat/completions", json=chat_payload())
    assert r.status_code == 429
    assert r.json()["error"]["type"] == "rate_limit"
    for name, value in headers.items():
        assert r.headers[name] == value


def test_upstream_500_json_passthrough(captured):
    body = {"error": {"message": "boom", "type": "server_error"}}
    client = _chat_client(captured, 500, body)
    r = client.post("/v1/chat/completions", json=chat_payload())
    assert r.status_code == 500
    assert r.json() == body


def test_upstream_timeout_returns_openai_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    client = client_for_handler(handler)
    r = client.post("/v1/chat/completions", json=chat_payload())
    assert r.status_code == 502
    err = error_body(r)
    assert_openai_error_shape(err)
    assert err["code"] == "upstream_unavailable"
    assert "timed out" in err["message"]


def test_models_endpoint_upstream_failure_openai_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = client_for_handler(handler)
    r = client.get("/v1/models")
    assert r.status_code == 502
    err = error_body(r)
    assert err["code"] == "upstream_unavailable"


def test_non_json_429_preserves_safe_headers():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            content=b"rate limit exceeded",
            headers={
                "content-type": "text/plain",
                "retry-after": "30",
                "x-ratelimit-remaining": "0",
                "x-request-id": "test-request",
            },
        )

    client = client_for_handler(handler)
    r = client.post("/v1/chat/completions", json=chat_payload())

    assert r.status_code == 429
    err = error_body(r)
    assert_openai_error_shape(err)
    assert err["type"] == "upstream_error"
    assert r.headers["retry-after"] == "30"
    assert r.headers["x-ratelimit-remaining"] == "0"
    assert r.headers["x-request-id"] == "test-request"
    assert r.headers["content-type"].startswith("application/json")
    forwarded = {k.lower() for k in r.headers}
    assert not forwarded & {"connection", "transfer-encoding", "keep-alive"}


def test_models_endpoint_passthrough_headers_and_body(captured):
    upstream_headers = {
        "content-type": "application/json; charset=utf-8",
        "x-request-id": "req-models",
        "x-ratelimit-limit-requests": "100",
        "x-ratelimit-remaining-requests": "99",
        "connection": "keep-alive, X-Hop-Meta",
        "x-hop-meta": "strip-me",
        "set-cookie": "sid=1",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(200, MODELS_RESPONSE, headers=upstream_headers)

    client = client_for_handler(handler)
    r = client.get("/v1/models")

    assert r.status_code == 200
    assert r.json() == MODELS_RESPONSE
    assert r.headers["content-type"] == "application/json; charset=utf-8"
    assert r.headers["x-request-id"] == "req-models"
    assert r.headers["x-ratelimit-limit-requests"] == "100"
    assert r.headers["x-ratelimit-remaining-requests"] == "99"
    forwarded = {k.lower() for k in r.headers}
    # content-length is legitimately recomputed by the ASGI layer for the
    # forwarded body; what matters is that we do not forward upstream values.
    forbidden = {"connection", "keep-alive", "x-hop-meta", "set-cookie"}
    assert not forwarded & forbidden


def test_no_exception_leakage_on_transport_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("secret internal host internal-host.local refused")

    client = client_for_handler(handler)
    r = client.post("/v1/chat/completions", json=chat_payload())
    text = r.text
    assert "Traceback" not in text
    assert error_body(r)["type"] in {"api_error", "internal_error"}


def test_hop_by_hop_headers_never_forwarded(captured):
    upstream_headers = {
        "content-type": "application/json",
        "transfer-encoding": "chunked",
        "connection": "keep-alive",
        "keep-alive": "timeout=5",
        "set-cookie": "session=xyz; HttpOnly",
        "upgrade": "h2c",
        "proxy-authenticate": "Basic realm=x",
        # preserved:
        "x-request-id": "req-123",
        "openai-processing-ms": "42",
        "x-ratelimit-limit-tokens": "32768",
        "content-length": str(len(str(CHAT_RESPONSE))),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=CHAT_RESPONSE, headers=upstream_headers)

    client = client_for_handler(handler)
    r = client.post("/v1/chat/completions", json=chat_payload())

    assert r.status_code == 200
    for stripped in ("transfer-encoding", "connection", "keep-alive", "set-cookie", "upgrade"):
        assert stripped not in {k.lower() for k in r.headers}
    assert r.headers["x-request-id"] == "req-123"
    assert r.headers["openai-processing-ms"] == "42"
    assert r.headers["x-ratelimit-limit-tokens"] == "32768"


def test_content_type_not_hardcoded_overridden(captured):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=CHAT_RESPONSE,
            headers={"content-type": "application/json; charset=utf-8"},
        )

    client = client_for_handler(handler)
    r = client.post("/v1/chat/completions", json=chat_payload())
    assert r.headers["content-type"] == "application/json; charset=utf-8"


def test_default_content_type_applied_when_missing(captured):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=CHAT_RESPONSE)

    client = client_for_handler(handler)
    r = client.post("/v1/chat/completions", json=chat_payload())
    assert r.headers["content-type"].startswith("application/json")


def test_streaming_upstream_error_before_start_passthrough(captured):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "overloaded", "type": "server"}})

    client = client_for_handler(handler)
    r = client.post("/v1/chat/completions", json=chat_payload(stream=True))
    assert r.status_code == 503
    assert r.json()["error"]["message"] == "overloaded"


def test_existing_fixture_suite_still_wired(captured):
    client = make_client(captured)
    r = client.post("/v1/chat/completions", json=chat_payload())
    assert r.status_code == 200
    payload = captured_json(captured[0])
    assert payload["model"] == "client-model"
