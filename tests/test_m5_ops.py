"""M5 API tests: observability, resource limits, readiness, breaker, stress."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager

import httpx
import pytest
from fastapi.testclient import TestClient

from context_proxy.config import (
    EndpointSettings,
    RateLimitSettings,
    ResilienceSettings,
    ServerSettings,
)
from context_proxy.main import create_app
from context_proxy.observability.metrics import REGISTRY
from tests.conftest import UPSTREAM, make_settings, upstream_handler


@pytest.fixture(autouse=True)
def _isolated_metrics_registry():
    """Zero the global registry around each test (review §2)."""
    REGISTRY.reset()
    yield
    REGISTRY.reset()


def build_client(
    captured: list[httpx.Request],
    *,
    settings_overrides: dict | None = None,
    llm_client: httpx.AsyncClient | None = None,
) -> TestClient:
    settings = make_settings()
    if settings_overrides:
        settings = settings.model_copy(update=settings_overrides)
    app = create_app(
        settings,
        llm_client=llm_client
        or httpx.AsyncClient(base_url=UPSTREAM, transport=upstream_handler(captured)),
        store=None,
    )
    return TestClient(app)


@contextmanager
def run_client(client: TestClient) -> Iterator[TestClient]:
    with client as running:
        yield running


def post_chat(client: TestClient, body: dict | None = None):
    return client.post("/v1/chat/completions", json=body or CHAT_BODY)


CHAT_BODY = {
    "model": "client-model",
    "messages": [{"role": "user", "content": "hi"}],
}


class TestObservabilityEndpoint:
    def test_metrics_exposes_request_counters(self, captured_requests):
        with run_client(build_client(captured_requests)) as client:
            assert client.post("/v1/chat/completions", json=CHAT_BODY).status_code == 200
            metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert metrics.headers["content-type"].startswith("text/plain")
        body = metrics.text
        assert 'context_proxy_http_requests_total{method="POST"' in body
        assert 'route="/v1/chat/completions"' in body
        assert "context_proxy_http_request_duration_seconds" in body

    def test_token_accounting_counts_upstream_usage(self, captured_requests):
        with run_client(build_client(captured_requests)) as client:
            assert client.post("/v1/chat/completions", json=CHAT_BODY).status_code == 200
            body = client.get("/metrics").text
        # conftest CHAT_RESPONSE reports prompt 5 / completion 2 tokens
        assert 'context_proxy_llm_tokens_total{direction="prompt"}' in body
        assert 'context_proxy_llm_tokens_total{direction="completion"}' in body

    def test_request_id_echoed_when_client_supplies_it(self, captured_requests):
        with run_client(build_client(captured_requests)) as client:
            response = client.post(
                "/v1/chat/completions",
                json=CHAT_BODY,
                headers={"X-Request-ID": "my-correlation-id"},
            )
        assert response.status_code == 200
        assert response.headers["x-request-id"] == "my-correlation-id"

    def test_readyz_reports_checks(self, captured_requests):
        with run_client(build_client(captured_requests)) as client:
            response = client.get("/readyz")
        assert response.status_code == 200
        payload = response.json()
        assert payload["ready"] is True
        assert payload["checks"]["database"] in {"ok", "degraded"}
        assert payload["checks"]["circuit_breaker"] == "closed"


class TestResourceLimits:
    def test_oversized_body_rejected_before_parsing(self, captured_requests):
        overrides = {
            "server": ServerSettings(max_body_bytes=64),
        }
        with run_client(
            build_client(captured_requests, settings_overrides=overrides)
        ) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "x" * 500}]},
            )
        assert response.status_code == 413
        error = response.json()["error"]
        assert error["code"] == "request_too_large"
        assert captured_requests == []

    def test_rate_limit_returns_openai_style_429_with_retry_after(
        self, captured_requests
    ):
        overrides = {
            "rate_limit": RateLimitSettings(enabled=True, requests_per_minute=60, burst=2),
        }
        with run_client(
            build_client(captured_requests, settings_overrides=overrides)
        ) as client:
            statuses = [
                client.post("/v1/chat/completions", json=CHAT_BODY).status_code
                for _ in range(4)
            ]
        assert statuses[:2] == [200, 200]
        assert 429 in statuses
        # rejection shape verified on a dedicated rejected request
        for _ in range(3):
            last = client.post("/v1/chat/completions", json=CHAT_BODY)
        assert last.status_code == 429
        assert last.json()["error"]["code"] == "rate_limit_exceeded"
        assert int(last.headers["retry-after"]) >= 1


class TestDiagnosticsAndRebuild:
    def test_diagnostics_shape_without_secrets(self, captured_requests):
        class RebuildableMemory:
            async def rebuild_vector_index(self, conversation_id=None, *, force=False):
                return {"chunks": 0, "memories": 0}

        app = create_app(
            make_settings(),
            llm_client=httpx.AsyncClient(
                base_url=UPSTREAM, transport=upstream_handler(captured_requests)
            ),
            store=None,
            memory_service=RebuildableMemory(),
        )
        with TestClient(app) as client:
            response = client.get("/internal/v1/diagnostics")
        assert response.status_code == 200
        payload = response.json()
        assert payload["database"]["available"] is False
        assert payload["context_engine"]["enabled"] is True
        assert payload["resilience"]["breaker_state"] in {
            "closed",
            "open",
            "half_open",
        }
        blob = json.dumps(payload).lower()
        assert "api_key" not in blob and "authorization" not in blob

    def test_rebuild_endpoint_reports_summary(self, captured_requests):
        calls: list[tuple[str | None, bool]] = []

        class RebuildableMemory:
            async def rebuild_vector_index(self, conversation_id=None, *, force=False):
                calls.append((conversation_id, force))
                return {"chunks": 3, "chunks_failed": 0, "memories": 5, "memories_failed": 0}

        app = create_app(
            make_settings(),
            llm_client=httpx.AsyncClient(
                base_url=UPSTREAM, transport=upstream_handler(captured_requests)
            ),
            store=None,
            memory_service=RebuildableMemory(),
        )
        with TestClient(app) as client:
            response = client.post("/internal/v1/index/rebuild?force=true")
            scoped = client.post(
                "/internal/v1/index/rebuild"
                "?conversation_id=aaaaaaaa-1111-4111-8111-111111111111"
            )
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "chunks": 3,
            "chunks_failed": 0,
            "memories": 5,
            "memories_failed": 0,
        }
        import uuid as _uuid

        assert calls[0] == (None, True)
        # scoped passthrough arrives as the parsed UUID
        assert str(calls[1][0]) == "aaaaaaaa-1111-4111-8111-111111111111"
        assert isinstance(calls[1][0], _uuid.UUID)
        assert calls[1][1] is False
        assert scoped.status_code == 200

    def test_rebuild_unavailable_without_memory_service(self, captured_requests):
        with run_client(build_client(captured_requests)) as client:
            response = client.post("/internal/v1/index/rebuild")
        assert response.status_code == 503


class TestCircuitBreakerIntegration:
    def test_breaker_fails_fast_after_transport_failures(self):
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            raise httpx.ConnectError("connection refused")

        settings = make_settings().model_copy(
            update={
                "resilience": ResilienceSettings(
                    max_retries=0,
                    backoff_base_seconds=0.0,
                    breaker_failure_threshold=2,
                    breaker_reset_seconds=3600,
                )
            }
        )
        app = create_app(
            settings,
            llm_client=httpx.AsyncClient(
                base_url=UPSTREAM, transport=httpx.MockTransport(handler)
            ),
            store=None,
        )
        with TestClient(app) as client:
            first = client.post("/v1/chat/completions", json=CHAT_BODY)
            second = client.post("/v1/chat/completions", json=CHAT_BODY)
            third = client.post("/v1/chat/completions", json=CHAT_BODY)

        assert first.status_code == 502
        assert second.status_code == 502
        # breaker open: fails fast WITHOUT touching the transport.
        # max_retries=0 → exactly one transport attempt per request.
        assert third.status_code == 502
        assert third.json()["error"]["code"] == "upstream_unavailable"
        assert attempts["n"] == 2

        metrics_text = client.get("/metrics").text
        assert 'context_proxy_circuit_state{state="open"} 1' in metrics_text

    def test_streaming_failure_maps_to_upstream_error(self, captured_requests):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down")

        settings = make_settings().model_copy(
            update={
                "resilience": ResilienceSettings(
                    max_retries=0,
                    backoff_base_seconds=0.0,
                    breaker_failure_threshold=10,
                )
            }
        )
        app = create_app(
            settings,
            llm_client=httpx.AsyncClient(
                base_url=UPSTREAM, transport=httpx.MockTransport(handler)
            ),
            store=None,
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions", json={**CHAT_BODY, "stream": True}
            )
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "upstream_unavailable"


def test_stress_concurrent_chat_completions_stay_consistent():
    """30 concurrent requests: every one answered, upstream hit exactly once."""
    captured: list[httpx.Request] = []
    app = create_app(
        make_settings(),
        llm_client=httpx.AsyncClient(
            base_url=UPSTREAM,
            transport=upstream_handler(captured, chat_status=200),
        ),
        store=None,
    )

    client = TestClient(app)

    results: list[int] = []
    lock = threading.Lock()

    def do_post():
        code = client.post("/v1/chat/completions", json=CHAT_BODY).status_code
        with lock:
            results.append(code)

    threads = [threading.Thread(target=do_post) for _ in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(results) == [200] * 30
    assert len(captured) == 30


class TestTokenAccountingExactlyOnce:
    """One upstream response -> exactly one token accounting (review §2)."""

    USAGE = {"prompt_tokens": 5, "completion_tokens": 2}

    @staticmethod
    def _handler(with_usage: bool):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path != "/v1/chat/completions":
                return httpx.Response(404, json={"error": {"message": "nf"}})
            payload = json.loads(request.content)
            if payload.get("stream") is True:
                async def sse():
                    yield b'data: {"id":"1","choices":[{"delta":{"content":"he"}}]}\n\n'
                    if with_usage:
                        yield (
                            b'data: {"id":"1","choices":[],'
                            b'"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n'
                        )
                    yield b"data: [DONE]\n\n"

                return httpx.Response(
                    200,
                    content=sse(),
                    headers={"content-type": "text/event-stream"},
                )
            response = {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
            }
            if with_usage:
                response["usage"] = dict(TestTokenAccountingExactlyOnce.USAGE)
            return httpx.Response(200, json=response)

        return handler

    @staticmethod
    def _token_counters(client: TestClient) -> dict[str, int]:
        values: dict[str, int] = {}
        for line in client.get("/metrics").text.splitlines():
            if line.startswith("context_proxy_llm_tokens_total{"):
                direction = line.split('direction="')[1].split('"')[0]
                values[direction] = int(line.rsplit(" ", 1)[1])
        return values

    def test_non_streaming_counts_once(self, captured_requests):
        with run_client(build_client(captured_requests)) as client:
            assert post_chat(client).status_code == 200
            counters = self._token_counters(client)
        assert counters == {"prompt": 5, "completion": 2}

    def test_non_streaming_with_store_counts_once(self, captured_requests):
        class OkStore:
            async def ensure_conversation(self, conversation_id):
                return None

            async def reconcile_history(self, conversation_id, messages, metadata=None):
                return []

        app = create_app(
            make_settings(),
            llm_client=httpx.AsyncClient(
                base_url=UPSTREAM,
                transport=httpx.MockTransport(self._handler(with_usage=True)),
            ),
            store=OkStore(),
        )
        with TestClient(app) as client:
            assert post_chat(client).status_code == 200
            counters = self._token_counters(client)
        assert counters == {"prompt": 5, "completion": 2}  # NOT 10/4

    def test_streaming_counts_once(self, captured_requests):
        llm_client = httpx.AsyncClient(
            base_url=UPSTREAM,
            transport=httpx.MockTransport(self._handler(with_usage=True)),
        )
        with run_client(
            build_client(captured_requests, llm_client=llm_client)
        ) as client:
            response = client.post(
                "/v1/chat/completions", json={**CHAT_BODY, "stream": True}
            )
        assert response.status_code == 200
        assert response.text.endswith("data: [DONE]\n\n")
        assert self._token_counters(client) == {"prompt": 5, "completion": 2}

    def test_streaming_without_usage_records_nothing(self, captured_requests):
        llm_client = httpx.AsyncClient(
            base_url=UPSTREAM,
            transport=httpx.MockTransport(self._handler(with_usage=False)),
        )
        with run_client(
            build_client(captured_requests, llm_client=llm_client)
        ) as client:
            response = client.post(
                "/v1/chat/completions", json={**CHAT_BODY, "stream": True}
            )
        assert response.status_code == 200
        assert self._token_counters(client) == {}


class TestBodyLimitWithoutContentLength:
    def test_chunked_body_over_limit_rejected_before_app(self, captured_requests):
        overrides = {"server": ServerSettings(max_body_bytes=128)}
        big_chunks = [b"x" * 64, b"y" * 64, b"z" * 64]  # 192 > 128

        def stream_body():
            yield from big_chunks

        with run_client(
            build_client(captured_requests, settings_overrides=overrides)
        ) as client:
            response = client.post("/v1/chat/completions", content=stream_body())
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "request_too_large"
        assert captured_requests == []  # upstream never reached

    def test_asgi_level_limit_without_content_length_header(self):
        """Pure-ASGI proof: no Content-Length header, oversized streamed body.

        Contract (final review §1): the middleware stops consuming the body
        AT the first violating chunk — no further receive() calls — never
        runs the application, and answers 413.
        """
        from context_proxy.observability.middleware import ObservabilityMiddleware

        reached_app = {"flag": False}

        async def app(scope, receive, send):  # pragma: no cover - must NOT run
            reached_app["flag"] = True
            while True:
                message = await receive()
                if not message.get("more_body"):
                    break
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        sent_messages: list[dict] = []
        receive_calls = {"n": 0}
        body_parts = [b"a" * 100, b"b" * 100]  # second chunk crosses limit=150

        async def receive():
            receive_calls["n"] += 1
            if body_parts:
                part = body_parts.pop(0)
                return {
                    "type": "http.request",
                    "body": part,
                    "more_body": bool(body_parts),
                }
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent_messages.append(message)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [],  # NO content-length header at all
            "client": ("test", 123),
            "state": {},
        }

        middleware = ObservabilityMiddleware(app, max_body_bytes=150)
        asyncio.run(middleware(scope, receive, send))

        assert reached_app["flag"] is False   # application never ran
        starts = [m for m in sent_messages if m["type"] == "http.response.start"]
        assert len(starts) == 1 and starts[0]["status"] == 413
        assert receive_calls["n"] == 2        # stopped AT the violating chunk

    def test_asgi_limit_stops_reading_on_huge_body(self):
        """A body far beyond the cap triggers an immediate stop (review §2).

        200 chunks of 100 bytes against a 150-byte cap: detection happens on
        the SECOND chunk; the remaining ~198 chunks are NEVER consumed.
        """
        from context_proxy.observability.middleware import ObservabilityMiddleware

        total_chunks = 200
        receive_calls = {"n": 0}
        remaining = {"count": total_chunks}

        async def receive():
            receive_calls["n"] += 1
            remaining["count"] -= 1
            more = remaining["count"] > 0
            return {"type": "http.request", "body": b"x" * 100, "more_body": more}

        async def app(scope, receive, send):  # pragma: no cover - must NOT run
            raise AssertionError("application must not run for oversized bodies")

        sent_statuses: list[int] = []

        async def send(message):
            if message["type"] == "http.response.start":
                sent_statuses.append(message["status"])

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [],
            "client": ("test", 123),
            "state": {},
        }

        middleware = ObservabilityMiddleware(app, max_body_bytes=150)
        asyncio.run(middleware(scope, receive, send))

        assert sent_statuses == [413]
        assert receive_calls["n"] == 2  # NOT all {total_chunks} messages

    def test_asgi_disconnect_during_body_never_reaches_app(self):
        """Client disconnect mid-upload: app NOT invoked, no synthetic body.

        http.disconnect must NEVER be converted into a normal end-of-body
        http.request (M5 final mini-fix §1-§2).
        """
        from context_proxy.observability.middleware import ObservabilityMiddleware

        reached_app = {"flag": False}
        forwarded: list[dict] = []

        async def app(scope, receive, send):  # pragma: no cover - must NOT run
            reached_app["flag"] = True
            while True:
                forwarded.append(await receive())
                if not forwarded[-1].get("more_body"):
                    break
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        sent_messages: list[dict] = []
        receive_sequence = [
            {"type": "http.request", "body": b"partial", "more_body": True},
            {"type": "http.disconnect"},
        ]

        async def receive():
            if receive_sequence:
                return receive_sequence.pop(0)
            return {"type": "http.disconnect"}

        async def send(message):
            sent_messages.append(message)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [],
            "client": ("test", 123),
            "state": {},
        }

        middleware = ObservabilityMiddleware(app, max_body_bytes=150)
        asyncio.run(middleware(scope, receive, send))

        assert reached_app["flag"] is False      # application never invoked
        assert forwarded == []                   # nothing synthesized/forwarded
        # no normal response either: the connection is already gone
        starts = [m for m in sent_messages if m["type"] == "http.response.start"]
        assert starts == []

    def test_asgi_under_limit_replays_body_to_app(self):
        """Under the cap: app runs and receives the exact buffered body."""
        from context_proxy.observability.middleware import ObservabilityMiddleware

        seen: list[dict] = []

        async def app(scope, receive, send):
            while True:
                message = await receive()
                seen.append(message)
                if not message.get("more_body"):
                    break
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        sent: list[dict] = []

        async def send(message):
            sent.append(message)

        body_parts = [b"a" * 50, b"b" * 50]

        async def receive():
            if body_parts:
                part = body_parts.pop(0)
                return {"type": "http.request", "body": part, "more_body": bool(body_parts)}
            return {"type": "http.request", "body": b"", "more_body": False}

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [],
            "client": ("test", 123),
            "state": {},
        }

        middleware = ObservabilityMiddleware(app, max_body_bytes=150)
        asyncio.run(middleware(scope, receive, send))

        bodies = b"".join(m.get("body") or b"" for m in seen)
        assert bodies == b"a" * 50 + b"b" * 50
        statuses = [m["status"] for m in sent if m["type"] == "http.response.start"]
        assert statuses == [200]


class TestDiagnosticsSecretHygiene:
    def test_realistic_secrets_never_leak(self, captured_requests):
        overrides = {
            "inference": EndpointSettings(
                base_url="http://upstream.test/v1",
                api_key="sk-live-supersecret-inference-key",
            ),
            "embeddings": EndpointSettings(
                base_url="http://embed.test/v1",
                api_key="sk-live-supersecret-embedding-key",
            ),
        }

        class RebuildableMemory:
            async def rebuild_vector_index(self, conversation_id=None, *, force=False):
                return {"chunks": 0, "chunks_failed": 0, "memories": 0, "memories_failed": 0}

        settings = make_settings().model_copy(update=overrides)
        app = create_app(
            settings,
            llm_client=httpx.AsyncClient(
                base_url=UPSTREAM, transport=upstream_handler(captured_requests)
            ),
            store=None,
            memory_service=RebuildableMemory(),
        )
        with TestClient(app) as client:
            response = client.get("/internal/v1/diagnostics")
        assert response.status_code == 200
        blob = json.dumps(response.json()).lower()
        for secret in (
            "sk-live-supersecret-inference-key",
            "sk-live-supersecret-embedding-key",
            "api_key",
            "authorization",
            "password",
            "secret",
            "token",
        ):
            assert secret not in blob


class TestRateLimiterIdentityPolicy:
    """Identity policy (documented on RateLimitSettings):

    X-Conversation-ID header -> per-conversation bucket;
    otherwise client host -> shared host bucket.
    Body-level conversation ids are intentionally ignored (pre-parse decision).
    """

    def test_buckets_isolated_per_conversation_identity(self, captured_requests):
        overrides = {
            "rate_limit": RateLimitSettings(enabled=True, requests_per_minute=60, burst=1),
        }
        # X-Conversation-ID must be a VALID UUID (M2 identity policy); the
        # limiter keys on the raw header value pre-validation.
        conv_a = "aaaaaaaa-1111-4111-8111-111111111111"
        conv_b = "bbbbbbbb-2222-4222-8222-222222222222"
        with run_client(
            build_client(captured_requests, settings_overrides=overrides)
        ) as client:
            first = client.post(
                "/v1/chat/completions", json=CHAT_BODY, headers={"X-Conversation-ID": conv_a}
            )
            second = client.post(
                "/v1/chat/completions", json=CHAT_BODY, headers={"X-Conversation-ID": conv_a}
            )
            other = client.post(
                "/v1/chat/completions", json=CHAT_BODY, headers={"X-Conversation-ID": conv_b}
            )
        assert first.status_code == 200
        assert second.status_code == 429  # bucket A exhausted...
        # post-04592c0 review §2: rotation can NO LONGER mint fresh quota —
        # the shared client/IP bucket aggregates both attempts, so a new
        # conversation id from the same host is throttled too.
        assert other.status_code == 429
        assert other.headers.get("Retry-After") is not None


class TestTokenAccountingStreamingWithStore:
    """Streaming through the persistence path: exactly-once (final review §3-§4)."""

    @staticmethod
    def _store(reconcile_raises: Exception | None = None):
        class Store:
            def __init__(self):
                self.calls = 0

            async def ensure_conversation(self, conversation_id):
                return None

            async def reconcile_history(self, conversation_id, messages, metadata=None):
                self.calls += 1
                if reconcile_raises is not None:
                    raise reconcile_raises
                return []

        return Store()

    def test_streaming_with_store_counts_once(self):
        handler = TestTokenAccountingExactlyOnce._handler(with_usage=True)
        llm_client = httpx.AsyncClient(
            base_url=UPSTREAM, transport=httpx.MockTransport(handler)
        )
        store = self._store()
        app = create_app(make_settings(), llm_client=llm_client, store=store)
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions", json={**CHAT_BODY, "stream": True}
            )
            counters = TestTokenAccountingExactlyOnce._token_counters(client)
        assert response.status_code == 200
        assert response.text.endswith("data: [DONE]\n\n")
        # persistence path taken: inbound reconciliation + assistant capture
        assert store.calls == 2
        assert counters == {"prompt": 5, "completion": 2}  # NOT 10/4

    def test_streaming_with_store_failure_counts_once_and_stream_survives(self):
        handler = TestTokenAccountingExactlyOnce._handler(with_usage=True)
        llm_client = httpx.AsyncClient(
            base_url=UPSTREAM, transport=httpx.MockTransport(handler)
        )
        from context_proxy.memory.errors import PersistenceInfrastructureError
        store = self._store(
            reconcile_raises=PersistenceInfrastructureError("persistence down")
        )
        app = create_app(make_settings(), llm_client=llm_client, store=store)
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions", json={**CHAT_BODY, "stream": True}
            )
            counters = TestTokenAccountingExactlyOnce._token_counters(client)
        assert response.status_code == 200
        # stream reached the client COMPLETE despite the persistence failure
        assert "data: [DONE]" in response.text
        # accounting happened exactly once even though persistence failed
        assert counters == {"prompt": 5, "completion": 2}
