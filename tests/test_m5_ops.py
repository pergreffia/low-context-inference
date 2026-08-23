"""M5 API tests: observability, resource limits, readiness, breaker, stress."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager

import httpx
from fastapi.testclient import TestClient

from context_proxy.config import (
    RateLimitSettings,
    ResilienceSettings,
    ServerSettings,
)
from context_proxy.main import create_app
from tests.conftest import UPSTREAM, make_settings, upstream_handler


def build_client(
    captured: list[httpx.Request],
    *,
    settings_overrides: dict | None = None,
) -> TestClient:
    settings = make_settings()
    if settings_overrides:
        settings = settings.model_copy(update=settings_overrides)
    app = create_app(
        settings,
        llm_client=httpx.AsyncClient(
            base_url=UPSTREAM, transport=upstream_handler(captured)
        ),
        store=None,
    )
    return TestClient(app)


@contextmanager
def run_client(client: TestClient) -> Iterator[TestClient]:
    with client as running:
        yield running


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
                return {"chunks": 3, "memories": 5}

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
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "chunks": 3, "memories": 5}
        assert calls == [(None, True)]

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
    import threading

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
