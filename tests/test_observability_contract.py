"""Observability/diagnostics contract (post-0876b10 review §12).

Endpoints answer, counters move, breaker state is reported, and — critically
— Prometheus label cardinality stays bounded: no conversation id, request id,
or arbitrary input may ever become a label.
"""

from __future__ import annotations

import httpx
from fastapi.testclient import TestClient

from context_proxy.main import create_app
from context_proxy.observability.metrics import REGISTRY
from tests.conftest import CHAT_RESPONSE, UPSTREAM, make_settings

CONV = "cccccccc-0000-0000-0000-000000000000"
REQ_ID = "arbitrary-request-id-0123456789abcdef"


def _handler_factory(state: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] = state.get("calls", 0) + 1
        if not state.get("up", True):
            raise httpx.ConnectError("down")
        return httpx.Response(200, json=CHAT_RESPONSE)

    return handler


def _app(state: dict):
    return create_app(
        make_settings(),
        llm_client=httpx.AsyncClient(
            base_url=UPSTREAM,
            transport=httpx.MockTransport(_handler_factory(state)),
        ),
    )


class TestEndpointContract:
    def test_healthz_shape(self):
        app = _app({})
        with TestClient(app) as client:
            response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_readyz_reports_breaker_state(self):
        app = _app({})
        with TestClient(app) as client:
            body = client.get("/readyz").json()
        assert body["ready"] is True
        assert body["checks"]["circuit_breaker"] == "closed"

    def test_metrics_endpoint_serves_registry(self):
        app = _app({})
        with TestClient(app) as client:
            response = client.get("/metrics")
        assert response.status_code == 200
        assert "# HELP" in response.text


class TestCounterMovement:
    def test_latency_degradation_and_tokens_recorded(self):
        REGISTRY.reset()
        app = _app({})
        with TestClient(app) as client:
            client.post("/v1/chat/completions",
                        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        text = REGISTRY.render()
        assert 'route="/v1/chat/completions"' in text          # latency histogram label
        assert "context_proxy_http_request_duration_seconds" in text
        assert 'context_proxy_llm_tokens_total{direction="prompt"}' in text

    def test_degradation_counter_on_breaker_reject(self):
        REGISTRY.reset()
        from context_proxy.config import ResilienceSettings

        settings = make_settings().model_copy(
            update={"resilience": ResilienceSettings(
                max_retries=0, breaker_failure_threshold=1)}
        )

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down")

        app = create_app(
            settings,
            llm_client=httpx.AsyncClient(
                base_url=UPSTREAM, transport=httpx.MockTransport(handler)
            ),
        )
        with TestClient(app) as client:
            first = client.post("/v1/chat/completions",
                                json={"model": "m", "messages": [{"role": "user"}]})
            second = client.post("/v1/chat/completions",
                                 json={"model": "m", "messages": [{"role": "user"}]})
        assert first.status_code == 502
        assert second.status_code == 502              # fail-fast via OPEN breaker
        text = REGISTRY.render()
        assert 'context_proxy_degradations_total{component="upstream_breaker_open"}' in text

    def test_capture_overflow_metric_after_overflow(self):
        REGISTRY.reset()
        from context_proxy.config import ServerSettings

        settings = make_settings().model_copy(
            update={"server": ServerSettings(port=8080, max_capture_bytes=256)}
        )

        def handler(request: httpx.Request) -> httpx.Response:
            async def agen():
                yield b'data: {"choices":[{"delta":{"content":"' + b"z" * 4096 + b'"}}]}\n\n'
                yield b"data: [DONE]\n\n"

            return httpx.Response(200, content=agen(),
                                  headers={"content-type": "text/event-stream"})

        app = create_app(
            settings,
            llm_client=httpx.AsyncClient(
                base_url=UPSTREAM, transport=httpx.MockTransport(handler)
            ),
        )
        conv = "dddddddd-0000-0000-0000-000000000000"
        with TestClient(app) as client, client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "m", "stream": True,
                  "messages": [{"role": "user", "content": "hi"}],
                  "conversation_id": conv},
        ) as response:
            b"".join(response.iter_bytes())
        text = REGISTRY.render()
        assert "context_proxy_assistant_capture_overflow_total 1" in text


class TestLabelCardinalityBounded:
    def test_no_conversation_id_or_request_id_ever_becomes_a_label(self):
        REGISTRY.reset()
        state: dict = {}
        app = _app(state)
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                    "conversation_id": CONV,
                },
                headers={"X-Request-ID": REQ_ID},
            )
            assert response.status_code == 200
            metrics_text = client.get("/metrics").text
        for forbidden in (CONV, REQ_ID):
            assert forbidden not in metrics_text       # cardinality stays closed

    def test_unknown_paths_map_to_low_cardinality_other(self):
        REGISTRY.reset()
        app = _app({})
        with TestClient(app) as client:
            client.get(f"/v1/whatever/{'x' * 500}/deeper")
            client.get("/totally/unknown/route")
            metrics_text = client.get("/metrics").text
        assert 'route="other"' in metrics_text
        assert "x" * 100 not in metrics_text           # raw path never a label
        assert "/totally/unknown" not in metrics_text

    def test_status_labels_are_real_http_statuses_only(self):
        REGISTRY.reset()
        state: dict = {"up": False}
        app = _app(state)
        with TestClient(app) as client:
            client.post("/v1/chat/completions",
                        json={"model": "m", "messages": [{"role": "user"}]})
        text = REGISTRY.render()
        statuses = [
            ln.split('status="')[1].split('"')[0]
            for ln in text.splitlines()
            if 'context_proxy_http_requests_total{' in ln and 'status="' in ln
        ]
        assert all(s.isdigit() and len(s) == 3 for s in statuses)
        assert "499" not in statuses                   # fabricated statuses stay dead
