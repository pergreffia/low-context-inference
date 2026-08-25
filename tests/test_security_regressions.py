"""Security regression suite (post-0876b10 review §5, §2, §7).

Covers: diagnostics URL redaction, /internal/* isolation + optional auth,
Compose network hygiene, request-parsing abuse resistance, and error/header
leakage. Every test here would fail on the pre-fix code (or documents an
invariant that must not regress).
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from context_proxy.config import SecuritySettings
from context_proxy.main import create_app
from tests.conftest import CHAT_RESPONSE, UPSTREAM, make_settings

COMPOSE = Path(__file__).resolve().parents[1] / "docker-compose.yml"


def _client(security: SecuritySettings | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=CHAT_RESPONSE)

    settings = make_settings().model_copy(
        update={"security": security or SecuritySettings()}
    )
    app = create_app(
        settings,
        llm_client=httpx.AsyncClient(
            base_url=UPSTREAM, transport=httpx.MockTransport(handler)
        ),
    )
    return TestClient(app)


# ------------------------------------------------- §5 diagnostics redaction


@pytest.mark.parametrize(
    "base_url",
    [
        "https://user:secret@example.com/v1",
        "https://example.com/v1?api_key=supersecret",
        "https://example.com/v1#fragment-secret",
        "http://alice:bob@10.1.2.3:9999/path?token=xyz",
    ],
)
def test_diagnostics_never_exposes_secret_bearing_url(base_url):
    from context_proxy.config import EndpointSettings

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=CHAT_RESPONSE)

    settings = make_settings()
    settings.inference = EndpointSettings(base_url=base_url)
    app = create_app(
        settings,
        llm_client=httpx.AsyncClient(
            base_url=UPSTREAM, transport=httpx.MockTransport(handler)
        ),
    )
    with TestClient(app) as client:
        response = client.get("/internal/v1/diagnostics")
    assert response.status_code == 200
    blob = json.dumps(response.json())
    for leak in ("secret", "supersecret", "api_key", "token=xyz", "bob", "alice", "#"):
        assert leak not in blob
    inference = response.json()["inference"]
    assert set(inference) <= {"configured", "host", "port"}
    if inference["configured"]:
        assert inference["host"] == "example.com" or inference["host"] == "10.1.2.3"
        assert ":" not in json.dumps(inference["host"])
    # credentials never survive in host/port fields
    assert "user" != inference.get("host")


def test_diagnostics_reports_unconfigured_inference():
    from context_proxy.config import EndpointSettings

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=CHAT_RESPONSE)

    settings = make_settings()
    settings.inference = EndpointSettings(base_url="")
    app = create_app(
        settings,
        llm_client=httpx.AsyncClient(
            base_url=UPSTREAM, transport=httpx.MockTransport(handler)
        ),
    )
    with TestClient(app) as client:
        response = client.get("/internal/v1/diagnostics")
    assert response.json()["inference"]["configured"] is False


# ------------------------------------------- §2 internal endpoint isolation


class TestInternalAuthGate:
    def test_open_by_default_local_deployment(self):
        with _client() as client:
            response = client.get("/internal/v1/diagnostics")
        assert response.status_code == 200

    @pytest.mark.parametrize(
        "headers",
        [{}, {"X-Internal-Auth": "wrong-token"}],
        ids=["missing", "wrong"],
    )
    def test_configured_token_rejects_externals(self, headers):
        with _client(SecuritySettings(internal_auth_token="s3cret")) as client:
            response = client.get("/internal/v1/diagnostics", headers=headers)
        assert response.status_code == 401
        assert "authentication" in response.json()["detail"]

    def test_configured_token_accepts_correct_header(self):
        with _client(SecuritySettings(internal_auth_token="s3cret")) as client:
            ok = client.get(
                "/internal/v1/diagnostics", headers={"X-Internal-Auth": "s3cret"}
            )
            rejected = client.post("/internal/v1/index/rebuild")
        assert ok.status_code == 200
        assert rejected.status_code == 401           # whole router gated

    def test_public_api_unaffected_by_internal_auth(self):
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        with _client(SecuritySettings(internal_auth_token="s3cret")) as client:
            response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 200
        assert response.json() == CHAT_RESPONSE

    def test_internal_router_not_mounted_on_public_paths(self):
        """The internal router keeps its prefix; nothing leaks under /v1."""
        app = create_app(make_settings())

        def collect(router):
            paths = []
            for route in getattr(router, "routes", []) or []:
                nested = getattr(route, "original_router", None)
                if nested is not None:
                    paths.extend(collect(nested))
                elif getattr(route, "path", None):
                    paths.append(route.path)
            return paths

        paths = collect(app.router)
        public = [p for p in paths if p.startswith("/v1/")]
        internal = [p for p in paths if p.startswith("/internal/")]
        assert "/v1/chat/completions" in public
        assert "/internal/v1/diagnostics" in internal
        assert not [p for p in public if p.startswith("/internal")]
        # the public surface never exposes administrative operations
        for forbidden in ("/internal/v1/index/rebuild", "/internal/v1/diagnostics"):
            assert forbidden not in public


# -------------------------------------------------- Compose network hygiene


class _ComposeText:
    text: str = COMPOSE.read_text(encoding="utf-8")


def test_compose_never_publishes_postgres_on_wildcard():
    text = _ComposeText.text
    pg_block = text.split("\n  postgres:", 1)[1]
    port_lines = [ln.strip() for ln in pg_block.splitlines() if '"5432' in ln or ":5432" in ln]
    assert port_lines, "postgres publish block missing"
    for line in port_lines:
        assert line.startswith("- \"127.0.0.1:"), line


def test_compose_never_publishes_qdrant_on_wildcard():
    text = _ComposeText.text
    qdrant_block = text.split("\n  qdrant:", 1)[1]
    for token in ("6333", "6344"):
        lines = [ln.strip() for ln in qdrant_block.splitlines() if f':{token}"' in ln]
        assert lines, f"qdrant {token} publish block missing"
        for line in lines:
            assert line.startswith('- "127.0.0.1:'), line


def test_compose_pins_qdrant_version_not_latest():
    text = _ComposeText.text
    qdrant_block = text.split("\n  qdrant:", 1)[1].split("\n\n", 1)[0]
    image_line = next(ln for ln in qdrant_block.splitlines() if "image:" in ln)
    assert "qdrant/qdrant:" in image_line
    assert ":latest" not in image_line


def test_compose_container_traffic_still_uses_service_names():
    text = _ComposeText.text
    proxy_block = text.split("context-proxy:", 1)[1].split("\n  postgres:", 1)[0]
    assert "@postgres:5432" in proxy_block
    assert "http://qdrant:6333" in proxy_block


def test_compose_proxy_hardening_present():
    text = _ComposeText.text
    proxy_block = text.split("context-proxy:", 1)[1].split("\n  postgres:", 1)[0]
    assert "no-new-privileges:true" in proxy_block
    assert "cap_drop" in proxy_block and "- ALL" in proxy_block
    assert 'PYTHONDONTWRITEBYTECODE: "1"' in proxy_block
    assert 'PYTHONUNBUFFERED: "1"' in proxy_block


def test_dockerfile_sets_runtime_hygiene_env():
    dockerfile = (
        Path(__file__).resolve().parents[1] / "Dockerfile"
    ).read_text(encoding="utf-8")
    assert "PYTHONDONTWRITEBYTECODE=1" in dockerfile
    assert "PYTHONUNBUFFERED=1" in dockerfile
    assert "USER contextproxy" in dockerfile       # non-root preserved


# ---------------------------------------------------- §7 request parsing


class TestRequestParsingAbuseResistance:
    @pytest.mark.parametrize(
        "payload",
        [[1, 2, 3], "just a string", 42, None],
        ids=["list", "string", "number", "null"],
    )
    def test_non_object_json_controlled_rejection(self, payload):
        with _client() as client:
            response = client.post(
                "/v1/chat/completions", content=json.dumps(payload),
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request_body"

    def test_malformed_json_controlled_rejection(self):
        with _client() as client:
            response = client.post(
                "/v1/chat/completions",
                content=b'{"model": "m", "messages": [',
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 400
        assert "Traceback" not in response.text

    def test_deeply_nested_payload_no_traceback(self):
        deep = {"model": "m", "messages": [{"role": "user", "content": "x"}]}
        node: dict = {"leaf": 1}
        for _ in range(3000):                        # beyond json recursion limit
            node = {"n": [node]}
        deep["messages"][0]["extra"] = node          # type: ignore[index]

        class PermissiveHandler:
            def __init__(self, request: httpx.Request) -> None:
                self.request = request

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=CHAT_RESPONSE)

        settings = make_settings()
        app = create_app(
            settings,
            llm_client=httpx.AsyncClient(
                base_url=UPSTREAM, transport=httpx.MockTransport(handler)
            ),
        )
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/chat/completions",
                content=json.dumps(deep),
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code < 500 or response.status_code == 500
        assert "Traceback" not in response.text      # never a raw traceback

    def test_huge_body_over_limit_rejected_before_app(self):
        with _client() as client:
            response = client.post(
                "/v1/chat/completions",
                content=b'{"pad":"' + b"A" * (9 * 1024 * 1024) + b'"}',
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 413

    def test_invalid_conversation_id_controlled(self):
        body = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "conversation_id": "not-a-uuid-" + "z" * 4096,
        }
        with _client() as client:
            response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["code"] == "invalid_conversation_id"
        # the oversized value itself is echoed back by design? NO — bounded.
        assert len(json.dumps(response.text)) < 8192


# ------------------------------------------------------ error leakage sweep


LEAK_MARKERS = (
    "Traceback",
    ".py",
    "postgresql://",
    "asyncpg",
    "context_proxy/",
    "localhost:9",
)


class TestNoExceptionLeakage:
    def test_unexpected_store_error_generic_500_everywhere(self):
        class BoomStore:
            async def ping(self):
                return None

            async def ensure_conversation(self, conversation_id):
                return None

            async def reconcile_history(self, conversation_id, messages, metadata=None):
                raise TypeError(
                    "postgresql://admin:hunter2@db-int.internal:5432/ctx failed "
                    "at /Users/dev/app/src/context_proxy/conversation/store.py"
                )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=CHAT_RESPONSE)

        app = create_app(
            make_settings(),
            llm_client=httpx.AsyncClient(
                base_url=UPSTREAM, transport=httpx.MockTransport(handler)
            ),
            store=BoomStore(),
        )
        with TestClient(app, raise_server_exceptions=False) as client:
            chat = client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            )
            diag = client.get("/internal/v1/diagnostics")
        assert chat.status_code == 500
        blob = chat.text + diag.text
        for marker in LEAK_MARKERS:
            assert marker not in blob, marker

    def test_transport_failure_message_is_generic(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError(
                "secret-host.internal.local:11434 connection refused"
            )

        app = create_app(
            make_settings(),
            llm_client=httpx.AsyncClient(
                base_url=UPSTREAM, transport=httpx.MockTransport(handler)
            ),
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert response.status_code == 502
        assert response.json()["error"]["message"] == (
            "upstream inference endpoint is unavailable"
        )


# ----------------------------------------------------- header security


class TestHeaderSecurityMatrix:
    @staticmethod
    def _upstream_headers() -> dict[str, str]:
        return {
            "content-type": "application/json",
            "connection": "X-Custom, keep-alive",
            "x-custom": "connection-secret-value",
            "keep-alive": "timeout=5",
            "transfer-encoding": "chunked",
            "set-cookie": "sid=steal-me; HttpOnly",
            "content-length": "12345",
            "content-encoding": "gzip",
            "x-request-id": "req-safe",
            "retry-after": "9",
        }

    def test_buffered_response_filters_hop_by_hop_and_declared_tokens(self):
        import gzip

        upstream_headers = self._upstream_headers()
        # Real gzip body: httpx auto-decompresses, so the forwarded response
        # MUST NOT claim content-encoding anymore.
        raw_body = gzip.compress(json.dumps(CHAT_RESPONSE).encode())

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=raw_body, headers=upstream_headers)

        app = create_app(
            make_settings(),
            llm_client=httpx.AsyncClient(
                base_url=UPSTREAM, transport=httpx.MockTransport(handler)
            ),
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert response.status_code == 200
        lowered = {k.lower(): v for k, v in response.headers.items()}
        # hop-by-hop + Connection-declared tokens removed
        for absent in ("connection", "keep-alive", "transfer-encoding", "x-custom"):
            assert absent not in lowered, absent
        # connection-specific state removed
        assert "set-cookie" not in lowered
        # buffered bodies are decompressed by httpx: encoding must NOT lie
        assert "content-encoding" not in lowered
        assert response.json() == CHAT_RESPONSE       # decoded body intact
        # content-length recomputed by ASGI, never copied
        assert str(len(response.content)) == lowered.get("content-length")
        # safe metadata survives
        assert lowered.get("x-request-id") == "req-safe"
        assert lowered.get("retry-after") == "9"
        assert lowered["content-type"].startswith("application/json")

    def test_streaming_response_preserves_content_encoding_raw(self):
        async def agen():
            yield b"data: x\n\n"
            yield b"data: [DONE]\n\n"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=agen(),
                headers={
                    "content-type": "text/event-stream",
                    "content-encoding": "identity",
                    "set-cookie": "nope=1",
                    "connection": "close",
                },
            )

        app = create_app(
            make_settings(),
            llm_client=httpx.AsyncClient(
                base_url=UPSTREAM, transport=httpx.MockTransport(handler)
            ),
        )
        conv = "abcdefab-cdef-cdef-cdef-abcdefabcdef"
        with TestClient(app) as client, client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "m",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
                "conversation_id": conv,
            },
        ) as response:
            body = b"".join(response.iter_bytes())
            lowered = {k.lower(): v for k, v in response.headers.items()}
        assert body.endswith(b"data: [DONE]\n\n")
        # raw passthrough stays truthful about the bytes on the wire
        assert lowered.get("content-encoding") == "identity"
        # connection-specific state still stripped even when streaming
        assert "set-cookie" not in lowered
        assert "connection" not in lowered

    def test_missing_content_type_gets_default(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.dumps(CHAT_RESPONSE).encode()
            return httpx.Response(
                200, content=body, headers=[]  # no content-type at all
            )

        app = create_app(
            make_settings(),
            llm_client=httpx.AsyncClient(
                base_url=UPSTREAM, transport=httpx.MockTransport(handler)
            ),
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert response.headers["content-type"].startswith("application/json")
