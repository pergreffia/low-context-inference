"""Same-proxy provider failure/recovery regressions (review §5).

The real-provider E2E previously suspended the proxy and booted a SECOND
proxy — it proved nothing about recovery of the running instance. This suite
is the authoritative deterministic contract:

    1. provider available          -> request succeeds
    2. provider unavailable        -> request fails per contract
                                      (502, code=upstream_unavailable)
    3. provider available again    -> request succeeds through the
                                      SAME app/proxy instance

Circuit breaker stays correct across the cycle (OPEN fails fast without an
upstream call; HALF_OPEN probe recovers to CLOSED) and persistence behavior
stays correct throughout (failed turns leave no partial assistant rows).
"""

from __future__ import annotations

import asyncio
import time

import httpx
from fastapi.testclient import TestClient

from context_proxy.config import ResilienceSettings
from context_proxy.main import create_app
from tests.conftest import CHAT_RESPONSE, UPSTREAM, make_settings

CONV_OK_1 = "aaaaaaa1-0000-0000-0000-000000000000"
CONV_DOWN_A = "aaaaaaa2-0000-0000-0000-000000000000"
CONV_DOWN_B = "aaaaaaa3-0000-0000-0000-000000000000"
CONV_RECOVERED = "aaaaaaa4-0000-0000-0000-000000000000"

BREAKER_RESET_SECONDS = 0.2


class ToggleableUpstream:
    """Deterministic healthy/unavailable/healthy switch."""

    def __init__(self) -> None:
        self.up = True
        self.attempts = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.attempts += 1
        if not self.up:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, json=CHAT_RESPONSE)


def recovery_client(upstream: ToggleableUpstream, store):
    settings = make_settings().model_copy(
        update={
            "resilience": ResilienceSettings(
                max_retries=0,
                backoff_base_seconds=0.0,
                breaker_failure_threshold=2,
                breaker_reset_seconds=BREAKER_RESET_SECONDS,
            )
        }
    )
    app = create_app(
        settings,
        llm_client=httpx.AsyncClient(
            base_url=UPSTREAM, transport=httpx.MockTransport(upstream)
        ),
        store=store,
    )
    return TestClient(app)


class RecordingStore:
    def __init__(self) -> None:
        self.conversations: dict[str, list[dict]] = {}

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



def persisted(store, conv: str) -> list[dict]:
    return asyncio.run(store.get_messages(conv))


def post(client: TestClient, conv: str):
    return client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [{"role": "user", "content": f"ping {conv}"}],
            "conversation_id": conv,
        },
    )


class TestSameProxyFailureRecovery:
    def test_full_cycle_through_one_app_instance(self):
        upstream = ToggleableUpstream()
        store = RecordingStore()

        # ONE client context == ONE proxy instance for the whole cycle.
        with recovery_client(upstream, store) as client:
            # -- phase 1: healthy -> success
            r1 = post(client, CONV_OK_1)
            assert r1.status_code == 200
            assert r1.json() == CHAT_RESPONSE
            assert [(m["role"], m["content"]) for m in persisted(store, CONV_OK_1)] == [
                ("user", f"ping {CONV_OK_1}"),
                ("assistant", "hello"),
            ]
            assert client.get("/readyz").json()["checks"]["circuit_breaker"] == "closed"

            # -- phase 2: provider down -> contract failure on the SAME proxy
            upstream.up = False
            attempts_before = upstream.attempts
            ra = post(client, CONV_DOWN_A)
            assert ra.status_code == 502
            body_a = ra.json()["error"]
            assert body_a["code"] == "upstream_unavailable"
            assert body_a["type"] == "api_error"
            assert body_a["message"] == "upstream inference endpoint is unavailable"
            rb = post(client, CONV_DOWN_B)   # second failure trips the breaker
            assert rb.status_code == 502
            assert client.get("/readyz").json()["checks"]["circuit_breaker"] == "open"

            # OPEN breaker fails fast WITHOUT touching the upstream
            attempts_after_open = upstream.attempts
            rc = post(client, CONV_DOWN_A)
            assert rc.status_code == 502
            assert rc.json()["error"]["code"] == "upstream_unavailable"
            assert upstream.attempts == attempts_after_open

            # failed turns persisted inbound user but NO partial assistant
            for conv in (CONV_DOWN_A, CONV_DOWN_B):
                roles = [m["role"] for m in persisted(store, conv)]
                assert roles == ["user"]

            # -- phase 3: provider back -> SAME instance recovers
            upstream.up = True
            time.sleep(BREAKER_RESET_SECONDS + 0.05)  # OPEN -> HALF_OPEN window

            rd = post(client, CONV_RECOVERED)
            assert rd.status_code == 200              # HALF_OPEN probe succeeds
            assert rd.json() == CHAT_RESPONSE
            assert client.get("/readyz").json()["checks"]["circuit_breaker"] == "closed"

            re_ = post(client, CONV_RECOVERED)
            assert re_.status_code == 200             # fully closed again
            # identical replay is idempotent: still exactly ONE exchange
            assert [(m["role"], m["content"]) for m in persisted(store, CONV_RECOVERED)] == [
                ("user", f"ping {CONV_RECOVERED}"),
                ("assistant", "hello"),
            ]

        # sanity: the outage actually reached the upstream before opening
        assert upstream.attempts > attempts_before

    def test_streaming_served_by_same_instance_after_recovery(self):
        upstream = ToggleableUpstream()
        store = RecordingStore()
        sse = (
            b'data: {"choices":[{"delta":{"role":"assistant","content":"hi"}}]}\n\n'
            b"data: [DONE]\n\n"
        )

        def handler(request: httpx.Request) -> httpx.Response:
            if not upstream.up:
                raise httpx.ConnectError("connection refused")

            async def agen():
                yield sse

            return httpx.Response(
                200,
                content=agen(),
                headers={"content-type": "text/event-stream"},
            )

        settings = make_settings().model_copy(
            update={"resilience": ResilienceSettings(max_retries=0)}
        )
        app = create_app(
            settings,
            llm_client=httpx.AsyncClient(
                base_url=UPSTREAM, transport=httpx.MockTransport(handler)
            ),
            store=store,
        )
        conv = "aaaaaaa5-0000-0000-0000-000000000000"
        payload = {
            "model": "m",
            "messages": [{"role": "user", "content": "stream"}],
            "conversation_id": conv,
            "stream": True,
        }
        with TestClient(app) as client:
            upstream.up = False
            r_fail = client.post("/v1/chat/completions", json=payload)
            assert r_fail.status_code == 502
            assert r_fail.json()["error"]["code"] == "upstream_unavailable"

            upstream.up = True
            with client.stream("POST", "/v1/chat/completions", json=payload) as ok:
                body = b"".join(ok.iter_bytes())
            assert body == sse                       # full stream, same instance
