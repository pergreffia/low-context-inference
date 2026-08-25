"""Bounded rate limiter regressions (post-0876b10 review §1).

Memory-DoS contract: identity cardinality is client-controlled
(X-Conversation-ID rotation), so bucket count is hard-capped by
`max_identities`, idle buckets expire via `identity_ttl_seconds`, oversized
identities are truncated, and evictions are observable as a metric.
"""

from __future__ import annotations

import threading

import httpx
import pytest
from fastapi.testclient import TestClient

from context_proxy.config import RateLimitSettings
from context_proxy.main import create_app
from context_proxy.observability.metrics import REGISTRY
from context_proxy.observability.ratelimit import RateLimiter
from tests.conftest import CHAT_RESPONSE, UPSTREAM, make_settings


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_limiter(**kwargs) -> tuple[RateLimiter, FakeClock]:
    clock = FakeClock()
    defaults = dict(requests_per_minute=60, burst=2, clock=clock)
    defaults.update(kwargs)
    return RateLimiter(**defaults), clock


# ------------------------------------------------------------ token bucket


class TestTokenBucketBasics:
    def test_creation_refill_rejection(self):
        limiter, clock = make_limiter(burst=2, requests_per_minute=6)  # 0.1 tokens/s
        assert limiter.allow("a") is True
        assert limiter.allow("a") is True
        assert limiter.allow("a") is False          # burst exhausted
        clock.advance(5)                             # +0.5 tokens
        assert limiter.allow("a") is False           # still under one token
        clock.advance(10)                            # +1.0 more -> a full token
        assert limiter.allow("a") is True

    def test_retry_after_counts_down(self):
        limiter, clock = make_limiter(burst=1)
        assert limiter.allow("k")
        first = limiter.retry_after("k")
        clock.advance(first)
        assert limiter.retry_after("k") < first or limiter.allow("k")


# ------------------------------------------------------------------ bounds


class TestBoundedIdentities:
    def test_ttl_expires_idle_buckets(self):
        limiter, clock = make_limiter(identity_ttl_seconds=10, max_identities=100)
        limiter.allow("a")
        limiter.allow("b")
        assert limiter.identity_count() == 2
        clock.advance(5)
        limiter.allow("a")                           # refresh a only
        clock.advance(6)                             # b now older than TTL
        limiter.allow("c")
        assert limiter.identity_count() == 2         # b expired, a+c live

    def test_capacity_eviction_respects_max_identities(self):
        limiter, _clock = make_limiter(max_identities=5, identity_ttl_seconds=10_000)
        for i in range(50):
            limiter.allow(f"id-{i}")
        assert limiter.identity_count() <= 5

    def test_thousands_of_rotated_ids_never_exceed_limit(self):
        limiter, _clock = make_limiter(max_identities=32, identity_ttl_seconds=10_000)
        for i in range(5000):
            limiter.allow(f"00000000-0000-0000-0000-{i:012d}")
        assert limiter.identity_count() == 32

    def test_evicted_identity_can_return_fresh(self):
        limiter, _clock = make_limiter(max_identities=1, burst=1)
        assert limiter.allow("first") is True
        limiter.allow("second")                      # evicts "first"
        assert limiter.identity_count() == 1
        assert limiter.allow("first") is True        # fresh bucket, full burst

    def test_oversized_identity_truncated(self):
        limiter, _clock = make_limiter(
            max_identity_chars=8, max_identities=10, identity_ttl_seconds=10_000
        )
        limiter.allow("x" * 100_000)
        limiter.allow("x" * 100_001)                 # same 8-char prefix bucket
        assert limiter.identity_count() == 1

    def test_evictions_counted_in_metrics(self):
        REGISTRY.reset()
        limiter, _clock = make_limiter(max_identities=2, identity_ttl_seconds=10_000)
        for i in range(6):
            limiter.allow(f"z-{i}")
        text = REGISTRY.render()
        line = next(
            ln for ln in text.splitlines()
            if ln.startswith("context_proxy_rate_limit_identities_evicted_total")
        )
        assert int(line.rsplit(" ", 1)[1]) >= 4      # at least the capacity evictions

    def test_concurrent_access_thread_safe_and_bounded(self):
        limiter, _clock = make_limiter(max_identities=64, identity_ttl_seconds=10_000)
        barrier = threading.Barrier(16)

        def hammer(thread_id: int):
            barrier.wait()                           # all threads start together
            for i in range(200):
                limiter.allow(f"t{thread_id}-k{i}")

        threads = [threading.Thread(target=hammer, args=(t,)) for t in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert limiter.identity_count() <= 64        # bound held under contention

    def test_invalid_construction_rejected(self):
        with pytest.raises(ValueError):
            RateLimiter(requests_per_minute=0, burst=1)
        with pytest.raises(ValueError):
            RateLimiter(requests_per_minute=60, burst=1, max_identities=0)
        with pytest.raises(ValueError):
            RateLimiter(requests_per_minute=60, burst=1, identity_ttl_seconds=0)


# --------------------------------------------------- two-dimension policy


class TestTwoDimensionAdmission:
    """Rotating X-Conversation-ID must not bypass the client-level bucket."""

    def test_rotation_cannot_bypass_client_quota(self):
        limiter, clock = make_limiter(burst=2, requests_per_minute=6)
        outcomes = [
            limiter.admit(client_key="10.0.0.1", conversation_key=f"conv-{i}")
            for i in range(6)
        ]
        allowed = [d for d in outcomes if d.allowed]
        rejected = [d for d in outcomes if not d.allowed]
        assert len(allowed) == 2                      # client burst, ONCE
        assert len(rejected) == 4
        assert all(d.scope == "client" for d in rejected)
        assert all(d.retry_after >= 1 for d in rejected)

    def test_rotation_does_not_reset_client_bucket(self):
        limiter, clock = make_limiter(burst=3, requests_per_minute=6)
        for i in range(3):                            # exhaust via rotation
            decision = limiter.admit("10.0.0.7", f"c{i}")
            assert decision.allowed
        # a brand-new conversation id still hits the SAME drained client bucket
        fresh = limiter.admit("10.0.0.7", "brand-new-conversation")
        assert fresh.allowed is False
        assert fresh.scope == "client"
        clock.advance(120)                            # full refill @0.1/s -> burst back
        assert limiter.admit("10.0.0.7", "another-new").allowed is True

    def test_conversation_bucket_still_isolates_identities(self):
        limiter, _clock = make_limiter(burst=1, requests_per_minute=6)
        # same conversation from DIFFERENT clients: conv bucket shared
        first = limiter.admit("10.0.0.1", "conv-shared")
        second = limiter.admit("10.0.0.2", "conv-shared")
        assert first.allowed and second.allowed is False
        assert second.scope == "conversation"
        # different conversations from different clients: fully independent
        assert limiter.admit("10.0.0.3", "conv-x").allowed
        assert limiter.admit("10.0.0.4", "conv-y").allowed

    def test_no_conversation_header_checks_client_only(self):
        limiter, _clock = make_limiter(burst=1, requests_per_minute=6)
        assert limiter.admit("10.0.0.9").allowed
        denied = limiter.admit("10.0.0.9")
        assert denied.allowed is False and denied.scope == "client"

    def test_rejected_attempt_consumes_passing_dimension(self):
        """Hammering loophole closed: alternating identities stay throttled."""
        limiter, clock = make_limiter(burst=2, requests_per_minute=6)
        assert limiter.admit("ip", "a").allowed       # client 2 left, conv-a 0
        assert limiter.admit("ip", "b").allowed       # client 1 left
        # conv-a refilled by now? No — but the point: each attempt burns a
        # client token even when the conversation bucket rejects.
        clock.advance(60)                             # ~6 tokens back on both dims
        burned = [limiter.admit("ip", f"k{i}") for i in range(8)]
        allowed = sum(1 for d in burned if d.allowed)
        assert allowed <= 2                           # client cap held overall

    def test_memory_bound_holds_across_both_dimensions(self):
        limiter, _clock = make_limiter(
            max_identities=16, identity_ttl_seconds=10_000
        )
        for i in range(500):
            limiter.admit(f"host-{i % 7}", f"rotating-{i}")   # many conversations
        assert limiter.identity_count() <= 16         # ONE table, both namespaces

    def test_legacy_allow_and_retry_after_compat(self):
        limiter, _clock = make_limiter(burst=1, requests_per_minute=6)
        assert limiter.allow("solo") is True
        assert limiter.allow("solo") is False
        assert limiter.retry_after("solo") >= 1


# -------------------------------------------------------------- route level


def _app_with_rate_limits(store=None):
    settings = make_settings().model_copy(
        update={
            "rate_limit": RateLimitSettings(
                enabled=True,
                requests_per_minute=600,
                burst=3,
                max_identities=4,
                identity_ttl_seconds=3600,
                max_identity_chars=128,
            ),
            "security": __import__(
                "context_proxy.config", fromlist=["SecuritySettings"]
            ).SecuritySettings(),
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=CHAT_RESPONSE)

    return create_app(
        settings,
        llm_client=httpx.AsyncClient(
            base_url=UPSTREAM, transport=httpx.MockTransport(handler)
        ),
        store=store,
    )


class TestRouteLevelBoundedRateLimit:
    def test_n_much_greater_than_max_identities_stays_bounded(self):
        """Rotating X-Conversation-ID cannot grow memory past the cap."""
        app = _app_with_rate_limits()
        with TestClient(app) as client:
            for i in range(200):                     # N >> max_identities=4
                client.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                    headers={"X-Conversation-ID": f"{i:08d}-aaaa"},
                )
        # The bounded-memory contract: 200 distinct identities, <=4 buckets.
        assert app.state.rate_limiter.identity_count() <= 4

    def test_same_identity_exhausts_bucket_and_gets_429(self):
        REGISTRY.reset()
        app = _app_with_rate_limits()
        with TestClient(app) as client:
            statuses = []
            for _ in range(5):
                response = client.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                    headers={"X-Conversation-ID": "fixed-conv-0001"},
                )
                statuses.append(response.status_code)
            assert 429 in statuses
            assert response.headers["Retry-After"]
        text = REGISTRY.render()
        line = next(
            ln for ln in text.splitlines()
            if ln.startswith("context_proxy_rate_limit_rejects_total")
        )
        assert int(line.rsplit(" ", 1)[1]) >= 1      # rejection metric wired

    def test_client_host_bucket_bounded_without_header(self):
        """No conversation header -> bounded per-host bucket still applies."""
        app = _app_with_rate_limits()
        with TestClient(app) as client:
            statuses = [
                client.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                ).status_code
                for _ in range(5)
            ]
        assert 429 in statuses                       # host bucket exhausts too

    def test_huge_header_value_handled_without_error(self):
        """Giant identity must never crash or grow unbounded keys."""
        app = _app_with_rate_limits()
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                headers={"X-Conversation-ID": "H" * 64_000},
            )
        # Controlled outcome (rate-limited, served, or rejected by identity
        # validation) — but never a traceback/5xx.
        assert response.status_code in (200, 400, 429)
        if response.status_code == 400:
            assert response.json()["error"]["code"] == "invalid_conversation_id"
        assert app.state.rate_limiter.identity_count() <= 4

    def test_rotation_cannot_bypass_client_quota_route_level(self):
        """Same client rotating conversation ids: 429 after the client burst."""
        REGISTRY.reset()
        before = _reject_metric_value()
        app = _app_with_rate_limits()                 # burst=3
        with TestClient(app) as client:               # all from one client host
            statuses = []
            for i in range(8):
                conv = f"00000000-0000-0000-0000-{i:012d}"
                response = client.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                    headers={"X-Conversation-ID": conv},
                )
                statuses.append(response.status_code)
        assert statuses[:3] == [200, 200, 200]        # client burst admitted once
        assert set(statuses[3:]) == {429}             # rotation buys NOTHING

        rejected = statuses[3:].count(429)
        after = _reject_metric_value()
        assert after - before == rejected            # exactly one metric per reject

        last = client.post(                           # Retry-After contract
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Conversation-ID": "00000000-0000-0000-0000-999999999999"},
        )
        assert last.status_code == 429
        error = last.json()["error"]
        assert error["code"] == "rate_limit_exceeded"
        assert error["type"] == "rate_limit_error"
        retry_after = int(last.headers["Retry-After"])
        assert retry_after >= 1


def _reject_metric_value() -> int:
    text = REGISTRY.render()
    line = next(
        (ln for ln in text.splitlines()
         if ln.startswith("context_proxy_rate_limit_rejects_total")),
        None,
    )
    return int(line.rsplit(" ", 1)[1]) if line else 0
