"""M5 unit tests: circuit breaker, bounded retries, rate limiter, metrics."""

from __future__ import annotations

import httpx
import pytest

from context_proxy.config import ContextSettings
from context_proxy.observability.logging_setup import redact_text
from context_proxy.observability.metrics import REGISTRY, Counter, Histogram
from context_proxy.observability.ratelimit import RateLimiter
from context_proxy.providers.errors import UpstreamUnavailable
from context_proxy.providers.resilience import CircuitBreaker, with_retries


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestCircuitBreaker:
    def test_opens_after_threshold_and_fails_fast(self):
        clock = FakeClock()
        breaker = CircuitBreaker(failure_threshold=3, reset_seconds=10, clock=clock)
        for _ in range(3):
            assert breaker.allow_attempt()
            breaker.record_failure()
        assert breaker.state == "open"
        assert not breaker.allow_attempt()

    def test_half_open_after_reset_then_closes_on_success(self):
        clock = FakeClock()
        breaker = CircuitBreaker(failure_threshold=2, reset_seconds=5, clock=clock)
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == "open"
        clock.advance(6)
        assert breaker.state == "half_open"
        assert breaker.allow_attempt()
        breaker.record_success()
        assert breaker.state == "closed"

    def test_half_open_failure_reopens(self):
        clock = FakeClock()
        breaker = CircuitBreaker(failure_threshold=1, reset_seconds=1, clock=clock)
        breaker.record_failure()
        clock.advance(2)
        assert breaker.state == "half_open"
        breaker.record_failure()
        assert breaker.state == "open"

    def test_success_resets_consecutive_failures(self):
        breaker = CircuitBreaker(failure_threshold=3, reset_seconds=1, clock=FakeClock())
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == "closed"  # only 2 consecutive now


@pytest.mark.anyio
async def test_retries_only_transport_errors_with_bounded_attempts():
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    attempts = 0

    async def flaky():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ConnectError("boom")
        return "ok"

    result = await with_retries(
        flaky,
        max_retries=2,
        backoff_base_seconds=0.1,
        backoff_max_seconds=1.0,
        retry_on=(httpx.HTTPError,),
        sleep=fake_sleep,
    )
    assert result == "ok"
    assert attempts == 3
    assert len(sleeps) == 2
    # full jitter bounds: 0 <= sleep <= min(cap, base*2^attempt)
    assert all(0 <= s <= 0.4 for s in sleeps)


@pytest.mark.anyio
async def test_http_error_response_is_never_retried():
    """An upstream ANSWER (mapped error) is terminal — no retries."""
    attempts = 0

    async def answered():
        nonlocal attempts
        attempts += 1
        raise UpstreamUnavailable("upstream returned 503")

    with pytest.raises(UpstreamUnavailable):
        await with_retries(
            answered,
            max_retries=3,
            backoff_base_seconds=0.01,
            backoff_max_seconds=0.02,
            retry_on=(httpx.HTTPError,),
        )
    assert attempts == 1


@pytest.mark.anyio
async def test_exhausted_retries_reraise_last_transport_error():
    async def always_down():
        raise httpx.ConnectTimeout("timeout")

    with pytest.raises(httpx.ConnectTimeout):
        await with_retries(
            always_down,
            max_retries=1,
            backoff_base_seconds=0.0,
            backoff_max_seconds=0.0,
            retry_on=(httpx.HTTPError,),
            sleep=_noop,
        )


async def _noop(_seconds: float) -> None:
    return None


class TestRateLimiter:
    def test_burst_allows_then_blocks(self):
        clock = FakeClock()
        limiter = RateLimiter(requests_per_minute=60, burst=3, clock=clock)
        assert [limiter.allow("k") for _ in range(3)] == [True, True, True]
        assert limiter.allow("k") is False

    def test_refill_over_time_is_deterministic(self):
        clock = FakeClock()
        # 60 rpm = 1 token/second; burst 2.
        limiter = RateLimiter(requests_per_minute=60, burst=2, clock=clock)
        assert limiter.allow("k")
        assert limiter.allow("k")
        assert not limiter.allow("k")
        clock.advance(1.0)
        assert limiter.allow("k")
        assert not limiter.allow("k")

    def test_keys_are_isolated(self):
        clock = FakeClock()
        limiter = RateLimiter(requests_per_minute=60, burst=1, clock=clock)
        assert limiter.allow("a")
        assert not limiter.allow("a")
        assert limiter.allow("b")

    def test_retry_after_positive(self):
        clock = FakeClock()
        limiter = RateLimiter(requests_per_minute=60, burst=1, clock=clock)
        limiter.allow("k")
        assert limiter.retry_after("k") >= 1


def test_counter_render_prometheus_text():
    counter = Counter("m5_test_total", "test counter", ("route",))
    counter.labels(route="/v1/chat/completions").inc()
    counter.labels(route="/v1/chat/completions").inc()
    lines = counter.render()
    assert "# TYPE m5_test_total counter" in lines
    assert 'm5_test_total{route="/v1/chat/completions"} 2' in "".join(lines)


def test_histogram_buckets_cumulative():
    histogram = Histogram(
        "m5_test_seconds", "test histogram", labelnames=("route",), buckets=(0.1, 1.0)
    )
    bound = histogram.labels(route="r")
    bound.observe(0.05)
    bound.observe(0.5)
    text = "\n".join(histogram.render())
    assert 'm5_test_seconds_bucket{route="r",le="0.1"} 1' in text
    assert 'm5_test_seconds_bucket{route="r",le="1.0"} 2' in text
    assert 'm5_test_seconds_bucket{route="r",le="+Inf"} 2' in text
    assert 'm5_test_seconds_count{route="r"} 2' in text


def test_registry_renders_all_collectors():
    text = REGISTRY.render()
    assert "context_proxy_http_requests_total" in text


def test_redaction_scrubs_bearer_tokens():
    leaked = "error: Bearer sk-supersecret-123 rejected by upstream"
    assert "sk-supersecret-123" not in redact_text(leaked)
    assert "[REDACTED]" in redact_text(leaked)


def test_config_validator_rejects_margin_over_limit():
    with pytest.raises(ValueError):
        ContextSettings(model_limit_tokens=1000, safety_margin_tokens=2000)


class TestHalfOpenSingleProbe:
    """HALF_OPEN admits exactly one concurrent probe (M5 review §3)."""

    def test_exactly_one_probe_among_concurrent_callers(self):
        import threading

        clock = FakeClock()
        breaker = CircuitBreaker(failure_threshold=1, reset_seconds=5, clock=clock)
        breaker.record_failure()  # OPEN
        clock.advance(6)          # reset elapsed -> next read becomes HALF_OPEN

        results: list[bool] = []
        lock = threading.Lock()
        start = threading.Barrier(16)

        def try_enter():
            start.wait()
            allowed = breaker.allow_attempt()
            with lock:
                results.append(allowed)

        threads = [threading.Thread(target=try_enter) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sorted(results, reverse=True)[0] is True   # exactly one probe...
        assert sum(results) == 1                          # ...all others fail fast

    def test_failed_probe_reopens_breaker(self):
        clock = FakeClock()
        breaker = CircuitBreaker(failure_threshold=1, reset_seconds=5, clock=clock)
        breaker.record_failure()
        clock.advance(6)
        assert breaker.state == "half_open"
        assert breaker.allow_attempt()  # the single probe
        breaker.record_failure()
        assert breaker.state == "open"
        assert not breaker.allow_attempt()

    def test_successful_probe_closes_breaker(self):
        clock = FakeClock()
        breaker = CircuitBreaker(failure_threshold=1, reset_seconds=5, clock=clock)
        breaker.record_failure()
        clock.advance(6)
        assert breaker.allow_attempt()
        breaker.record_success()
        assert breaker.state == "closed"
        assert breaker.allow_attempt()

    def test_second_concurrent_probe_fails_fast_while_first_in_flight(self):
        clock = FakeClock()
        breaker = CircuitBreaker(failure_threshold=1, reset_seconds=5, clock=clock)
        breaker.record_failure()
        clock.advance(6)
        assert breaker.allow_attempt() is True   # probe reserved
        for _ in range(10):
            assert breaker.allow_attempt() is False  # everyone else blocked


class TestRecursiveRedaction:
    """Centralized recursive redaction (M5 review §9)."""

    def test_bearer_token_scrubbed_from_text(self):
        from context_proxy.observability.logging_setup import redact_text

        out = redact_text("upstream said Bearer abc.def-ghi is invalid")
        assert "abc.def-ghi" not in out
        assert "Bearer" in out

    def test_sensitive_keys_masked_at_any_depth(self):
        import json

        from context_proxy.observability.logging_setup import redact

        payload = {
            "api_key": "sk-top-level",
            "nested": {
                "authorization": "Bearer deep-secret",
                "inner": {"client_secret": "hunter2", "note": "Bearer visible-scrubbed"},
                "list": [{"password": "p@ss"}, "plain", 42],
            },
            "safe": "value",
        }
        out = json.loads(json.dumps(redact(payload)))
        assert out["api_key"] == "[REDACTED]"
        assert out["nested"]["authorization"] == "[REDACTED]"
        assert out["nested"]["inner"]["client_secret"] == "[REDACTED]"
        assert "visible-scrubbed" not in out["nested"]["inner"]["note"]
        assert out["nested"]["list"][0]["password"] == "[REDACTED]"
        assert out["nested"]["list"][1] == "plain"
        assert out["nested"]["list"][2] == 42
        assert out["safe"] == "value"

    def test_credential_field_name_is_redacted(self):
        from context_proxy.observability.logging_setup import redact

        assert redact({"credential": "anything-here"}) == {"credential": "[REDACTED]"}
