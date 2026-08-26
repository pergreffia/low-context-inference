"""Final hardening pass regressions (post-e4f5cad review).

H1  breaker probe-reservation leak on cancellation/unclassified errors;
L4  X-Request-ID sanitization/bounding;
L1  FastAPI docs endpoints disabled in production mode;
M3  runtime fail-closed guard surviving `model_copy` validator bypass;
L2  boolean `n` rejection pin; L3 empty-messages contract pin;
M1  amortized TTL sweep proof for the rate limiter.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from context_proxy.config import SecuritySettings
from context_proxy.main import create_app
from context_proxy.providers.resilience import CircuitBreaker
from tests.conftest import CHAT_RESPONSE, UPSTREAM, make_settings

# ------------------------------------------------------------- H1 breaker


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _provider_for_breaker(clock: FakeClock, script: list):
    """Provider executing the scripted outcomes in order.

    Script entries: "connect" | "runtime" | "cancel" | "postsend" | 200.
    """
    settings = make_settings().model_copy(
        update={"resilience": __import__(
            "context_proxy.config", fromlist=["ResilienceSettings"]
        ).ResilienceSettings(max_retries=0)}
    )

    def handler(request: httpx.Request) -> httpx.Response:
        outcome = script.pop(0)
        if outcome == "connect":
            raise httpx.ConnectError("refused")
        if outcome == "runtime":
            raise RuntimeError("unexpected internal error")
        if outcome == "cancel":
            raise asyncio.CancelledError()
        if outcome == "postsend":
            response = httpx.Response(200)

            async def stall():
                raise httpx.ReadTimeout("stalled")

            response.aread = stall  # type: ignore[method-assign]
            return response
        return httpx.Response(200, json=CHAT_RESPONSE)

    from context_proxy.config import ResilienceSettings
    from context_proxy.providers.llm import OpenAICompatibleLLMProvider

    client = httpx.AsyncClient(
        base_url="http://upstream.test/v1",
        transport=httpx.MockTransport(handler),
    )
    provider = OpenAICompatibleLLMProvider(
        settings.inference,
        client=client,
        resilience=ResilienceSettings(max_retries=0),
        breaker=CircuitBreaker(failure_threshold=1, reset_seconds=50, clock=clock),
    )
    return provider


def _complete(provider):
    return asyncio.run(provider.complete({"model": "m", "messages": []}))


@pytest.mark.parametrize("probe_failure", ["runtime", "cancel", "postsend"])
def test_probe_reservation_never_leaks(probe_failure):
    """H1: any non-classified probe exit releases the HALF_OPEN reservation."""
    clock = FakeClock()
    script = ["connect"]                          # trip the breaker immediately
    provider = _provider_for_breaker(clock, script)
    breaker = provider._breaker

    from context_proxy.providers.errors import UpstreamUnavailable

    with pytest.raises(UpstreamUnavailable):
        _complete(provider)
    assert breaker.state == "open"
    assert breaker._probe_reserved is False

    clock.advance(60)                             # OPEN -> HALF_OPEN window

    # THE probe: this provider call exits WITHOUT a classified outcome.
    script.append(probe_failure)
    if probe_failure == "cancel":
        with pytest.raises(asyncio.CancelledError):
            _complete(provider)                   # cancellation propagates
    elif probe_failure == "postsend":
        with pytest.raises(UpstreamUnavailable):  # mapped, no retry
            _complete(provider)
    else:
        with pytest.raises(RuntimeError):         # unexpected error propagates
            _complete(provider)

    assert breaker._probe_reserved is False       # THE invariant: never stuck

    # the freed probe slot is immediately reservable and succeeds -> CLOSED
    assert breaker.allow_attempt() is True
    breaker.release_probe()                       # free it for the real call
    script.append(200)
    status, _h, _b = _complete(provider)
    assert status == 200
    assert breaker.state == "closed"


def test_probe_reservation_semantics_unit_level():
    """Direct reservation lifecycle: held -> exclusive -> releasable."""
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, reset_seconds=50, clock=clock)
    breaker.record_failure()
    assert breaker.state == "open"
    assert breaker.allow_attempt() is False       # OPEN fails fast

    clock.advance(60)                             # window elapsed
    assert breaker.allow_attempt() is True        # probe reserved
    assert breaker._probe_reserved is True
    assert breaker.allow_attempt() is False       # exclusive: second caller denied

    breaker.release_probe()
    assert breaker._probe_reserved is False
    assert breaker.allow_attempt() is True        # immediately reservable again
    breaker.release_probe()

    # idempotent outside any reservation
    breaker.release_probe()
    assert breaker._probe_reserved is False


def test_classified_outcomes_keep_normal_semantics():
    clock = FakeClock()
    script = ["connect", 200]
    provider = _provider_for_breaker(clock, script)
    breaker = provider._breaker
    from context_proxy.providers.errors import UpstreamUnavailable

    with pytest.raises(UpstreamUnavailable):
        _complete(provider)
    assert breaker.state == "open"
    clock.advance(60)
    script.append(200)
    status, _h, _b = _complete(provider)          # HALF_OPEN probe succeeds
    assert status == 200
    assert breaker.state == "closed"


# --------------------------------------------------- L4 X-Request-ID policy


def _client_with_echo_handler():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=CHAT_RESPONSE)

    app = create_app(
        make_settings(),
        llm_client=httpx.AsyncClient(
            base_url=UPSTREAM, transport=httpx.MockTransport(handler)
        ),
    )
    return TestClient(app)


class TestRequestIdPolicy:
    def test_oversized_request_id_falls_back_to_generated(self):
        with _client_with_echo_handler() as client:
            response = client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                headers={"X-Request-ID": "A" * 65_536},
            )
        assert response.status_code == 200            # no failure
        rid = response.headers["X-Request-ID"]
        assert rid != "A" * 65_536
        assert len(rid) <= 128                        # bounded

    @pytest.mark.parametrize(
        "bad_id", ["line1\nline2", "carriage\rreturn", "null\x00byte", "\t tab"]
    )
    def test_control_characters_fall_back_to_generated(self, bad_id):
        with _client_with_echo_handler() as client:
            response = client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                headers={"X-Request-ID": bad_id},
            )
        assert response.status_code == 200
        assert response.headers["X-Request-ID"] != bad_id

    def test_valid_request_id_echoed_unchanged(self):
        with _client_with_echo_handler() as client:
            response = client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                headers={"X-Request-ID": "req-123_ABC.01"},
            )
        assert response.headers["X-Request-ID"] == "req-123_ABC.01"

    def test_boundary_length_accepted(self):
        ok = "r" * 128
        with _client_with_echo_handler() as client:
            response = client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                headers={"X-Request-ID": ok},
            )
        assert response.headers["X-Request-ID"] == ok


# ------------------------------------------- L1 docs endpoints by mode


class TestDocsEndpointsByMode:
    def test_production_disables_docs_redoc_openapi(self):
        settings = make_settings().model_copy(
            update={
                "security": SecuritySettings(
                    mode="production", internal_auth_token="t"
                )
            }
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=CHAT_RESPONSE)

        app = create_app(
            settings,
            llm_client=httpx.AsyncClient(
                base_url=UPSTREAM, transport=httpx.MockTransport(handler)
            ),
        )
        with TestClient(app) as client:
            for path in ("/docs", "/redoc", "/openapi.json"):
                assert client.get(path).status_code == 404, path
            # public API unaffected
            ok = client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert ok.status_code == 200

    def test_development_keeps_docs_available(self):
        app = create_app(make_settings())
        with TestClient(app) as client:
            assert client.get("/openapi.json").status_code == 200
            assert client.get("/docs").status_code == 200


# ------------------------------- M3 runtime guard vs model_copy bypass


class TestProductionGuardSurvivesModelCopy:
    def test_runtime_fail_closed_when_validator_bypassed(self):
        """Programmatic settings construction skips validators: the runtime
        gate must hold. `SecuritySettings.model_construct` (and any
        `model_copy(update=...)` over an already-built Settings) produce
        production-without-token objects that never saw the validator."""
        base = make_settings()                       # valid development settings
        bypassed_security = SecuritySettings.model_construct(
            mode="production", internal_auth_token=""
        )
        bypassed = base.model_copy(update={"security": bypassed_security})
        assert bypassed.security.internal_auth_token == ""
        assert bypassed.security.mode == "production"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=CHAT_RESPONSE)

        app = create_app(
            bypassed,
            llm_client=httpx.AsyncClient(
                base_url=UPSTREAM, transport=httpx.MockTransport(handler)
            ),
        )
        with TestClient(app) as client:
            internal = client.get("/internal/v1/diagnostics")
            public = client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert internal.status_code == 503           # fail closed at runtime
        assert "misconfigured" in internal.json()["detail"]
        assert public.status_code == 200             # public surface untouched

    def test_production_with_token_still_works_at_runtime(self):
        settings = make_settings().model_copy(
            update={
                "security": SecuritySettings(
                    mode="production", internal_auth_token="tok"
                )
            }
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=CHAT_RESPONSE)

        app = create_app(
            settings,
            llm_client=httpx.AsyncClient(
                base_url=UPSTREAM, transport=httpx.MockTransport(handler)
            ),
        )
        with TestClient(app) as client:
            ok = client.get(
                "/internal/v1/diagnostics", headers={"X-Internal-Auth": "tok"}
            )
            denied = client.get("/internal/v1/diagnostics")
        assert ok.status_code == 200
        assert denied.status_code == 401


# ------------------------------------- L2/L3 validation contract pins


class TestValidationContractPins:
    @staticmethod
    def _post(client: TestClient, body: dict):
        return client.post("/v1/chat/completions", json=body)

    def test_boolean_n_rejected(self):
        with _client_with_echo_handler() as client:
            for value in (True, False):
                response = self._post(
                    client, {"model": "m", "n": value,
                             "messages": [{"role": "user", "content": "hi"}]}
                )
            assert response.status_code == 400
            assert "'n' must be an integer" in response.json()["error"]["message"]

    def test_integer_n_contract_unchanged(self):
        with _client_with_echo_handler() as client:
            accepted = self._post(
                client, {"model": "m", "n": 1,
                         "messages": [{"role": "user", "content": "hi"}]}
            )
            zero = self._post(
                client, {"model": "m", "n": 0,
                         "messages": [{"role": "user", "content": "hi"}]}
            )
            two = self._post(
                client, {"model": "m", "n": 2,
                         "messages": [{"role": "user", "content": "hi"}]}
            )
            string_n = self._post(
                client, {"model": "m", "n": "1",
                         "messages": [{"role": "user", "content": "hi"}]}
            )
        assert accepted.status_code == 200
        assert zero.status_code == 200                # passthrough contract
        assert two.status_code == 400                 # only n=1 supported
        assert "only n=1" in two.json()["error"]["message"]
        assert string_n.status_code == 400

    def test_empty_messages_rejected_locally(self):
        with _client_with_echo_handler() as client:
            response = self._post(client, {"model": "m", "messages": []})
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["code"] == "invalid_request_body"
        assert "must not be empty" in error["message"]

    def test_missing_messages_rejected_identically(self):
        with _client_with_echo_handler() as client:
            response = self._post(client, {"model": "m"})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request_body"


# --------------------------------------- M1 amortized rate-limit sweep


class TestAmortizedSweep:
    def test_expiry_deferred_between_sweep_windows(self):
        from tests.test_rate_limiter_bounds import make_limiter

        limiter, clock = make_limiter(
            burst=10,
            requests_per_minute=600,
            identity_ttl_seconds=100,
            max_identities=100,
        )
        limiter.admit("A")                            # t=0, sweep anchored t=0

        clock.advance(60)
        limiter.admit("B")                            # 60 >= ttl/2(50): sweep runs
        assert limiter.identity_count() == 2          # A idle 60 < 100: kept

        clock.advance(45)                             # t=105
        limiter.admit("C")                            # 45 < 50: NO sweep
        # A has been idle 105 > ttl — yet still present: expiry DEFERRED.
        assert limiter.identity_count() == 3

        clock.advance(45)                             # t=150, 90 >= 50: sweep
        limiter.admit("D")
        identities = limiter.identity_count()
        # A (idle 150 > 100) expired now; B(90)/C(45)/D fresh remain.
        assert identities == 3

        # hard capacity bound independent of sweeping
        tight, c2 = make_limiter(
            burst=10,
            requests_per_minute=600,
            identity_ttl_seconds=10_000,
            max_identities=4,
        )
        for i in range(20):
            tight.admit(f"k{i}")
        assert tight.identity_count() <= 4
        del c2
