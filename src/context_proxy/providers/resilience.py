"""Upstream resilience primitives (M5): circuit breaker + bounded retries.

Retry policy (master prompt §12): ONLY transport-level failures that occur
before any response byte is received are retried (connect errors, connect
timeouts). An upstream HTTP error response is an ANSWER — never retried.
Streaming: no retry once the stream opened; only the pre-stream send phase is
covered.

The circuit breaker is per provider endpoint: after `failure_threshold`
consecutive transport failures it opens and fails fast; after
`reset_seconds` it half-opens and lets one attempt through. State changes are
published as gauges. A monotonic clock is injectable so tests are exact.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Awaitable, Callable

from context_proxy.observability.metrics import set_circuit_state

logger = logging.getLogger(__name__)

CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"


class CircuitBreaker:
    """Thread-safe breaker with an atomic HALF_OPEN probe reservation.

    Contract: OPEN fails fast; after reset_seconds it transitions to
    HALF_OPEN where EXACTLY ONE concurrent caller may probe upstream — every
    other caller fails fast until the probe resolves. Probe success closes
    the circuit; probe failure reopens it.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        reset_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self._threshold = failure_threshold
        self._reset_seconds = reset_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._state = CLOSED
        self._consecutive_failures = 0
        self._opened_at = 0.0
        self._probe_reserved = False

    @property
    def state(self) -> str:
        with self._lock:
            return self._current_state()

    def _current_state(self) -> str:
        # Caller must hold the lock.
        if self._state == OPEN and (self._clock() - self._opened_at) >= self._reset_seconds:
            self._transition(HALF_OPEN)
        return self._state

    def _transition(self, state: str) -> None:
        # Caller must hold the lock.
        self._state = state
        if state == OPEN:
            self._opened_at = self._clock()
            self._probe_reserved = False
        elif state == CLOSED:
            self._probe_reserved = False
        logger.info("circuit_state_changed", extra={"state": state})
        set_circuit_state(state)

    def allow_attempt(self) -> bool:
        """Atomically reserve the right to attempt a call.

        False when OPEN, or when HALF_OPEN and the single probe is already
        reserved by another in-flight request.
        """
        with self._lock:
            state = self._current_state()
            if state == OPEN:
                return False
            if state == HALF_OPEN:
                if self._probe_reserved:
                    return False
                self._probe_reserved = True
                return True
            return True

    def record_success(self) -> None:
        with self._lock:
            if self._state != CLOSED:
                self._transition(CLOSED)
            self._consecutive_failures = 0
            self._probe_reserved = False

    def record_failure(self) -> None:
        with self._lock:
            self._probe_reserved = False
            self._consecutive_failures += 1
            if self._state == HALF_OPEN:
                self._transition(OPEN)
            elif (
                self._state == CLOSED
                and self._consecutive_failures >= self._threshold
            ):
                self._transition(OPEN)

    def release_probe(self) -> None:
        """Release a HALF_OPEN probe reservation without recording an outcome.

        Safety net for attempts that end without a classifiable result — task
        cancellation (client disconnects cancel handler tasks) or unexpected
        internal errors. Without this the breaker could stay pinned in
        HALF_OPEN forever: `_current_state()` only transitions OPEN→HALF_OPEN,
        so a leaked reservation would block all traffic indefinitely.
        Idempotent; no-op outside HALF_OPEN reservations.
        """
        with self._lock:
            self._probe_reserved = False


async def with_retries[T](
    operation: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    backoff_base_seconds: float,
    backoff_max_seconds: float,
    retry_on: tuple[type[BaseException], ...],
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> T:
    """Run `operation`, retrying only `retry_on` failures with full jitter.

    Full-jitter exponential backoff: sleep = uniform(0, min(cap, base*2^try)).
    Deterministic in distribution bounds; jitter prevents thundering herds.
    """
    do_sleep = sleep or _async_sleep
    last_error: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return await operation()
        except retry_on as exc:
            last_error = exc
            if attempt == max_retries:
                break
            cap = min(backoff_max_seconds, backoff_base_seconds * (2**attempt))
            await do_sleep(random.uniform(0, cap))
    assert last_error is not None
    raise last_error


async def _async_sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)
