"""In-process token-bucket rate limiter (M5, master prompt §12).

Single-instance scope by design (§2.6): no Redis. Deterministic under an
injectable clock so tests never sleep.

Two-dimension admission (post-04592c0 review §2): a request is admitted only
when BOTH applicable buckets have a token:

    client/IP bucket        (always — the stable principal)
    + conversation bucket   (when X-Conversation-ID is present)

Rotating X-Conversation-ID can no longer mint fresh quota: the client-level
bucket aggregates all of a host's traffic.

Bounded-memory contract (post-0876b10 review §1, preserved): both dimensions
live in ONE capped table (namespaced keys), so memory stays O(max_identities):

- `max_identities`: at most that many live buckets across BOTH namespaces;
  when full, the least recently ACTIVE bucket is evicted;
- `identity_ttl_seconds`: idle buckets are dropped opportunistically;
- `max_identity_chars`: keys are truncated so oversized headers can never
  turn into unbounded dict keys.

A single short-lived lock guards the table; no I/O and no allocation beyond
two floats per bucket happens inside it.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from threading import Lock

from context_proxy.observability.metrics import RATE_LIMIT_EVICTED_TOTAL

logger = logging.getLogger(__name__)

DEFAULT_MAX_IDENTITY_CHARS = 256

_CLIENT_PREFIX = "c\x00"
_CONVERSATION_PREFIX = "v\x00"

SCOPE_CLIENT = "client"
SCOPE_CONVERSATION = "conversation"


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    """Outcome of a two-dimension admission check."""

    allowed: bool
    scope: str | None = None          # which bucket rejected (None if allowed)
    retry_after: int = 1


class RateLimiter:
    def __init__(
        self,
        requests_per_minute: int,
        burst: int,
        clock=time.monotonic,
        *,
        max_identities: int = 10_000,
        identity_ttl_seconds: float = 3600.0,
        max_identity_chars: int = DEFAULT_MAX_IDENTITY_CHARS,
    ):
        if requests_per_minute <= 0 or burst <= 0:
            raise ValueError("requests_per_minute and burst must be positive")
        if max_identities < 1:
            raise ValueError("max_identities must be >= 1")
        if identity_ttl_seconds <= 0:
            raise ValueError("identity_ttl_seconds must be positive")
        if max_identity_chars < 1:
            raise ValueError("max_identity_chars must be >= 1")
        self._rate = requests_per_minute / 60.0  # tokens per second
        self._burst = float(burst)
        self._clock = clock
        self._max_identities = max_identities
        self._ttl = identity_ttl_seconds
        self._max_key_chars = max_identity_chars
        self._tokens: dict[str, float] = defaultdict(lambda: self._burst)
        self._updated: dict[str, float] = {}
        self._lock = Lock()
        # Amortized TTL sweep (final hardening pass): a full-table scan under
        # the global lock on EVERY admission is O(max_identities) per request.
        # Sweep at most once per half-TTL window instead — expiry semantics
        # shift by at most that window, the hard capacity bound is unaffected.
        self._last_sweep = self._clock()
        self._sweep_interval = self._ttl / 2.0

    # ------------------------------------------------------------ internals

    def _normalize(self, key: str) -> str:
        # Truncation collapses arbitrarily long identities onto a bounded
        # prefix: giant headers cannot create giant keys.
        return key[: self._max_key_chars]

    def _expire_stale(self, now: float) -> None:
        # Caller must hold the lock. TTL sweep keeps truly idle buckets from
        # lingering even when capacity is never reached.
        stale = [k for k, last in self._updated.items() if now - last > self._ttl]
        for k in stale:
            del self._tokens[k]
            del self._updated[k]
        if stale:
            RATE_LIMIT_EVICTED_TOTAL.inc(len(stale))
            logger.info("rate_limit_buckets_expired", extra={"count": len(stale)})

    def _make_room(self, incoming: str, now: float) -> None:
        # Caller must hold the lock, key absent, table at capacity. Evict the
        # least recently active bucket (LRU by last refill); the incoming
        # identity wins over the coldest resident one.
        while len(self._tokens) >= self._max_identities:
            coldest = min(self._updated, key=self._updated.get)  # type: ignore[arg-type]
            del self._tokens[coldest]
            del self._updated[coldest]
            RATE_LIMIT_EVICTED_TOTAL.inc()
            logger.debug("rate_limit_bucket_evicted", extra={"reason": "capacity"})

    def _consume(self, namespaced_key: str, now: float) -> bool:
        # Caller must hold the lock.
        if now - self._last_sweep >= self._sweep_interval:
            self._expire_stale(now)
            self._last_sweep = now
        if namespaced_key not in self._updated:
            self._make_room(namespaced_key, now)
        last = self._updated.get(namespaced_key)
        tokens = self._tokens[namespaced_key]
        if last is not None:
            tokens = min(self._burst, tokens + (now - last) * self._rate)
        self._updated[namespaced_key] = now
        if tokens >= 1.0:
            self._tokens[namespaced_key] = tokens - 1.0
            return True
        self._tokens[namespaced_key] = tokens
        return False

    def _seconds_until_token(self, namespaced_key: str) -> int:
        # Caller must hold the lock.
        tokens = self._tokens.get(namespaced_key, self._burst)
        deficit = 1.0 - tokens
        if deficit <= 0:
            return 1
        return max(1, int(deficit / self._rate) + 1)

    # ------------------------------------------------------------- public API

    def admit(
        self,
        client_key: str,
        conversation_key: str | None = None,
    ) -> AdmissionDecision:
        """Consume one token from the client bucket and, when present, from
        the conversation bucket. Rejected with the failing scope otherwise.

        A rejected request still consumes the passing dimension's token: an
        attempt is an attempt, and this closes the hammering loophole where a
        caller alternates between two identities to keep both buckets full.
        """
        client_ns = _CLIENT_PREFIX + self._normalize(client_key)
        conversation_ns = (
            _CONVERSATION_PREFIX + self._normalize(conversation_key)
            if conversation_key
            else None
        )
        now = self._clock()
        with self._lock:
            if not self._consume(client_ns, now):
                return AdmissionDecision(
                    allowed=False,
                    scope=SCOPE_CLIENT,
                    retry_after=self._seconds_until_token(client_ns),
                )
            if conversation_ns is not None and not self._consume(conversation_ns, now):
                return AdmissionDecision(
                    allowed=False,
                    scope=SCOPE_CONVERSATION,
                    retry_after=self._seconds_until_token(conversation_ns),
                )
        return AdmissionDecision(allowed=True)

    def allow(self, key: str) -> bool:
        """Legacy single-identity check (compat): one bucket, no second dim."""
        return self.admit(key).allowed

    def retry_after(self, key: str) -> int:
        """Seconds until the next token (rounded up, at least 1)."""
        with self._lock:
            return self._seconds_until_token(_CLIENT_PREFIX + self._normalize(key))

    def identity_count(self) -> int:
        """Live bucket count across BOTH dimensions; <= max_identities."""
        with self._lock:
            return len(self._tokens)

    def reset(self) -> None:
        with self._lock:
            self._tokens.clear()
            self._updated.clear()
