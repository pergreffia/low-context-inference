"""In-process token-bucket rate limiter (M5, master prompt §12).

Single-instance scope by design (§2.6): no Redis. Deterministic under an
injectable clock so tests never sleep.

Bounded-memory contract (post-0876b10 review §1): identity cardinality is
client-controlled (X-Conversation-ID rotation), so the bucket table is hard-
capped:

- `max_identities`: at most that many live buckets; when full, the least
  recently ACTIVE identity is evicted before a new one is created;
- `identity_ttl_seconds`: buckets idle longer than the TTL are dropped
  opportunistically on every admission decision;
- `max_identity_chars`: keys are truncated so oversized/invalid headers can
  never turn into unbounded dict keys.

All three bounds make worst-case memory O(max_identities) regardless of
client behavior. A single short-lived lock guards both dicts; no I/O and no
allocation beyond the two floats per bucket happens inside it.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from threading import Lock

from context_proxy.observability.metrics import RATE_LIMIT_EVICTED_TOTAL

logger = logging.getLogger(__name__)

DEFAULT_MAX_IDENTITY_CHARS = 256


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

    def allow(self, key: str) -> bool:
        """Consume one token; False when the bucket is empty."""
        key = self._normalize(key)
        now = self._clock()
        with self._lock:
            self._expire_stale(now)
            if key not in self._updated:
                self._make_room(key, now)
            last = self._updated.get(key)
            tokens = self._tokens[key]
            if last is not None:
                tokens = min(self._burst, tokens + (now - last) * self._rate)
            self._updated[key] = now
            if tokens >= 1.0:
                self._tokens[key] = tokens - 1.0
                return True
            self._tokens[key] = tokens
            return False

    def retry_after(self, key: str) -> int:
        """Seconds until the next token (rounded up, at least 1)."""
        with self._lock:
            tokens = self._tokens.get(self._normalize(key), self._burst)
        deficit = 1.0 - tokens
        if deficit <= 0:
            return 1
        return max(1, int(deficit / self._rate) + 1)

    def identity_count(self) -> int:
        """Live bucket count (observability/tests; bounded by max_identities)."""
        with self._lock:
            return len(self._tokens)

    def reset(self) -> None:
        with self._lock:
            self._tokens.clear()
            self._updated.clear()
