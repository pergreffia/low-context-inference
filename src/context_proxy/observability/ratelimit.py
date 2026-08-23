"""In-process token-bucket rate limiter (M5, master prompt §12).

Single-instance scope by design (§2.6): no Redis. Deterministic under an
injectable clock so tests never sleep.
"""

from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock


class RateLimiter:
    def __init__(self, requests_per_minute: int, burst: int, clock=time.monotonic):
        if requests_per_minute <= 0 or burst <= 0:
            raise ValueError("requests_per_minute and burst must be positive")
        self._rate = requests_per_minute / 60.0  # tokens per second
        self._burst = float(burst)
        self._clock = clock
        self._tokens: dict[str, float] = defaultdict(lambda: self._burst)
        self._updated: dict[str, float] = {}
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        """Consume one token; False when the bucket is empty."""
        now = self._clock()
        with self._lock:
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
            tokens = self._tokens.get(key, self._burst)
        deficit = 1.0 - tokens
        if deficit <= 0:
            return 1
        return max(1, int(deficit / self._rate) + 1)

    def reset(self) -> None:
        with self._lock:
            self._tokens.clear()
            self._updated.clear()
