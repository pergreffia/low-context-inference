"""Domain errors for the Memory Service (master prompt §18).

Expected retrieval failures are semantically distinct from programming
errors: callers may degrade to raw/recent context on RetrievalError, but an
unexpected TypeError/AttributeError must propagate untouched.
"""

from __future__ import annotations


class RetrievalError(Exception):
    """Expected retrieval infrastructure failure (PostgreSQL leg down, etc.).

    Raising this type signals "retrieval unavailable" — a degraded but
    understood state. Anything else must not be converted into degradation.
    """

    def __init__(self, message: str, *, cause: Exception | None = None):
        self.cause = cause
        super().__init__(message)
