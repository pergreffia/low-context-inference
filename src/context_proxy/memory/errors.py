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


class VectorStoreError(Exception):
    """Expected vector-store infrastructure failure (M5 final review §1).

    Raised by QdrantVectorStore for transport/timeouts/HTTP status problems:
    callers may degrade to lexical/raw retrieval. Programming errors
    (TypeError, AttributeError, malformed payloads) must NOT be wrapped —
    they propagate so bugs surface instead of masquerading as outages.
    """

    def __init__(self, message: str, *, cause: Exception | None = None):
        self.cause = cause
        super().__init__(message)


class EmbeddingProviderError(Exception):
    """Expected embedding-provider infrastructure failure (M6 review).

    Transport errors, timeouts, HTTP status failures and malformed provider
    payloads are degradable to lexical-only retrieval. Programming errors
    (TypeError, KeyError on internal data structures) propagate untouched.
    """

    def __init__(self, message: str, *, cause: Exception | None = None):
        self.cause = cause
        super().__init__(message)
