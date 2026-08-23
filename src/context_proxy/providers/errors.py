from __future__ import annotations

import httpx


class ContextProxyError(Exception):
    """Base class for proxy errors rendered as OpenAI-compatible errors."""


class UpstreamHTTPError(ContextProxyError):
    def __init__(self, status_code: int, body: bytes, content_type: str = "application/json"):
        self.status_code = status_code
        self.body = body
        self.content_type = content_type
        super().__init__(f"upstream returned {status_code}")


class UpstreamUnavailable(ContextProxyError):
    pass


def map_upstream_error(exc: httpx.HTTPError) -> ContextProxyError:
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return UpstreamUnavailable(f"cannot connect to upstream endpoint: {exc}")
    if isinstance(exc, httpx.TimeoutException):
        return UpstreamUnavailable(f"upstream request timed out: {exc}")
    return UpstreamUnavailable(f"upstream transport error: {exc}")
