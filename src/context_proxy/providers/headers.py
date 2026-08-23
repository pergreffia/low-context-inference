"""Explicit response-header forwarding policy.

The proxy is transparent: upstream application/protocol headers are preserved
(rate-limit, retry, tracing, provider metadata). Hop-by-hop and
connection-specific headers are never forwarded (RFC 9110 §7.6.1).
"""

from __future__ import annotations

from collections.abc import Mapping

HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)

# Always stripped from forwarded responses.
STRIPPED_RESPONSE_HEADERS = HOP_BY_HOP_HEADERS | {
    "content-length",  # recomputed by the ASGI server for the forwarded body
    "set-cookie",  # upstream cookies are connection-specific state
}

DEFAULT_CONTENT_TYPE = "application/json"


def filter_response_headers(
    headers: Mapping[str, str],
    *,
    keep_content_encoding: bool,
) -> dict[str, str]:
    """Return headers safe to forward downstream.

    keep_content_encoding: buffered responses are auto-decompressed by httpx,
    so their content-encoding must be dropped to stay truthful about the body.
    Raw streaming passthrough keeps it.
    """
    stripped = STRIPPED_RESPONSE_HEADERS
    if not keep_content_encoding:
        stripped = stripped | {"content-encoding"}
    return {name: value for name, value in headers.items() if name.lower() not in stripped}


def ensure_content_type(
    headers: dict[str, str],
    default: str = DEFAULT_CONTENT_TYPE,
) -> dict[str, str]:
    if not any(name.lower() == "content-type" for name in headers):
        headers["Content-Type"] = default
    return headers


def get_header(headers: Mapping[str, str], name: str) -> str | None:
    for key, value in headers.items():
        if key.lower() == name.lower():
            return value
    return None
