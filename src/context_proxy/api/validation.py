"""Compatibility shim for request-shape validation.

The public chat-completions boundary is intentionally protocol-transparent.
The proxy must not reject or normalize provider-specific OpenAI-compatible
fields, tool calls, content parts, roles, or sampling parameters merely because
its local model of the wire format is narrower than the upstream provider.

JSON decoding and the top-level object check belong to ``parse_json_body``.
Everything else is owned by the configured upstream endpoint.
"""

from __future__ import annotations

from typing import Any


class PayloadValidationError(ValueError):
    """Retained for backwards compatibility with callers/tests."""


def validate_chat_payload(payload: dict[str, Any]) -> None:
    """Intentionally perform no protocol validation.

    The proxy may inspect a payload for context assembly, but it must never
    turn a provider-specific request into a proxy-generated 400 just because
    an OpenAI-compatible field has an unfamiliar shape. In particular, tool
    definitions/calls, content parts, roles, ``n``, and unknown extension
    fields are opaque transport data here.

    ``parse_json_body`` already guarantees that ``payload`` is a JSON object.
    """
    return None
