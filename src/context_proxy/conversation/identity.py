"""Conversation identification for stateless OpenAI-compatible clients.

The OpenAI protocol has no conversation concept, so the proxy accepts an
explicit identifier (body field `conversation_id` or `X-Conversation-ID`
header). When absent, each request is treated as a new conversation. The
resolved id is echoed back via the `X-Conversation-ID` response header so
clients can continue a conversation.

The body field is stripped before forwarding: upstream endpoints receive only
valid OpenAI fields.
"""

from __future__ import annotations

import uuid

from fastapi import Request

CONVERSATION_HEADER = "x-conversation-id"
RESPONSE_CONVERSATION_HEADER = "X-Conversation-ID"


class InvalidConversationId(ValueError):
    pass


def resolve_conversation_id(request: Request, payload: dict) -> tuple[str, dict]:
    """Return (conversation_id, sanitized_payload)."""
    raw = payload.pop("conversation_id", None)
    raw = raw or request.headers.get(CONVERSATION_HEADER)
    if raw is None:
        return str(uuid.uuid4()), payload
    if not isinstance(raw, str) or len(raw) > 64:
        raise InvalidConversationId("conversation_id must be a short string")
    try:
        uuid.UUID(raw)
    except ValueError as exc:
        raise InvalidConversationId(f"invalid conversation_id: {raw!r}") from exc
    return raw, payload
