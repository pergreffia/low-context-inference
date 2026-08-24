"""Conversation identification for stateless OpenAI-compatible clients.

Precedence (M2.1 §2):

    1. body field `conversation_id`
    2. `X-Conversation-ID` header
    3. configured session identity header (`conversation.client_id_header`)
    4. generated UUID

Explicit conversation ids (1–2) must be valid UUIDs. Session identities (3)
are opaque tokens mapped deterministically to a UUID so the same stable
client identity always resolves to the same conversation, without forcing a
UUID format onto callers.

Semantics:
- stable identity -> same conversation on every request;
- no identity -> every request gets a fresh conversation (no implicit
  cross-request continuity);
- the resolved id is echoed via `X-Conversation-ID` for clients that want to
  pin it; HTTP does not guarantee that any client reuses it.

The body field is stripped before forwarding: upstream endpoints receive only
valid OpenAI fields.
"""

from __future__ import annotations

import uuid

from fastapi import Request

CONVERSATION_HEADER = "x-conversation-id"
RESPONSE_CONVERSATION_HEADER = "X-Conversation-ID"

_SESSION_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://context-proxy.local/sessions")


class InvalidConversationId(ValueError):
    pass


def _validated_explicit_id(raw: str) -> str:
    if not isinstance(raw, str) or not raw or len(raw) > 64:
        raise InvalidConversationId("conversation_id must be a short non-empty string")
    try:
        uuid.UUID(raw)
    except ValueError as exc:
        raise InvalidConversationId(f"invalid conversation_id: {raw!r}") from exc
    return raw


def _session_conversation_id(token: str, max_chars: int) -> str:
    token = token.strip()
    if not token or len(token) > max_chars:
        raise InvalidConversationId(
            f"session identity header must be a short non-empty string "
            f"(max {max_chars} characters)"
        )
    return str(uuid.uuid5(_SESSION_NAMESPACE, token))


def resolve_conversation_id(request: Request, payload: dict, settings=None) -> tuple[str, dict]:
    """Return (conversation_id, sanitized_payload)."""
    raw = payload.pop("conversation_id", None)
    raw = raw or request.headers.get(CONVERSATION_HEADER)
    if raw is not None:
        return _validated_explicit_id(raw), payload

    client_header = getattr(settings.conversation, "client_id_header", None) if settings else None
    if client_header:
        session_token = request.headers.get(client_header.lower())
        if session_token:
            max_chars = int(
                getattr(
                    settings.conversation,
                    "max_session_identity_chars",
                    128,
                )
            )
            return _session_conversation_id(session_token, max_chars), payload

    return str(uuid.uuid4()), payload
