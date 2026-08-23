from __future__ import annotations

import pytest

from context_proxy.conversation.identity import (
    InvalidConversationId,
    resolve_conversation_id,
)


class _FakeRequest:
    def __init__(self, headers: dict[str, str] | None = None):
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}

    def get(self, name: str, default=None):
        return self.headers.get(name.lower(), default)


def _request(headers: dict[str, str] | None = None):
    req = _FakeRequest(headers)
    return type("R", (), {"headers": req.headers})


BODY_CONV = "11111111-1111-1111-1111-111111111111"
HEADER_CONV = "22222222-2222-2222-2222-222222222222"


def test_body_conversation_id_wins_and_is_stripped():
    payload = {"model": "m", "messages": [], "conversation_id": BODY_CONV}
    request = _request({"X-Conversation-ID": HEADER_CONV})
    conv_id, sanitized = resolve_conversation_id(request, payload)
    assert conv_id == BODY_CONV
    assert "conversation_id" not in sanitized


def test_header_used_when_no_body_field():
    payload = {"model": "m", "messages": []}
    conv_id, _ = resolve_conversation_id(
        _request({"x-conversation-id": "33333333-3333-3333-3333-333333333333"}), payload
    )
    assert conv_id == "33333333-3333-3333-3333-333333333333"


def test_absent_generates_new_uuid_each_time():
    a, _ = resolve_conversation_id(_request(), {"messages": []})
    b, _ = resolve_conversation_id(_request(), {"messages": []})
    assert a != b


def test_invalid_uuid_rejected():
    with pytest.raises(InvalidConversationId):
        resolve_conversation_id(_request(), {"conversation_id": "not-a-uuid"})
