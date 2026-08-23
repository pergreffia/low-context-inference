from __future__ import annotations

import uuid as uuid_mod

import pytest

from context_proxy.config import ConversationSettings, Settings
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


def _settings(client_header: str | None = "X-Session-ID") -> Settings:
    return Settings(
        _env_file=None,
        conversation=ConversationSettings(client_id_header=client_header or ""),
    )


SESSION_A = "session-token-A"
SESSION_B = "session-token-B"


def test_precedence_body_beats_everything():
    payload = {"conversation_id": "11111111-1111-1111-1111-111111111111"}
    request = _request(
        {
            "X-Conversation-ID": "22222222-2222-2222-2222-222222222222",
            "X-Session-ID": SESSION_A,
        }
    )
    conv_id, _ = resolve_conversation_id(request, payload, _settings())
    assert conv_id == "11111111-1111-1111-1111-111111111111"


def test_precedence_explicit_header_beats_client_identity():
    request = _request(
        {"X-Conversation-ID": "22222222-2222-2222-2222-222222222222", "X-Session-ID": SESSION_A}
    )
    conv_id, _ = resolve_conversation_id(request, {}, _settings())
    assert conv_id == "22222222-2222-2222-2222-222222222222"


def test_precedence_client_identity_before_uuid():
    request = _request({"X-Session-ID": SESSION_A})
    first, _ = resolve_conversation_id(request, {}, _settings())
    second, _ = resolve_conversation_id(_request({"X-Session-ID": SESSION_A}), {}, _settings())
    assert first == second  # stable identity -> stable conversation
    other, _ = resolve_conversation_id(_request({"X-Session-ID": SESSION_B}), {}, _settings())
    assert other != first  # distinct identities -> distinct conversations


def test_session_identity_maps_to_valid_uuid_deterministically():
    request = _request({"X-Session-ID": SESSION_A})
    conv_id, _ = resolve_conversation_id(request, {}, _settings())
    uuid_mod.UUID(conv_id)  # valid UUID
    again, _ = resolve_conversation_id(_request({"X-Session-ID": SESSION_A}), {}, _settings())
    assert again == conv_id


def test_no_identity_generates_fresh_uuids():
    a, _ = resolve_conversation_id(_request(), {}, _settings())
    b, _ = resolve_conversation_id(_request(), {}, _settings())
    assert a != b


def test_disabled_client_header_skips_session_identity():
    settings = _settings(client_header=None)
    request = _request({"X-Session-ID": SESSION_A})
    a, _ = resolve_conversation_id(request, {}, settings)
    b, _ = resolve_conversation_id(_request({"X-Session-ID": SESSION_A}), {}, settings)
    assert a != b


def test_non_string_body_value_rejected():
    import pytest

    with pytest.raises(InvalidConversationId):
        resolve_conversation_id(_request(), {"conversation_id": 12345}, _settings())


def test_oversized_explicit_id_rejected():
    import pytest

    with pytest.raises(InvalidConversationId):
        resolve_conversation_id(_request(), {"conversation_id": "a" * 65}, _settings())


def test_malformed_explicit_uuid_rejected():
    import pytest

    with pytest.raises(InvalidConversationId):
        resolve_conversation_id(_request({"X-Conversation-ID": "zzz-not-uuid"}), {}, _settings())


def test_oversized_session_token_rejected():
    import pytest

    with pytest.raises(InvalidConversationId):
        resolve_conversation_id(_request({"X-Session-ID": "x" * 129}), {}, _settings())
