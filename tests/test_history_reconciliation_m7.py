from __future__ import annotations

from typing import Any

import httpx
import pytest
from conftest import CHAT_RESPONSE, client_for_handler

from context_proxy.conversation.reconciliation import reconcile_projection
from context_proxy.conversation.store import HistoryDivergenceError


def msg(role: str, content: Any, **extra: Any) -> dict[str, Any]:
    return {"role": role, "content": content, **extra}


def summary() -> dict[str, Any]:
    return msg(
        "assistant",
        """## Objective
- continue the task

## Important Details
- preserve the important context

## Work State
### Completed
- previous work
### Active
- current work
### Blocked
- (none)

## Next Move
1. continue

## Relevant Files
- src/example.py
""",
    )


def test_reasoning_difference_is_projection_equivalent():
    persisted = [msg("user", "hello"), msg("assistant", "world", reasoning_content="A")]
    incoming = [msg("user", "hello"), msg("assistant", "world", reasoning_content="B")]

    result = reconcile_projection(persisted, incoming)

    assert result.mode == "exact"
    assert result.append_from is None


def test_text_content_scalar_and_parts_are_equivalent():
    persisted = [msg("user", "hello")]
    incoming = [msg("user", [{"type": "text", "text": "hello"}])]

    result = reconcile_projection(persisted, incoming)

    assert result.mode == "exact"


def test_compaction_accepts_summary_and_preserved_tail():
    persisted = [
        msg("user", "A"),
        msg("assistant", "A1"),
        msg("user", "B"),
        msg("assistant", "B1"),
        msg("user", "C"),
        msg("assistant", "C1"),
    ]
    incoming = [
        msg("user", "A"),
        msg("assistant", "A1"),
        summary(),
        msg("user", "C"),
        msg("assistant", "C1"),
        msg("user", "new request"),
    ]

    result = reconcile_projection(persisted, incoming)

    assert result.mode == "compacted"
    assert result.append_from == 5


def test_compaction_without_new_tail_does_not_append_summary_or_tail():
    persisted = [msg("user", "A"), msg("assistant", "A1"), msg("user", "B"), msg("assistant", "B1")]
    incoming = [msg("user", "A"), summary(), msg("user", "B"), msg("assistant", "B1")]

    result = reconcile_projection(persisted, incoming)

    assert result.mode == "compacted"
    assert result.append_from == len(incoming)


def test_truncated_suffix_is_accepted():
    persisted = [msg("user", "A"), msg("assistant", "A1"), msg("user", "B"), msg("assistant", "B1")]
    incoming = [msg("user", "B"), msg("assistant", "B1")]

    result = reconcile_projection(persisted, incoming)

    assert result.mode == "truncate"
    assert result.append_from is None


def test_unanchored_rewrite_is_rejected():
    persisted = [msg("user", "A"), msg("assistant", "A1"), msg("user", "B"), msg("assistant", "B1")]
    incoming = [msg("user", "A"), msg("assistant", "different"), msg("user", "B"), msg("assistant", "B1")]

    result = reconcile_projection(persisted, incoming)

    assert result.mode == "conflict"


def test_unknown_tool_result_cannot_be_accepted_as_projection():
    persisted = [
        msg("user", "call tool"),
        msg(
            "assistant",
            None,
            tool_calls=[{"id": "call-1", "type": "function", "function": {"name": "x", "arguments": "{}"}}],
        ),
        msg("tool", "ok", tool_call_id="call-1"),
    ]
    incoming = [
        msg("user", "call tool"),
        msg(
            "assistant",
            None,
            tool_calls=[{"id": "call-1", "type": "function", "function": {"name": "x", "arguments": "{}"}}],
        ),
        msg("tool", "bad", tool_call_id="unknown"),
    ]

    result = reconcile_projection(persisted, incoming)

    assert result.mode == "conflict"


class ProjectionFakeStore:
    def __init__(self) -> None:
        self.conversations: dict[str, list[dict[str, Any]]] = {}

    async def ensure_conversation(self, conversation_id: str) -> None:
        self.conversations.setdefault(conversation_id, [])

    async def reconcile_history(self, conversation_id: str, messages: list[dict[str, Any]], metadata=None):
        persisted = self.conversations.setdefault(conversation_id, [])
        result = reconcile_projection(persisted, messages)
        if result.mode == "conflict":
            index = min(len(persisted), len(messages))
            raise HistoryDivergenceError(
                conversation_id,
                index,
                persisted=persisted[index] if index < len(persisted) else None,
                incoming=messages[index] if index < len(messages) else None,
            )
        if result.append_from is None or result.append_from >= len(messages):
            return []
        suffix = messages[result.append_from :]
        persisted.extend(suffix)
        return [f"msg-{len(persisted) - len(suffix) + i + 1}" for i in range(len(suffix))]

    async def get_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        return list(self.conversations.get(conversation_id, []))


def test_route_accepts_compacted_opencode_projection_and_persists_only_new_tail():
    store = ProjectionFakeStore()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=CHAT_RESPONSE)

    client = client_for_handler(handler, store=store)
    conversation_id = "77777777-7777-7777-7777-777777777777"

    with client:
        first = [msg("user", "A"), msg("assistant", "A1"), msg("user", "B"), msg("assistant", "B1")]
        store.conversations[conversation_id] = first.copy()
        projected = [
            msg("user", "A"),
            summary(),
            msg("user", "B"),
            msg("assistant", "B1"),
            msg("user", "new"),
        ]
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": projected, "conversation_id": conversation_id},
        )

    assert response.status_code == 200
    # The summary never replaces persisted history; only the new user turn and
    # the normal assistant response are appended.
    assert store.conversations[conversation_id][:4] == first
    assert store.conversations[conversation_id][4]["content"] == "new"
    assert store.conversations[conversation_id][5]["role"] == "assistant"


@pytest.mark.parametrize(
    "incoming",
    [
        [msg("user", "X")],
        [msg("user", "A"), msg("assistant", "X"), msg("user", "B")],
    ],
)
def test_real_rewrite_still_conflicts(incoming):
    persisted = [msg("user", "A"), msg("assistant", "A1"), msg("user", "B")]
    assert reconcile_projection(persisted, incoming).mode == "conflict"
