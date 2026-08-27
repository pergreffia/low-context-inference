"""Classify model-facing requests that are not canonical conversation turns."""

from __future__ import annotations

from typing import Any

# OpenCode's built-in title invocation prepends this user message to the real
# conversation. It is an auxiliary model projection, not canonical history.
_OPEN_CODE_TITLE_PREFIX = "Generate a title for this conversation:"


def _user_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [message for message in messages if message.get("role") == "user"]


def _text_content(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def is_auxiliary_projection(messages: list[dict[str, Any]]) -> bool:
    """Return True when *messages* are a known auxiliary LLM projection.

    The classifier intentionally relies on request shape/content rather than
    agent names. OpenCode agents are user-configurable and the same agent name
    may be used by multiple independent child sessions.

    The title marker must be the first user message and there must be another
    user message after it. This avoids treating a normal user message that
    happens to start with the same phrase as an auxiliary request.
    """
    users = _user_messages(messages)
    if len(users) < 2:
        return False
    return _text_content(users[0]).strip().startswith(_OPEN_CODE_TITLE_PREFIX)
