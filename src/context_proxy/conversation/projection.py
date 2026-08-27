"""Classify model-facing requests that are not canonical conversation turns."""

from __future__ import annotations

from typing import Any

# OpenCode's built-in title invocation prepends this user message to the real
# conversation. It is an auxiliary model projection, not canonical history.
_OPEN_CODE_TITLE_PREFIX = "Generate a title for this conversation:"


def is_auxiliary_projection(messages: list[dict[str, Any]]) -> bool:
    """Return True when *messages* are a known auxiliary LLM projection.

    The classifier intentionally relies on request shape/content rather than
    agent names. OpenCode agents are user-configurable and the same agent name
    may be used by multiple independent child sessions.
    """
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip().startswith(_OPEN_CODE_TITLE_PREFIX):
            return True
        if isinstance(content, list):
            text = "\n".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
            if text.strip().startswith(_OPEN_CODE_TITLE_PREFIX):
                return True
    return False
