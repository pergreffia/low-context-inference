"""Retrieval query extraction shared by production and preview paths (M4 §11.11).

Single source of truth: both the chat-completions request path and the
internal context preview endpoint must derive the identical textual query.
Only textual parts of multimodal content contribute; image parts and unknown
part types are ignored for retrieval purposes (they still pass through the
proxy untouched — M6 will give them first-class treatment).
"""

from __future__ import annotations

from typing import Any


def extract_retrieval_query(messages: list[dict[str, Any]]) -> str:
    """Text of the latest user message; multimodal parts contribute text only."""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                str(part.get("text") or "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ).strip()
        return ""
    return ""
