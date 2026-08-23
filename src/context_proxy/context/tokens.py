"""Token accounting for the context budget (master prompt §15)."""

from __future__ import annotations

import json
import math
from typing import Any

CHARS_PER_TOKEN = 4
MESSAGE_OVERHEAD_TOKENS = 4
TOOL_CALL_OVERHEAD_TOKENS = 8
TOOL_DEF_OVERHEAD_TOKENS = 8


class TokenCounter:
    """Heuristic estimator: ~4 chars per token plus structural overhead.

    Deliberately model-agnostic (works for any tokenizer family) and
    deterministic. Exact counts are not required for safety because the
    configured safety_margin_tokens absorbs estimation error.
    """

    def text(self, value: str | None) -> int:
        if not value:
            return 0
        return max(1, math.ceil(len(value) / CHARS_PER_TOKEN))

    def _serialized(self, value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, str):
            return self.text(value)
        return self.text(json.dumps(value, separators=(",", ":"), ensure_ascii=False))

    def message(self, message: dict[str, Any]) -> int:
        tokens = MESSAGE_OVERHEAD_TOKENS
        tokens += self._serialized(message.get("content"))
        for tool_call in message.get("tool_calls") or []:
            tokens += TOOL_CALL_OVERHEAD_TOKENS
            function = tool_call.get("function") or {}
            tokens += self.text(function.get("name"))
            tokens += self._serialized(function.get("arguments"))
        if message.get("tool_call_id"):
            tokens += self.text(message["tool_call_id"])
        if message.get("name"):
            tokens += self.text(message["name"])
        return tokens

    def messages(self, messages: list[dict[str, Any]]) -> int:
        return sum(self.message(m) for m in messages)

    def tools(self, tools: list[dict[str, Any]] | None) -> int:
        if not tools:
            return 0
        total = 0
        for tool in tools:
            total += TOOL_DEF_OVERHEAD_TOKENS + self._serialized(tool)
        return total
