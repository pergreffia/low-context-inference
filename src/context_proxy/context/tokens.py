from __future__ import annotations

import json
import math
from typing import Any

CHARS_PER_TOKEN = 4
MESSAGE_OVERHEAD_TOKENS = 4
TOOL_CALL_OVERHEAD_TOKENS = 8
TOOL_DEF_OVERHEAD_TOKENS = 8
IMAGE_PART_TOKENS = 1024
UNKNOWN_PART_TOKENS = 16


class TokenCounter:
    """Deterministic, model-agnostic token estimate with opaque-data safety."""

    def text(self, value: Any) -> int:
        if value is None or value == "":
            return 0
        if not isinstance(value, str):
            value = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        return max(1, math.ceil(len(value) / CHARS_PER_TOKEN))

    def _serialized(self, value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return self._parts(value)
        try:
            serialized = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        except (TypeError, ValueError):
            return UNKNOWN_PART_TOKENS
        return self.text(serialized)

    @staticmethod
    def _part_type(part: Any) -> str | None:
        if isinstance(part, dict) and isinstance(part.get("type"), str):
            return part["type"]
        return None

    def _parts(self, parts: list[Any]) -> int:
        total = 0
        for part in parts:
            kind = self._part_type(part)
            if kind == "text":
                text_value = part.get("text")
                total += (
                    self.text(text_value)
                    if isinstance(text_value, str)
                    else UNKNOWN_PART_TOKENS
                )
            elif kind == "image_url":
                total += IMAGE_PART_TOKENS
            else:
                total += UNKNOWN_PART_TOKENS
        return total

    def message(self, message: dict[str, Any]) -> int:
        tokens = MESSAGE_OVERHEAD_TOKENS
        if not isinstance(message, dict):
            return tokens + UNKNOWN_PART_TOKENS

        tokens += self._serialized(message.get("content"))
        tool_calls = message.get("tool_calls") or []
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                tokens += TOOL_CALL_OVERHEAD_TOKENS
                if not isinstance(tool_call, dict):
                    tokens += UNKNOWN_PART_TOKENS
                    continue
                function = tool_call.get("function")
                if isinstance(function, dict):
                    tokens += self.text(function.get("name"))
                    tokens += self._serialized(function.get("arguments"))
                else:
                    tokens += self._serialized(function)
        else:
            tokens += UNKNOWN_PART_TOKENS

        tokens += self.text(message.get("tool_call_id")) if message.get("tool_call_id") else 0
        tokens += self.text(message.get("name")) if message.get("name") else 0
        return tokens

    def messages(self, messages: list[dict[str, Any]]) -> int:
        return sum(self.message(m) for m in messages)

    def tools(self, tools: list[dict[str, Any]] | None) -> int:
        if not tools:
            return 0
        return sum(TOOL_DEF_OVERHEAD_TOKENS + self._serialized(tool) for tool in tools)
