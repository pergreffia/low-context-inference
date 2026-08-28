"""Token accounting for the context budget (master prompt §15)."""

from __future__ import annotations

import json
import math
from typing import Any

CHARS_PER_TOKEN = 4
MESSAGE_OVERHEAD_TOKENS = 4
TOOL_CALL_OVERHEAD_TOKENS = 8
TOOL_DEF_OVERHEAD_TOKENS = 8
# Multimodal content parts (M6): a deterministic, model-agnostic estimate per
# image part. Base64 payloads are NEVER serialized into the estimate — a
# data URL would otherwise inflate the count by ~chars/4 and starve the
# budget. Unknown part types cost the same flat placeholder: opaque parts
# stay in the request (M6.1 transparency) while accounting stays sane.
IMAGE_PART_TOKENS = 1024
UNKNOWN_PART_TOKENS = 16


class TokenCounter:
    """Heuristic estimator: ~4 chars per token plus structural overhead.

    Deliberately model-agnostic (works for any tokenizer family) and
    deterministic. Exact counts are not required for safety because the
    configured safety_margin_tokens absorbs estimation error.

    The estimator is deliberately defensive: unknown provider-specific
    structures are charged conservatively instead of being validated,
    normalized, or allowed to break the passthrough request.
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
        if isinstance(value, list):
            # Multimodal content parts (M6 §13.1): text parts counted as
            # text; image/unknown parts via flat estimates so base64 blobs
            # never distort the budget.
            return self._parts(value)
        return self.text(json.dumps(value, separators=(",", ":"), ensure_ascii=False))

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
                    # Unknown tool-call containers are opaque. Estimate their
                    # serialized representation without inspecting semantics.
                    tokens += self._serialized(function)
        else:
            tokens += UNKNOWN_PART_TOKENS

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
