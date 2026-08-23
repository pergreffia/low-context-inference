"""Assistant-response capture for persistence.

Streaming responses pass through byte-for-byte; a side channel accumulates the
SSE payload so the raw assistant message can be persisted once the stream
finishes. The passthrough is never rewritten or delayed.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from context_proxy.providers.base import LLMStream

logger = logging.getLogger(__name__)


class AssistantCapture:
    """Reconstructs an OpenAI assistant message from streamed SSE chunks."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._content_parts: list[str] = []
        self._tool_calls: dict[int, dict[str, Any]] = {}
        self._finish_reason: str | None = None
        self._saw_any_choice = False

    def feed(self, chunk: bytes) -> None:
        self._buffer.extend(chunk)

    def finalize(self) -> dict[str, Any] | None:
        for event in self._buffer.decode("utf-8", errors="replace").split("\n\n"):
            for line in event.splitlines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    payload = json.loads(data)
                except ValueError:
                    continue  # non-SSE noise is persisted nowhere but passed through
                choices = payload.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                self._saw_any_choice = True
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    self._content_parts.append(delta["content"])
                for tool_call in delta.get("tool_calls") or []:
                    index = tool_call.get("index", 0)
                    slot = self._tool_calls.setdefault(
                        index,
                        {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    if tool_call.get("id"):
                        slot["id"] = tool_call["id"]
                    function = tool_call.get("function") or {}
                    if function.get("name"):
                        slot["function"]["name"] += function["name"]
                    if function.get("arguments"):
                        slot["function"]["arguments"] += function["arguments"]
                if choice.get("finish_reason"):
                    self._finish_reason = choice["finish_reason"]

        if not self._saw_any_choice:
            return None
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(self._content_parts) or None,
        }
        if self._tool_calls:
            message["tool_calls"] = [
                self._tool_calls[index]
                for index in sorted(self._tool_calls)
                if self._tool_calls[index]["id"]
            ]
        return message


class PersistingLLMStream:
    """Wraps an LLMStream; passes bytes through untouched, persists at the end."""

    def __init__(
        self,
        inner: LLMStream,
        on_finished: Callable[[dict[str, Any] | None], Awaitable[None]],
    ):
        self._inner = inner
        self._on_finished = on_finished

    @property
    def status_code(self) -> int:
        return self._inner.status_code

    @property
    def media_type(self) -> str:
        return self._inner.media_type

    def passthrough_headers(self) -> dict[str, str]:
        return self._inner.passthrough_headers()

    async def iter_bytes(self):
        capture = AssistantCapture()
        completed = False
        try:
            async for chunk in self._inner.iter_bytes():
                capture.feed(chunk)
                yield chunk
            completed = True
        finally:
            message = capture.finalize() if completed else None
            if not completed:
                logger.warning("assistant_stream_incomplete_persistence_skipped")
            try:
                await self._on_finished(message)
            except Exception as exc:  # noqa: BLE001 - persistence must not break streaming
                logger.warning("assistant_persistence_failed", extra={"error": str(exc)})
