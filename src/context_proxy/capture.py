"""Assistant-response capture for persistence.

Streaming responses pass through byte-for-byte; a side channel accumulates the
SSE payload so the raw assistant message can be persisted once the stream
finishes. The passthrough is never rewritten or delayed.

Semantic state preserved (M2.1 §4): role, accumulated content, tool_calls,
finish_reason, refusal, usage (final usage chunk), model. Transport framing
(SSE event boundaries) is not stored.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from context_proxy.conversation.store import HistoryDivergenceError
from context_proxy.memory.errors import PersistenceInfrastructureError
from context_proxy.providers.base import LLMStream

logger = logging.getLogger(__name__)


class AssistantCapture:
    """Reconstructs OpenAI semantic state from streamed SSE chunks."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._role: str | None = None
        self._content_parts: list[str] = []
        self._refusal_parts: list[str] = []
        self._tool_calls: dict[int, dict[str, Any]] = {}
        self._finish_reason: str | None = None
        self._usage: dict[str, Any] | None = None
        self._model: str | None = None
        self._saw_any_choice = False

    def feed(self, chunk: bytes) -> None:
        self._buffer.extend(chunk)

    def finalize(self) -> dict[str, Any] | None:
        """Return the reconstructed assistant message, or None if nothing usable."""
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
                self._absorb(payload)

        if not self._saw_any_choice:
            return None
        return self._build_message()

    def response_metadata(self) -> dict[str, Any]:
        """Response-level state that is not part of the assistant message."""
        metadata: dict[str, Any] = {}
        if self._finish_reason is not None:
            metadata["finish_reason"] = self._finish_reason
        if self._usage is not None:
            metadata["usage"] = self._usage
        if self._model is not None:
            metadata["model"] = self._model
        return metadata

    def _absorb(self, payload: dict[str, Any]) -> None:
        if payload.get("usage"):
            self._usage = payload["usage"]
        if payload.get("model"):
            self._model = payload["model"]
        choices = payload.get("choices") or []
        if not choices:
            return
        choice = choices[0]
        self._saw_any_choice = True
        delta = choice.get("delta") or choice.get("message") or {}
        if delta.get("role"):
            self._role = delta["role"]
        if delta.get("content"):
            self._content_parts.append(delta["content"])
        if delta.get("refusal"):
            self._refusal_parts.append(delta["refusal"])
        for tool_call in delta.get("tool_calls") or []:
            index = tool_call.get("index", 0)
            slot = self._tool_calls.setdefault(index, {})
            if tool_call.get("id"):
                slot["id"] = tool_call["id"]
            if tool_call.get("type"):
                slot["type"] = tool_call["type"]
            # Preserve unknown transport fields from the FIRST fragment only;
            # never invent or discard semantic fields.
            for key, value in tool_call.items():
                if key not in ("index", "id", "type", "function", "custom"):
                    slot.setdefault(key, value)
            function = tool_call.get("function")
            if isinstance(function, dict):
                fn = slot.setdefault(
                    "function", {"name": "", "arguments": ""}
                )
                if function.get("name"):
                    fn["name"] += function["name"]
                if function.get("arguments"):
                    fn["arguments"] += function["arguments"]
            custom = tool_call.get("custom")
            if isinstance(custom, dict):
                cc = slot.setdefault("custom", {"name": "", "input": ""})
                if custom.get("name"):
                    cc["name"] += custom["name"]
                if custom.get("input"):
                    cc["input"] += custom["input"]
        if choice.get("finish_reason"):
            self._finish_reason = choice["finish_reason"]

    def _build_message(self) -> dict[str, Any]:
        content = "".join(self._content_parts) or None
        refusal = "".join(self._refusal_parts) or None
        tool_calls: list[dict[str, Any]] = []
        for index in sorted(self._tool_calls):
            call = self._tool_calls[index]
            if not call.get("id"):
                continue
            if call.get("type") == "custom" or "custom" in call:
                custom = call.get("custom") or {}
                rebuilt: dict[str, Any] = {
                    "id": call["id"],
                    "type": "custom",
                    "custom": {
                        "name": custom.get("name", ""),
                        "input": custom.get("input", ""),
                    },
                }
            else:
                function = call.get("function") or {}
                rebuilt = {
                    "id": call["id"],
                    "type": call.get("type", "function"),
                    "function": {
                        "name": function.get("name", ""),
                        "arguments": function.get("arguments", ""),
                    },
                }
            # unknown transport-preserved fields ride along untouched
            for key, value in call.items():
                if key not in ("id", "type", "function", "custom"):
                    rebuilt[key] = value
            tool_calls.append(rebuilt)
        message: dict[str, Any] = {
            "role": self._role or "assistant",
            "content": content,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        if refusal:
            message["refusal"] = refusal
        return message


CaptureResult = tuple[dict[str, Any] | None, dict[str, Any]]


class PersistingLLMStream:
    """Wraps an LLMStream; passes bytes through untouched, persists at the end."""

    def __init__(
        self,
        inner: LLMStream,
        on_finished: Callable[[dict[str, Any] | None, dict[str, Any]], Awaitable[None]],
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
            metadata = capture.response_metadata() if completed else {}
            if not completed:
                logger.warning("assistant_stream_incomplete_persistence_skipped")
            try:
                await self._on_finished(message, metadata)
            except HistoryDivergenceError as exc:
                # Classification guard: the route callback normally handles this
                # itself; if a raw reconciliation callback ever lets one escape,
                # it must still NOT be misclassified as a persistence failure.
                logger.warning(
                    "assistant_persistence_conflict",
                    extra={"conversation_id": exc.conversation_id, "index": exc.index},
                )
            except PersistenceInfrastructureError as exc:
                # Expected infrastructure failure: passthrough already done,
                # degrade with the standard event.
                logger.warning(
                    "assistant_persistence_failed",
                    extra={"error": str(exc)},
                )
            except Exception as exc:
                # The passthrough is sacred and already completed: we never
                # break the stream. But unexpected programming errors are NOT
                # logged as ordinary infrastructure degradation — they get a
                # dedicated error event with traceback so bugs stay visible
                # (M6-final review §3).
                logger.error(
                    "assistant_persistence_programming_error",
                    extra={"error": str(exc), "error_type": type(exc).__name__},
                    exc_info=True,
                )
