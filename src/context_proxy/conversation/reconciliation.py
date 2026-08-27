"""Projection-aware conversation history reconciliation."""

from dataclasses import dataclass
import hashlib
import json
import logging
from typing import Any

PRUNED_TOOL_RESULT = "[Old tool result content cleared]"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReconciliationResult:
    """Result of reconciling a client history projection."""

    mode: str
    append_from: int | None = None


def _text_content(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return None
    parts: list[str] = []
    for part in value:
        if not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str):
            return None
        parts.append(part["text"])
    return "".join(parts)


def canonical_message(message: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize only known representation-level differences."""
    result = dict(message)
    for key in ("reasoning_content", "reasoning", "reasoning_text"):
        result.pop(key, None)
    text = _text_content(result.get("content"))
    if text is not None:
        result["content"] = text
    if result.get("role") == "assistant" and result.get("tool_calls") and result.get("content") is None:
        result["content"] = ""
    return result


def _canonical_fingerprint(message: dict[str, Any]) -> str:
    payload = json.dumps(canonical_message(message), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _canonical_differing_fields(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    keys = set(left) | set(right)
    return sorted(key for key in keys if left.get(key) != right.get(key))


def _content_shape(value: Any) -> tuple[str, int, bool, bool, str | None]:
    if value is None:
        return "null", 0, True, False, None
    if isinstance(value, str):
        return "string", len(value), len(value) == 0, False, None
    if isinstance(value, list):
        return "list", len(value), len(value) == 0, False, ",".join(
            str(item.get("type", "?")) if isinstance(item, dict) else type(item).__name__ for item in value
        )
    return type(value).__name__, 0, False, False, None


def _tool_shape(value: Any) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    if not isinstance(value, list):
        return 0, (), ()
    ids: list[str] = []
    names: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("id"), str):
            ids.append(item["id"])
        function = item.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.append(function["name"])
    return len(value), tuple(ids), tuple(names)


def _tool_details(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    details: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        arguments = function.get("arguments") if isinstance(function, dict) else None
        argument_payload = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")) if not isinstance(arguments, str) else arguments
        argument_bytes = argument_payload.encode("utf-8")
        item_payload = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        details.append({
            "keys": tuple(sorted(item.keys())),
            "function_keys": tuple(sorted(function.keys())) if isinstance(function, dict) else (),
            "arguments_type": type(arguments).__name__ if arguments is not None else "null",
            "arguments_len": len(argument_payload) if arguments is not None else 0,
            "arguments_sha256": hashlib.sha256(argument_bytes).hexdigest()[:16] if arguments is not None else None,
            "tool_sha256": hashlib.sha256(item_payload.encode("utf-8")).hexdigest()[:16],
        })
    return tuple(details)


def _argument_shape(value: Any) -> tuple[str, int, str | None, tuple[str, ...], tuple[str, ...]]:
    if isinstance(value, str):
        raw = value
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return "str", len(raw), hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16], (), ()
        if isinstance(parsed, dict):
            return "str-json-object", len(raw), hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16], tuple(sorted(parsed)), tuple(
                key for key, item in parsed.items() if isinstance(item, (str, int, float, bool)) or item is None
            )
        return f"str-json-{type(parsed).__name__}", len(raw), hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16], (), ()
    if isinstance(value, dict):
        return "dict", len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))), None, tuple(sorted(value)), tuple(
            key for key, item in value.items() if isinstance(item, (str, int, float, bool)) or item is None
        )
    return type(value).__name__, 0, None, (), ()


def _tool_argument_details(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    details: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        arguments = function.get("arguments") if isinstance(function, dict) else None
        shape = _argument_shape(arguments)
        details.append({
            "id": item.get("id"),
            "name": function.get("name") if isinstance(function, dict) else None,
            "argument_shape": shape,
        })
    return tuple(details)


def _diagnose_difference(index: int, persisted: dict[str, Any], incoming: dict[str, Any]) -> None:
    left = canonical_message(persisted)
    right = canonical_message(incoming)
    persisted_content = _content_shape(left.get("content"))
    incoming_content = _content_shape(right.get("content"))
    persisted_tools = _tool_shape(left.get("tool_calls"))
    incoming_tools = _tool_shape(right.get("tool_calls"))
    logger.warning(
        "history_reconciliation_message_difference index=%s persisted_role=%s incoming_role=%s "
        "persisted_keys=%s incoming_keys=%s canonical_persisted_sha256=%s canonical_incoming_sha256=%s "
        "canonical_different_fields=%s persisted_content_shape=%s incoming_content_shape=%s "
        "persisted_tool_shape=%s incoming_tool_shape=%s persisted_tool_details=%s incoming_tool_details=%s "
        "persisted_tool_argument_details=%s incoming_tool_argument_details=%s",
        index,
        persisted.get("role"),
        incoming.get("role"),
        sorted(left),
        sorted(right),
        _canonical_fingerprint(persisted),
        _canonical_fingerprint(incoming),
        _canonical_differing_fields(left, right),
        persisted_content,
        incoming_content,
        persisted_tools,
        incoming_tools,
        _tool_details(left.get("tool_calls")),
        _tool_details(right.get("tool_calls")),
        _tool_argument_details(left.get("tool_calls")),
        _tool_argument_details(right.get("tool_calls")),
    )


def equivalent(a: dict[str, Any], b: dict[str, Any]) -> bool:
    left = canonical_message(a)
    right = canonical_message(b)
    if left == right:
        return True
    if left.get("role") == right.get("role") == "tool":
        tool_call_id = left.get("tool_call_id")
        return bool(tool_call_id and tool_call_id == right.get("tool_call_id") and right.get("content") == PRUNED_TOOL_RESULT)
    return False


def _is_compaction_summary(message: dict[str, Any]) -> bool:
    if message.get("summary") is True or str(message.get("mode", "")).lower() == "compaction":
        return True
    metadata = message.get("metadata")
    if isinstance(metadata, dict) and (metadata.get("compaction") is True or metadata.get("compaction_continue") is True):
        return True
    text = _text_content(message.get("content"))
    return bool(text and "The conversation history before this point was compacted into the following summary:" in text and "<summary>" in text and "</summary>" in text)


def _prefix_len(persisted: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> int:
    limit = min(len(persisted), len(incoming))
    index = 0
    while index < limit:
        if not equivalent(persisted[index], incoming[index]):
            _diagnose_difference(index, persisted[index], incoming[index])
            break
        index += 1
    return index


def _suffix_match(persisted: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> tuple[int, int] | None:
    p = len(persisted) - 1
    i = len(incoming) - 1
    while p >= 0 and i >= 0 and equivalent(persisted[p], incoming[i]):
        p -= 1
        i -= 1
    if p == len(persisted) - 1:
        return None
    return p + 1, i + 1


def _anchor_before_tail(persisted: list[dict[str, Any]], incoming: list[dict[str, Any]], prefix: int, *, max_scan: int = 128) -> tuple[int, int, int] | None:
    if prefix >= len(incoming) or prefix >= len(persisted):
        return None
    persisted_end = len(persisted)
    lower = max(prefix + 1, len(incoming) - max_scan)
    for incoming_end in range(len(incoming), lower - 1, -1):
        p = persisted_end - 1
        i = incoming_end - 1
        if p < prefix or i < prefix:
            continue
        while p >= prefix and i >= prefix and equivalent(persisted[p], incoming[i]):
            p -= 1
            i -= 1
        if i < incoming_end - 1:
            return p + 1, i + 1, incoming_end
    return None


def reconcile_projection(persisted: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> ReconciliationResult:
    """Reconcile a client history projection without mutating persisted history."""
    if not incoming:
        return ReconciliationResult("exact")
    if not persisted:
        return ReconciliationResult("append", 0)
    prefix = _prefix_len(persisted, incoming)
    if prefix == len(incoming):
        return ReconciliationResult("exact" if len(incoming) == len(persisted) else "truncate")
    if prefix == len(persisted):
        return ReconciliationResult("append", len(persisted))
    anchor = _anchor_before_tail(persisted, incoming, prefix)
    if anchor is not None:
        persisted_start, incoming_start, incoming_end = anchor
        gap = incoming[prefix:incoming_start]
        if persisted_start > prefix:
            if any(_is_compaction_summary(m) for m in gap):
                return ReconciliationResult("compacted", incoming_end)
            if incoming_start == 0:
                return ReconciliationResult("truncate" if incoming_end == len(incoming) else "truncate_append", incoming_end if incoming_end != len(incoming) else None)
    suffix = _suffix_match(persisted, incoming)
    if suffix is not None and suffix[1] == 0 and suffix[0] > 0:
        return ReconciliationResult("truncate")
    return ReconciliationResult("conflict")
