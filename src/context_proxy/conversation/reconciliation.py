"""Projection-aware conversation history reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
        if not isinstance(part, dict):
            return None
        if part.get("type") != "text" or not isinstance(part.get("text"), str):
            return None
        parts.append(part["text"])
    return "".join(parts)


def canonical_message(message: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize only known representation-level differences.

    Reasoning is intentionally not part of conversation identity: providers and
    clients may normalize, omit, or reconstruct it. Text scalar/content-parts
    are canonicalized to the same text. Other fields remain significant.
    """
    result = dict(message)
    for key in ("reasoning_content", "reasoning", "reasoning_text"):
        result.pop(key, None)
    text = _text_content(result.get("content"))
    if text is not None:
        result["content"] = text
    return result


def equivalent(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return canonical_message(a) == canonical_message(b)


def _is_compaction_summary(message: dict[str, Any]) -> bool:
    """Recognize explicit or OpenCode-compatible compaction summaries."""
    if message.get("summary") is True:
        return True
    if str(message.get("mode", "")).lower() == "compaction":
        return True
    metadata = message.get("metadata")
    if isinstance(metadata, dict):
        if metadata.get("compaction") is True or metadata.get("compaction_continue") is True:
            return True

    text = _text_content(message.get("content"))
    if not text:
        return False
    # OpenCode's current compaction summary template uses these five stable
    # headings. Keep recognition strict so arbitrary assistant text cannot
    # become a compaction escape hatch.
    required = (
        "## Objective",
        "## Important Details",
        "## Work State",
        "## Next Move",
        "## Relevant Files",
    )
    return all(marker in text for marker in required)


def _prefix_len(persisted: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> int:
    limit = min(len(persisted), len(incoming))
    index = 0
    while index < limit and equivalent(persisted[index], incoming[index]):
        index += 1
    return index


def _suffix_match(persisted: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> tuple[int, int] | None:
    """Return the longest suffix shared by both sequences in O(n)."""
    p = len(persisted) - 1
    i = len(incoming) - 1
    while p >= 0 and i >= 0 and equivalent(persisted[p], incoming[i]):
        p -= 1
        i -= 1
    if p == len(persisted) - 1:
        return None
    return p + 1, i + 1


def _anchor_before_tail(
    persisted: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    prefix: int,
    *,
    max_scan: int = 128,
) -> tuple[int, int, int] | None:
    """Find a persisted suffix anchor occurring before a new incoming tail.

    Returns (persisted_start, incoming_start, incoming_end). The bounded scan
    is only used after prefix mismatch, so the normal replay/append path stays
    linear. A compaction summary is required by the caller before accepting the
    anchor, preventing arbitrary rewrites from being treated as projections.
    """
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


def reconcile_projection(
    persisted: list[dict[str, Any]], incoming: list[dict[str, Any]]
) -> ReconciliationResult:
    """Reconcile a client projection without mutating persisted history.

    Modes:
      exact     - complete canonical replay;
      append    - persisted history is a canonical prefix;
      truncate  - incoming is a persisted suffix;
      compacted - a recognized summary replaces a persisted region and a
                  persisted suffix remains anchored, optionally followed by new
                  incoming messages;
      conflict  - no safe continuity proof exists.
    """
    if not incoming:
        return ReconciliationResult("exact")
    if not persisted:
        return ReconciliationResult("append", 0)

    prefix = _prefix_len(persisted, incoming)
    if prefix == len(incoming):
        if len(incoming) == len(persisted):
            return ReconciliationResult("exact")
        return ReconciliationResult("truncate")
    if prefix == len(persisted):
        return ReconciliationResult("append", len(persisted))

    # First try a compacted projection. A prefix anchor plus a suffix anchor
    # and an explicit/strongly recognizable summary are required.
    anchor = _anchor_before_tail(persisted, incoming, prefix)
    if anchor is not None:
        persisted_start, incoming_start, incoming_end = anchor
        gap = incoming[prefix:incoming_start]
        if persisted_start > prefix and any(_is_compaction_summary(m) for m in gap):
            return ReconciliationResult("compacted", incoming_end)

    # A pure tail projection is safe when the incoming history ends at a
    # persisted suffix. No summary is needed because no rewrite is implied.
    suffix = _suffix_match(persisted, incoming)
    if suffix is not None and suffix[0] > 0:
        return ReconciliationResult("truncate")

    return ReconciliationResult("conflict")
