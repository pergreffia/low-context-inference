"""Projection-aware conversation history reconciliation.

The durable conversation is the source of truth. Client-provided history is a
model-context projection and may legitimately normalize reasoning/content or
compact/truncate older turns.

The algorithm is deliberately conservative: exact/canonical matches are
preferred; truncation requires a persisted suffix anchor; compaction requires
both prefix/suffix anchors plus an explicit or strongly recognizable summary
message. Unanchored rewrites remain conflicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReconciliationResult:
    """Result of reconciling an incoming client projection."""

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
    """Return the representation used for semantic history comparison.

    Reasoning is intentionally excluded because provider/client pipelines may
    normalize, omit, or reconstruct it. Text-only content scalar/parts forms
    are normalized to one string. All other fields remain significant.
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
    """Recognize explicit or OpenCode-compatible compaction summaries.

    Explicit metadata is preferred. OpenCode's current compaction prompt emits
    a stable Markdown structure beginning with Objective/Important Details and
    containing Work State/Next Move/Relevant Files. This fallback is intentionally
    strict so arbitrary assistant text cannot become a compaction escape hatch.
    """
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


def _suffix_match(
    persisted: list[dict[str, Any]], incoming: list[dict[str, Any]],
) -> tuple[int, int] | None:
    """Return (persisted_start, incoming_start) for the longest suffix anchor."""
    max_len = min(len(persisted), len(incoming))
    for length in range(max_len, 0, -1):
        p_start = len(persisted) - length
        i_start = len(incoming) - length
        if all(equivalent(persisted[p], incoming[i]) for p, i in zip(
            range(p_start, len(persisted)), range(i_start, len(incoming))
        )):
            return p_start, i_start
    return None


def reconcile_projection(
    persisted: list[dict[str, Any]], incoming: list[dict[str, Any]]
) -> ReconciliationResult:
    """Reconcile client projection without mutating persisted history.

    Modes:
      exact     - complete canonical replay;
      append    - canonical persisted prefix followed by new messages;
      truncate  - incoming is a persisted suffix;
      compacted - prefix and suffix anchors survive a summarized gap, with the
                  incoming gap containing an explicit/recognized summary;
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

    # A compacted projection may replace an older persisted region while
    # retaining a recent suffix. Require a real prefix anchor, a real suffix
    # anchor, and a recognized summary in the replaced incoming region.
    suffix = _suffix_match(persisted[prefix:], incoming[prefix:])
    if suffix is not None:
        persisted_start, incoming_start = suffix
        persisted_start += prefix
        incoming_start += prefix
        if persisted_start > prefix and incoming_start > prefix:
            gap = incoming[prefix:incoming_start]
            if any(_is_compaction_summary(message) for message in gap):
                return ReconciliationResult("compacted", len(incoming))

    # A pure tail projection is safe even when there is no common prefix.
    suffix = _suffix_match(persisted, incoming)
    if suffix is not None and suffix[0] > 0:
        return ReconciliationResult("truncate")

    return ReconciliationResult("conflict")
