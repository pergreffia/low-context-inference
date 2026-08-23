"""Candidate model for the Context Assembly Engine (master prompt §11.2).

Every context source is normalized into a Candidate carrying enough metadata
for deterministic fusion, deduplication, scoring, and packing:

- system messages;
- tool definitions;
- pinned context;
- the current request;
- recent raw interaction units;
- retrieved memories;
- retrieved historical chunks.

Candidates never mutate raw conversation state: they are projections used
exclusively to assemble the upstream request.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from context_proxy.context.tokens import TokenCounter
from context_proxy.memory.models import RetrievedItem


class CandidateSource(StrEnum):
    SYSTEM = "system"
    TOOL_DEFINITIONS = "tool_definitions"
    PINNED = "pinned"
    CURRENT_REQUEST = "current_request"
    RECENT_TURN = "recent_turn"
    MEMORY = "memory"
    CHUNK = "chunk"


# Packing tiers (master prompt §11.7, §11.8). Lower value = packed earlier =
# dropped later when budget is insufficient. Documented rationale:
#
#   1 system/tool definitions  — protocol requirements of the provider call;
#   2 pinned                   — explicit operator curating, cheap to keep;
#   3 current request          — the reason for this inference; losing it
#                                breaks semantics, so it only yields to 1–2;
#   4 recent turns             — freshest raw truth, keeps dialogue coherent;
#   5 memories/chunks          — derived compression; most disposable.
TIER_BY_SOURCE: dict[CandidateSource, int] = {
    CandidateSource.SYSTEM: 1,
    CandidateSource.TOOL_DEFINITIONS: 1,
    CandidateSource.PINNED: 2,
    CandidateSource.CURRENT_REQUEST: 3,
    CandidateSource.RECENT_TURN: 4,
    CandidateSource.MEMORY: 5,
    CandidateSource.CHUNK: 5,
}

_WHITESPACE = re.compile(r"\s+")


def canonical_text(value: str) -> str:
    """Whitespace-collapsed lowercase form used for duplicate detection."""
    return _WHITESPACE.sub(" ", value).strip().lower()


def message_texts(messages: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> str:
    """Role+content projection of a message list, order-preserving.

    Used both for candidate rendering and canonical comparison, so a chunk
    storing JSON-lines of persisted messages compares equal to the recent
    window holding the same interaction.
    """
    parts: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(
                str(part.get("text") or "")
                for part in content
                if isinstance(part, dict)
            )
        parts.append(f"{message.get('role')}: {content if isinstance(content, str) else ''}")
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            parts.append(
                f"tool_call: {function.get('name')} {function.get('arguments')}"
            )
    return "\n".join(parts)


def chunk_canonical_text(raw_content: str) -> str:
    """Canonical form of a stored chunk (JSON-lines of persisted messages)."""
    lines: list[dict[str, Any]] = []
    for line in raw_content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except ValueError:
            return canonical_text(raw_content)
        if isinstance(parsed, dict):
            lines.append(parsed)
    if not lines:
        return canonical_text(raw_content)
    return canonical_text(message_texts(lines))


@dataclass(frozen=True)
class Candidate:
    source: CandidateSource
    key: str  # stable identity; final tie-breaker for determinism
    tokens: int
    tier: int
    render: tuple[dict[str, Any], ...] = ()  # messages emitted into the request
    text: str = ""  # canonical text for dedup + lexical similarity
    components: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def conversation_id(self) -> str | None:
        return self.metadata.get("conversation_id")


@dataclass(frozen=True)
class DroppedCandidate:
    key: str
    source: CandidateSource
    reason: str


def candidate_from_retrieved(item: RetrievedItem, counter: TokenCounter) -> Candidate:
    """Project a RetrievedItem into a retrieval Candidate.

    The rendered message is one system block; token cost is computed from its
    exact rendered form so budget accounting matches what is sent upstream.
    Supersession enforcement lives upstream of this projection: the memory
    service only returns active records (PostgreSQL status filter) and the
    engine additionally drops any id listed as superseded by the caller.
    """
    label = f"[{item.item_type}:{item.kind} {item.id}]"
    content = f"{label} {item.content}"
    message = {"role": "system", "content": content}
    source = (
        CandidateSource.MEMORY if item.item_type == "memory" else CandidateSource.CHUNK
    )
    # dedup_text: comparable against raw-window canonical text. Chunks store
    # JSON-lines of persisted messages, so they are normalized through the
    # same role+content projection as recent units; memories compare directly.
    dedup_text = (
        chunk_canonical_text(item.content)
        if item.item_type == "chunk"
        else canonical_text(item.content)
    )
    return Candidate(
        source=source,
        key=item.id,
        tokens=counter.message(message),
        tier=TIER_BY_SOURCE[source],
        render=(message,),
        text=canonical_text(content),
        components=dict(item.components),
        metadata={
            "conversation_id": item.conversation_id,
            "kind": item.kind,
            "score": item.score,
            "dedup_text": dedup_text,
        },
    )
