"""Typed schemas for the Memory Service (master prompt §10, §33).

No arbitrary dicts cross the internal API boundary.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class MemoryKind(StrEnum):
    DECISION = "decision"
    CONSTRAINT = "constraint"
    FACT = "fact"
    TASK = "task"
    BUG = "bug"
    IMPLEMENTATION = "implementation"
    TOOL_RESULT = "tool_result"
    EPISODE_SUMMARY = "episode_summary"
    CONVERSATION_SUMMARY = "conversation_summary"


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RESOLVED = "resolved"
    OBSOLETE = "obsolete"


# Static type priority for ranking (§19 type_weight term). Documented order:
# durable decisions/constraints outrank facts; raw-ish artifacts rank lowest.
TYPE_PRIORITY: dict[str, float] = {
    "decision": 0.9,
    "constraint": 0.85,
    "task": 0.7,
    "bug": 0.7,
    "episode_summary": 0.6,
    "conversation_summary": 0.6,
    "fact": 0.5,
    "implementation": 0.5,
    "chunk": 0.4,
    "tool_result": 0.3,
}


class MemoryCreate(BaseModel):
    kind: MemoryKind
    content: str = Field(min_length=1)
    conversation_id: str
    source_message_ids: list[str] = Field(default_factory=list)
    importance: float = Field(default=0.0, ge=0.0, le=1.0)
    supersedes: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SupersedeRequest(BaseModel):
    status: MemoryStatus = MemoryStatus.SUPERSEDED
    superseded_by: str | None = None


class RetrievedItem(BaseModel):
    item_type: Literal["chunk", "memory"]
    id: str
    conversation_id: str
    kind: str
    content: str
    score: float
    components: dict[str, float] = Field(default_factory=dict)
    source_message_ids: list[str] = Field(default_factory=list)


class RetrievalResponse(BaseModel):
    query: str
    items: list[RetrievedItem]
