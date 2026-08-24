"""Context Assembly Engine (master prompt §11, M4).

Dedicated domain component deciding what context reaches the model:

    candidate fusion -> deduplication -> supersession filtering
        -> relevance scoring -> MMR diversity -> budget allocation
        -> packing -> ContextPlan

Invariants:

- the final context never exceeds the usable token budget;
- the current request is preserved whenever it can fit; if the mandatory
  prefix (system + tools + pinned + current request) alone cannot fit, the
  build fails with ContextOverflowError and nothing is sent;
- interaction units stay atomic: a recent turn is never split;
- raw conversation state is never mutated — dedup/drop affect only the
  assembled request;
- conversation isolation: retrieval candidates from other conversations are
  rejected before scoring (§19);
- determinism: identical inputs always yield an identical ContextPlan.

Category precedence when the budget is insufficient (§11.7, §11.8):

    1. system messages + tool definitions   (provider-call requirements)
    2. pinned context                       (explicit operator curation)
    3. the current request                  (reason for this inference)
    4. recent raw turns, newest first       (freshest raw truth)
    5. retrieved memories/chunks, MMR order (derived, most disposable)

The current request is STRUCTURALLY distinct from history: `build` takes it
as a separate argument and models it as exactly one mandatory atomic
candidate that can never be re-selected as a recent turn. Recent turns pack
as ONE contiguous newest-side window; when an old unit no longer fits,
everything older is dropped too. Contiguous history beats maximal fill.
Under pressure recent raw context displaces retrieved blocks — the
documented trade-off, not an accident.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from context_proxy.config import AssemblySettings, RetrievalSettings
from context_proxy.context.candidates import (
    TIER_BY_SOURCE,
    Candidate,
    CandidateClassification,
    CandidateSource,
    DroppedCandidate,
    candidate_from_retrieved,
    canonical_text,
    content_texts,
    message_texts,
)
from context_proxy.context.mmr import cosine_similarity, mmr_select
from context_proxy.context.planner import Unit, segment_messages
from context_proxy.context.scoring import relevance_score
from context_proxy.context.tokens import TokenCounter
from context_proxy.memory.models import RetrievedItem

logger = logging.getLogger(__name__)

_RETRIEVAL_SOURCES = frozenset({CandidateSource.MEMORY, CandidateSource.CHUNK})

CURRENT_REQUEST_KEY = "current_request"


def separate_current_request(
    messages: list[dict[str, Any]],
    counter: TokenCounter | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split an inbound payload into (history, current_request).

    The current request is the logical interaction started by the LAST user
    message (including attached tool calls/results) — robust even when a
    trailing system message follows it. Trailing system messages remain
    system context in history; they never leak into the request unit.
    """
    last_user_index = -1
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            last_user_index = index
            break
    if last_user_index == -1:
        # No user interaction at all: everything is history.
        return list(messages), []

    # The request spans from the last user message up to the payload end,
    # EXCEPT a maximal suffix of bare system messages, which remain system
    # context in history.
    end = len(messages)
    trailing_systems: list[dict[str, Any]] = []
    while end - 1 > last_user_index and messages[end - 1].get("role") == "system":
        trailing_systems.insert(0, messages[end - 1])
        end -= 1

    history = [*messages[:last_user_index], *trailing_systems]
    return history, list(messages[last_user_index:end])


class ContextOverflowError(Exception):
    """The mandatory context cannot fit the usable budget (master §9)."""

    def __init__(self, required_tokens: int, usable_budget: int):
        self.required_tokens = required_tokens
        self.usable_budget = usable_budget
        super().__init__(
            f"context requires {required_tokens} tokens, "
            f"usable budget is {usable_budget}"
        )


@dataclass(frozen=True)
class SelectedItem:
    key: str
    source: CandidateSource
    tokens: int
    score: float


@dataclass
class ContextPlan:
    """Structured assembly result with full diagnostics (§11.10)."""

    messages: list[dict[str, Any]]
    selected_items: list[SelectedItem]
    dropped_items: list[DroppedCandidate]
    token_estimate: int
    tools_tokens: int
    budget: dict[str, int]
    rationale: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def within_budget(self) -> bool:
        return self.token_estimate <= self.budget["usable"]

    def debug_view(self) -> dict[str, Any]:
        """Internal representation for the preview endpoint; no raw content."""
        return {
            "selected": [
                {
                    "key": item.key,
                    "source": item.source.value,
                    "tokens": item.tokens,
                    "score": item.score,
                }
                for item in self.selected_items
            ],
            "dropped": [
                {"key": d.key, "source": d.source.value, "reason": d.reason}
                for d in self.dropped_items
            ],
            "token_estimate": self.token_estimate,
            "tools_tokens": self.tools_tokens,
            "budget": dict(self.budget),
            "rationale": self.rationale,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass
class _Allocation:
    """Intermediate packing state before rendering."""

    ordered: list[Candidate] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)


class ContextAssemblyEngine:
    """Synchronous, deterministic assembler; retrieval happens before build."""

    def __init__(
        self,
        *,
        usable_budget: int,
        settings: AssemblySettings,
        retrieval_settings: RetrievalSettings,
        counter: TokenCounter | None = None,
        similarity: Any = cosine_similarity,
    ):
        if usable_budget <= 0:
            raise ValueError("usable_budget must be positive")
        self._usable = usable_budget
        self._settings = settings
        self._retrieval = retrieval_settings
        self._counter = counter or TokenCounter()
        self._similarity = similarity

    # ------------------------------------------------------------------ build

    def build(
        self,
        *,
        history: list[dict[str, Any]],
        current_request: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        pinned: list[dict[str, Any]] | None = None,
        retrieved: list[RetrievedItem] | None = None,
        superseded_memory_ids: set[str] | None = None,
        conversation_id: str | None = None,
    ) -> ContextPlan:
        """Assemble the final upstream message list within the usable budget.

        `history` and `current_request` are structurally disjoint: the engine
        models the request as exactly one mandatory atomic candidate and
        never re-derives it from the history tail.
        """
        fused = self._fuse_candidates(
            history, current_request, tools, pinned, retrieved or []
        )

        dropped: dict[str, DroppedCandidate] = {}
        fused = self._filter_superseded(fused, superseded_memory_ids or set(), dropped)
        fused = self._filter_foreign(fused, conversation_id, dropped)
        fused, duplicates = self._deduplicate(fused)
        dropped.update(duplicates)

        scored = {c.key: relevance_score(c, self._retrieval) for c in fused}
        allocation = self._allocate(fused, scored, dropped)
        plan = self._make_plan(allocation, scored, dropped)
        logger.info(
            "context_assembled",
            extra={
                "conversation_id": conversation_id,
                "selected": len(plan.selected_items),
                "dropped": len(plan.dropped_items),
                "token_estimate": plan.token_estimate,
                "usable_budget": self._usable,
                "rationale": plan.rationale,
            },
        )
        return plan

    # ----------------------------------------------------------------- fusion

    def _fuse_candidates(
        self,
        history: list[dict[str, Any]],
        current_request: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        pinned: list[dict[str, Any]] | None,
        retrieved: list[RetrievedItem],
    ) -> list[Candidate]:
        candidates: list[Candidate] = []

        units: list[Unit] = segment_messages(history, self._counter)
        for index, unit in enumerate(units):
            source = (
                CandidateSource.SYSTEM if unit.is_system else CandidateSource.RECENT_TURN
            )
            candidates.append(
                Candidate(
                    source=source,
                    key=f"unit:{index}",
                    tokens=unit.tokens,
                    tier=TIER_BY_SOURCE[source],
                    render=tuple(unit.messages),
                    text=canonical_text(message_texts(list(unit.messages))),
                    metadata={
                        "position": index,
                        # Bare per-part texts (M6-aware: images contribute
                        # fingerprint tokens) let memory records restating a
                        # turn be recognized as duplicates.
                        "content_texts": [
                            text
                            for m in unit.messages
                            for text in content_texts(m)
                        ],
                    },
                )
            )

        # The current request is exactly one atomic mandatory candidate.
        if current_request:
            candidates.append(
                Candidate(
                    source=CandidateSource.CURRENT_REQUEST,
                    key=CURRENT_REQUEST_KEY,
                    tokens=self._counter.messages(current_request),
                    tier=TIER_BY_SOURCE[CandidateSource.CURRENT_REQUEST],
                    render=tuple(current_request),
                    text=canonical_text(message_texts(current_request)),
                    metadata={"position": len(units)},
                )
            )

        if tools:
            candidates.append(
                Candidate(
                    source=CandidateSource.TOOL_DEFINITIONS,
                    key="tool_definitions",
                    tokens=self._counter.tools(tools),
                    tier=TIER_BY_SOURCE[CandidateSource.TOOL_DEFINITIONS],
                    metadata={"is_tools": True},
                )
            )

        for index, message in enumerate(pinned or []):
            candidates.append(
                Candidate(
                    source=CandidateSource.PINNED,
                    key=f"pinned:{index}",
                    tokens=self._counter.message(message),
                    tier=TIER_BY_SOURCE[CandidateSource.PINNED],
                    render=(message,),
                    text=canonical_text(message_texts([message])),
                    metadata={"position": index},
                )
            )

        for item in retrieved:
            candidates.append(candidate_from_retrieved(item, self._counter))

        return candidates

    # -------------------------------------------------------------- filtering

    @staticmethod
    def _filter_superseded(
        candidates: list[Candidate],
        superseded_memory_ids: set[str],
        dropped: dict[str, DroppedCandidate],
    ) -> list[Candidate]:
        kept = []
        for candidate in candidates:
            if (
                candidate.source == CandidateSource.MEMORY
                and candidate.key in superseded_memory_ids
            ):
                dropped[candidate.key] = DroppedCandidate(
                    candidate.key, candidate.source, "superseded"
                )
                continue
            kept.append(candidate)
        return kept

    @staticmethod
    def _filter_foreign(
        candidates: list[Candidate],
        conversation_id: str | None,
        dropped: dict[str, DroppedCandidate],
    ) -> list[Candidate]:
        """Reject retrieval blocks from any other conversation (§19)."""
        if conversation_id is None:
            return candidates
        kept = []
        for candidate in candidates:
            foreign = (
                candidate.source in _RETRIEVAL_SOURCES
                and candidate.conversation_id != conversation_id
            )
            if foreign:
                dropped[candidate.key] = DroppedCandidate(
                    candidate.key, candidate.source, "foreign_conversation"
                )
                continue
            kept.append(candidate)
        return kept

    @staticmethod
    def _deduplicate(
        candidates: list[Candidate],
    ) -> tuple[list[Candidate], dict[str, DroppedCandidate]]:
        """Cross-source duplicate removal over the assembled set only.

        Raw inbound units are authoritative and never dropped as duplicates.
        A retrieval block equal to already-present raw text — or to another
        retrieval block — is removed. Exact-match only: partial overlap is
        left to MMR redundancy pressure (documented limitation).
        """
        seen_raw: set[str] = set()
        seen_retrieval: set[str] = set()
        duplicates: dict[str, DroppedCandidate] = {}
        kept: list[Candidate] = []
        for candidate in candidates:
            if candidate.tier <= TIER_BY_SOURCE[CandidateSource.RECENT_TURN]:
                if candidate.text:
                    seen_raw.add(candidate.text)
                seen_raw.update(candidate.metadata.get("content_texts") or [])
                kept.append(candidate)
                continue
            dedup_text: str | None = candidate.metadata.get("dedup_text")
            if not dedup_text:
                kept.append(candidate)
                continue
            if dedup_text in seen_raw or dedup_text in seen_retrieval:
                duplicates[candidate.key] = DroppedCandidate(
                    candidate.key, candidate.source, "duplicate"
                )
                continue
            seen_retrieval.add(dedup_text)
            kept.append(candidate)
        return kept, duplicates

    # -------------------------------------------------------- budget + packing

    def _allocate(
        self,
        candidates: list[Candidate],
        scores: dict[str, float],
        dropped: dict[str, DroppedCandidate],
    ) -> _Allocation:
        """Deterministic tiered allocation (see module docstring precedence)."""
        mandatory: list[Candidate] = []
        recent: list[Candidate] = []
        retrieval: list[Candidate] = []
        tools_candidate: Candidate | None = None
        for candidate in candidates:
            if candidate.source == CandidateSource.TOOL_DEFINITIONS:
                tools_candidate = candidate
            elif candidate.tier <= TIER_BY_SOURCE[CandidateSource.PINNED]:
                mandatory.append(candidate)
            elif candidate.source == CandidateSource.CURRENT_REQUEST:
                mandatory.append(candidate)
            elif candidate.source == CandidateSource.RECENT_TURN:
                recent.append(candidate)
            else:
                retrieval.append(candidate)

        tools_tokens = tools_candidate.tokens if tools_candidate else 0
        mandatory_tokens = sum(c.tokens for c in mandatory)
        if mandatory_tokens + tools_tokens > self._usable:
            raise ContextOverflowError(mandatory_tokens + tools_tokens, self._usable)

        remaining = self._usable - mandatory_tokens - tools_tokens

        # Tier 4: one contiguous newest-side window of whole units.
        kept_recent: list[Candidate] = []
        recent_tokens = 0
        for candidate in reversed(recent):
            if recent_tokens + candidate.tokens > remaining:
                break
            kept_recent.append(candidate)
            recent_tokens += candidate.tokens
        kept_recent.reverse()  # restore oldest -> newest order
        remaining -= recent_tokens
        kept_keys = {c.key for c in kept_recent}
        for candidate in recent:
            if candidate.key not in kept_keys:
                dropped[candidate.key] = DroppedCandidate(
                    candidate.key, candidate.source, "budget"
                )

        # Tier 5: MMR diversity, then whole-item packing under the cap.
        picked_retrieval: list[Candidate] = []
        retrieved_tokens = 0
        if retrieval and remaining > 0 and self._settings.max_retrieved_items > 0:
            ranked = mmr_select(
                [(scores[c.key], c.key) for c in retrieval],
                self._similarity,
                limit=min(self._settings.max_retrieved_items, len(retrieval)),
                lam=self._settings.mmr_lambda,
            )
            by_key = {c.key: c for c in retrieval}
            retrieved_cap = min(remaining, self._settings.retrieved_budget_tokens)
            for key in ranked:
                candidate = by_key[key]
                if retrieved_tokens + candidate.tokens > retrieved_cap:
                    dropped[key] = DroppedCandidate(key, candidate.source, "budget")
                    continue
                picked_retrieval.append(candidate)
                retrieved_tokens += candidate.tokens
        picked_keys = {c.key for c in picked_retrieval}
        for candidate in retrieval:
            if candidate.key not in picked_keys and candidate.key not in dropped:
                dropped[candidate.key] = DroppedCandidate(
                    candidate.key, candidate.source, "not_selected"
                )

        # Render order: system+pinned (position order), retrieval blocks,
        # recent window oldest->newest, current request last.
        head = sorted(
            (c for c in mandatory),
            key=lambda c: (c.tier, c.metadata.get("position", 0), c.key),
        )
        tail = next(
            (c for c in head if c.source == CandidateSource.CURRENT_REQUEST), None
        )
        head_no_request = [c for c in head if c is not tail]
        ordered = (
            head_no_request
            + picked_retrieval
            + kept_recent
            + ([tail] if tail is not None else [])
        )
        return _Allocation(
            ordered=ordered,
            stats={
                "recent_tokens": recent_tokens,
                "retrieved_tokens": retrieved_tokens,
                "mandatory_tokens": mandatory_tokens,
                "tools_tokens": tools_tokens,
            },
        )

    # ------------------------------------------------------------------- plan

    def _make_plan(
        self,
        allocation: _Allocation,
        scores: dict[str, float],
        dropped: dict[str, DroppedCandidate],
    ) -> ContextPlan:
        stats = allocation.stats
        tools_tokens = stats["tools_tokens"]
        messages: list[dict[str, Any]] = []
        token_estimate = 0
        selected_items: list[SelectedItem] = []
        for candidate in allocation.ordered:
            if (
                candidate.classification == CandidateClassification.DERIVED_CONTEXT
                and candidate.render
                and candidate.render[0].get("role") == "system"
                and "<retrieved_context>" not in str(
                    candidate.render[0].get("content") or ""
                )
            ):
                # Structural guard against future regressions: derived data
                # must never pose as a bare trusted system instruction.
                raise RuntimeError(
                    f"derived candidate {candidate.key!r} rendered as an "
                    "undelimited system instruction"
                )
            messages.extend(candidate.render)
            token_estimate += candidate.tokens
            if candidate.source != CandidateSource.TOOL_DEFINITIONS:
                selected_items.append(
                    SelectedItem(
                        key=candidate.key,
                        source=candidate.source,
                        tokens=candidate.tokens,
                        score=scores.get(candidate.key, 0.0),
                    )
                )
        token_estimate += tools_tokens

        if token_estimate > self._usable:  # structural invariant, never expected
            raise ContextOverflowError(token_estimate, self._usable)

        rationale = (
            f"selected {len(selected_items)} items "
            f"(recent={stats['recent_tokens']}tok, "
            f"retrieved={stats['retrieved_tokens']}tok, "
            f"mandatory={stats['mandatory_tokens']}tok, "
            f"tools={tools_tokens}tok); dropped {len(dropped)}"
        )
        diagnostics = {
            **stats,
            "mmr_lambda": self._settings.mmr_lambda,
            "max_retrieved_items": self._settings.max_retrieved_items,
            "retrieved_budget_tokens": self._settings.retrieved_budget_tokens,
            "weights": {
                "semantic": self._retrieval.semantic_weight,
                "lexical": self._retrieval.lexical_weight,
                "recency": self._retrieval.recency_weight,
                "importance": self._retrieval.importance_weight,
                "type": self._retrieval.type_weight,
            },
            "drop_reasons": sorted(
                {f"{d.reason}" for d in dropped.values()}
            ),
        }
        return ContextPlan(
            messages=messages,
            selected_items=selected_items,
            dropped_items=[dropped[k] for k in sorted(dropped)],
            token_estimate=token_estimate,
            tools_tokens=tools_tokens,
            budget={"usable": self._usable},
            rationale=rationale,
            diagnostics=diagnostics,
        )
