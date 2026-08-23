"""Deterministic relevance scoring for assembled candidates (master prompt §11.5).

Weights mirror RetrievalSettings so hybrid retrieval and assembly share one
configuration surface. Missing components contribute zero. Arithmetic is over
pre-rounded component values with a final 6-dp round: identical inputs always
produce identical scores, and ordering ties are broken by candidate key.
"""

from __future__ import annotations

from context_proxy.config import RetrievalSettings
from context_proxy.context.candidates import Candidate, CandidateSource

# Baseline relevance for non-retrieval candidates by tier: raw context that
# was explicitly provided (system/pinned/current/recent) outranks derived
# retrieval blocks once both compete for the same budget slice.
BASE_SCORE_BY_SOURCE: dict[CandidateSource, float] = {
    CandidateSource.SYSTEM: 1.0,
    CandidateSource.TOOL_DEFINITIONS: 1.0,
    CandidateSource.PINNED: 0.95,
    CandidateSource.CURRENT_REQUEST: 1.0,
    CandidateSource.RECENT_TURN: 0.9,
    CandidateSource.MEMORY: 0.0,
    CandidateSource.CHUNK: 0.0,
}


def relevance_score(candidate: Candidate, weights: RetrievalSettings) -> float:
    """Weighted sum of the candidate's score components; deterministic."""
    base = BASE_SCORE_BY_SOURCE.get(candidate.source, 0.0)
    if base > 0.0:
        return round(base, 6)
    c = candidate.components
    score = (
        weights.semantic_weight * c.get("semantic", 0.0)
        + weights.lexical_weight * c.get("lexical", 0.0)
        + weights.recency_weight * c.get("recency", 0.0)
        + weights.importance_weight * c.get("importance", 0.0)
        + weights.type_weight * c.get("type_priority", 0.0)
    )
    return round(score, 6)
