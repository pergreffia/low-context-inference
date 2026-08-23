"""Maximal Marginal Relevance selection (master prompt §11.6).

Greedy, deterministic MMR: after the first pick, each step selects

    argmax(lambda * relevance - (1 - lambda) * max_similarity_to_selected)

with ties broken by (-relevance, key) so equal-score candidates resolve by
stable ID and never by hash/row order. Similarity is token-set cosine over
lowercase word tokens: dependency-free and stable across runs.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable

_TOKEN = re.compile(r"[a-z0-9]+")


def _vector(text: str) -> dict[str, int]:
    return Counter(_TOKEN.findall(text))


def cosine_similarity(a: str, b: str) -> float:
    """Token-set cosine in [0, 1]; empty inputs are maximally dissimilar."""
    va, vb = _vector(a), _vector(b)
    if not va or not vb:
        return 0.0
    dot = sum(count * vb.get(token, 0) for token, count in va.items())
    if dot == 0:
        return 0.0
    norm_a = math.sqrt(sum(c * c for c in va.values()))
    norm_b = math.sqrt(sum(c * c for c in vb.values()))
    similarity = dot / (norm_a * norm_b)
    return max(0.0, min(1.0, similarity))


def mmr_select(
    scored: list[tuple[float, str]],
    similarity: Callable[[str, str], float],
    *,
    limit: int,
    lam: float = 0.7,
) -> list[str]:
    """Return up to `limit` keys selected for relevance + low redundancy.

    `scored` pairs (relevance_score, key); key must be unique. Deterministic:
    candidates are ordered by (-score, key) before greedy iteration and every
    argmax uses the same explicit tie-break.
    """
    if limit <= 0 or not scored:
        return []
    remaining = sorted(scored, key=lambda pair: (-pair[0], pair[1]))
    first_score, first_key = remaining.pop(0)
    selected: list[str] = [first_key]
    while remaining and len(selected) < limit:
        best_key: str | None = None
        best_rank: tuple[float, float, str] | None = None
        for score, key in remaining:
            max_sim = max((similarity(key, chosen) for chosen in selected), default=0.0)
            mmr = lam * score - (1 - lam) * max_sim
            rank = (-mmr, -score, key)
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_key = key
        assert best_key is not None  # remaining is non-empty
        remaining = [(s, k) for s, k in remaining if k != best_key]
        selected.append(best_key)
    return selected
