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

import hashlib
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


def _sha256_hex(value: str) -> str:
    """Full SHA-256 hex digest — collision-resistant identity component."""
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _part_fingerprint(part: dict[str, Any]) -> str:
    """Deterministic fingerprint of an opaque multimodal part (M6 review §3).

    Canonical JSON (sorted keys, tight separators) makes the hash independent
    of JSON key order; the raw payload never enters logs or canonical text.
    """
    canonical_json = json.dumps(
        part, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return _sha256_hex(canonical_json)


def _image_fingerprint(url: str) -> str:
    """Full-length stable fingerprint of an image source URL."""
    return _sha256_hex(url)


def message_texts(messages: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> str:
    """Role+content projection of a message list, order-preserving.

    Used both for candidate rendering and canonical comparison, so a chunk
    storing JSON-lines of persisted messages compares equal to the recent
    window holding the same interaction.

    Multimodal parts (M6 §13.1): text parts contribute their text; image and
    unknown parts contribute fingerprints — same text but different payloads
    must NOT collapse into one identity.

    Tool calls (M6 review §1) are ALWAYS rendered after the content,
    whatever its shape: name + arguments form the semantic identity (ids are
    transport noise).
    """
    parts: list[str] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if isinstance(content, str):
            rendered_content = content
        elif isinstance(content, list):
            rendered_parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                kind = part.get("type")
                if kind == "text":
                    rendered_parts.append(str(part.get("text") or ""))
                elif kind == "image_url":
                    image_url = part.get("image_url") or {}
                    url = (
                        image_url.get("url", "")
                        if isinstance(image_url, dict)
                        else str(image_url)
                    )
                    rendered_parts.append(f"[image:{_image_fingerprint(str(url))}]")
                else:
                    rendered_parts.append(
                        f"[{kind or 'unknown'}:{_part_fingerprint(part)}]"
                    )
            rendered_content = " ".join(rendered_parts)
        else:
            rendered_content = ""
        parts.append(f"{role}: {rendered_content}")
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            parts.append(
                f"tool_call: {function.get('name')} {function.get('arguments')}"
            )
    return "\n".join(parts)


def content_texts(message: dict[str, Any]) -> list[str]:
    """Canonical per-part texts of ONE message (dedup keys, M6-aware).

    Text parts contribute their text; image parts their full fingerprint;
    unknown parts a type+fingerprint token so payload changes change identity.
    """
    content = message.get("content")
    if isinstance(content, str):
        text_value = canonical_text(content)
        return [text_value] if text_value else []
    if isinstance(content, list):
        out: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            kind = part.get("type")
            if kind == "text":
                text_value = canonical_text(str(part.get("text") or ""))
                if text_value:
                    out.append(text_value)
            elif kind == "image_url":
                image_url = part.get("image_url") or {}
                url = (
                    image_url.get("url", "")
                    if isinstance(image_url, dict)
                    else str(image_url)
                )
                out.append(f"[image:{_image_fingerprint(str(url))}]")
            else:
                out.append(f"[{kind or 'unknown'}:{_part_fingerprint(part)}]")
        return out
    return []


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


class CandidateClassification(StrEnum):
    """Trust boundary for rendered context (M0–M6 review P1).

    Retrieved memories/chunks derive from user-controlled conversation
    content: they must NEVER be promoted to trusted system instructions.
    They are delimited derived data instead.
    """

    TRUSTED_INSTRUCTION = "trusted_instruction"
    DERIVED_CONTEXT = "derived_context"
    UNTRUSTED_USER_DATA = "untrusted_user_data"


# Static mapping: prevents future candidate builders from accidentally
# promoting derived data to system-level instructions.
CLASSIFICATION_BY_SOURCE: dict[CandidateSource, CandidateClassification] = {
    CandidateSource.SYSTEM: CandidateClassification.TRUSTED_INSTRUCTION,
    CandidateSource.TOOL_DEFINITIONS: CandidateClassification.TRUSTED_INSTRUCTION,
    CandidateSource.PINNED: CandidateClassification.TRUSTED_INSTRUCTION,
    CandidateSource.CURRENT_REQUEST: CandidateClassification.UNTRUSTED_USER_DATA,
    CandidateSource.RECENT_TURN: CandidateClassification.UNTRUSTED_USER_DATA,
    CandidateSource.MEMORY: CandidateClassification.DERIVED_CONTEXT,
    CandidateSource.CHUNK: CandidateClassification.DERIVED_CONTEXT,
}


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

    @property
    def classification(self) -> CandidateClassification:
        explicit = self.metadata.get("classification")
        if isinstance(explicit, CandidateClassification):
            return explicit
        return CLASSIFICATION_BY_SOURCE[self.source]


@dataclass(frozen=True)
class DroppedCandidate:
    key: str
    source: CandidateSource
    reason: str


def candidate_from_retrieved(item: RetrievedItem, counter: TokenCounter) -> Candidate:
    """Project a RetrievedItem into a retrieval Candidate.

    The rendered message is one untrusted user-role block carrying a
    provenance header (`[retrieved …]`); token cost is computed from its
    exact rendered form so budget accounting matches what is sent upstream.
    Supersession enforcement lives upstream of this projection: the memory
    service only returns active records (PostgreSQL status filter) and the
    engine additionally drops any id listed as superseded by the caller.

    Identity (M4 review §6): chunks carry their authoritative message span
    (conversation_id + start_seq/end_seq); memories carry a stable content
    fingerprint. Identical content is NOT identical authoritative history —
    raw interactions are only ever matched through their stored spans.
    Trust boundary (final review P1): retrieved blocks are DERIVED, untrusted
    data derived from user-controlled content. They render as **user-role**
    messages with a provenance header line: every provider treats user
    content as untrusted by definition, so retrieved text can never be
    mistaken for a trusted system instruction regardless of its content
    (delimiters alone are not a security boundary and are not used as one).
    Raw retrieved text is preserved verbatim.

    Ordering invariant (post-024d014 review): retrieved blocks always pack
    BEFORE the current request, so the final user turn the model sees is the
    actual live request — never a retrieved block.
    """
    label = f"[retrieved {item.item_type}:{item.kind} id={item.id}]"
    content = f"{label}\n{item.content}"
    message = {"role": "user", "content": content}
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
            "fingerprint": content_fingerprint(dedup_text),
            "start_seq": item.start_seq,
            "end_seq": item.end_seq,
        },
    )


def content_fingerprint(text: str) -> str:
    """Stable fingerprint of canonical content for derived candidates."""
    digest = hashlib.sha256(canonical_text(text).encode("utf-8")).hexdigest()
    return digest[:16]
