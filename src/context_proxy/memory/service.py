"""Memory Service (master prompt §9–§12, §17, §28).

Turn-based conversation chunking, memory records with supersession, and
hybrid retrieval (Qdrant semantic leg + PostgreSQL full-text lexical leg,
fused with configurable weights). Qdrant is a derived index: PostgreSQL holds
the source of truth and everything is rebuildable from it.

Degradation: when the embedding endpoint or Qdrant is unavailable, retrieval
falls back to lexical-only; indexing skips the vector leg and logs a warning.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import asyncpg

from context_proxy.config import RetrievalSettings
from context_proxy.context.tokens import TokenCounter
from context_proxy.memory.embeddings import OpenAICompatibleEmbeddingProvider
from context_proxy.memory.models import TYPE_PRIORITY, MemoryCreate, MemoryStatus, RetrievedItem
from context_proxy.memory.qdrant import QdrantVectorStore

logger = logging.getLogger(__name__)


def now_utc() -> datetime:
    return datetime.now(UTC)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


class MemoryService:
    def __init__(
        self,
        pool: asyncpg.Pool,
        embedder: OpenAICompatibleEmbeddingProvider,
        vector_store: QdrantVectorStore,
        retrieval_settings: RetrievalSettings | None = None,
        max_embed_chars: int = 8000,
    ):
        self._pool = pool
        self._embedder = embedder
        self._qdrant = vector_store
        self._retrieval = retrieval_settings or RetrievalSettings()
        self._max_embed_chars = max_embed_chars
        self._counter = TokenCounter()

    # ------------------------------------------------------------------ chunks

    async def index_completed_turns(self, conversation_id: str) -> int:
        """Chunk every completed turn not yet indexed; returns chunks created.

        A turn = user message plus the following non-user messages (tool calls
        stay attached to their results). The trailing turn is never chunked —
        it is still the live recent interaction. System messages are excluded:
        they remain permanent raw context (priority 1).
        """
        rows = await self._pool.fetch(
            """
            SELECT id, seq, role, content FROM messages
            WHERE conversation_id = $1::uuid AND role <> 'system'
            ORDER BY seq
            """,
            conversation_id,
        )
        if not rows:
            return 0
        last_seq = rows[-1]["seq"]

        units: list[tuple[int, list[asyncpg.Record]]] = []
        current: list[asyncpg.Record] = []
        for row in rows:
            if row["role"] == "user" and current:
                units.append((current[0]["seq"], current))
                current = []
            current.append(row)
        if current:
            units.append((current[0]["seq"], current))

        existing = {
            r["start_seq"]
            for r in await self._pool.fetch(
                "SELECT start_seq FROM conversation_chunks WHERE conversation_id = $1::uuid",
                conversation_id,
            )
        }

        created = 0
        for start_seq, unit_rows in units:
            if start_seq in existing:
                continue  # idempotent replay
            if unit_rows[-1]["seq"] >= last_seq:
                continue  # trailing turn = live recent interaction, keep raw
            created += await self._index_unit(conversation_id, start_seq, unit_rows)
        return created

    async def _index_unit(
        self, conversation_id: str, start_seq: int, unit_rows: list[asyncpg.Record]
    ) -> int:
        messages = [json.loads(r["content"]) for r in unit_rows]
        raw_content = "\n".join(json.dumps(m, ensure_ascii=False) for m in messages)
        token_count = self._counter.messages(messages)
        message_ids = [str(r["id"]) for r in unit_rows]

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT id FROM conversations WHERE id = $1::uuid FOR UPDATE",
                    conversation_id,
                )
                chunk_id = await conn.fetchval(
                    """
                    INSERT INTO conversation_chunks
                        (conversation_id, start_seq, message_ids, raw_content, token_count)
                    VALUES ($1::uuid, $2, $3::uuid[], $4, $5)
                    ON CONFLICT (conversation_id, start_seq) DO NOTHING
                    RETURNING id
                    """,
                    conversation_id,
                    start_seq,
                    message_ids,
                    raw_content,
                    token_count,
                )
        if chunk_id is None:
            return 0  # raced or already present: idempotent no-op
        await self._embed_and_upsert(
            point_id=str(chunk_id),
            text=raw_content,
            payload={
                "conversation_id": conversation_id,
                "kind": "chunk",
                "chunk_id": str(chunk_id),
                "start_seq": start_seq,
            },
        )
        logger.info(
            "chunk_indexed",
            extra={"conversation_id": conversation_id, "chunk_id": str(chunk_id)},
        )
        return 1

    # ---------------------------------------------------------------- memories

    async def create_memory(self, spec: MemoryCreate) -> str:
        """Insert an active memory; optionally supersede another record.

        Superseded/obsolete records are NEVER deleted — only excluded from
        active retrieval (master prompt §10, §12).
        """
        memory_id = str(uuid.uuid4())
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Memory may reference a conversation that has no rows yet
                # (e.g. distilled knowledge); bootstrap the parent row.
                await conn.execute(
                    "INSERT INTO conversations (id) VALUES ($1::uuid) ON CONFLICT (id) DO NOTHING",
                    spec.conversation_id,
                )
                await conn.fetchval(
                    "SELECT id FROM conversations WHERE id = $1::uuid FOR UPDATE",
                    spec.conversation_id,
                )
                if spec.supersedes:
                    # Pre-check inside the tx: an unknown target is a client
                    # error (404), not a database integrity accident.
                    target = await conn.fetchval(
                        "SELECT id FROM memory_records WHERE id = $1::uuid FOR UPDATE",
                        uuid.UUID(spec.supersedes),
                    )
                    if target is None:
                        raise ValueError(f"supersedes target {spec.supersedes} not found")
                await conn.execute(
                    """
                    INSERT INTO memory_records
                        (id, conversation_id, kind, content, source_message_ids,
                         importance, status, supersedes, metadata)
                    VALUES ($1::uuid, $2::uuid, $3, $4, $5::uuid[], $6, 'active',
                            $7::uuid, $8::jsonb)
                    """,
                    memory_id,
                    spec.conversation_id,
                    spec.kind.value,
                    spec.content,
                    spec.source_message_ids,
                    spec.importance,
                    uuid.UUID(spec.supersedes) if spec.supersedes else None,
                    json.dumps(spec.metadata, ensure_ascii=False),
                )
                if spec.supersedes:
                    await conn.execute(
                        """
                        UPDATE memory_records
                        SET status = 'superseded', superseded_by = $1::uuid
                        WHERE id = $2::uuid AND status <> 'superseded'
                        """,
                        memory_id,
                        uuid.UUID(spec.supersedes),
                    )
        await self._embed_and_upsert(
            point_id=memory_id,
            text=spec.content,
            payload={
                "conversation_id": spec.conversation_id,
                "kind": spec.kind.value,
                "memory_id": memory_id,
            },
        )
        logger.info(
            "memory_created",
            extra={
                "memory_id": memory_id,
                "kind": spec.kind.value,
                "conversation_id": spec.conversation_id,
            },
        )
        return memory_id

    async def supersede_memory(
        self, memory_id: str, status: MemoryStatus = MemoryStatus.OBSOLETE
    ) -> bool:
        result = await self._pool.execute(
            """
            UPDATE memory_records SET status = $2
            WHERE id = $1::uuid AND status = 'active'
            """,
            memory_id,
            status.value,
        )
        updated = result == "UPDATE 1"
        if updated:
            logger.info(
                "memory_superseded", extra={"memory_id": memory_id, "status": status.value}
            )
        return updated

    # --------------------------------------------------------------- retrieval

    async def retrieve(
        self, query: str, conversation_id: str, limit: int | None = None
    ) -> list[RetrievedItem]:
        """Hybrid pipeline (§17): semantic + lexical -> fusion -> metadata
        filtering (same conversation, active memories only = supersession
        filtering) -> weighted ranking (§19)."""
        limit = limit or self._retrieval.limit_default
        pool_size = self._retrieval.candidate_pool

        query_vec = await self._safe_embed(query)
        semantic_scores: dict[str, float] = {}
        if query_vec is not None:
            try:
                vec_hits = await self._qdrant.search(
                    query_vec, limit=pool_size, conversation_id=conversation_id
                )
                for hit in vec_hits:
                    payload = hit.get("payload") or {}
                    key = str(payload.get("chunk_id") or payload.get("memory_id") or "")
                    if key:
                        semantic_scores[key] = min(1.0, max(0.0, float(hit["score"])))
            except Exception as exc:  # noqa: BLE001 - degrade to lexical (§31)
                logger.warning("vector_search_unavailable", extra={"error": str(exc)})

        lexical = await self._lexical_search(query, conversation_id, pool_size)
        max_rank = max((h["rank"] for h in lexical), default=0.0)

        candidates: dict[str, dict[str, Any]] = {
            hit["id"]: hit for hit in lexical
        }
        # semantic-only hits need their source rows fetched from PostgreSQL
        missing = [k for k in semantic_scores if k not in candidates]
        if missing:
            candidates.update(await self._fetch_by_ids(conversation_id, missing))

        now = now_utc()
        items: list[RetrievedItem] = []
        for key, row in candidates.items():
            sem = semantic_scores.get(key, 0.0)
            lex = (row["rank"] / max_rank) if max_rank > 0 else 0.0
            if sem <= 0.0 and lex <= 0.0:
                continue  # relevance over utilization (§22 baseline)
            age_days = 0.0
            if row.get("created_at") is not None:
                age_days = max(0.0, (now - _as_utc(row["created_at"])).total_seconds() / 86400.0)
            components = {
                "semantic": round(sem, 6),
                "lexical": round(lex, 6),
                "recency": round(1.0 / (1.0 + age_days), 6),
                "importance": round(float(row.get("importance") or 0.0), 6),
                "type_priority": TYPE_PRIORITY.get(row.get("kind") or "chunk", 0.4),
            }
            cfg = self._retrieval
            score = (
                cfg.semantic_weight * components["semantic"]
                + cfg.lexical_weight * components["lexical"]
                + cfg.recency_weight * components["recency"]
                + cfg.importance_weight * components["importance"]
                + cfg.type_weight * components["type_priority"]
            )
            items.append(
                RetrievedItem(
                    item_type=row["item_type"],
                    id=key,
                    conversation_id=conversation_id,
                    kind=row.get("kind") or "chunk",
                    content=row.get("content") or "",
                    score=round(score, 6),
                    components=components,
                    source_message_ids=[str(x) for x in row.get("source_message_ids") or []],
                )
            )
        items.sort(key=lambda i: (-i.score, i.id))
        return items[:limit]

    async def _fetch_by_ids(
        self, conversation_id: str, ids: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Resolve semantic-only hits against the source of truth; enforces
        active-status filtering for memories (supersession filtering)."""
        out: dict[str, dict[str, Any]] = {}
        try:
            uuids = [uuid.UUID(x) for x in ids]
        except ValueError:
            return out
        mem_rows = await self._pool.fetch(
            """
            SELECT id, kind, content, importance, created_at, source_message_ids
            FROM memory_records
            WHERE conversation_id = $1::uuid AND status = 'active' AND id = ANY($2::uuid[])
            """,
            conversation_id,
            uuids,
        )
        for r in mem_rows:
            out[str(r["id"])] = {
                "item_type": "memory",
                "kind": r["kind"],
                "content": r["content"],
                "importance": r["importance"],
                "created_at": r["created_at"],
                "source_message_ids": r["source_message_ids"] or [],
                "rank": 0.0,
            }
        chunk_rows = await self._pool.fetch(
            """
            SELECT id, raw_content, created_at, message_ids
            FROM conversation_chunks
            WHERE conversation_id = $1::uuid AND id = ANY($2::uuid[])
            """,
            conversation_id,
            uuids,
        )
        for r in chunk_rows:
            out[str(r["id"])] = {
                "item_type": "chunk",
                "kind": "chunk",
                "content": r["raw_content"],
                "importance": 0.0,
                "created_at": r["created_at"],
                "source_message_ids": [str(x) for x in (r["message_ids"] or [])],
                "rank": 0.0,
            }
        return out

    async def _lexical_search(
        self, query: str, conversation_id: str, limit: int
    ) -> list[dict[str, Any]]:
        ts_query = "websearch_to_tsquery('simple', $2)"
        rows = await self._pool.fetch(
            f"""
            SELECT id, kind, content AS text_content, importance, created_at,
                   source_message_ids, ts_rank(ts, {ts_query}) AS rank
            FROM memory_records
            WHERE conversation_id = $1::uuid AND status = 'active'
              AND ts @@ {ts_query}
            """,
            conversation_id,
            query,
        )
        out = [
            {
                "id": str(r["id"]),
                "item_type": "memory",
                "kind": r["kind"],
                "content": r["text_content"],
                "importance": r["importance"],
                "created_at": r["created_at"],
                "source_message_ids": r["source_message_ids"] or [],
                "rank": float(r["rank"]),
            }
            for r in rows
        ]
        chunk_rows = await self._pool.fetch(
            f"""
            SELECT id, raw_content, created_at, message_ids,
                   ts_rank(ts, {ts_query}) AS rank
            FROM conversation_chunks
            WHERE conversation_id = $1::uuid AND ts @@ {ts_query}
            """,
            conversation_id,
            query,
        )
        out += [
            {
                "id": str(r["id"]),
                "item_type": "chunk",
                "kind": "chunk",
                "content": r["raw_content"],
                "importance": 0.0,
                "created_at": r["created_at"],
                "source_message_ids": [str(x) for x in (r["message_ids"] or [])],
                "rank": float(r["rank"]),
            }
            for r in chunk_rows
        ]
        out.sort(key=lambda h: -h["rank"])
        return out[:limit]

    # ----------------------------------------------------------------- helpers

    async def _safe_embed(self, text: str) -> list[float] | None:
        try:
            vectors = await self._embedder.embed([text[: self._max_embed_chars]])
            return vectors[0]
        except Exception as exc:  # noqa: BLE001 - degrade to lexical-only (§31)
            logger.warning("embedding_unavailable", extra={"error": str(exc)})
            return None

    async def _embed_and_upsert(self, point_id: str, text: str, payload: dict[str, Any]) -> None:
        vector = await self._safe_embed(text)
        if vector is None:
            return  # row remains rebuildable from PostgreSQL (§32)
        try:
            await self._qdrant.upsert(
                [{"id": point_id, "vector": vector, "payload": payload}],
                vector_size=len(vector),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "vector_index_unavailable", extra={"point_id": point_id, "error": str(exc)}
            )
