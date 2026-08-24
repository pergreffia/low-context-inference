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
from context_proxy.memory.errors import EmbeddingProviderError, RetrievalError, VectorStoreError
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
        embed_batch_size: int = 64,
    ):
        self._pool = pool
        self._embedder = embedder
        self._qdrant = vector_store
        self._retrieval = retrieval_settings or RetrievalSettings()
        self._max_embed_chars = max_embed_chars
        self._embed_batch_size = max(1, embed_batch_size)
        self._counter = TokenCounter()

    # ------------------------------------------------------------------ chunks

    async def index_completed_turns(self, conversation_id: uuid.UUID | str) -> int:
        """Chunk new completed turns, then durably vector-index pending chunks.

        Two distinct progress concepts (M3 final review §2):

        - conversations.last_chunked_seq: PostgreSQL chunking watermark;
        - conversation_chunks.vector_indexed_at: NULL until embedding AND
          Qdrant upsert both succeeded for that chunk.

        Every invocation retries chunks whose vector_indexed_at is NULL —
        partial embedding/Qdrant failures are recovered automatically without
        a manual rebuild. Concurrency: chunking takes the conversation row
        lock; duplicate Qdrant upserts are harmless and the marker update only
        fires after success.
        """
        conversation_id = str(conversation_id)
        created, _settled_end = await self._chunk_new_turns(conversation_id)
        indexed = await self._index_pending_chunks(conversation_id)
        if created or indexed:
            logger.info(
                "indexing_pass_completed",
                extra={
                    "conversation_id": conversation_id,
                    "chunks_created": created,
                    "vectors_indexed": indexed,
                },
            )
        return created

    async def _chunk_new_turns(self, conversation_id: str) -> tuple[int, int | None]:
        """Create missing turn chunks; returns (created, settled_end_seq).

        Incremental: scans from the user message at-or-before the chunking
        watermark so a turn straddling the boundary is reconsidered once it
        completes (its insert no-ops via the unique constraint). The trailing
        unit stays raw — it is still the live interaction.
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT id FROM conversations WHERE id = $1::uuid FOR UPDATE",
                    conversation_id,
                )
                watermark = await conn.fetchval(
                    "SELECT last_chunked_seq FROM conversations WHERE id = $1::uuid",
                    conversation_id,
                )
                # Window starts at the user message AT OR BEFORE the watermark:
                # a turn straddling the previous boundary must be reconsidered
                # once it completed (its chunk insert no-ops if already done).
                boundary = await conn.fetchval(
                    """
                    SELECT COALESCE(MAX(seq), 0) FROM messages
                    WHERE conversation_id = $1::uuid AND role = 'user'
                      AND seq <= COALESCE($2, -1)
                    """,
                    conversation_id,
                    watermark,
                )
                rows = await conn.fetch(
                    """
                    SELECT id, seq, role, content FROM messages
                    WHERE conversation_id = $1::uuid AND role <> 'system'
                      AND seq >= $2
                    ORDER BY seq
                    """,
                    conversation_id,
                    boundary,
                )
                if not rows:
                    return 0, watermark if watermark is not None else None

                units: list[tuple[int, list[asyncpg.Record]]] = []
                current: list[asyncpg.Record] = []
                for row in rows:
                    if row["role"] == "user" and current:
                        units.append((current[0]["seq"], current))
                        current = []
                    current.append(row)
                if current:
                    units.append((current[0]["seq"], current))

                # The LAST unit is the live interaction: never chunked now.
                settled_units = units[:-1]
                created = 0
                settled_end: int | None = None
                for start_seq, unit_rows in settled_units:
                    chunk_id = await self._insert_chunk(
                        conn, conversation_id, start_seq, unit_rows
                    )
                    if chunk_id is not None:
                        created += 1
                    settled_end = unit_rows[-1]["seq"]

                if settled_end is not None and (
                    watermark is None or settled_end > watermark
                ):
                    await conn.execute(
                        """
                        UPDATE conversations SET last_chunked_seq = $2
                        WHERE id = $1::uuid
                        """,
                        conversation_id,
                        settled_end,
                    )
                return created, settled_end

    async def _index_pending_chunks(self, conversation_id: str) -> int:
        """Embed+upsert every chunk still pending; mark after success.

        Embedding is BATCHED (hardening P2.2): one provider call per slice of
        up to EMBED_BATCH_SIZE texts, preserving input/output ordering.
        Embedding failure fails the whole slice (chunks stay retryable);
        Qdrant failures are per-chunk.
        """
        rows = await self._pool.fetch(
            """
            SELECT id, raw_content, start_seq FROM conversation_chunks
            WHERE conversation_id = $1::uuid AND vector_indexed_at IS NULL
            ORDER BY start_seq
            LIMIT 100
            """,
            conversation_id,
        )
        indexed = 0
        for batch_start in range(0, len(rows), self._embed_batch_size):
            batch = rows[batch_start : batch_start + self._embed_batch_size]
            vectors = await self._embed_batch([r["raw_content"] for r in batch])
            if vectors is None:
                continue  # whole slice stays pending (embedding leg down)
            for row, vector in zip(batch, vectors, strict=True):
                if await self._index_chunk(
                    conversation_id,
                    str(row["id"]),
                    row["raw_content"],
                    row["start_seq"],
                    vector=vector,
                ):
                    indexed += 1
        return indexed

    async def embed_texts_in_batches(
        self, texts: list[str]
    ) -> list[list[float]] | None:
        """Batch-embed texts in slices of `_embed_batch_size`, order-preserving.

        Returns None when the embedding leg fails for any slice (callers keep
        their items retryable); empty input short-circuits to [].
        """
        if not texts:
            return []
        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), self._embed_batch_size):
            slice_vectors = await self._safe_embed_batch(
                texts[start : start + self._embed_batch_size]
            )
            if slice_vectors is None:
                return None
            all_vectors.extend(slice_vectors)
        return all_vectors

    async def _safe_embed_batch(
        self, texts: list[str]
    ) -> list[list[float]] | None:
        """One provider call for `texts`; None on typed embedding failure."""
        if not texts:
            return []
        try:
            truncated = [t[: self._max_embed_chars] for t in texts]
            vectors = await self._embedder.embed(truncated)
        except EmbeddingProviderError as exc:
            logger.warning(
                "embedding_unavailable", extra={"count": len(texts), "error": str(exc)}
            )
            return None
        if len(vectors) != len(texts):
            logger.warning(
                "embedding_count_mismatch",
                extra={"expected": len(texts), "received": len(vectors)},
            )
            return None
        return vectors

    async def _embed_batch(
        self, texts: list[str]
    ) -> list[list[float]] | None:
        """Alias kept for internal callers; see embed_texts_in_batches."""
        return await self._safe_embed_batch(texts)

    async def _index_chunk(
        self,
        conversation_id: str,
        chunk_id: str,
        raw_content: str,
        start_seq: int | None = None,
        *,
        vector: list[float] | None = None,
    ) -> bool:
        """Embed (or reuse a batched vector) + upsert one chunk.

        Marks vector_indexed_at only after a successful Qdrant upsert. When
        `vector` is supplied the embedding step is skipped (batched path).
        """
        if vector is None:
            vector = await self._safe_embed(raw_content)
        if vector is None:
            return False  # embedding failed: Qdrant not called, stays pending
        try:
            await self._qdrant.upsert(
                [
                    {
                        "id": chunk_id,
                        "vector": vector,
                        "payload": {
                            "conversation_id": conversation_id,
                            "kind": "chunk",
                            "chunk_id": chunk_id,
                            "start_seq": start_seq,
                        },
                    }
                ],
                vector_size=len(vector),
            )
        except VectorStoreError as exc:
            # Expected vector-store outage: chunk stays pending, retried by a
            # future indexing pass (§8). Programming errors propagate.
            logger.warning(
                "vector_index_unavailable", extra={"chunk_id": chunk_id, "error": str(exc)}
            )
            return False
        await self._pool.execute(
            """
            UPDATE conversation_chunks SET vector_indexed_at = now()
            WHERE id = $1::uuid AND vector_indexed_at IS NULL
            """,
            uuid.UUID(chunk_id),
        )
        return True

    async def _insert_chunk(
        self,
        conn: asyncpg.Connection,
        conversation_id: str,
        start_seq: int,
        unit_rows: list[asyncpg.Record],
    ) -> Any:
        """Insert one turn chunk (caller holds lock+tx); returns new id or None."""
        messages = [json.loads(r["content"]) for r in unit_rows]
        raw_content = "\n".join(json.dumps(m, ensure_ascii=False) for m in messages)
        token_count = self._counter.messages(messages)
        message_ids = [str(r["id"]) for r in unit_rows]
        return await conn.fetchval(
            """
            INSERT INTO conversation_chunks
                (conversation_id, start_seq, end_seq, message_ids, raw_content, token_count)
            VALUES ($1::uuid, $2, $3, $4::uuid[], $5, $6)
            ON CONFLICT (conversation_id, start_seq) DO NOTHING
            RETURNING id
            """,
            conversation_id,
            start_seq,
            unit_rows[-1]["seq"],
            message_ids,
            raw_content,
            token_count,
        )

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
                    # Cross-conversation targets are rejected BEFORE any write:
                    # the target stays untouched and no new memory is created.
                    target = await conn.fetchrow(
                        """
                        SELECT id, conversation_id FROM memory_records
                        WHERE id = $1::uuid FOR UPDATE
                        """,
                        uuid.UUID(spec.supersedes),
                    )
                    if target is None:
                        raise ValueError(f"supersedes target {spec.supersedes} not found")
                    if str(target["conversation_id"]) != spec.conversation_id:
                        raise ValueError(
                            "supersedes target belongs to a different conversation"
                        )
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
        self,
        memory_id: uuid.UUID | str,
        status: MemoryStatus = MemoryStatus.OBSOLETE,
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
        self, query: str, conversation_id: uuid.UUID | str, limit: int | None = None
    ) -> list[RetrievedItem]:
        """Hybrid pipeline (§17): semantic + lexical -> fusion -> metadata
        filtering (same conversation, active memories only = supersession
        filtering) -> weighted ranking (§19)."""
        conversation_id = str(conversation_id)
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
            except VectorStoreError as exc:
                # Expected vector-store outage: degrade to lexical (§10.4).
                # Programming errors from the store propagate untouched.
                logger.warning("vector_search_unavailable", extra={"error": str(exc)})

        # PostgreSQL legs are typed-failure territory: an expected database
        # outage becomes RetrievalError (degradable), while programming errors
        # (TypeError, AttributeError, ...) propagate untouched (§18).
        try:
            lexical = await self._lexical_search(query, conversation_id, pool_size)
            max_rank = max((h["rank"] for h in lexical), default=0.0)

            candidates: dict[str, dict[str, Any]] = {
                hit["id"]: hit for hit in lexical
            }
            # semantic-only hits need their source rows fetched from PostgreSQL
            missing = [k for k in semantic_scores if k not in candidates]
            if missing:
                candidates.update(await self._fetch_by_ids(conversation_id, missing))
        except (asyncpg.PostgresError, asyncpg.InterfaceError) as exc:
            raise RetrievalError(
                f"lexical retrieval unavailable: {exc}", cause=exc
            ) from exc

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
                    start_seq=row.get("start_seq"),
                    end_seq=row.get("end_seq"),
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
                "start_seq": None,
                "end_seq": None,
            }
        chunk_rows = await self._pool.fetch(
            """
            SELECT id, raw_content, created_at, message_ids, start_seq, end_seq
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
                "start_seq": r["start_seq"],
                "end_seq": r["end_seq"],
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
                "start_seq": None,
                "end_seq": None,
            }
            for r in rows
        ]
        chunk_rows = await self._pool.fetch(
            f"""
            SELECT id, raw_content, created_at, message_ids, start_seq, end_seq,
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
                "start_seq": r["start_seq"],
                "end_seq": r["end_seq"],
            }
            for r in chunk_rows
        ]
        out.sort(key=lambda h: -h["rank"])
        return out[:limit]

    # ----------------------------------------------------------------- rebuild

    async def rebuild_vector_index(
        self,
        conversation_id: uuid.UUID | str | None = None,
        *,
        force: bool = False,
    ) -> dict[str, int]:
        """Rebuild the derived Qdrant index from PostgreSQL (M5 §12).

        PostgreSQL is authoritative; this procedure exists precisely because
        the vector index is rebuildable. With force=True every chunk's
        vector_indexed_at is reset first so already-marked chunks are re-upserted
        (idempotent point ids make duplicate upserts harmless). Memories are
        always re-embedded+upserted (they carry no durable marker). Scope is
        per conversation by default; conversation_id=None rebuilds everything.

        Error classification (M5 review): expected VectorStoreError failures
        are counted in `chunks_failed` / `memories_failed` and the rebuild
        continues; unexpected programming errors propagate.
        """
        scope_clause = "AND conversation_id = $1::uuid" if conversation_id else ""
        args: list[Any] = []
        if conversation_id:
            args.append(str(conversation_id))

        if force:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    f"""
                    UPDATE conversation_chunks SET vector_indexed_at = NULL
                    WHERE TRUE {scope_clause}
                    """,
                    *args,
                )

        chunk_rows = await self._pool.fetch(
            f"""
            SELECT id, conversation_id, raw_content, start_seq FROM conversation_chunks
            WHERE TRUE {scope_clause}
            ORDER BY conversation_id, start_seq
            """,
            *args,
        )
        memory_rows = await self._pool.fetch(
            f"""
            SELECT id, conversation_id, kind, content FROM memory_records
            WHERE status = 'active' {scope_clause}
            """,
            *args,
        )
        summary = {
            "chunks": 0,
            "chunks_failed": 0,
            "memories": 0,
            "memories_failed": 0,
        }
        for batch_start in range(0, len(chunk_rows), self._embed_batch_size):
            chunk_batch = chunk_rows[batch_start : batch_start + self._embed_batch_size]
            vectors = await self._embed_batch([r["raw_content"] for r in chunk_batch])
            if vectors is None:
                summary["chunks_failed"] += len(chunk_batch)
                continue
            for row, vector in zip(chunk_batch, vectors, strict=True):
                ok = await self._rebuild_chunk(
                    str(row["conversation_id"]),
                    str(row["id"]),
                    row["raw_content"],
                    row["start_seq"],
                    vector=vector,
                )
                summary["chunks" if ok else "chunks_failed"] += 1

        memory_texts = [r["content"][: self._max_embed_chars] for r in memory_rows]
        vectors_by_memory = await self._embed_batch(memory_texts)
        if vectors_by_memory is None:
            summary["memories"] = len(memory_rows)
            summary["memories_failed"] = len(memory_rows)
        else:
            for row, vector in zip(memory_rows, vectors_by_memory, strict=True):
                try:
                    await self._qdrant.upsert(
                        [
                            {
                                "id": str(row["id"]),
                                "vector": vector,
                                "payload": {
                                    "conversation_id": str(row["conversation_id"]),
                                    "kind": row["kind"],
                                    "memory_id": str(row["id"]),
                                },
                            }
                        ],
                        vector_size=len(vector),
                    )
                except VectorStoreError as exc:
                    # Expected provider outage: count it and keep rebuilding.
                    logger.warning(
                        "rebuild_memory_upsert_failed", extra={"error": str(exc)}
                    )
                    summary["memories_failed"] += 1
                summary["memories"] += 1
        logger.info(
            "vector_index_rebuilt",
            extra={**summary, "force": force},
        )
        return summary

    async def _rebuild_chunk(
        self,
        conversation_id: str,
        chunk_id: str,
        raw_content: str,
        start_seq: int | None,
        *,
        vector: list[float] | None = None,
    ) -> bool:
        """Upsert one chunk during a rebuild; typed failure handling.

        Unlike _index_chunk (which keeps M3's blanket retry semantics), the
        rebuild path must surface programming errors: only VectorStoreError is
        treated as an expected infrastructure failure here.
        """
        if vector is None:
            vector = await self._safe_embed(raw_content)
        if vector is None:
            return False  # embedding leg unavailable: stays retryable
        try:
            await self._qdrant.upsert(
                [
                    {
                        "id": chunk_id,
                        "vector": vector,
                        "payload": {
                            "conversation_id": conversation_id,
                            "kind": "chunk",
                            "chunk_id": chunk_id,
                            "start_seq": start_seq,
                        },
                    }
                ],
                vector_size=len(vector),
            )
        except VectorStoreError as exc:
            logger.warning("rebuild_chunk_upsert_failed", extra={"error": str(exc)})
            return False
        await self._pool.execute(
            """
            UPDATE conversation_chunks SET vector_indexed_at = now()
            WHERE id = $1::uuid AND vector_indexed_at IS NULL
            """,
            uuid.UUID(chunk_id),
        )
        return True

    # ----------------------------------------------------------------- helpers

    async def _safe_embed(self, text: str) -> list[float] | None:
        try:
            vectors = await self._embedder.embed([text[: self._max_embed_chars]])
            return vectors[0]
        except EmbeddingProviderError as exc:
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
        except VectorStoreError as exc:
            # Expected vector-store outage at create-time: row stays
            # rebuildable from PostgreSQL (§32). Programming errors propagate.
            logger.warning(
                "vector_index_unavailable", extra={"point_id": point_id, "error": str(exc)}
            )
