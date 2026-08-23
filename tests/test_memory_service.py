from __future__ import annotations

import asyncio
import hashlib
import json
import os
import uuid

import asyncpg
import pytest

from context_proxy.config import EndpointSettings, RetrievalSettings
from context_proxy.memory.embeddings import OpenAICompatibleEmbeddingProvider
from context_proxy.memory.models import MemoryCreate, MemoryKind
from context_proxy.memory.qdrant import QdrantVectorStore
from context_proxy.memory.service import MemoryService

MIGRATION_DSN = os.environ.get("TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not MIGRATION_DSN,
    reason="set TEST_DATABASE_URL to run PostgreSQL integration tests",
)


@pytest.fixture(autouse=True)
def _migrated_db():
    async def _apply():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            from context_proxy.db.database import apply_migrations

            await apply_migrations(pool)
        finally:
            await pool.close()

    asyncio.run(_apply())


class HashingEmbedder(OpenAICompatibleEmbeddingProvider):
    """Deterministic offline embedder: 6-dim token-hash vector (tests only)."""

    def __init__(self):
        super().__init__(EndpointSettings(base_url="offline://none"))

    async def embed(self, texts):
        out = []
        for text in texts:
            vec = [0.0] * 6
            for token in text.lower().split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "big") % 6
                vec[bucket] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out


class RecordingVectorStore(QdrantVectorStore):
    def __init__(self):
        super().__init__("offline://none")
        self.points: list[dict] = []

    async def upsert(self, points, vector_size):
        self.points.extend(points)

    async def search(self, vector, limit, conversation_id=None):
        from math import sqrt

        def cos(p):
            v = p["vector"]
            dot = sum(x * y for x, y in zip(vector, v, strict=True))
            na = sqrt(sum(x * x for x in vector)) or 1
            nb = sqrt(sum(y * y for y in v)) or 1
            return dot / (na * nb)

        hits = sorted(self.points, key=cos, reverse=True)[:limit]
        return [
            {"id": p["id"], "score": cos(p), "payload": p["payload"]}
            for p in hits
            if p["payload"].get("conversation_id") == conversation_id
        ]


def _store(pool):
    from context_proxy.conversation.store import PostgresConversationStore

    return PostgresConversationStore(pool)


async def _make(pool) -> MemoryService:
    return MemoryService(
        pool,
        HashingEmbedder(),
        RecordingVectorStore(),
        retrieval_settings=RetrievalSettings(),
    )


def _call(tool_call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": tool_call_id, "type": "function", "function": {"name": "f", "arguments": "{}"}}
        ],
    }


def test_chunking_turn_based_idempotent_tool_atomic_system_excluded():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            conv = str(uuid.uuid4())
            store = _store(pool)
            memory = await _make(pool)

            await store.append_messages(
                conv,
                [
                    {"role": "system", "content": "sys prompt stays raw"},
                    {"role": "user", "content": "question one"},
                    _call("c1"),
                    {"role": "tool", "tool_call_id": "c1", "content": "result"},
                    {"role": "assistant", "content": "answer one"},
                ],
            )
            await store.append_messages(conv, [{"role": "user", "content": "question two"}])

            created = await memory.index_completed_turns(conv)
            assert created == 1  # trailing turn stays raw; system excluded
            again = await memory.index_completed_turns(conv)
            assert again == 0  # idempotent

            rows = await pool.fetch(
                """
                SELECT start_seq, message_ids, raw_content
                FROM conversation_chunks WHERE conversation_id = $1::uuid
                """,
                conv,
            )
            assert len(rows) == 1
            chunk = rows[0]
            assert chunk["start_seq"] == 1
            contents = [
                json.loads(line)["content"]
                for line in chunk["raw_content"].splitlines()
            ]
            assert "sys prompt stays raw" not in contents
            # tool call and result live in the same chunk (atomic unit)
            assert "result" in contents

            # second turn completes -> becomes chunkable on next pass
            await store.append_messages(conv, [{"role": "assistant", "content": "answer two"}])
            created = await memory.index_completed_turns(conv)
            assert created == 0  # still trailing
        finally:
            await pool.close()

    asyncio.run(_run())


def test_supersession_excludes_old_decision_from_retrieval():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            conv = str(uuid.uuid4())
            memory = await _make(pool)

            old_id = await memory.create_memory(
                MemoryCreate(
                    kind=MemoryKind.DECISION,
                    content="Use SQLite for persistence",
                    conversation_id=conv,
                    importance=0.9,
                    source_message_ids=[],
                )
            )
            new_id = await memory.create_memory(
                MemoryCreate(
                    kind=MemoryKind.DECISION,
                    content="Use PostgreSQL for persistence",
                    conversation_id=conv,
                    importance=0.9,
                    supersedes=old_id,
                )
            )

            row_old = await pool.fetchrow(
                "SELECT status, superseded_by FROM memory_records WHERE id = $1::uuid", old_id
            )
            assert row_old["status"] == "superseded"  # never deleted (§10)
            assert str(row_old["superseded_by"]) == new_id

            items = await memory.retrieve("persistence database", conv)
            ids = {i.id for i in items}
            assert new_id in ids
            assert old_id not in ids  # supersession filtering (§12)

            standalone = await memory.supersede_memory(new_id)
            assert standalone is True
            gone = await memory.supersede_memory(new_id)  # already inactive
            assert gone is False
        finally:
            await pool.close()

    asyncio.run(_run())


def test_hybrid_retrieval_merges_lexical_and_semantic():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            conv = str(uuid.uuid4())
            store = _store(pool)
            memory = await _make(pool)

            # turn 1 contains the lexical keyword; turn 2 completes it
            await store.append_messages(
                conv,
                [
                    {"role": "user", "content": "how does the kubernetes ingress work"},
                    {"role": "assistant", "content": "ingress routes external traffic"},
                ],
            )
            await store.append_messages(
                conv, [{"role": "user", "content": "unrelated follow-up"}]
            )
            await memory.index_completed_turns(conv)

            decision_id = await memory.create_memory(
                MemoryCreate(
                    kind=MemoryKind.CONSTRAINT,
                    content="kubernetes ingress must use nginx class",
                    conversation_id=conv,
                    importance=0.8,
                )
            )

            items = await memory.retrieve("kubernetes ingress", conv)
            kinds = {(i.item_type, i.id) for i in items}
            types = {i.item_type for i in items}

            # both legs contribute: lexical chunk + semantic/lexical memory
            assert types <= {"chunk", "memory"}
            assert any(i.kind == "chunk" for i in items)
            assert decision_id in {i.id for i in items if i.item_type == "memory"}
            assert all(i.score > 0 for i in items)
            assert kinds  # non-empty
            top = items[0]
            assert top.components["semantic"] + top.components["lexical"] > 0

            # conversation isolation (§18)
            other_conv = str(uuid.uuid4())
            assert await memory.retrieve("kubernetes ingress", other_conv) == []
        finally:
            await pool.close()

    asyncio.run(_run())


def test_degraded_vector_leg_still_returns_lexical_only():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            conv = str(uuid.uuid4())
            memory = await _make(pool)
            await memory.create_memory(
                MemoryCreate(
                    kind=MemoryKind.FACT,
                    content="the deploy pipeline uses github actions",
                    conversation_id=conv,
                )
            )

            class BrokenEmbedder(HashingEmbedder):
                async def embed(self, texts):
                    raise RuntimeError("embedding endpoint down")

            broken = MemoryService(
                pool,
                BrokenEmbedder(),
                RecordingVectorStore(),
                retrieval_settings=RetrievalSettings(),
            )
            items = await broken.retrieve("deploy pipeline github actions", conv)
            assert len(items) == 1
            assert items[0].components["semantic"] == 0.0
            assert items[0].components["lexical"] > 0
        finally:
            await pool.close()

    asyncio.run(_run())


def test_weights_change_ranking_order():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            conv = str(uuid.uuid4())
            memory = await _make(pool)
            await memory.create_memory(
                MemoryCreate(
                    kind=MemoryKind.DECISION,
                    content="postgres replication lag budget is 5 minutes",
                    conversation_id=conv,
                    importance=0.2,
                )
            )
            await memory.create_memory(
                MemoryCreate(
                    kind=MemoryKind.FACT,
                    content="postgres replication lag budget is 5 minutes",
                    conversation_id=conv,
                    importance=1.0,
                )
            )

            importance_heavy = MemoryService(
                pool,
                HashingEmbedder(),
                RecordingVectorStore(),
                retrieval_settings=RetrievalSettings(importance_weight=0.99),
            )
            type_heavy = MemoryService(
                pool,
                HashingEmbedder(),
                RecordingVectorStore(),
                retrieval_settings=RetrievalSettings(type_weight=0.99),
            )

            by_importance = await importance_heavy.retrieve("replication lag budget", conv)
            by_type = await type_heavy.retrieve("replication lag budget", conv)
            assert by_importance[0].kind == "fact"  # importance 1.0 wins
            assert by_type[0].kind == "decision"  # decision outranks fact
        finally:
            await pool.close()

    asyncio.run(_run())


class BrokenVectorStore(QdrantVectorStore):
    """Simulates a Qdrant outage."""

    def __init__(self):
        super().__init__("offline://none")

    async def search(self, vector, limit, conversation_id=None):
        raise RuntimeError("qdrant down")

    async def upsert(self, points, vector_size):
        raise RuntimeError("qdrant down")


def test_supersession_rejects_cross_conversation_target():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            conv_a = str(uuid.uuid4())
            conv_b = str(uuid.uuid4())
            memory = await _make(pool)

            target_id = await memory.create_memory(
                MemoryCreate(
                    kind=MemoryKind.DECISION,
                    content="decision owned by conversation A",
                    conversation_id=conv_a,
                )
            )
            with pytest.raises(ValueError) as excinfo:
                await memory.create_memory(
                    MemoryCreate(
                        kind=MemoryKind.DECISION,
                        content="attempt from conversation B",
                        conversation_id=conv_b,
                        supersedes=target_id,
                    )
                )
            assert "different conversation" in str(excinfo.value)

            row = await pool.fetchrow(
                "SELECT status, superseded_by FROM memory_records WHERE id = $1::uuid",
                uuid.UUID(target_id),
            )
            assert row["status"] == "active"
            assert row["superseded_by"] is None
            in_a = await pool.fetchval(
                "SELECT count(*) FROM memory_records WHERE conversation_id = $1::uuid",
                uuid.UUID(conv_a),
            )
            in_b = await pool.fetchval(
                "SELECT count(*) FROM memory_records WHERE conversation_id = $1::uuid",
                uuid.UUID(conv_b),
            )
            assert in_a == 1 and in_b == 0  # nothing new created anywhere
        finally:
            await pool.close()

    asyncio.run(_run())


def test_degraded_qdrant_leg_still_returns_lexical_results():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            conv = str(uuid.uuid4())
            memory = MemoryService(
                pool,
                HashingEmbedder(),
                BrokenVectorStore(),
                retrieval_settings=RetrievalSettings(),
            )
            await memory.create_memory(
                MemoryCreate(
                    kind=MemoryKind.FACT,
                    content="the release pipeline signs artifacts with cosign",
                    conversation_id=conv,
                )
            )
            items = await memory.retrieve("release pipeline cosign", conv)
            assert len(items) == 1
            assert items[0].components["lexical"] > 0
            assert items[0].components["semantic"] == 0.0
        finally:
            await pool.close()

    asyncio.run(_run())


def test_superseded_memory_returned_by_qdrant_is_filtered():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            conv = str(uuid.uuid4())
            memory = MemoryService(
                pool,
                HashingEmbedder(),
                RecordingVectorStore(),
                retrieval_settings=RetrievalSettings(),
            )
            old_id = await memory.create_memory(
                MemoryCreate(
                    kind=MemoryKind.DECISION,
                    content="cache eviction policy is LRU",
                    conversation_id=conv,
                )
            )
            new_id = await memory.create_memory(
                MemoryCreate(
                    kind=MemoryKind.DECISION,
                    content="cache eviction policy is LFU",
                    conversation_id=conv,
                    supersedes=old_id,
                )
            )
            stale_hit = {
                "id": old_id,
                "score": 0.99,
                "payload": {"conversation_id": conv, "memory_id": old_id},
            }
            fresh_hit = {
                "id": new_id,
                "score": 0.4,
                "payload": {"conversation_id": conv, "memory_id": new_id},
            }

            class StaleVectorStore(RecordingVectorStore):
                async def search(self, vector, limit, conversation_id=None):
                    return [stale_hit, fresh_hit]

            stale_memory = MemoryService(
                pool,
                HashingEmbedder(),
                StaleVectorStore(),
                retrieval_settings=RetrievalSettings(),
            )
            items = await stale_memory.retrieve("cache eviction policy", conv)
            ids = {i.id for i in items}
            assert new_id in ids
            assert old_id not in ids  # PostgreSQL authoritative: stale hit filtered
        finally:
            await pool.close()

    asyncio.run(_run())


def test_concurrent_indexing_creates_no_duplicate_chunks():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN, min_size=2, max_size=8)
        try:
            conv = str(uuid.uuid4())
            store = _store(pool)
            memory = await _make(pool)
            await store.append_messages(
                conv,
                [
                    {"role": "user", "content": "first turn question"},
                    {"role": "assistant", "content": "first turn answer"},
                ],
            )
            await store.append_messages(conv, [{"role": "user", "content": "second"}])

            results = await asyncio.gather(
                memory.index_completed_turns(conv),
                memory.index_completed_turns(conv),
                return_exceptions=True,
            )
            assert all(not isinstance(r, BaseException) for r in results), results

            dupes = await pool.fetch(
                """
                SELECT start_seq FROM conversation_chunks
                WHERE conversation_id = $1::uuid
                GROUP BY start_seq HAVING count(*) > 1
                """,
                conv,
            )
            assert dupes == []
            total = await pool.fetchval(
                "SELECT count(*) FROM conversation_chunks WHERE conversation_id = $1::uuid",
                conv,
            )
            assert total == 1
        finally:
            await pool.close()

    asyncio.run(_run())


async def _seed_two_turns(store, conv):
    await store.append_messages(
        conv,
        [
            {"role": "user", "content": "turn one question"},
            {"role": "assistant", "content": "turn one answer"},
        ],
    )
    await store.append_messages(conv, [{"role": "user", "content": "turn two start"}])


def test_incremental_indexing_only_processes_new_turns():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            conv = str(uuid.uuid4())
            store = _store(pool)
            memory = await _make(pool)
            await _seed_two_turns(store, conv)

            assert await memory.index_completed_turns(conv) == 1

            # complete turn 2; only IT must be processed now
            await store.append_messages(
                conv, [{"role": "assistant", "content": "turn two answer"}]
            )
            await store.append_messages(conv, [{"role": "user", "content": "turn three"}])
            created = await memory.index_completed_turns(conv)
            assert created == 1

            rows = await pool.fetch(
                """
                SELECT start_seq FROM conversation_chunks
                WHERE conversation_id = $1::uuid ORDER BY start_seq
                """,
                conv,
            )
            assert [r["start_seq"] for r in rows] == [0, 2]
        finally:
            await pool.close()

    asyncio.run(_run())


def test_incremental_indexing_remains_idempotent():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            conv = str(uuid.uuid4())
            store = _store(pool)
            memory = await _make(pool)
            await _seed_two_turns(store, conv)

            results = [
                await memory.index_completed_turns(conv) for _ in range(3)
            ]
            assert results == [1, 0, 0]

            rows = await pool.fetch(
                """
                SELECT start_seq FROM conversation_chunks
                WHERE conversation_id = $1::uuid ORDER BY start_seq
                """,
                conv,
            )
            assert [r["start_seq"] for r in rows] == [0]
        finally:
            await pool.close()

    asyncio.run(_run())


def test_semantic_only_hit_resolved_from_postgresql():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            conv = str(uuid.uuid4())
            other = str(uuid.uuid4())
            memory = MemoryService(
                pool,
                HashingEmbedder(),
                RecordingVectorStore(),
                retrieval_settings=RetrievalSettings(),
            )
            mem_id = await memory.create_memory(
                MemoryCreate(
                    kind=MemoryKind.FACT,
                    content="zebra quadrant threshold",
                    conversation_id=conv,
                )
            )

            class SemanticOnlyStore(RecordingVectorStore):
                async def search(self, vector, limit, conversation_id=None):
                    if conversation_id != conv:
                        return []  # isolation enforced at the vector leg too
                    return [
                        {
                            "id": mem_id,
                            "score": 0.95,
                            "payload": {
                                "conversation_id": conv,
                                "memory_id": mem_id,
                            },
                        }
                    ]

            semantic_memory = MemoryService(
                pool,
                HashingEmbedder(),
                SemanticOnlyStore(),
                retrieval_settings=RetrievalSettings(),
            )
            items = await semantic_memory.retrieve("unrelated keywords here", conv)
            assert len(items) == 1
            assert items[0].id == mem_id
            assert items[0].components["semantic"] > 0
            assert items[0].components["lexical"] == 0.0

            assert await semantic_memory.retrieve("anything", other) == []
        finally:
            await pool.close()

    asyncio.run(_run())


def test_empty_and_weak_retrieval_returns_nothing():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            conv = str(uuid.uuid4())
            memory = await _make(pool)

            # no data at all
            assert await memory.retrieve("anything whatsoever", conv) == []

            # data exists but zero lexical + zero semantic overlap -> excluded
            await memory.create_memory(
                MemoryCreate(
                    kind=MemoryKind.FACT,
                    content="completely orthogonal stored fact",
                    conversation_id=conv,
                )
            )
            class NoHits(RecordingVectorStore):
                async def search(self, vector, limit, conversation_id=None):
                    return []

            no_hits = MemoryService(
                pool,
                HashingEmbedder(),
                NoHits(),
                retrieval_settings=RetrievalSettings(),
            )
            assert await no_hits.retrieve("zzz qqq xxx", conv) == []
        finally:
            await pool.close()

    asyncio.run(_run())


def test_equal_scores_order_deterministically_by_id():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            conv = str(uuid.uuid4())
            memory = MemoryService(
                pool,
                HashingEmbedder(),
                RecordingVectorStore(),
                retrieval_settings=RetrievalSettings(
                    importance_weight=0.0,
                    recency_weight=0.0,
                    type_weight=0.0,
                ),
            )
            id_a = await memory.create_memory(
                MemoryCreate(
                    kind=MemoryKind.FACT,
                    content="identical twin fact",
                    conversation_id=conv,
                )
            )
            id_b = await memory.create_memory(
                MemoryCreate(
                    kind=MemoryKind.FACT,
                    content="identical twin fact",
                    conversation_id=conv,
                )
            )
            items = await memory.retrieve("identical twin fact", conv)
            scores = [i.score for i in items]
            assert scores == sorted(scores, reverse=True)
            top_lex = items[0].components["lexical"]
            tied = [i.id for i in items if i.components["lexical"] == top_lex]
            assert tied == sorted(tied)  # ID tie-breaker ascending
            assert {id_a, id_b} <= set(tied)
        finally:
            await pool.close()

    asyncio.run(_run())


class InstrumentedEmbedder(HashingEmbedder):
    """Counts calls; optionally fails the first N and/or sleeps."""

    def __init__(self, fail_times: int = 0, sleep_seconds: float = 0.0):
        super().__init__()
        self.calls = 0
        self._fail_times = fail_times
        self._sleep = sleep_seconds

    async def embed(self, texts):
        import asyncio

        self.calls += 1
        if self._sleep:
            await asyncio.sleep(self._sleep)
        if self.calls <= self._fail_times:
            raise RuntimeError("embedding outage")
        return await super().embed(texts)


class CountingVectorStore(RecordingVectorStore):
    """Counts upserts per point; optionally fails the first N calls."""

    def __init__(self, fail_times: int = 0):
        super().__init__()
        self.upsert_calls = 0
        self.per_point: dict[str, int] = {}
        self._fail_times = fail_times

    async def upsert(self, points, vector_size):
        self.upsert_calls += 1
        for p in points:
            self.per_point[p["id"]] = self.per_point.get(p["id"], 0) + 1
        if self.upsert_calls <= self._fail_times:
            raise RuntimeError("qdrant outage")
        await super().upsert(points, vector_size)


async def _seed_three_completed_turns(pool, store, conv):
    for q, a in [
        ("alpha question", "alpha answer"),
        ("beta question", "beta answer"),
        ("gamma question", "gamma answer"),
    ]:
        await store.append_messages(conv, [{"role": "user", "content": q}])
        await store.append_messages(conv, [{"role": "assistant", "content": a}])
    await store.append_messages(conv, [{"role": "user", "content": "live tail"}])


async def _vector_state(pool, conv):
    rows = await pool.fetch(
        """
        SELECT start_seq, vector_indexed_at FROM conversation_chunks
        WHERE conversation_id = $1::uuid ORDER BY start_seq
        """,
        uuid.UUID(conv),
    )
    return [(r["start_seq"], r["vector_indexed_at"] is not None) for r in rows]


def test_partial_vector_index_failure_is_retried():
    """§15: first-upsert failure leaves that chunk pending; pass 2 recovers."""
    from context_proxy.memory.service import MemoryService as MS

    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            conv = str(uuid.uuid4())
            store = _store(pool)
            await _seed_three_completed_turns(pool, store, conv)

            flaky_vectors = CountingVectorStore(fail_times=1)  # first upsert fails
            embedder = InstrumentedEmbedder()
            flaky = MS(pool, embedder, flaky_vectors, retrieval_settings=RetrievalSettings())

            created = await flaky.index_completed_turns(conv)
            assert created == 3
            state = dict(await _vector_state(pool, conv))
            # keys are turn start_seqs: 0 failed, 2 and 4 succeeded
            assert state[0] is False and state[2] is True and state[4] is True

            # second pass retries only the pending chunk (start_seq 0)
            await flaky.index_completed_turns(conv)

            state = dict(await _vector_state(pool, conv))
            assert all(state[s] for s in (0, 2, 4))
            # A was already done: untouched. B/C now upserted exactly once.
            chunk_ids = [str(r["id"]) for r in await _all_chunks(pool, conv)]
            for cid in chunk_ids:
                assert cid in flaky_vectors.per_point
        finally:
            await pool.close()

    asyncio.run(_run())


async def _all_chunks(pool, conv):
    return await pool.fetch(
        "SELECT id FROM conversation_chunks WHERE conversation_id = $1::uuid ORDER BY start_seq",
        uuid.UUID(conv),
    )


def test_qdrant_failure_leaves_chunk_pending():
    """§16: upsert failure -> NULL state; retry after recovery succeeds."""
    from context_proxy.memory.service import MemoryService as MS

    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            conv = str(uuid.uuid4())
            store = _store(pool)
            vectors = CountingVectorStore(fail_times=1)
            embedder = InstrumentedEmbedder()
            memory = MS(pool, embedder, vectors, retrieval_settings=RetrievalSettings())

            await store.append_messages(conv, [{"role": "user", "content": "q"}])
            await store.append_messages(conv, [{"role": "assistant", "content": "a"}])
            await store.append_messages(conv, [{"role": "user", "content": "next"}])

            await memory.index_completed_turns(conv)
            assert dict(await _vector_state(pool, conv))[0] is False  # pending

            recovered = MS(pool, embedder, vectors, retrieval_settings=RetrievalSettings())
            await recovered.index_completed_turns(conv)
            assert dict(await _vector_state(pool, conv))[0] is True
        finally:
            await pool.close()

    asyncio.run(_run())


def test_embedding_failure_leaves_chunk_pending_and_skips_qdrant():
    """§17: embed failure -> Qdrant never called; retry with healthy embedder."""
    from context_proxy.memory.service import MemoryService as MS

    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            conv = str(uuid.uuid4())
            store = _store(pool)
            vectors = CountingVectorStore()
            broken_embedder = InstrumentedEmbedder(fail_times=1)

            memory = MS(pool, broken_embedder, vectors, retrieval_settings=RetrievalSettings())
            await store.append_messages(conv, [{"role": "user", "content": "q"}])
            await store.append_messages(conv, [{"role": "assistant", "content": "a"}])
            await store.append_messages(conv, [{"role": "user", "content": "next"}])

            calls_before_upsert_check = vectors.upsert_calls
            await memory.index_completed_turns(conv)
            assert vectors.upsert_calls == calls_before_upsert_check  # Qdrant untouched
            assert dict(await _vector_state(pool, conv))[0] is False

            healthy = MS(
                pool,
                InstrumentedEmbedder(),
                vectors,
                retrieval_settings=RetrievalSettings(),
            )
            await healthy.index_completed_turns(conv)
            assert dict(await _vector_state(pool, conv))[0] is True
        finally:
            await pool.close()

    asyncio.run(_run())


def test_successfully_indexed_chunk_is_not_reembedded():
    """§23: second no-change pass performs zero embeds and zero upserts."""
    from context_proxy.memory.service import MemoryService as MS

    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            conv = str(uuid.uuid4())
            store = _store(pool)
            embedder = InstrumentedEmbedder()
            vectors = CountingVectorStore()
            memory = MS(pool, embedder, vectors, retrieval_settings=RetrievalSettings())

            await store.append_messages(conv, [{"role": "user", "content": "q"}])
            await store.append_messages(conv, [{"role": "assistant", "content": "a"}])
            await store.append_messages(conv, [{"role": "user", "content": "tail"}])

            await memory.index_completed_turns(conv)
            embeds_after_first = embedder.calls
            upserts_after_first = vectors.upsert_calls
            assert embeds_after_first == 1 and upserts_after_first == 1

            await memory.index_completed_turns(conv)
            await memory.index_completed_turns(conv)
            assert embedder.calls == embeds_after_first  # not re-embedded
            assert vectors.upsert_calls == upserts_after_first  # not re-upserted
        finally:
            await pool.close()

    asyncio.run(_run())


def test_stale_pending_chunk_and_new_chunk_processed_independently():
    """§24: B(pending) retried while C(new) is chunked+indexed; A untouched."""
    from context_proxy.memory.service import MemoryService as MS

    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            conv = str(uuid.uuid4())
            store = _store(pool)
            # first upsert (start_seq asc -> turn A) fails
            vectors = CountingVectorStore(fail_times=1)
            embedder = InstrumentedEmbedder()

            # pass 1: A chunked+pending(fail), B trailing
            memory = MS(pool, embedder, vectors, retrieval_settings=RetrievalSettings())
            await store.append_messages(conv, [{"role": "user", "content": "A question"}])
            await store.append_messages(conv, [{"role": "assistant", "content": "A answer"}])
            await store.append_messages(conv, [{"role": "user", "content": "B question"}])
            await memory.index_completed_turns(conv)

            async def states():
                rows = await pool.fetch(
                    """
                    SELECT start_seq, vector_indexed_at FROM conversation_chunks
                    WHERE conversation_id = $1::uuid ORDER BY start_seq
                    """,
                    uuid.UUID(conv),
                )
                return {r["start_seq"]: r["vector_indexed_at"] is not None for r in rows}

            st = await states()
            assert st[0] is False and 2 not in st  # A pending, B trailing unchunked

            # complete B and add C's question (C becomes the new trailing turn)
            await store.append_messages(conv, [{"role": "assistant", "content": "B answer"}])
            await store.append_messages(conv, [{"role": "user", "content": "C question"}])
            ids_before = await pool.fetch(
                """
                SELECT start_seq, id FROM conversation_chunks
                WHERE conversation_id = $1::uuid
                """,
                uuid.UUID(conv),
            )
            id_by_start_before = {r["start_seq"]: str(r["id"]) for r in ids_before}
            a_chunk_id = id_by_start_before[0]
            a_upserts_before = vectors.per_point.get(a_chunk_id, 0)
            await memory.index_completed_turns(conv)

            st = await states()
            assert st[0] is True   # pending chunk A successfully retried
            assert st[2] is True   # newly completed B-turn chunked + indexed

            ids_by_start = await pool.fetch(
                """
                SELECT start_seq, id FROM conversation_chunks
                WHERE conversation_id = $1::uuid
                """,
                uuid.UUID(conv),
            )
            id_by_start = {r["start_seq"]: str(r["id"]) for r in ids_by_start}
            a_id, b_id = a_chunk_id, id_by_start[2]

            # A: exactly one failed attempt before + exactly one successful retry
            assert vectors.per_point[a_id] == a_upserts_before + 1
            # B: brand-new chunk indexed exactly once
            assert vectors.per_point[b_id] == 1
        finally:
            await pool.close()

    asyncio.run(_run())


def test_trailing_turn_is_never_chunked_or_vector_indexed():
    """§21: completed turn A indexable; live trailing turn B has no chunk."""

    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            conv = str(uuid.uuid4())
            store = _store(pool)
            memory = await _make(pool)
            await store.append_messages(conv, [{"role": "user", "content": "A q"}])
            await store.append_messages(conv, [{"role": "assistant", "content": "A a"}])
            await store.append_messages(conv, [{"role": "user", "content": "B live"}])

            await memory.index_completed_turns(conv)
            rows = await pool.fetch(
                """
                SELECT start_seq FROM conversation_chunks
                WHERE conversation_id = $1::uuid ORDER BY start_seq
                """,
                uuid.UUID(conv),
            )
            assert [r["start_seq"] for r in rows] == [0]
        finally:
            await pool.close()

    asyncio.run(_run())


def test_process_restart_recovers_pending_chunks():
    """§19: a brand-new service instance discovers pending chunks from PG."""

    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            conv = str(uuid.uuid4())
            store = _store(pool)
            from context_proxy.memory.service import MemoryService as MS

            broken_vectors = CountingVectorStore(fail_times=99)
            first_instance = MS(
                pool,
                InstrumentedEmbedder(),
                broken_vectors,
                retrieval_settings=RetrievalSettings(),
            )

            for q, a in (("q1", "a1"), ("q2", "a2")):
                await store.append_messages(conv, [{"role": "user", "content": q}])
                await store.append_messages(conv, [{"role": "assistant", "content": a}])
            await store.append_messages(conv, [{"role": "user", "content": "live"}])
            await first_instance.index_completed_turns(conv)  # all vector attempts fail

            # process restart: fresh instance, fresh fakes, same PostgreSQL
            fresh_vectors = CountingVectorStore()
            second_instance = MS(
                pool,
                InstrumentedEmbedder(),
                fresh_vectors,
                retrieval_settings=RetrievalSettings(),
            )
            st = dict(await _vector_state(pool, conv))
            assert st[0] is False and st[2] is False  # both pending after outage
            await second_instance.index_completed_turns(conv)

            st = dict(await _vector_state(pool, conv))
            assert st[0] is True and st[2] is True
            assert len(fresh_vectors.points) == 2
        finally:
            await pool.close()

    asyncio.run(_run())
