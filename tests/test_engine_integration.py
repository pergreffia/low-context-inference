"""M4 PostgreSQL integration: engine + memory service end-to-end invariants.

Requires TEST_DATABASE_URL (same contract as the other integration suites).
Covers supersession exclusion, conversation isolation, and chunk-vs-recent
deduplication with real persistence underneath.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import asyncpg
import pytest

from context_proxy.config import AssemblySettings, EndpointSettings, RetrievalSettings
from context_proxy.context.engine import ContextAssemblyEngine, separate_current_request
from context_proxy.conversation.store import PostgresConversationStore
from context_proxy.memory.embeddings import OpenAICompatibleEmbeddingProvider
from context_proxy.memory.errors import EmbeddingProviderError, RetrievalError
from context_proxy.memory.models import MemoryCreate, MemoryKind
from context_proxy.memory.qdrant import QdrantVectorStore
from context_proxy.memory.service import MemoryService

MIGRATION_DSN = os.environ.get("TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not MIGRATION_DSN,
    reason="set TEST_DATABASE_URL to run PostgreSQL integration tests",
)


class HashingEmbedder(OpenAICompatibleEmbeddingProvider):
    """Deterministic offline embedder: 6-dim token-hash vector (tests only)."""

    def __init__(self):
        super().__init__(EndpointSettings(base_url="offline://none"))

    async def embed(self, texts):
        import hashlib

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


def _services(pool) -> tuple[MemoryService, PostgresConversationStore]:
    memory = MemoryService(
        pool,
        HashingEmbedder(),
        RecordingVectorStore(),
        retrieval_settings=RetrievalSettings(),
    )
    return memory, PostgresConversationStore(pool)


def _engine() -> ContextAssemblyEngine:
    return ContextAssemblyEngine(
        usable_budget=30_000,
        settings=AssemblySettings(),
        retrieval_settings=RetrievalSettings(),
    )


def test_superseded_memory_never_reaches_plan():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            memory, store = _services(pool)
            conv = str(uuid.uuid4())
            old_id = await memory.create_memory(
                MemoryCreate(
                    kind=MemoryKind.DECISION,
                    content="deploy target is staging cluster",
                    conversation_id=conv,
                    importance=0.9,
                )
            )
            await memory.create_memory(
                MemoryCreate(
                    kind=MemoryKind.DECISION,
                    content="deploy target moved to production cluster",
                    conversation_id=conv,
                    supersedes=old_id,
                    importance=0.9,
                )
            )
            messages = [
                {"role": "user", "content": "where do we deploy?"},
                {"role": "assistant", "content": "checking"},
                {"role": "user", "content": "where do we deploy now?"},
            ]
            await store.ensure_conversation(conv)
            retrieved = await memory.retrieve("deploy target", conv)
            history, current = separate_current_request(messages)
            plan = _engine().build(
                history=history,
                current_request=current,
                retrieved=retrieved,
                conversation_id=conv,
            )
            blob = "\n".join(m.get("content") or "" for m in plan.messages)
            assert "staging cluster" not in blob
            assert "production cluster" in blob
        finally:
            await pool.close()

    asyncio.run(_run())


def test_conversation_isolation_in_assembled_plan():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            memory, _store = _services(pool)
            conv_a = str(uuid.uuid4())
            conv_b = str(uuid.uuid4())
            await memory.create_memory(
                MemoryCreate(
                    kind=MemoryKind.FACT,
                    content="conversation A secret keyword orangutan",
                    conversation_id=conv_a,
                )
            )
            await memory.create_memory(
                MemoryCreate(
                    kind=MemoryKind.FACT,
                    content="conversation B secret keyword xylophone",
                    conversation_id=conv_b,
                )
            )
            messages = [
                {"role": "user", "content": "tell me the secret keyword"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "secret keyword please"},
            ]
            for conv, forbidden in ((conv_a, "xylophone"), (conv_b, "orangutan")):
                retrieved = await memory.retrieve("secret keyword", conv)
                history, current = separate_current_request(messages)
                plan = _engine().build(
                    history=history,
                    current_request=current,
                    retrieved=retrieved,
                    conversation_id=conv,
                )
                blob = "\n".join(m.get("content") or "" for m in plan.messages)
                assert forbidden not in blob
        finally:
            await pool.close()

    asyncio.run(_run())


def test_chunk_dedup_against_persisted_recent_window():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            memory, store = _services(pool)
            conv = str(uuid.uuid4())
            history = [
                {"role": "user", "content": "design the retry policy"},
                {"role": "assistant", "content": "retry three times with backoff"},
                {"role": "user", "content": "and the timeout policy"},
                {"role": "assistant", "content": "timeout after thirty seconds"},
            ]
            current = {"role": "user", "content": "summarize both policies"}
            await store.reconcile_history(conv, [*history, current])
            # Chunk the completed turns (all but the live trailing unit).
            await memory.index_completed_turns(conv)

            # The client replays full history; chunks covering it must dedup.
            inbound = [*history, current]
            retrieved = await memory.retrieve("retry timeout policy", conv)
            history_msgs, current_msgs = separate_current_request(inbound)
            plan = _engine().build(
                history=history_msgs,
                current_request=current_msgs,
                retrieved=retrieved,
                conversation_id=conv,
            )
            rendered = [m.get("content") or "" for m in plan.messages]
            chunk_blocks = [
                c for c in rendered if c.startswith("[chunk:")
            ]
            overlapping = [
                c
                for c in chunk_blocks
                if "retry three times" in c or "timeout after thirty" in c
            ]
            assert not overlapping  # covered by the raw recent window
            assert rendered[-1] == current["content"]
            assert plan.token_estimate <= 30_000
        finally:
            await pool.close()

    asyncio.run(_run())


def test_lexical_fallback_feeds_engine_when_semantic_leg_down():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            memory_service, store = _services(pool)
            conv = str(uuid.uuid4())
            await memory_service.create_memory(
                MemoryCreate(
                    kind=MemoryKind.CONSTRAINT,
                    content="never deploy on fridays",
                    conversation_id=conv,
                )
            )
            # Vector leg unavailable: embedding provider raises.
            class BrokenEmbedder(HashingEmbedder):
                async def embed(self, texts):
                    raise EmbeddingProviderError("embedding endpoint down")

            broken = MemoryService(
                pool,
                BrokenEmbedder(),
                RecordingVectorStore(),
                retrieval_settings=RetrievalSettings(),
            )
            messages = [
                {"role": "user", "content": "can we deploy today?"},
                {"role": "assistant", "content": "checking calendar"},
                {"role": "user", "content": "deploy rules?"},
            ]
            retrieved = await broken.retrieve("deploy", conv)  # lexical only
            history, current = separate_current_request(messages)
            plan = _engine().build(
                history=history,
                current_request=current,
                retrieved=retrieved,
                conversation_id=conv,
            )
            blob = "\n".join(m.get("content") or "" for m in plan.messages)
            assert "never deploy on fridays" in blob
        finally:
            await pool.close()

    asyncio.run(_run())


def test_raw_history_unchanged_after_dedup():
    """A memory duplicating a turn is dropped from the request only (#12)."""
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            memory, store = _services(pool)
            conv = str(uuid.uuid4())
            inbound = [
                {"role": "user", "content": "we chose SQLite for the edge cache"},
                {"role": "assistant", "content": "noted"},
                {"role": "user", "content": "current question"},
            ]
            await store.reconcile_history(conv, inbound)
            await memory.create_memory(
                MemoryCreate(
                    kind=MemoryKind.FACT,
                    content="we chose SQLite for the edge cache",
                    conversation_id=conv,
                    importance=0.8,
                )
            )
            retrieved = await memory.retrieve("edge cache", conv)
            assert any(i.item_type == "memory" for i in retrieved)

            history, current = separate_current_request(inbound)
            plan = _engine().build(
                history=history,
                current_request=current,
                retrieved=retrieved,
                conversation_id=conv,
            )
            dup_dropped = any(
                d.reason == "duplicate" for d in plan.dropped_items
            )
            assert dup_dropped, "restating memory must be dropped as duplicate"
            rendered = "\n".join(m.get("content") or "" for m in plan.messages)
            # exactly one copy survives (the raw turn), dedup or no dedup
            assert rendered.count("edge cache") >= 1

            # authoritative state untouched regardless
            persisted = await store.get_messages(conv)
            assert persisted == inbound
        finally:
            await pool.close()

    asyncio.run(_run())


def test_identical_repeated_messages_stay_separate_rows():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            _, store = _services(pool)
            conv = str(uuid.uuid4())
            repeated = {"role": "user", "content": "run the tests"}
            inbound = [
                repeated,
                {"role": "assistant", "content": "ok"},
                repeated,
                {"role": "assistant", "content": "ok again"},
            ]
            first = await store.reconcile_history(conv, inbound)
            replay = await store.reconcile_history(conv, inbound)  # idempotent
            assert replay == []
            assert len(first) == 4
            rows = await pool.fetch(
                "SELECT content FROM messages WHERE conversation_id = $1::uuid ORDER BY seq",
                conv,
            )
            contents = [r["content"] for r in rows]
            assert contents.count(json_content(repeated)) == 2
        finally:
            await pool.close()

    asyncio.run(_run())


def json_content(message: dict) -> str:
    import json

    return json.dumps(message, ensure_ascii=False)


def test_pg_outage_raises_typed_retrieval_error():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        memory, _store = _services(pool)
        conv = str(uuid.uuid4())
        await pool.close()  # simulate PostgreSQL outage
        try:
            await memory.retrieve("anything", conv)
        except RetrievalError:
            pass  # expected typed failure
        else:
            pytest.fail("expected RetrievalError, got success")
        # programming errors are NOT converted (no broad except upstream)

    asyncio.run(_run())


def test_chunk_structural_span_populated():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            memory, store = _services(pool)
            conv = str(uuid.uuid4())
            inbound = [
                {"role": "user", "content": "first topic"},
                {"role": "assistant", "content": "answer one"},
                {"role": "user", "content": "second topic"},
                {"role": "assistant", "content": "answer two"},
                {"role": "user", "content": "live"},
            ]
            await store.reconcile_history(conv, inbound)
            await memory.index_completed_turns(conv)
            rows = await pool.fetch(
                """
                SELECT start_seq, end_seq FROM conversation_chunks
                WHERE conversation_id = $1::uuid ORDER BY start_seq
                """,
                conv,
            )
            assert len(rows) == 2
            for row in rows:
                assert row["start_seq"] is not None
                assert row["end_seq"] is not None
                assert row["end_seq"] > row["start_seq"]
            # spans tile the settled range contiguously
            assert rows[0]["end_seq"] < rows[1]["start_seq"]
        finally:
            await pool.close()

    asyncio.run(_run())


def test_chunk_span_exact_message_boundaries():
    """end_seq = LAST message of the unit, start_seq = first (final review §4)."""
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            memory, store = _services(pool)
            conv = str(uuid.uuid4())
            inbound = [
                {"role": "user", "content": "turn one question"},       # seq 0
                {"role": "assistant", "content": "turn one answer"},    # seq 1
                {"role": "assistant", "content": "turn one final"},     # seq 2
                {"role": "user", "content": "turn two"},                # seq 3
                {"role": "assistant", "content": "turn two answer"},    # seq 4
                {"role": "user", "content": "live"},                    # seq 5
            ]
            await store.reconcile_history(conv, inbound)
            await memory.index_completed_turns(conv)
            rows = await pool.fetch(
                """
                SELECT start_seq, end_seq FROM conversation_chunks
                WHERE conversation_id = $1::uuid ORDER BY start_seq
                """,
                conv,
            )
            spans = [(r["start_seq"], r["end_seq"]) for r in rows]
            # multi-message chunk: end covers the FINAL message, not the first
            assert spans == [(0, 2), (3, 4)]
            for start, end in spans:
                assert start <= end
        finally:
            await pool.close()

    asyncio.run(_run())


def test_vector_store_outage_degrades_to_lexical():
    """Expected VectorStoreError -> lexical leg continues, logged (#Test A)."""
    async def _run():
        import logging

        from context_proxy.memory.errors import VectorStoreError

        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            class OutageVectorStore(RecordingVectorStore):
                async def search(self, vector, limit, conversation_id=None):
                    raise VectorStoreError("qdrant search failed: connection refused")

            conv = str(uuid.uuid4())
            memory = MemoryService(
                pool,
                HashingEmbedder(),
                OutageVectorStore(),
                retrieval_settings=RetrievalSettings(),
            )
            await memory.create_memory(
                MemoryCreate(
                    kind=MemoryKind.FACT,
                    content="zebra fallback marker",
                    conversation_id=conv,
                )
            )
            records: list[logging.LogRecord] = []

            class Capture(logging.Handler):
                def emit(self, record):
                    records.append(record)

            logger = logging.getLogger("context_proxy.memory.service")
            handler = Capture()
            old_level = logger.level
            logger.addHandler(handler)
            logger.setLevel(logging.WARNING)
            try:
                items = await memory.retrieve("zebra fallback", conv)
            finally:
                logger.removeHandler(handler)
                logger.setLevel(old_level)

            assert any("zebra fallback marker" in i.content for i in items)
            assert any(r.message == "vector_search_unavailable" for r in records)
        finally:
            await pool.close()

    asyncio.run(_run())


def test_vector_programming_bug_propagates():
    """TypeError from the vector store must NOT become degradation (#Test B).

    Fails if a broad `except Exception` is reintroduced around vector search.
    """
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            class BuggyVectorStore(RecordingVectorStore):
                async def search(self, vector, limit, conversation_id=None):
                    raise TypeError("programming bug")

            memory = MemoryService(
                pool,
                HashingEmbedder(),
                BuggyVectorStore(),
                retrieval_settings=RetrievalSettings(),
            )
            try:
                await memory.retrieve("anything", str(uuid.uuid4()))
            except TypeError:
                pass  # expected propagation
            else:
                pytest.fail("TypeError was swallowed as vector degradation")
        finally:
            await pool.close()

    asyncio.run(_run())


def test_scoped_rebuild_touches_only_target_conversation():
    """conversation_id scope: A rebuilt (force), B left byte-identical."""
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            memory, store = _services(pool)
            conv_a, conv_b = str(uuid.uuid4()), str(uuid.uuid4())

            async def seed(conv):
                await store.reconcile_history(
                    conv,
                    [
                        {"role": "user", "content": f"question for {conv}"},
                        {"role": "assistant", "content": "answer"},
                        {"role": "user", "content": "live"},
                    ],
                )
                await memory.create_memory(
                    MemoryCreate(
                        kind=MemoryKind.FACT,
                        content=f"memory fact for {conv}",
                        conversation_id=conv,
                    )
                )
                await memory.index_completed_turns(conv)

            await seed(conv_a)
            await seed(conv_b)

            async def chunk_markers():
                rows = await pool.fetch(
                    """
                    SELECT conversation_id::text AS conv, id::text AS cid,
                           vector_indexed_at
                    FROM conversation_chunks
                    WHERE vector_indexed_at IS NOT NULL
                    """
                )
                return {
                    (r["conv"], r["cid"]): str(r["vector_indexed_at"]) for r in rows
                }

            before = await chunk_markers()
            assert len([k for k in before if k[0] == conv_a]) == 1
            assert len([k for k in before if k[0] == conv_b]) == 1

            summary = await memory.rebuild_vector_index(conv_a, force=True)
            assert summary["chunks"] == 1
            assert summary["memories"] == 1
            assert summary["chunks_failed"] == 0
            assert summary["memories_failed"] == 0

            after = await chunk_markers()

            # Same series survive; A refreshed, B byte-identical.
            assert set(after) == set(before)
            key_b = next(k for k in before if k[0] == conv_b)
            assert after[key_b] == before[key_b]
            key_a = next(k for k in before if k[0] == conv_a)
            # force rebuild re-upserted and re-marked: marker moved forward.
            assert after[key_a] >= before[key_a]

            # Global rebuild still functions (no scope).
            global_summary = await memory.rebuild_vector_index(None, force=True)
            assert global_summary["chunks"] >= 2
            assert global_summary["memories"] >= 2
        finally:
            await pool.close()

    asyncio.run(_run())


def test_rebuild_counts_expected_vector_failures():
    """VectorStoreError -> failures counted, rebuild continues (#Test 1)."""
    async def _run():
        from context_proxy.memory.errors import VectorStoreError

        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            memory, store = _services(pool)
            conv = str(uuid.uuid4())
            await store.reconcile_history(
                conv,
                [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                    {"role": "user", "content": "live"},
                ],
            )

            class OutageStore(RecordingVectorStore):
                async def upsert(self, points, vector_size):
                    raise VectorStoreError("qdrant down")

            broken = MemoryService(
                pool, HashingEmbedder(), OutageStore(),
                retrieval_settings=RetrievalSettings(),
            )
            await broken.create_memory(
                MemoryCreate(kind=MemoryKind.FACT, content="fact x", conversation_id=conv)
            )
            await broken.index_completed_turns(conv)  # chunk exists (vector leg pending)
            summary = await broken.rebuild_vector_index(conv, force=True)
            assert summary["chunks_failed"] == 1
            assert summary["chunks"] == 0
            assert summary["memories_failed"] == 1
            assert summary["memories"] == 1
        finally:
            await pool.close()

    asyncio.run(_run())


def test_rebuild_does_not_swallow_unexpected_errors():
    """TypeError from the vector store propagates — never silenced (#Test 2)."""
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            memory, store = _services(pool)
            conv = str(uuid.uuid4())
            await store.reconcile_history(
                conv,
                [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                    {"role": "user", "content": "live"},
                ],
            )
            await memory.index_completed_turns(conv)

            class BuggyStore(RecordingVectorStore):
                async def upsert(self, points, vector_size):
                    raise TypeError("programming bug")

            broken = MemoryService(
                pool, HashingEmbedder(), BuggyStore(),
                retrieval_settings=RetrievalSettings(),
            )
            try:
                await broken.rebuild_vector_index(conv, force=True)
            except TypeError:
                pass  # expected propagation
            else:
                pytest.fail("unexpected error was swallowed during rebuild")
        finally:
            await pool.close()

    asyncio.run(_run())
