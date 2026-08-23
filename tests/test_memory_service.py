from __future__ import annotations

import asyncio
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
                vec[hash(token) % 6] += 1.0
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
