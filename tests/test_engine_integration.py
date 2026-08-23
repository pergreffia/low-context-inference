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
from context_proxy.context.engine import ContextAssemblyEngine
from context_proxy.conversation.store import PostgresConversationStore
from context_proxy.memory.embeddings import OpenAICompatibleEmbeddingProvider
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
            plan = _engine().build(
                messages=messages,
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
                plan = _engine().build(
                    messages=messages, retrieved=retrieved, conversation_id=conv
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
            plan = _engine().build(
                messages=inbound, retrieved=retrieved, conversation_id=conv
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
                    raise RuntimeError("embedding endpoint down")

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
            plan = _engine().build(
                messages=messages, retrieved=retrieved, conversation_id=conv
            )
            blob = "\n".join(m.get("content") or "" for m in plan.messages)
            assert "never deploy on fridays" in blob
        finally:
            await pool.close()

    asyncio.run(_run())
