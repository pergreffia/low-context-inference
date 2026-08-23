from __future__ import annotations

import asyncio
import json
import os
import uuid

import asyncpg
import httpx
import pytest
from fastapi.testclient import TestClient

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


class HashingEmbedder:
    """Deterministic offline embedder (tests only): 6-dim token-hash vectors."""

    async def embed(self, texts):
        out = []
        for text in texts:
            vec = [0.0] * 6
            for token in text.lower().split():
                vec[hash(token) % 6] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out


class RecordingVectorStore:
    def __init__(self):
        self.points: list[dict] = []

    async def upsert(self, points, vector_size):
        self.points.extend(points)

    async def search(self, vector, limit, conversation_id=None):
        return []


def _upstream(request: httpx.Request) -> httpx.Response:
    payload = json.loads(request.content)
    if payload.get("stream") is True:

        async def agen():
            yield b'data: {"choices":[{"delta":{"content":"streamed reply"}}]}\n\n'
            yield b"data: [DONE]\n\n"

        return httpx.Response(
            200, content=agen(), headers={"content-type": "text/event-stream"}
        )
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-9",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "reply one"},
                    "finish_reason": "stop",
                }
            ],
        },
    )


async def _await_coro(coro):
    return await coro


class AppContext:
    """App whose store/memory live inside the TestClient portal loop."""

    def __init__(self):
        from context_proxy.config import DatabaseSettings, Settings
        from context_proxy.main import create_app

        self.settings = Settings(
            _env_file=None,
            database=DatabaseSettings(url=MIGRATION_DSN),
        )
        self.app = create_app(
            self.settings,
            llm_client=httpx.AsyncClient(
                base_url=str(self.settings.inference.base_url),
                transport=httpx.MockTransport(_upstream),
            ),
        )
        self.client = TestClient(self.app)

    def run_async(self, coro):
        """Run a coroutine on the portal loop (where the PG pool lives)."""
        import functools

        return self.client.portal.start_task_soon(
            functools.partial(_await_coro, coro)
        ).result()

    def install_memory(self):
        """Replace the lifespan-built memory service with offline fakes."""
        from context_proxy.memory.service import MemoryService

        pool = self.app.state.database.pool
        assert pool is not None
        self.pool = pool
        self.store = self.app.state.store
        memory = MemoryService(pool, HashingEmbedder(), RecordingVectorStore())
        self.memory = memory
        self.app.state.memory = memory
        return memory


def test_internal_memory_api_and_retrieval_e2e():
    ctx = AppContext()
    with ctx.client as client:
        ctx.install_memory()
        conv = str(uuid.uuid4())

        # create via internal API
        r = client.post(
            "/internal/v1/memories",
            json={
                "kind": "decision",
                "content": "the api gateway uses token auth",
                "conversation_id": conv,
                "importance": 0.9,
            },
        )
        assert r.status_code == 200
        memory_id = r.json()["id"]

        # supersede via second creation referencing the first
        r2 = client.post(
            "/internal/v1/memories",
            json={
                "kind": "decision",
                "content": "the api gateway uses jwt auth",
                "conversation_id": conv,
                "supersedes": memory_id,
            },
        )
        assert r2.status_code == 200

        # retrieval excludes the superseded decision (supersession filtering)
        r3 = client.get(
            "/internal/v1/retrieval",
            params={"q": "api gateway auth", "conversation_id": conv},
        )
        assert r3.status_code == 200
        ids = {i["id"] for i in r3.json()["items"]}
        assert r2.json()["id"] in ids
        assert memory_id not in ids

        # unknown supersedes target -> 404
        r4 = client.post(
            "/internal/v1/memories",
            json={
                "kind": "fact",
                "content": "orphan",
                "conversation_id": conv,
                "supersedes": str(uuid.uuid4()),
            },
        )
        assert r4.status_code == 404


def test_supersede_endpoint_and_isolation():
    ctx = AppContext()
    with ctx.client as client:
        ctx.install_memory()
        conv_a = str(uuid.uuid4())
        conv_b = str(uuid.uuid4())

        task_id = client.post(
            "/internal/v1/memories",
            json={"kind": "task", "content": "todo item", "conversation_id": conv_a},
        ).json()["id"]

        r = client.post(
            f"/internal/v1/memories/{task_id}/supersede",
            json={"status": "resolved"},
        )
        assert r.status_code == 200
        items = client.get(
            "/internal/v1/retrieval",
            params={"q": "todo item", "conversation_id": conv_a},
        ).json()["items"]
        assert task_id not in {i["id"] for i in items}

        # conversation isolation: B never sees A's memories
        assert (
            client.get(
                "/internal/v1/retrieval",
                params={"q": "todo item", "conversation_id": conv_b},
            ).json()["items"]
            == []
        )


def test_indexing_endpoint_and_chunk_in_retrieval():
    ctx = AppContext()
    with ctx.client as client:
        ctx.install_memory()
        conv = str(uuid.uuid4())

        store_sync = client.app.state.store

        seed = store_sync.append_messages(
            conv,
            [
                {"role": "user", "content": "tell me about the api gateway"},
                {"role": "assistant", "content": "it uses jwt auth"},
                {"role": "user", "content": "next topic please"},
            ],
        )
        ctx.run_async(seed)

        r = client.post(f"/internal/v1/conversations/{conv}/index")
        assert r.status_code == 200
        assert r.json()["chunks_created"] == 1

        r2 = client.get(
            "/internal/v1/retrieval",
            params={"q": "api gateway", "conversation_id": conv},
        )
        kinds = {i["item_type"] for i in r2.json()["items"]}
        assert "chunk" in kinds

        again = client.post(f"/internal/v1/conversations/{conv}/index")
        assert again.json()["chunks_created"] == 0  # idempotent


def test_chat_completion_auto_indexes_completed_turns():
    ctx = AppContext()
    with ctx.client as client:
        ctx.install_memory()
        pool = ctx.pool
        conv = str(uuid.uuid4())
        store_sync = client.app.state.store

        def seed(messages):
            return store_sync.append_messages(conv, messages)

        ctx.run_async(seed([{"role": "user", "content": "alpha question"}]))

        # response completes turn 1; auto-index runs but turn is trailing...
        r1 = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "alpha question"}],
                "conversation_id": conv,
            },
        )
        assert r1.status_code == 200

        async def count_chunks() -> int:
            rows = await pool.fetch(
                "SELECT id FROM conversation_chunks WHERE conversation_id = $1::uuid",
                conv,
            )
            return len(rows)

        assert ctx.run_async(count_chunks()) == 0

        # ...until it becomes complete during the next request's inbound sync
        ctx.run_async(seed([{"role": "user", "content": "beta question"}]))
        r2 = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [
                    {"role": "user", "content": "alpha question"},
                    {"role": "assistant", "content": "reply one"},
                    {"role": "user", "content": "beta question"},
                ],
                "conversation_id": conv,
            },
        )
        assert r2.status_code == 200

        async def seqs() -> list[int]:
            rows = await pool.fetch(
                "SELECT start_seq FROM conversation_chunks WHERE conversation_id = $1::uuid",
                conv,
            )
            return [r["start_seq"] for r in rows]

        assert ctx.run_async(seqs()) == [0]
