from __future__ import annotations

import asyncio
import hashlib
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
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "big") % 6
                vec[bucket] += 1.0
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


class FakeOwnedClient:
    """Minimal stand-in for an owned httpx.AsyncClient (lifecycle tests)."""

    def __init__(self):
        self.aclose_called = False
        self.headers: dict[str, str] = {}

    async def aclose(self):
        self.aclose_called = True


def test_lifespan_does_not_close_injected_clients():
    from context_proxy.config import DatabaseSettings, Settings
    from context_proxy.main import create_app

    settings = Settings(_env_file=None, database=DatabaseSettings(url=MIGRATION_DSN))
    embed_fake = FakeOwnedClient()
    qdrant_fake = FakeOwnedClient()
    app = create_app(
        settings,
        llm_client=httpx.AsyncClient(base_url="http://up.test/v1"),
        embedding_client=embed_fake,
        qdrant_client=qdrant_fake,
    )
    with TestClient(app):
        pass  # lifespan runs and shuts down
    assert embed_fake.aclose_called is False
    assert qdrant_fake.aclose_called is False


def test_lifespan_closes_owned_embedding_and_qdrant_clients():
    from context_proxy.config import DatabaseSettings, Settings
    from context_proxy.main import create_app

    settings = Settings(_env_file=None, database=DatabaseSettings(url=MIGRATION_DSN))
    app = create_app(settings)
    with TestClient(app) as client:
        memory = client.app.state.memory
        assert memory is not None

    # owned clients must be closed after shutdown
    assert memory._embedder._client.is_closed
    assert memory._qdrant._client.is_closed


def test_memory_service_unavailable_returns_503():
    ctx = AppContext()
    with ctx.client as client:
        client.app.state.memory = None
        r = client.get(
            "/internal/v1/retrieval",
            params={"q": "anything", "conversation_id": str(uuid.uuid4())},
        )
        assert r.status_code == 503


def test_internal_api_validation_errors():
    ctx = AppContext()
    with ctx.client as client:
        ctx.install_memory()
        conv = str(uuid.uuid4())

        cases = [
            (
                "/internal/v1/memories",
                {"kind": "nonsense", "content": "x", "conversation_id": conv},
                422,
            ),
            (
                "/internal/v1/memories",
                {"kind": "fact", "content": "", "conversation_id": conv},
                422,
            ),
            (
                "/internal/v1/memories",
                {"kind": "fact", "content": "x", "conversation_id": conv, "importance": 1.5},
                422,
            ),
            (
                "/internal/v1/memories",
                {"kind": "fact", "content": "x", "conversation_id": "not-a-uuid"},
                400,
            ),
            (
                "/internal/v1/memories",
                {
                    "kind": "fact",
                    "content": "x",
                    "conversation_id": conv,
                    "supersedes": "bad",
                },
                400,
            ),
        ]
        for url, body, expected in cases:
            r = client.post(url, json=body)
            assert r.status_code == expected, (url, body, r.status_code)

        r = client.post(f"/internal/v1/memories/{'z' * 8}/supersede", json={})
        assert r.status_code == 400

        r = client.get(
            "/internal/v1/retrieval",
            params={"q": "q", "conversation_id": conv, "limit": 0},
        )
        assert r.status_code == 422

        r = client.get(
            "/internal/v1/retrieval", params={"q": "", "conversation_id": conv}
        )
        assert r.status_code == 422

        r = client.get("/internal/v1/retrieval", params={"q": "x", "conversation_id": "zz"})
        assert r.status_code == 400


class SlowHashingEmbedder(HashingEmbedder):
    def __init__(self, delay: float):
        self.delay = delay
        self.cancelled = False

    async def embed(self, texts):
        import asyncio

        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return await super().embed(texts)


def test_index_timeout_leaves_chunks_retryable():
    """§18/§26: timeout -> response OK, chunk pending, cleanup happens,
    later retry with healthy provider marks it indexed."""

    from context_proxy.config import DatabaseSettings, Settings
    from context_proxy.main import create_app
    from context_proxy.memory.service import MemoryService

    settings = Settings(_env_file=None, database=DatabaseSettings(url=MIGRATION_DSN))
    settings.memory.index_timeout_seconds = 0.05

    app = create_app(
        settings,
        llm_client=httpx.AsyncClient(
            base_url=str(settings.inference.base_url),
            transport=httpx.MockTransport(_upstream),
        ),
    )
    client = TestClient(app)
    conv = str(uuid.uuid4())

    with client:
        store = client.app.state.store
        pool = client.app.state.database.pool
        slow_embedder = SlowHashingEmbedder(delay=0.5)
        client.app.state.memory = MemoryService(
            pool,
            slow_embedder,
            RecordingVectorStore(),
            retrieval_settings=client.app.state.settings.retrieval,
        )

        async def seed(messages):
            await store.append_messages(conv, messages)

        async def states():
            rows = await pool.fetch(
                """
                SELECT start_seq, vector_indexed_at FROM conversation_chunks
                WHERE conversation_id = $1::uuid ORDER BY start_seq
                """,
                uuid.UUID(conv),
            )
            return [(r["start_seq"], r["vector_indexed_at"]) for r in rows]

        def run_async(coro):
            return client.portal.start_task_soon(lambda: coro).result()

        run_async(seed([{"role": "user", "content": "alpha question"}]))
        r1 = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "alpha question"}],
                "conversation_id": conv,
            },
        )
        assert r1.status_code == 200

        # complete turn 1 (assistant already persisted by r1); the next
        # auto-index pass times out
        run_async(seed([{"role": "user", "content": "beta question"}]))
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

        st = run_async(states())
        assert len(st) == 1 and st[0][0] == 0
        assert st[0][1] is None  # timed out -> still pending
        assert slow_embedder.cancelled is True  # cancellation cleaned up

    # fresh process-style retry: healthy provider marks the chunk indexed
    app2 = create_app(
        Settings(_env_file=None, database=DatabaseSettings(url=MIGRATION_DSN)),
        llm_client=httpx.AsyncClient(
            base_url="http://up.test/v1",
            transport=httpx.MockTransport(_upstream),
        ),
    )
    with TestClient(app2) as client2:
        pool = client2.app.state.database.pool
        fast_memory = MemoryService(
            pool,
            HashingEmbedder(),
            RecordingVectorStore(),
            retrieval_settings=client2.app.state.settings.retrieval,
        )

        async def reindex():
            return await fast_memory.index_completed_turns(conv)

        fut = client2.portal.start_task_soon(lambda: reindex())
        created = fut.result()

        async def states():
            rows = await pool.fetch(
                """
                SELECT start_seq, vector_indexed_at FROM conversation_chunks
                WHERE conversation_id = $1::uuid ORDER BY start_seq
                """,
                uuid.UUID(conv),
            )
            return [(r["start_seq"], r["vector_indexed_at"] is not None) for r in rows]

        fut2 = client2.portal.start_task_soon(lambda: states())
        st = fut2.result()
    assert created == 0
    assert st == [(0, True)]
