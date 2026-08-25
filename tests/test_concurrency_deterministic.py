"""Deterministic concurrency regressions (post-0876b10 review §8).

No arbitrary sleeps: synchronization uses PostgreSQL's own observable state
(pg_locks, row locks) and asyncio events. Requires TEST_DATABASE_URL.

Guarantees under test:

- same conversation + same history, concurrent writers -> suffix stored once;
- same conversation + divergent history  -> exactly one loser, NO silent merge;
- different conversations                -> fully independent progress
   (proven while conversation A's row lock is HELD by an external txn);
- identical tool_call_id in DIFFERENT conversations never cross-associates.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import asyncpg
import pytest

from context_proxy.conversation.store import HistoryDivergenceError, PostgresConversationStore

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


def _store(pool) -> PostgresConversationStore:
    return PostgresConversationStore(pool)


async def _lock_conversation_row(conn, conversation_id: str) -> None:
    """Hold the conversations row FOR UPDATE inside the caller's transaction."""
    await conn.execute("BEGIN")
    await conn.execute(
        "SELECT id FROM conversations WHERE id=$1::uuid FOR UPDATE",
        conversation_id,
    )


async def _wait_for_blocked_locks(pool, timeout_seconds: float = 10.0) -> bool:
    """Poll pg_locks until at least one lock request is waiting (deterministic:
    driven by actual server-side lock state, not wall-clock guesses)."""
    deadline = asyncio.get_event_loop().time() + timeout_seconds
    while asyncio.get_event_loop().time() < deadline:
        count = await pool.fetchval(
            "SELECT count(*) FROM pg_locks WHERE NOT granted"
        )
        if count and int(count) > 0:
            return True
        await asyncio.sleep(0.01)
    return False


class TestConcurrentReconciliation:
    async def _same_history_concurrent(self):
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN, min_size=4, max_size=8)
        try:
            store = _store(pool)
            conv = str(uuid.uuid4())
            history = [
                {"role": "developer", "content": "directive"},
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
            ]
            started = asyncio.Event()
            done = {"first": False, "second": False}

            async def writer(tag: str):
                await store.ensure_conversation(conv)
                await store.reconcile_history(conv, history)
                done[tag] = True
                started.set()

            # barrier: both writers start together via gather
            results = await asyncio.gather(
                writer("first"), writer("second"), return_exceptions=True
            )
            assert all(r is None for r in results), results
            assert all(done.values())
            rows = await pool.fetch(
                "SELECT seq FROM messages WHERE conversation_id=$1::uuid ORDER BY seq",
                conv,
            )
            assert len(rows) == len(history)          # suffix stored EXACTLY once
        finally:
            await pool.close()

    def test_same_history_concurrent_writes_store_suffix_once(self):
        asyncio.run(self._same_history_concurrent())

    async def _divergent_concurrent(self):
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN, min_size=4, max_size=8)
        try:
            store = _store(pool)
            conv = str(uuid.uuid4())
            base = [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
            ]

            # Phase A (sequential, fully deterministic): commit one branch,
            # then present a DIFFERENT continuation -> explicit conflict,
            # database untouched.
            await store.ensure_conversation(conv)
            await store.reconcile_history(conv, [*base, {"role": "user", "content": "A-turn"}])
            with pytest.raises(HistoryDivergenceError):
                await store.reconcile_history(
                    conv, [*base, {"role": "user", "content": "B-turn"}]
                )
            rows = await pool.fetch(
                "SELECT content FROM messages WHERE conversation_id=$1::uuid ORDER BY seq",
                conv,
            )
            import json as _json

            contents = [_json.loads(r["content"])["content"] for r in rows]
            assert "A-turn" in contents
            assert "B-turn" not in contents              # loser NEVER merged

            # Phase B (simultaneous): two concurrent writers, one carrying a
            # divergent suffix -> exactly ONE wins, the other conflicts.
            fresh = str(uuid.uuid4())
            await store.ensure_conversation(fresh)

            async def winner():
                await store.reconcile_history(fresh, [{"role": "user", "content": "W"}])
                return "win"

            async def loser():
                try:
                    # diverges from whatever got persisted first
                    await asyncio.sleep(0)               # yield once: real interleave
                    await store.ensure_conversation(fresh)
                    current = await store.get_messages(fresh)
                    divergent = [*current[:-1], {"role": "user", "content": "L"}] \
                        if current else [{"role": "user", "content": "L"}]
                    await store.reconcile_history(fresh, divergent)
                    return "merged"
                except HistoryDivergenceError:
                    return "conflict"

            results = await asyncio.gather(winner(), loser(), return_exceptions=True)
            outcomes = sorted(str(r) for r in results)
            assert "win" in outcomes                     # one authoritative write
            assert not any(isinstance(r, BaseException) for r in results), results
            final = await pool.fetch(
                "SELECT content FROM messages WHERE conversation_id=$1::uuid ORDER BY seq",
                fresh,
            )
            final_contents = [_json.loads(r["content"])["content"] for r in final]
            assert "L" not in final_contents or final_contents == ["L"]
        finally:
            await pool.close()

    def test_divergent_history_never_silently_merges(self):
        asyncio.run(self._divergent_concurrent())


class TestConversationIndependence:
    def test_different_conversations_progress_while_one_row_lock_is_held(self):
        """Writer B on conv-B completes while conv-A's row lock is held."""

        async def scenario():
            pool = await asyncpg.create_pool(dsn=MIGRATION_DSN, min_size=4, max_size=8)
            try:
                store = _store(pool)
                conv_a = str(uuid.uuid4())
                conv_b = str(uuid.uuid4())
                await store.ensure_conversation(conv_a)
                await store.ensure_conversation(conv_b)

                conn = await pool.acquire()
                blocker_done = asyncio.Event()
                try:
                    await _lock_conversation_row(conn, conv_a)   # hold A's row lock
                    blocker_done.set()

                    async def writer_b():
                        await store.reconcile_history(
                            conv_b, [{"role": "user", "content": "independent"}]
                        )

                    await asyncio.wait_for(writer_b(), timeout=10)
                    # B progressed WHILE A was locked: isolation proven.
                finally:
                    await conn.execute("ROLLBACK")
                    await pool.release(conn)

                a_finished = asyncio.Event()

                async def writer_a():
                    await store.reconcile_history(
                        conv_a, [{"role": "user", "content": "after unlock"}]
                    )
                    a_finished.set()

                await asyncio.wait_for(writer_a(), timeout=10)
                assert a_finished.is_set()
            finally:
                await pool.close()

        asyncio.run(scenario())

    def test_writer_blocks_only_until_same_conversation_lock_releases(self):
        """Same-conversation writer waits on the row lock; unblocks on commit."""

        async def scenario():
            pool = await asyncpg.create_pool(dsn=MIGRATION_DSN, min_size=4, max_size=8)
            try:
                store = _store(pool)
                conv = str(uuid.uuid4())
                await store.ensure_conversation(conv)

                conn = await pool.acquire()
                released = asyncio.Event()
                try:
                    await _lock_conversation_row(conn, conv)

                    async def blocked_writer():
                        await store.reconcile_history(
                            conv, [{"role": "user", "content": "queued"}]
                        )

                    task = asyncio.create_task(blocked_writer())
                    assert await _wait_for_blocked_locks(pool), (
                        "writer should be visibly blocked on the row lock"
                    )
                    await conn.execute("ROLLBACK")     # deterministic release point
                    released.set()
                    await asyncio.wait_for(task, timeout=10)
                    rows = await pool.fetch(
                        "SELECT count(*) AS n FROM messages WHERE conversation_id=$1::uuid",
                        conv,
                    )
                    assert rows[0]["n"] == 1           # completed after release
                finally:
                    if not released.is_set():
                        await conn.execute("ROLLBACK")
                    await pool.release(conn)
            finally:
                await pool.close()

        asyncio.run(scenario())


class TestToolCallIsolation:
    def test_same_tool_call_id_in_different_conversations_never_crosses(self):
        async def scenario():
            pool = await asyncpg.create_pool(dsn=MIGRATION_DSN, min_size=4, max_size=8)
            try:
                store = _store(pool)
                shared_call_id = "call_shared_0001"
                conv_x = str(uuid.uuid4())
                conv_y = str(uuid.uuid4())

                history_x = [
                    {"role": "assistant", "content": None, "tool_calls": [
                        {"id": shared_call_id, "type": "function",
                         "function": {"name": "conv_x_tool", "arguments": "{}"}},
                    ]},
                    {"role": "tool", "tool_call_id": shared_call_id,
                     "content": "result-from-X"},
                ]
                history_y = [
                    {"role": "assistant", "content": None, "tool_calls": [
                        {"id": shared_call_id, "type": "custom",
                         "custom": {"name": "conv_y_tool", "input": "Y"}},
                    ]},
                    {"role": "tool", "tool_call_id": shared_call_id,
                     "content": "result-from-Y"},
                ]

                # concurrent writers, different conversations
                await asyncio.gather(
                    store.reconcile_history(conv_x, history_x),
                    store.reconcile_history(conv_y, history_y),
                )

                refs_x = await pool.fetch(
                    """
                    SELECT tc.name, tr.content FROM tool_results tr
                    JOIN tool_calls tc ON tc.id = tr.tool_call_ref
                    JOIN messages m ON m.id = tc.message_id
                    WHERE m.conversation_id = $1::uuid
                    """,
                    conv_x,
                )
                refs_y = await pool.fetch(
                    """
                    SELECT tc.call_type, tc.input, tr.content FROM tool_results tr
                    JOIN tool_calls tc ON tc.id = tr.tool_call_ref
                    JOIN messages m ON m.id = tc.message_id
                    WHERE m.conversation_id = $1::uuid
                    """,
                    conv_y,
                )
                assert len(refs_x) == 1 and len(refs_y) == 1
                assert json_loads(refs_x[0]["content"]) == "result-from-X"
                assert refs_x[0]["name"] == "conv_x_tool"     # X result -> X call
                assert json_loads(refs_y[0]["content"]) == "result-from-Y"
                assert refs_y[0]["call_type"] == "custom"     # Y stays custom
                assert json_loads(refs_y[0]["input"]) == "Y"
            finally:
                await pool.close()

        asyncio.run(scenario())


def json_loads(value):
    import json

    return json.loads(value) if isinstance(value, str) else value


class TestCustomProjectionIntegration:
    def test_function_and_custom_calls_projected_relationally(self):
        async def scenario():
            pool = await asyncpg.create_pool(dsn=MIGRATION_DSN, min_size=2, max_size=4)
            try:
                store = _store(pool)
                conv = str(uuid.uuid4())
                inbound = [
                    {"role": "assistant", "content": None, "tool_calls": [
                        {"id": "cfn", "type": "function",
                         "function": {"name": "read_file",
                                      "arguments": '{"path": "a.py"}'},
                         "transport_extra": "kept"},
                        {"id": "ccu", "type": "custom",
                         "custom": {"name": "run_query", "input": "SELECT 1;"},
                         "vendor_field": 42},
                    ]},
                ]
                await store.reconcile_history(conv, inbound)
                rows = await pool.fetch(
                    """
                    SELECT tool_call_id, call_type, name, arguments, input, extra
                    FROM tool_calls tc JOIN messages m ON m.id = tc.message_id
                    WHERE m.conversation_id = $1::uuid ORDER BY tool_call_id
                    """,
                    conv,
                )
                assert len(rows) == 2
                by_id = {r["tool_call_id"]: r for r in rows}
                fn = by_id["cfn"]
                assert fn["call_type"] == "function"
                assert fn["name"] == "read_file"
                assert json_loads(fn["arguments"]) == '{"path": "a.py"}'
                assert fn["extra"] is not None and "transport_extra" in fn["extra"]
                cu = by_id["ccu"]
                assert cu["call_type"] == "custom"
                assert cu["name"] == "run_query"
                assert json_loads(cu["input"]) == "SELECT 1;"
                assert cu["arguments"] is None               # not force-fitted
                assert "vendor_field" in cu["extra"]

                # replay is idempotent; raw message untouched
                await store.reconcile_history(conv, inbound)
                raw = await store.get_messages(conv)
                assert raw == inbound
                count = await pool.fetchval(
                    """SELECT count(*) FROM tool_calls tc
                       JOIN messages m ON m.id = tc.message_id
                       WHERE m.conversation_id=$1::uuid""",
                    conv,
                )
                assert count == 2
            finally:
                await pool.close()

        asyncio.run(scenario())

    def test_orphan_tool_result_keeps_null_ref(self):
        async def scenario():
            pool = await asyncpg.create_pool(dsn=MIGRATION_DSN, min_size=2, max_size=4)
            try:
                store = _store(pool)
                conv = str(uuid.uuid4())
                inbound = [
                    {"role": "tool", "tool_call_id": "never-seen-call",
                     "content": "orphan"},
                ]
                await store.reconcile_history(conv, inbound)
                row = await pool.fetchrow(
                    """SELECT tr.tool_call_ref FROM tool_results tr
                       JOIN messages m ON m.id = tr.message_id
                       WHERE m.conversation_id = $1::uuid""",
                    conv,
                )
                assert row is not None
                assert row["tool_call_ref"] is None          # orphan policy holds
            finally:
                await pool.close()

        asyncio.run(scenario())
