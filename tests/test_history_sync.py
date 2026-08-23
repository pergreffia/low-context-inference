from __future__ import annotations

import asyncio
import os
import uuid

import asyncpg
import pytest

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


def _store(pool):
    from context_proxy.conversation.store import PostgresConversationStore

    return PostgresConversationStore(pool)


async def _clean(pool, *convs: str) -> None:
    for conv in convs:
        await pool.execute(
            """
            DELETE FROM tool_results WHERE message_id IN
                (SELECT id FROM messages WHERE conversation_id = $1::uuid)
            """,
            conv,
        )
        await pool.execute(
            """
            DELETE FROM tool_calls WHERE message_id IN
                (SELECT id FROM messages WHERE conversation_id = $1::uuid)
            """,
            conv,
        )
        await pool.execute("DELETE FROM messages WHERE conversation_id = $1::uuid", conv)


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


def test_full_history_requests_are_idempotent():
    """10.1: three full-history turns end with exactly the logical messages."""

    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            conv = str(uuid.uuid4())
            await _clean(pool, conv)
            store = _store(pool)

            turn1 = [_user("u1")]
            turn2 = [*turn1, _assistant("a1"), _user("u2")]
            turn3 = [*turn2, _assistant("a2"), _user("u3")]

            await store.reconcile_history(conv, turn1)
            await store.reconcile_history(conv, turn2)
            ids = await store.reconcile_history(conv, turn3)

            persisted = await store.get_messages(conv)
            assert persisted == turn3
            rows = await pool.fetch(
                "SELECT seq, role FROM messages WHERE conversation_id = $1::uuid ORDER BY seq",
                conv,
            )
            assert [r["seq"] for r in rows] == [0, 1, 2, 3, 4]
            # only the last suffix was newly written
            assert len(ids) == 2
        finally:
            await pool.close()

    asyncio.run(_run())


def test_identical_content_stays_distinct():
    """10.2: repeated 'hello' user messages are never deduplicated away."""

    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            conv = str(uuid.uuid4())
            await _clean(pool, conv)
            store = _store(pool)
            history = [_user("hello"), _assistant("hi"), _user("hello")]
            await store.reconcile_history(conv, history)
            await store.reconcile_history(conv, history)  # idempotent replay
            persisted = await store.get_messages(conv)
            assert persisted == history
            assert sum(1 for m in persisted if m["content"] == "hello") == 2
        finally:
            await pool.close()

    asyncio.run(_run())


def test_divergent_history_rejected_without_side_effects():
    """10.3: [A,B,X] against persisted [A,B,C] -> error, DB untouched."""

    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            conv = str(uuid.uuid4())
            await _clean(pool, conv)
            store = _store(pool)
            persisted = [_user("A"), _assistant("B"), _user("C")]
            await store.reconcile_history(conv, persisted)

            with pytest.raises(Exception) as excinfo:
                await store.reconcile_history(conv, [_user("A"), _assistant("B"), _user("X")])
            assert excinfo.value.index == 2

            assert await store.get_messages(conv) == persisted  # unchanged
        finally:
            await pool.close()

    asyncio.run(_run())


def test_shorter_consistent_history_is_noop():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            conv = str(uuid.uuid4())
            await _clean(pool, conv)
            store = _store(pool)
            history = [_user("A"), _assistant("B")]
            await store.reconcile_history(conv, history)
            assert await store.reconcile_history(conv, [_user("A")]) == []
            assert await store.get_messages(conv) == history
        finally:
            await pool.close()

    asyncio.run(_run())


def test_orphan_tool_result_handled_safely():
    """10.15 (orphan side): unknown call id -> NULL ref, raw still persisted."""

    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            conv = str(uuid.uuid4())
            await _clean(pool, conv)
            store = _store(pool)
            orphan = [{"role": "tool", "tool_call_id": "call_unknown", "content": "x"}]

            ids = await store.append_messages(conv, orphan)  # must not raise
            row = await pool.fetchrow(
                """
                SELECT tr.tool_call_ref, tr.tool_call_id
                FROM tool_results tr
                JOIN messages m ON m.id = tr.message_id
                WHERE m.conversation_id = $1::uuid AND m.id = $2::uuid
                """,
                conv,
                ids[0],
            )
            assert row["tool_call_id"] == "call_unknown"
            assert row["tool_call_ref"] is None  # not attached to an unrelated call
        finally:
            await pool.close()

    asyncio.run(_run())


def test_tool_result_ref_resolved_within_conversation():
    """10.15: ref points at the call row of the SAME conversation only."""

    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            conv_a, conv_b = str(uuid.uuid4()), str(uuid.uuid4())
            await _clean(pool, conv_a, conv_b)
            store = _store(pool)
            call = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_same_id",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            }
            result = [{"role": "tool", "tool_call_id": "call_same_id", "content": "r"}]

            await store.append_messages(conv_b, [_user("seed b")])  # unrelated conv
            await store.append_messages(conv_a, [_user("q"), call])
            await store.append_messages(conv_a, result)

            row = await pool.fetchrow(
                """
                SELECT tc.id AS call_row
                FROM tool_results tr
                JOIN messages m ON m.id = tr.message_id
                JOIN tool_calls tc ON tc.id = tr.tool_call_ref
                WHERE m.conversation_id = $1::uuid
                """,
                conv_a,
            )
            assert row is not None
            owner = await pool.fetchval(
                """
                SELECT m.conversation_id
                FROM tool_calls tc
                JOIN messages m ON m.id = tc.message_id
                WHERE tc.id = $1
                """,
                row["call_row"],
            )
            assert str(owner) == conv_a
        finally:
            await pool.close()

    asyncio.run(_run())


def test_multi_conversation_isolation():
    """10.20: conversations A and B never leak into each other."""

    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            conv_a, conv_b = str(uuid.uuid4()), str(uuid.uuid4())
            await _clean(pool, conv_a, conv_b)
            store = _store(pool)
            history_a = [_user("alpha question"), _assistant("alpha answer")]
            history_b = [_user("beta question"), _assistant("beta answer")]

            await store.reconcile_history(conv_a, history_a)
            await store.reconcile_history(conv_b, history_b)

            assert await store.get_messages(conv_a) == history_a
            assert await store.get_messages(conv_b) == history_b
            assert all("beta" not in m["content"] for m in await store.get_messages(conv_a))
            assert all("alpha" not in m["content"] for m in await store.get_messages(conv_b))
        finally:
            await pool.close()

    asyncio.run(_run())


def test_response_metadata_persisted():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            conv = str(uuid.uuid4())
            await _clean(pool, conv)
            store = _store(pool)
            await store.append_messages(
                conv,
                [_assistant("done")],
                metadata={"finish_reason": "stop", "usage": {"total_tokens": 7}, "model": "m1"},
            )
            row = await pool.fetchrow(
                "SELECT metadata FROM messages WHERE conversation_id = $1::uuid", conv
            )
            import json

            metadata = json.loads(row["metadata"])
            assert metadata["finish_reason"] == "stop"
            assert metadata["usage"] == {"total_tokens": 7}
            assert metadata["model"] == "m1"
        finally:
            await pool.close()

    asyncio.run(_run())
