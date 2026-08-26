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

CONV = "77777777-7777-7777-7777-777777777777"


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


def test_conversation_roundtrip_with_tool_parts():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            # FK-safe cleanup so the test is re-runnable on a dirty database.
            await pool.execute(
                """
                DELETE FROM tool_results WHERE message_id IN
                    (SELECT id FROM messages WHERE conversation_id = $1::uuid)
                """,
                CONV,
            )
            await pool.execute(
                """
                DELETE FROM tool_calls WHERE message_id IN
                    (SELECT id FROM messages WHERE conversation_id = $1::uuid)
                """,
                CONV,
            )
            await pool.execute("DELETE FROM messages WHERE conversation_id = $1::uuid", CONV)
            store = _store(pool)

            inbound = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "read a.py"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
                        }
                    ],
                },
            ]
            call_id = inbound[2]["tool_calls"][0]["id"]
            inbound.append({"role": "tool", "tool_call_id": call_id, "content": "contents"})

            await store.append_messages(CONV, inbound)
            await store.append_messages(CONV, [{"role": "assistant", "content": "done"}])

            rebuilt = await store.get_messages(CONV)
            assert rebuilt == inbound + [{"role": "assistant", "content": "done"}]

            row = await pool.fetchrow(
                """
                SELECT tr.tool_call_ref, tc.name
                FROM tool_results tr
                JOIN messages m ON m.id = tr.message_id
                LEFT JOIN tool_calls tc ON tc.id = tr.tool_call_ref
                WHERE m.conversation_id = $1::uuid AND tr.tool_call_id = $2
                """,
                CONV,
                call_id,
            )
            assert row is not None and row["tool_call_ref"] is not None
            assert row["name"] == "read_file"

            count = await pool.fetchval(
                "SELECT count(*) FROM tool_calls tc JOIN messages m ON m.id = tc.message_id"
                " WHERE m.conversation_id = $1::uuid",
                CONV,
            )
            assert count == 1
        finally:
            await pool.close()

    asyncio.run(_run())


def test_append_preserves_order_across_batches():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            conv = str(uuid.uuid4())
            store = _store(pool)
            await store.append_messages(conv, [{"role": "user", "content": "1"}])
            await store.append_messages(conv, [{"role": "assistant", "content": "2"}])
            await store.append_messages(conv, [{"role": "user", "content": "3"}])
            rebuilt = await store.get_messages(conv)
            assert [m["content"] for m in rebuilt] == ["1", "2", "3"]
        finally:
            await pool.close()

    asyncio.run(_run())


def test_reconcile_accepts_opencode_projection_without_rewriting_raw_history():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            conv = str(uuid.uuid4())
            store = _store(pool)
            persisted = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world", "reasoning_content": "A"},
            ]
            await store.append_messages(conv, persisted)

            incoming = [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                {"role": "assistant", "content": "world", "reasoning_content": "B"},
                {"role": "user", "content": "next"},
            ]
            await store.reconcile_history(conv, incoming)

            rebuilt = await store.get_messages(conv)
            assert rebuilt[:2] == persisted
            assert rebuilt[2] == incoming[2]
        finally:
            await pool.close()

    asyncio.run(_run())


def test_reconcile_rejects_real_rewrite():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            conv = str(uuid.uuid4())
            store = _store(pool)
            persisted = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "original"},
            ]
            await store.append_messages(conv, persisted)

            with pytest.raises(Exception, match="diverges from persisted history"):
                await store.reconcile_history(
                    conv,
                    [
                        persisted[0],
                        {"role": "assistant", "content": "rewritten"},
                    ],
                )
        finally:
            await pool.close()

    asyncio.run(_run())
