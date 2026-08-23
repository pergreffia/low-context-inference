from __future__ import annotations

import asyncio
import os

import asyncpg
import pytest

MIGRATION_DSN = os.environ.get("TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not MIGRATION_DSN,
    reason="set TEST_DATABASE_URL to run PostgreSQL integration tests",
)


def test_migrations_idempotent_and_schema_present():
    async def _run():
        from context_proxy.db.database import apply_migrations

        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            first = await apply_migrations(pool)
            second = await apply_migrations(pool)
            applied = {
                row["name"]
                for row in await pool.fetch("SELECT name FROM schema_migrations")
            }
            tables = {
                row["tablename"]
                for row in await pool.fetch(
                    "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
                )
            }
            return first, second, applied, tables  # type: ignore[return-value]
        finally:
            await pool.close()

    first, second, applied, tables = asyncio.run(_run())
    # Re-runnable: on a fresh DB first applies everything, otherwise it is empty.
    assert {"0001_init.sql", "0002_tool_result_integrity.sql"} <= applied | set(first)
    assert second == []  # never reapplied within a single process
    expected = {
        "schema_migrations",
        "conversations",
        "messages",
        "tool_calls",
        "tool_results",
        "conversation_chunks",
        "memory_records",
        "summaries",
    }
    assert expected <= tables


def test_message_role_constraint():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            conv_id = await pool.fetchval("INSERT INTO conversations DEFAULT VALUES RETURNING id")
            with pytest.raises(asyncpg.CheckViolationError):
                await pool.execute(
                    "INSERT INTO messages (conversation_id, seq, role, content)"
                    " VALUES ($1, 1, 'wizard', '{}'::jsonb)",
                    conv_id,
                )
        finally:
            await pool.close()

    asyncio.run(_run())


def test_tool_result_fk_and_uniqueness_enforced():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            conv_id = await pool.fetchval("INSERT INTO conversations DEFAULT VALUES RETURNING id")
            assistant_msg = await pool.fetchval(
                "INSERT INTO messages (conversation_id, seq, role, content)"
                " VALUES ($1, 1, 'assistant', '{}'::jsonb) RETURNING id",
                conv_id,
            )
            tool_msg = await pool.fetchval(
                "INSERT INTO messages (conversation_id, seq, role, content)"
                " VALUES ($1, 2, 'tool', '{}'::jsonb) RETURNING id",
                conv_id,
            )
            call_row = await pool.fetchval(
                "INSERT INTO tool_calls (message_id, tool_call_id, name, arguments)"
                " VALUES ($1, 'call_1', 'read_file', '{}'::jsonb) RETURNING id",
                assistant_msg,
            )

            # Valid: result linked to its call via surrogate key.
            await pool.execute(
                "INSERT INTO tool_results (message_id, tool_call_ref, tool_call_id, content)"
                " VALUES ($1, $2, 'call_1', '{}'::jsonb)",
                tool_msg,
                call_row,
            )

            # Orphaned reference rejected.
            bogus = "00000000-0000-0000-0000-000000000000"
            with pytest.raises(asyncpg.ForeignKeyViolationError):
                await pool.execute(
                    "INSERT INTO tool_results (message_id, tool_call_ref, tool_call_id, content)"
                    " VALUES ($1, $2, 'call_2', '{}'::jsonb)",
                    tool_msg,
                    bogus,
                )

            # Duplicate result for same call within one tool message rejected.
            with pytest.raises(asyncpg.UniqueViolationError):
                await pool.execute(
                    "INSERT INTO tool_results (message_id, tool_call_ref, tool_call_id, content)"
                    " VALUES ($1, $2, 'call_1', '{}'::jsonb)",
                    tool_msg,
                    call_row,
                )
        finally:
            await pool.close()

    asyncio.run(_run())
