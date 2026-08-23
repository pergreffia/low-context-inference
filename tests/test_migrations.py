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


async def _run_twice() -> tuple[list[str], list[str]]:
    from context_proxy.db.database import apply_migrations

    pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
    try:
        first = await apply_migrations(pool)
        second = await apply_migrations(pool)
        tables = {
            row["tablename"]
            for row in await pool.fetch(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            )
        }
        return first, second, tables  # type: ignore[return-value]
    finally:
        await pool.close()


def test_migrations_idempotent_and_schema_present():
    first, second, tables = asyncio.run(_run_twice())
    assert "0001_init.sql" in first
    assert second == []
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
