from __future__ import annotations

import logging
from pathlib import Path

import asyncpg

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


async def apply_migrations(pool: asyncpg.Pool, migrations_dir: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply pending .sql migrations in filename order, tracked in schema_migrations."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                name       TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        applied = {row["name"] for row in await conn.fetch("SELECT name FROM schema_migrations")}
        completed: list[str] = []
        for path in sorted(migrations_dir.glob("*.sql")):
            if path.name in applied:
                continue
            sql = path.read_text(encoding="utf-8")
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute(
                    "INSERT INTO schema_migrations (name) VALUES ($1)", path.name
                )
            completed.append(path.name)
            logger.info("applied_migration", extra={"migration": path.name})
        return completed


class Database:
    """Asyncpg pool lifecycle + startup migrations.

    Startup is best-effort: if PostgreSQL is unreachable the proxy keeps serving
    inference passthrough and reports degraded state on /healthz (master prompt §31).
    """

    def __init__(self, settings):
        self._settings = settings
        self._pool: asyncpg.Pool | None = None

    @property
    def pool(self) -> asyncpg.Pool | None:
        return self._pool

    @property
    def available(self) -> bool:
        return self._pool is not None

    async def start(self) -> bool:
        try:
            self._pool = await asyncpg.create_pool(
                dsn=self._settings.url,
                min_size=self._settings.min_pool_size,
                max_size=self._settings.max_pool_size,
                timeout=self._settings.connect_timeout_seconds,
            )
            await apply_migrations(self._pool)
        except Exception as exc:  # noqa: BLE001 - degradation is intentional (§31)
            logger.warning("postgres_unavailable", extra={"error": str(exc)})
            self._pool = None
            return False
        return True

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def ping(self) -> None:
        if self._pool is None:
            raise RuntimeError("database unavailable")
        await self._pool.execute("SELECT 1")
