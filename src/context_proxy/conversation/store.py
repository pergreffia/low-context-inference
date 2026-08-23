"""PostgreSQL-backed ConversationStore (source of truth, master prompt §7–§8).

Raw message content is stored verbatim as JSONB: the original conversation is
always reconstructable. Tool calls/results are additionally normalized into
their relational tables for integrity and future retrieval (M3+).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)


class PostgresConversationStore:
    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def ping(self) -> None:
        await self._pool.execute("SELECT 1")

    async def ensure_conversation(self, conversation_id: str) -> None:
        async with self._pool.acquire() as conn:
            await self._ensure_conversation(conn, conversation_id)

    @staticmethod
    async def _ensure_conversation(conn: asyncpg.Connection, conversation_id: str) -> None:
        await conn.execute(
            """
            INSERT INTO conversations (id) VALUES ($1::uuid)
            ON CONFLICT (id) DO UPDATE SET updated_at = now()
            """,
            conversation_id,
        )

    async def append_messages(
        self, conversation_id: str, messages: list[dict[str, Any]]
    ) -> list[str]:
        """Persist raw messages in order; extract tool calls/results.

        Returns the persisted message ids.
        """
        if not messages:
            return []
        message_ids: list[str] = []
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await self._ensure_conversation(conn, conversation_id)
                seq = await conn.fetchval(
                    "SELECT COALESCE(MAX(seq), -1) FROM messages WHERE conversation_id = $1::uuid",
                    conversation_id,
                )
                for message in messages:
                    seq += 1
                    message_id = await conn.fetchval(
                        """
                        INSERT INTO messages (conversation_id, seq, role, content)
                        VALUES ($1::uuid, $2, $3, $4::jsonb)
                        RETURNING id
                        """,
                        conversation_id,
                        seq,
                        message.get("role"),
                        json.dumps(message, ensure_ascii=False),
                    )
                    message_ids.append(str(message_id))
                    await self._persist_tool_parts(conn, conversation_id, message_id, message)
        logger.info(
            "messages_persisted",
            extra={"conversation_id": conversation_id, "count": len(message_ids)},
        )
        return message_ids

    @staticmethod
    async def _persist_tool_parts(
        conn: asyncpg.Connection,
        conversation_id: str,
        message_id,
        message: dict[str, Any],
    ) -> None:
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            await conn.execute(
                """
                INSERT INTO tool_calls (message_id, tool_call_id, name, arguments)
                VALUES ($1, $2, $3, $4::jsonb)
                ON CONFLICT (message_id, tool_call_id) DO NOTHING
                """,
                message_id,
                tool_call.get("id"),
                function.get("name"),
                json.dumps(function.get("arguments"), ensure_ascii=False),
            )
        if message.get("role") == "tool" and message.get("tool_call_id"):
            # Link the result to its call row (surrogate key, migration 0002).
            tool_call_ref = await conn.fetchval(
                """
                SELECT tc.id FROM tool_calls tc
                JOIN messages m ON m.id = tc.message_id
                WHERE m.conversation_id = $1::uuid AND tc.tool_call_id = $2
                ORDER BY tc.created_at DESC
                LIMIT 1
                """,
                conversation_id,
                message["tool_call_id"],
            )
            await conn.execute(
                """
                INSERT INTO tool_results (message_id, tool_call_ref, tool_call_id, content)
                VALUES ($1, $2, $3, $4::jsonb)
                """,
                message_id,
                tool_call_ref,
                message["tool_call_id"],
                json.dumps(message.get("content"), ensure_ascii=False),
            )

    async def get_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        """Reconstruct the raw conversation in order."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT content FROM messages WHERE conversation_id = $1::uuid ORDER BY seq",
                conversation_id,
            )
        return [json.loads(row["content"]) for row in rows]
