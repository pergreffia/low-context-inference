"""PostgreSQL-backed ConversationStore (source of truth, master prompt §7–§8).

Raw message content is stored verbatim as JSONB: the original conversation is
always reconstructable. Tool calls/results are additionally normalized into
their relational tables for integrity and future retrieval (M3+).

Concurrency (M2.2): every write path takes a row-level lock on the
conversation row (`SELECT ... FOR UPDATE`) inside the same transaction that
reads and writes history, so reconciliation is atomic per conversation across
multiple proxy processes. Different conversations never block each other.

Orphan tool-result policy: a tool result whose `tool_call_id` has no matching
call in the same conversation keeps `tool_call_ref = NULL` — the raw message
is preserved (source of truth), never attached to an unrelated call, and an
`orphan_tool_result` warning is logged with the conversation and call ids.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)


def _message_fingerprint(message: dict[str, Any]) -> str:
    """Deterministic, content-hiding diagnostic hash (sha256, first 16 hex).

    Serialization is canonical (sorted keys, compact separators) so equal
    messages always produce equal fingerprints. Hashes let operators tell
    'same message replayed' apart from 'actually different' without ever
    exposing prompts, responses or tool arguments in logs/errors.
    """
    canonical = json.dumps(
        message, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


_MISSING = object()


def _differing_fields(persisted: dict[str, Any], incoming: dict[str, Any]) -> list[str]:
    """Top-level keys whose values differ (names only — never values)."""
    union = set(persisted) | set(incoming)
    return sorted(
        key
        for key in union
        if persisted.get(key, _MISSING) != incoming.get(key, _MISSING)
    )


class HistoryDivergenceError(Exception):
    """Incoming history conflicts with persisted history at some index.

    The database is left untouched; the caller must reject the request
    (master prompt: raw history is the source of truth and is never rewritten).
    Diagnostics carry deterministic fingerprints and differing top-level field
    NAMES only — never message contents.
    """

    def __init__(
        self,
        conversation_id: str,
        index: int,
        *,
        persisted: dict[str, Any] | None = None,
        incoming: dict[str, Any] | None = None,
    ):
        self.conversation_id = conversation_id
        self.index = index
        self.persisted_hash: str | None = None
        self.incoming_hash: str | None = None
        self.different_fields: list[str] = []
        suffix = ""
        if persisted is not None and incoming is not None:
            self.persisted_hash = _message_fingerprint(persisted)
            self.incoming_hash = _message_fingerprint(incoming)
            self.different_fields = _differing_fields(persisted, incoming)
            suffix = (
                f" [persisted_sha256={self.persisted_hash} "
                f"incoming_sha256={self.incoming_hash} "
                f"different_fields={self.different_fields}]"
            )
        super().__init__(
            f"conversation {conversation_id}: incoming message {index} "
            f"diverges from persisted history{suffix}"
        )


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

    @staticmethod
    async def _lock_conversation(conn: asyncpg.Connection, conversation_id: str) -> None:
        """Row-level lock; serializes writers of one conversation only."""
        await conn.fetchval(
            "SELECT id FROM conversations WHERE id = $1::uuid FOR UPDATE",
            conversation_id,
        )

    async def append_messages(
        self,
        conversation_id: str,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Persist raw messages in order; extract tool calls/results.

        Concurrency-safe: seq derivation happens under the conversation row
        lock inside this transaction (M2.2 §4), so independent writers can
        never compute the same next seq.
        """
        if not messages:
            return []
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await self._ensure_conversation(conn, conversation_id)
                await self._lock_conversation(conn, conversation_id)
                return await self._insert_messages(
                    conn, conversation_id, messages, metadata
                )

    async def reconcile_history(
        self,
        conversation_id: str,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Idempotently sync a full client history (M2.1 §1, M2.2 §1).

        Full-history clients resend the entire conversation each turn: only the
        new suffix is appended. Comparison is positional raw-content equality
        (parsed JSON), never content-based deduplication — identical messages
        may legitimately occur multiple times.

        Atomic per conversation: lock -> read -> compare -> append -> commit in
        ONE transaction. A concurrent writer waits on the row lock, then re-reads
        the committed history before comparing, so identical replays stay
        idempotent and divergent histories can never silently merge.
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await self._ensure_conversation(conn, conversation_id)
                await self._lock_conversation(conn, conversation_id)
                persisted = await self._fetch_messages(conn, conversation_id)
                overlap = min(len(persisted), len(messages))
                for index in range(overlap):
                    if persisted[index] != messages[index]:
                        raise HistoryDivergenceError(
                            conversation_id,
                            index,
                            persisted=persisted[index],
                            incoming=messages[index],
                        )
                suffix = messages[len(persisted):]
                if not suffix:
                    return []
                return await self._insert_messages(conn, conversation_id, suffix, metadata)

    async def _fetch_messages(
        self, conn: asyncpg.Connection, conversation_id: str
    ) -> list[dict[str, Any]]:
        rows = await conn.fetch(
            "SELECT content FROM messages WHERE conversation_id = $1::uuid ORDER BY seq",
            conversation_id,
        )
        return [json.loads(row["content"]) for row in rows]

    async def _insert_messages(
        self,
        conn: asyncpg.Connection,
        conversation_id: str,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Insert a suffix; caller holds the conversation lock and transaction."""
        if not messages:
            return []
        message_ids: list[str] = []
        meta_json = json.dumps(metadata, ensure_ascii=False) if metadata else "{}"
        seq = await conn.fetchval(
            "SELECT COALESCE(MAX(seq), -1) FROM messages WHERE conversation_id = $1::uuid",
            conversation_id,
        )
        for message in messages:
            seq += 1
            message_id = await conn.fetchval(
                """
                INSERT INTO messages (conversation_id, seq, role, content, metadata)
                VALUES ($1::uuid, $2, $3, $4::jsonb, $5::jsonb)
                RETURNING id
                """,
                conversation_id,
                seq,
                message.get("role"),
                json.dumps(message, ensure_ascii=False),
                meta_json,
            )
            message_ids.append(str(message_id))
            await self._persist_tool_parts(conn, conversation_id, message_id, message)
            await self._persist_media_parts(conn, conversation_id, message_id, message)
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
        """Relational projection of assistant tool_calls + tool results.

        Supports function AND custom calls (post-0876b10 review §6); unknown
        call shapes keep their extra transport fields verbatim in `extra`.
        Purely a projection — the raw message in messages.jsonb is never
        touched and remains the source of truth.
        """
        for tool_call in message.get("tool_calls") or []:
            call_type = tool_call.get("type")
            if call_type is None:
                call_type = "custom" if "custom" in tool_call else "function"
            if call_type == "custom":
                custom = tool_call.get("custom") or {}
                name = custom.get("name")
                arguments = None
                call_input = json.dumps(custom.get("input"), ensure_ascii=False)
            else:
                function = tool_call.get("function") or {}
                name = function.get("name")
                arguments = json.dumps(function.get("arguments"), ensure_ascii=False)
                call_input = None
            extra = {
                key: value
                for key, value in tool_call.items()
                if key not in ("id", "type", "function", "custom")
            }
            await conn.execute(
                """
                INSERT INTO tool_calls (message_id, tool_call_id, call_type,
                                        name, arguments, input, extra)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7::jsonb)
                ON CONFLICT (message_id, tool_call_id) DO NOTHING
                """,
                message_id,
                tool_call.get("id"),
                str(call_type),
                name,
                arguments,
                call_input,
                json.dumps(extra, ensure_ascii=False),
            )
        if message.get("role") == "tool" and message.get("tool_call_id"):
            # Association rule: newest matching call within the SAME
            # conversation (created_at DESC, then id DESC as a total-order
            # tie-break — identical timestamps must stay deterministic).
            # Duplicate call ids resolve deterministically to the most recent
            # call; results never cross conversations.
            tool_call_ref = await conn.fetchval(
                """
                SELECT tc.id FROM tool_calls tc
                JOIN messages m ON m.id = tc.message_id
                WHERE m.conversation_id = $1::uuid AND tc.tool_call_id = $2
                ORDER BY tc.created_at DESC, tc.id DESC
                LIMIT 1
                """,
                conversation_id,
                message["tool_call_id"],
            )
            if tool_call_ref is None:
                logger.warning(
                    "orphan_tool_result",
                    extra={
                        "conversation_id": conversation_id,
                        "tool_call_id": message["tool_call_id"],
                        "message_id": str(message_id),
                    },
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
            return await self._fetch_messages(conn, conversation_id)

    @staticmethod
    async def _persist_media_parts(
        conn: asyncpg.Connection,
        conversation_id: str,
        message_id,
        message: dict[str, Any],
    ) -> None:
        """Register multimodal parts against their message (M6 §13.2).

        The raw content (including full data URLs) stays verbatim in
        messages.jsonb; this registry is a queryable index associated with the
        interaction unit. Insert is idempotent per (message, part_index).
        """
        import hashlib

        content = message.get("content")
        if not isinstance(content, list):
            return
        for index, part in enumerate(content):
            if not isinstance(part, dict):
                continue  # unknown parts stay opaque in raw storage only
            kind = part.get("type")
            if kind != "image_url":
                continue
            image_url = part.get("image_url") or {}
            url = (
                str(image_url.get("url", ""))
                if isinstance(image_url, dict)
                else str(image_url)
            )
            source = "data" if url.startswith("data:") else "url"
            payload_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
            await conn.execute(
                """
                INSERT INTO conversation_media
                    (conversation_id, message_id, part_index, kind,
                     source, media_hash, source_size)
                VALUES ($1::uuid, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (message_id, part_index) DO NOTHING
                """,
                conversation_id,
                message_id,
                index,
                str(kind),
                source,
                payload_hash,
                len(url),
            )
