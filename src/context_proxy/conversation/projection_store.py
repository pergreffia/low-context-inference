"""PostgreSQL conversation store with projection-aware reconciliation."""

from __future__ import annotations

from typing import Any

from context_proxy.conversation.reconciliation import reconcile_projection
from context_proxy.conversation.store import (
    HistoryDivergenceError,
    PostgresConversationStore,
)


class ProjectionAwareConversationStore(PostgresConversationStore):
    """Use the durable transcript as source of truth while accepting projections."""

    async def reconcile_history(
        self,
        conversation_id: str,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await self._ensure_conversation(conn, conversation_id)
                await self._lock_conversation(conn, conversation_id)
                persisted = await self._fetch_messages(conn, conversation_id)
                result = reconcile_projection(persisted, messages)

                if result.mode == "conflict":
                    index = min(len(persisted), len(messages))
                    persisted_message = (
                        persisted[index] if index < len(persisted) else None
                    )
                    incoming_message = (
                        messages[index] if index < len(messages) else None
                    )
                    raise HistoryDivergenceError(
                        conversation_id,
                        index,
                        persisted=persisted_message,
                        incoming=incoming_message,
                    )

                if result.append_from is None or result.append_from >= len(messages):
                    return []

                return await self._insert_messages(
                    conn,
                    conversation_id,
                    messages[result.append_from :],
                    metadata,
                )
