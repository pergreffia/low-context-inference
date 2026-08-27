"""PostgreSQL conversation store with projection-aware reconciliation."""

from __future__ import annotations

import logging
from typing import Any

from context_proxy.conversation.reconciliation import reconcile_projection
from context_proxy.conversation.store import (
    HistoryDivergenceError,
    PostgresConversationStore,
)

logger = logging.getLogger(__name__)


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
                logger.warning(
                    "history_reconciliation_result conversation_id=%s mode=%s append_from=%s persisted_len=%s incoming_len=%s",
                    conversation_id,
                    result.mode,
                    result.append_from,
                    len(persisted),
                    len(messages),
                )

                if result.mode == "conflict":
                    prefix_len = min(len(persisted), len(messages))
                    index = prefix_len
                    for candidate in range(prefix_len):
                        if persisted[candidate] != messages[candidate]:
                            index = candidate
                            break
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
                        persisted_len=len(persisted),
                        incoming_len=len(messages),
                        prefix_len=index,
                    )

                if result.append_from is None or result.append_from >= len(messages):
                    return []

                return await self._insert_messages(
                    conn,
                    conversation_id,
                    messages[result.append_from :],
                    metadata,
                )
