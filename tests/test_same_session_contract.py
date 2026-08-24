"""Same-session concurrency contract (final review P3) — PostgreSQL tests.

CONTRACT (Option B — client-side sequencing):

    Clients MUST serialize requests per conversation. The server guarantees
    *state consistency* under concurrency: every write takes the conversation
    row lock inside one reconciliation transaction, divergent histories are
    rejected (HistoryDivergenceError -> 409 history_conflict), and conflicting
    assistant continuations are logged as assistant_persistence_conflict while
    the losing response still reaches its client.

    The server does NOT guarantee logical ordering of concurrent inference
    calls for the same conversation, and never holds a database lock across
    an LLM call. Different conversations are fully independent.

Requires TEST_DATABASE_URL.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import asyncpg
import pytest

from context_proxy.conversation.store import (
    HistoryDivergenceError,
    PostgresConversationStore,
)

MIGRATION_DSN = os.environ.get("TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not MIGRATION_DSN,
    reason="set TEST_DATABASE_URL to run PostgreSQL integration tests",
)


def _store(pool) -> PostgresConversationStore:
    return PostgresConversationStore(pool)


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def test_concurrent_different_inbound_same_conversation_cannot_merge():
    """Two divergent histories on one conversation: exactly one wins."""

    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            store = _store(pool)
            conv = str(uuid.uuid4())
            base_a = [_msg("user", "add X")]
            chain_a = [*base_a, _msg("assistant", "added X")]

            async def writer_b():
                # B starts from a stale empty view and appends its own turn.
                await asyncio.sleep(0.01)
                try:
                    await store.reconcile_history(
                        conv, [_msg("user", "remove X"), _msg("assistant", "removed X")]
                    )
                    return "ok"
                except HistoryDivergenceError as exc:
                    return f"diverged@{exc.index}"

            results = await asyncio.gather(
                store.reconcile_history(conv, chain_a),
                writer_b(),
            )
            persisted_first, outcome_b = results
            assert len(persisted_first) == 2          # A committed its suffix
            assert isinstance(outcome_b, str) and outcome_b.startswith("diverged")

            history = await store.get_messages(conv)
            assert history == chain_a                  # single source of truth
            assert not any(m["content"] == "removed X" for m in history)
        finally:
            await pool.close()

    asyncio.run(_run())


def test_conflicting_assistant_continuations_keep_single_truth():
    """Same inbound, two concurrent continuations -> exactly one persists."""

    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            store = _store(pool)
            conv = str(uuid.uuid4())
            inbound = [_msg("user", "question")]

            async def continuation(text: str):
                await asyncio.sleep(0.01)
                try:
                    await store.reconcile_history(conv, [*inbound, _msg("assistant", text)])
                    return "persisted"
                except HistoryDivergenceError:
                    return "conflict"

            # Both writers see the same committed inbound (empty), so the
            # FIRST reconcile also commits the user message; the loser then
            # diverges at the assistant index.
            results = await asyncio.gather(
                continuation("answer A"),
                continuation("answer B"),
            )
            outcomes = sorted(results)
            assert outcomes[0] == "conflict"
            assert outcomes[1] == "persisted"

            history = await store.get_messages(conv)
            answers = [m["content"] for m in history if m["role"] == "assistant"]
            assert len(answers) == 1                     # exactly one truth
            assert answers[0] in {"answer A", "answer B"}
            assert history[0]["content"] == "question"   # no lost/duplicated user msg
            assert len(history) == 2
        finally:
            await pool.close()

    asyncio.run(_run())


def test_different_conversations_fully_parallel_and_independent():
    """Cross-conversation isolation under parallel load (final review P3)."""

    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            store = _store(pool)

            async def drive(conv: str, marker: str):
                inbound = [
                    _msg("user", f"hello {marker}"),
                    _msg("assistant", f"reply {marker}"),
                ]
                await store.reconcile_history(conv, inbound)
                # interleaved second turn
                await store.reconcile_history(
                    conv, [*inbound, _msg("user", f"again {marker}")]
                )
                return conv

            convs = [str(uuid.uuid4()) for _ in range(6)]
            done = await asyncio.gather(*(drive(c, str(i)) for i, c in enumerate(convs)))
            assert done == convs

            for index, conv in enumerate(convs):
                history = await store.get_messages(conv)
                contents = [m["content"] for m in history]
                assert contents == [
                    f"hello {index}",
                    f"reply {index}",
                    f"again {index}",
                ]
                # no cross-conversation contamination
                assert not any(str(other) in str(history) for other in convs if other != conv)
        finally:
            await pool.close()

    asyncio.run(_run())


def test_reconciliation_is_idempotent_under_parallel_replays():
    """Identical replays racing stay idempotent: no duplicated messages."""

    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            store = _store(pool)
            conv = str(uuid.uuid4())
            inbound = [
                _msg("user", "shared question"),
                {"role": "assistant", "content": None,
                 "tool_calls": [{"id": "c1", "type": "function",
                                 "function": {"name": "f", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "r"},
                _msg("assistant", "final"),
            ]

            results = await asyncio.gather(
                *[store.reconcile_history(conv, inbound) for _ in range(8)]
            )
            appended_counts = [len(r) for r in results]
            # exactly ONE racer appends the full suffix; all others are no-ops
            assert appended_counts.count(4) == 1
            assert all(count == 0 for count in appended_counts if count != 4)

            rows = await pool.fetch(
                "SELECT seq FROM messages WHERE conversation_id=$1::uuid ORDER BY seq",
                conv,
            )
            assert [r["seq"] for r in rows] == [0, 1, 2, 3]
        finally:
            await pool.close()

    asyncio.run(_run())
