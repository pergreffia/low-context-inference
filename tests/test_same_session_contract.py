"""Same-session concurrency contract — DETERMINISTIC synchronization.

Contract (Option B, documented in README): clients MUST serialize requests
per conversation; the server guarantees state consistency only. These tests
coordinate workers with asyncio gates (no wall-clock sleeps) so the race is
reproducible on every run.

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


class Gate:
    """Deterministic N-worker release: every worker registers, then all go."""

    def __init__(self, workers: int):
        self._workers = workers
        self._ready = 0
        self._event = asyncio.Event()

    async def wait_go(self) -> None:
        self._ready += 1
        if self._ready >= self._workers:
            self._event.set()
        await self._event.wait()


def test_divergent_histories_race_without_merge():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            store = _store(pool)
            conv = str(uuid.uuid4())
            gate = Gate(2)
            outcomes: list[str] = [None, None]  # type: ignore[list-item]

            chain_a = [_msg("user", "add X"), _msg("assistant", "added X")]
            chain_b = [_msg("user", "remove X"), _msg("assistant", "removed X")]

            async def worker_a():
                await gate.wait_go()
                try:
                    await store.reconcile_history(conv, chain_a)
                    outcomes[0] = "committed"
                except HistoryDivergenceError:
                    outcomes[0] = "diverged"

            async def worker_b():
                await gate.wait_go()
                try:
                    await store.reconcile_history(conv, chain_b)
                    outcomes[1] = "committed"
                except HistoryDivergenceError:
                    outcomes[1] = "diverged"

            await asyncio.gather(worker_a(), worker_b())

            # exactly one committed; the divergent one rejected
            assert sorted(outcomes) == ["committed", "diverged"]
            history = await store.get_messages(conv)
            assert history in (chain_a, chain_b)   # single truth, no merge
            assert len(history) == 2               # nothing lost/duplicated
        finally:
            await pool.close()

    asyncio.run(_run())


def test_conflicting_assistant_continuations_exactly_one_authoritative():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            store = _store(pool)
            conv = str(uuid.uuid4())
            inbound = [_msg("user", "question")]
            gate = Gate(2)
            results: list[str] = [None, None]  # type: ignore[list-item]

            async def continuation(text: str, slot: int):
                await gate.wait_go()
                try:
                    await store.reconcile_history(
                        conv, [*inbound, _msg("assistant", text)]
                    )
                    results[slot] = "persisted"
                except HistoryDivergenceError:
                    # documented conflict signal: losing response was still
                    # deliverable to its client per the M2.3 contract
                    results[slot] = "conflict"

            await asyncio.gather(
                continuation("answer A", 0),
                continuation("answer B", 1),
            )
            assert sorted(results) == ["conflict", "persisted"]

            history = await store.get_messages(conv)
            answers = [m["content"] for m in history if m["role"] == "assistant"]
            assert len(answers) == 1
            assert len(history) == 2                       # user + one assistant
            assert history[0]["content"] == "question"     # inbound not lost
        finally:
            await pool.close()

    asyncio.run(_run())


def test_identical_concurrent_replays_idempotent_contiguous():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            store = _store(pool)
            conv = str(uuid.uuid4())
            workers = 8
            inbound = [
                _msg("user", "shared question"),
                {"role": "assistant", "content": None,
                 "tool_calls": [{"id": "c1", "type": "function",
                                 "function": {"name": "f", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "r"},
                _msg("assistant", "final"),
            ]
            gate = Gate(workers)

            async def racer() -> int:
                await gate.wait_go()
                return len(await store.reconcile_history(conv, inbound))

            counts = await asyncio.gather(*(racer() for _ in range(workers)))
            assert sorted(counts) == [0] * (workers - 1) + [4]

            rows = await pool.fetch(
                "SELECT seq FROM messages WHERE conversation_id=$1::uuid ORDER BY seq",
                conv,
            )
            assert [r["seq"] for r in rows] == [0, 1, 2, 3]  # contiguous, unique
        finally:
            await pool.close()

    asyncio.run(_run())


def test_different_conversations_progress_independently():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            store = _store(pool)
            conversations = [str(uuid.uuid4()) for _ in range(6)]
            gate = Gate(len(conversations))

            async def drive(index: int):
                conv = conversations[index]
                marker = str(index)
                await gate.wait_go()  # all six start together
                inbound = [_msg("user", f"hello {marker}"),
                           _msg("assistant", f"reply {marker}")]
                await store.reconcile_history(conv, inbound)
                await store.reconcile_history(
                    conv, [*inbound, _msg("user", f"again {marker}")]
                )
                return conv

            done = await asyncio.gather(
                *(drive(i) for i in range(len(conversations)))
            )
            assert done == conversations

            for index, conv in enumerate(conversations):
                contents = [m["content"] for m in await store.get_messages(conv)]
                assert contents == [
                    f"hello {index}", f"reply {index}", f"again {index}"
                ]
                others = ", ".join(str(c) for c in conversations if c != conv)
                assert others not in str(contents)  # no cross-contamination
        finally:
            await pool.close()

    asyncio.run(_run())
