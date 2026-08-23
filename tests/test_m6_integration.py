"""M6 PostgreSQL integration: multimodal persistence & media registry.

Requires TEST_DATABASE_URL (same contract as the other integration suites).
"""

from __future__ import annotations

import asyncio
import os
import uuid

import asyncpg
import pytest

from context_proxy.conversation.store import PostgresConversationStore

MIGRATION_DSN = os.environ.get("TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not MIGRATION_DSN,
    reason="set TEST_DATABASE_URL to run PostgreSQL integration tests",
)

DATA_URL = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAf"
    "FcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _store(pool) -> PostgresConversationStore:
    return PostgresConversationStore(pool)


def test_media_registry_and_raw_reconstruction():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            store = _store(pool)
            conv = str(uuid.uuid4())
            inbound = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is wrong here?"},
                        {"type": "image_url", "image_url": {"url": DATA_URL}},
                    ],
                },
                {"role": "assistant", "content": "a bug in the loop"},
                {"role": "user", "content": "plain follow-up"},
                {"role": "assistant", "content": "ok"},
            ]
            await store.reconcile_history(conv, inbound)
            # replay: media registry stays deduplicated too
            await store.reconcile_history(conv, inbound)

            rows = await pool.fetch(
                """
                SELECT part_index, kind, source, byte_size
                FROM conversation_media
                WHERE conversation_id = $1::uuid
                ORDER BY message_id, part_index
                """,
                conv,
            )
            assert len(rows) == 1  # one image part across the whole conversation
            row = rows[0]
            assert row["part_index"] == 1  # second part of the first message
            assert row["kind"] == "image_url"
            assert row["source"] == "data"
            assert row["byte_size"] == len(DATA_URL)

            # authoritative reconstruction is EXACT, images included
            persisted = await store.get_messages(conv)
            assert persisted == inbound
        finally:
            await pool.close()

    asyncio.run(_run())


def test_remote_image_registered_as_url_source():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            store = _store(pool)
            conv = str(uuid.uuid4())
            inbound = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "check this"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/shot.png"},
                        },
                    ],
                }
            ]
            await store.reconcile_history(conv, inbound)
            row = await pool.fetchrow(
                "SELECT source FROM conversation_media WHERE conversation_id=$1::uuid",
                conv,
            )
            assert row["source"] == "url"
        finally:
            await pool.close()

    asyncio.run(_run())


def test_media_isolation_between_conversations():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            store = _store(pool)
            conv_a, conv_b = str(uuid.uuid4()), str(uuid.uuid4())

            async def seed(conv):
                await store.reconcile_history(
                    conv,
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"shot of {conv}"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": DATA_URL},
                                },
                            ],
                        }
                    ],
                )

            await seed(conv_a)
            await seed(conv_b)

            count_a = await pool.fetchval(
                "SELECT count(*) FROM conversation_media WHERE conversation_id=$1::uuid",
                conv_a,
            )
            count_b = await pool.fetchval(
                "SELECT count(*) FROM conversation_media WHERE conversation_id=$1::uuid",
                conv_b,
            )
            assert count_a == 1 and count_b == 1

            # conversation A's raw history contains no reference to B
            history_a = await store.get_messages(conv_a)
            assert conv_b not in str(history_a)
        finally:
            await pool.close()

    asyncio.run(_run())


def test_chunking_covers_multimodal_turns():
    async def _run():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            from test_engine_integration import (
                HashingEmbedder,
                RecordingVectorStore,
            )

            from context_proxy.config import RetrievalSettings
            from context_proxy.memory.service import MemoryService

            memory = MemoryService(
                pool,
                HashingEmbedder(),
                RecordingVectorStore(),
                retrieval_settings=RetrievalSettings(),
            )
            store = _store(pool)
            conv = str(uuid.uuid4())
            await store.reconcile_history(
                conv,
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "review the screenshot"},
                            {"type": "image_url", "image_url": {"url": DATA_URL}},
                        ],
                    },
                    {"role": "assistant", "content": "done reviewing"},
                    {"role": "user", "content": "live tail"},
                ],
            )
            created = await memory.index_completed_turns(conv)
            assert created == 1
            chunk = await pool.fetchrow(
                "SELECT raw_content FROM conversation_chunks WHERE conversation_id=$1::uuid",
                conv,
            )
            stored = [json.loads(line) for line in chunk["raw_content"].splitlines()]
            # image preserved verbatim inside the derived chunk as well
            assert stored[0]["content"][1]["image_url"]["url"] == DATA_URL
        finally:
            await pool.close()

    import json

    asyncio.run(_run())
