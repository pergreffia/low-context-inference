"""Multimodal matrix regressions (post-0876b10 review §11).

Covers image_url parts in every supported shape, retrieval text-extraction,
raw preservation through the proxy, and the PostgreSQL media registry
(integration-gated). The raw content is ALWAYS the source of truth.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid

import httpx
import pytest
from fastapi.testclient import TestClient

from context_proxy.context.query import extract_retrieval_query
from context_proxy.main import create_app
from tests.conftest import CHAT_RESPONSE, UPSTREAM, make_settings

MIGRATION_DSN = os.environ.get("TEST_DATABASE_URL", "")

PG_REQUIRED = pytest.mark.skipif(
    not MIGRATION_DSN, reason="set TEST_DATABASE_URL to run PostgreSQL integration tests"
)

DATA_URL = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAf"
    "FcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
REMOTE_URL = "https://cdn.example.com/images/photo.jpg"


def _app_with_capture(captured: list[httpx.Request], store=None):
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=CHAT_RESPONSE)

    return create_app(
        make_settings(),
        llm_client=httpx.AsyncClient(
            base_url=UPSTREAM, transport=httpx.MockTransport(handler)
        ),
        store=store,
    )


# --------------------------------------------------------- query extraction


class TestRetrievalQueryExtraction:
    def test_text_only_content_extracted(self):
        messages = [{"role": "user", "content": "what does this chart show?"}]
        assert extract_retrieval_query(messages) == "what does this chart show?"

    def test_multimodal_contributes_only_text_parts(self):
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "compare"},
                {"type": "image_url", "image_url": {"url": DATA_URL}},
                {"type": "text", "text": "these two"},
                {"type": "unknown_future_part", "blob": {"x": 1}},
            ],
        }]
        query = extract_retrieval_query(messages)
        assert query == "compare these two"
        assert DATA_URL not in query
        assert "unknown" not in query

    def test_image_only_message_yields_empty_query_no_crash(self):
        messages = [{
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": REMOTE_URL}}],
        }]
        assert extract_retrieval_query(messages) == ""

    def test_remote_image_url_never_leaks_into_query(self):
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": REMOTE_URL + "?token=secret"}},
            ],
        }]
        query = extract_retrieval_query(messages)
        assert query == "describe"
        assert "token" not in query and "secret" not in query

    def test_empty_content_list_gives_empty_query(self):
        assert extract_retrieval_query([{"role": "user", "content": []}]) == ""


# ------------------------------------------------------------ proxy passthrough


class TestMultimodalPassthroughMatrix:
    @pytest.mark.parametrize(
        "content",
        [
            # data URL
            [{"type": "text", "text": "t1"}, {"type": "image_url",
                                              "image_url": {"url": DATA_URL}}],
            # remote URL with query secret (must survive verbatim — it is raw data)
            [{"type": "image_url", "image_url": {"url": REMOTE_URL + "?sig=abc"}}],
            # unknown future part type stays intact
            [{"type": "acme_hologram", "depth_map": [1, 2, 3]}],
            # text + multiple images
            [
                {"type": "text", "text": "two images"},
                {"type": "image_url", "image_url": {"url": DATA_URL}},
                {"type": "image_url", "image_url": {"url": REMOTE_URL}},
            ],
            # empty content list
            [],
        ],
        ids=["data-url", "remote-url", "unknown-part", "multi-image", "empty-list"],
    )
    def test_raw_content_forwarded_and_persisted_verbatim(self, content):
        captured: list[httpx.Request] = []

        class Store:
            def __init__(self):
                self.rows: dict[str, list] = {}

            async def ping(self):
                return None

            async def ensure_conversation(self, cid):
                self.rows.setdefault(cid, [])

            async def reconcile_history(self, cid, msgs, metadata=None):
                bucket = self.rows.setdefault(cid, [])
                overlap = min(len(bucket), len(msgs))
                for i in range(overlap):
                    if bucket[i] != msgs[i]:
                        from context_proxy.conversation.store import HistoryDivergenceError

                        raise HistoryDivergenceError(cid, i)
                bucket.extend(msgs[len(bucket):])
                return []

            async def get_messages(self, cid):
                return list(self.rows.get(cid, []))

        store = Store()
        app = _app_with_capture(captured, store)
        message = {"role": "user", "content": content}
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [message]},
            )
        assert response.status_code == 200
        forwarded = json.loads(captured[0].content)["messages"]
        assert forwarded == [message]                 # byte-identical upstream
        assert store.rows[next(iter(store.rows))][0]["content"] == content

    def test_developer_with_image_parts_stays_trusted_instruction(self):
        captured: list[httpx.Request] = []
        dev = {
            "role": "developer",
            "content": [
                {"type": "text", "text": "reference layout attached"},
                {"type": "image_url", "image_url": {"url": DATA_URL}},
            ],
        }
        app = _app_with_capture(captured)
        with TestClient(app) as client:
            client.post("/v1/chat/completions",
                        json={"model": "m", "messages": [dev, {"role": "user", "content": "?"}]})
        sent = json.loads(captured[-1].content)["messages"]
        assert sent[0] == dev


# ----------------------------------------------------- media registry (PG)


class FakeStoreForRegistry:
    pass


@PG_REQUIRED
class TestMediaRegistryIntegration:
    def test_media_index_sources_hash_and_part_order(self):
        import asyncpg

        from context_proxy.conversation.store import PostgresConversationStore

        async def _run():
            pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
            try:
                store = PostgresConversationStore(pool)
                conv = str(uuid.uuid4())
                inbound = [
                    {"role": "user", "content": [
                        {"type": "text", "text": "look"},
                        {"type": "image_url", "image_url": {"url": DATA_URL}},
                        {"type": "image_url", "image_url": {"url": REMOTE_URL}},
                        {"type": "not_a_real_type", "payload": {"keep": "me"}},
                    ]},
                    {"role": "assistant", "content": "ok"},
                ]
                await store.reconcile_history(conv, inbound)

                rows = await pool.fetch(
                    """
                    SELECT part_index, kind, source, media_hash, source_size
                    FROM conversation_media WHERE conversation_id=$1::uuid
                    ORDER BY part_index
                    """,
                    conv,
                )
                assert len(rows) == 2                     # only image parts registered
                assert [(r["part_index"], r["source"]) for r in rows] == [
                    (1, "data"), (2, "url")
                ]
                import hashlib

                expected_data_hash = hashlib.sha256(DATA_URL.encode()).hexdigest()
                expected_remote_hash = hashlib.sha256(REMOTE_URL.encode()).hexdigest()
                assert rows[0]["media_hash"] == expected_data_hash
                assert rows[1]["media_hash"] == expected_remote_hash
                assert rows[1]["source_size"] == len(REMOTE_URL)
                # deterministic across replays
                await store.reconcile_history(conv, inbound)   # idempotent replay
                again = await pool.fetch(
                    "SELECT count(*) AS n FROM conversation_media WHERE conversation_id=$1::uuid",
                    conv,
                )
                assert again[0]["n"] == 2

                # raw reconstruction untouched, unknown part included
                persisted = await store.get_messages(conv)
                assert persisted == inbound
            finally:
                await pool.close()

        asyncio.run(_run())

    def test_very_large_data_url_bounded_behavior(self):
        """Multi-MB data URLs persist raw + registry row without blowing up."""
        import asyncpg

        from context_proxy.conversation.store import PostgresConversationStore

        giant = "data:image/png;base64," + "A" * (3 * 1024 * 1024)

        async def _run():
            pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
            try:
                store = PostgresConversationStore(pool)
                conv = str(uuid.uuid4())
                inbound = [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": giant}},
                ]}]
                await store.reconcile_history(conv, inbound)
                rows = await pool.fetch(
                    "SELECT source, source_size FROM conversation_media "
                    "WHERE conversation_id=$1::uuid",
                    conv,
                )
                assert rows[0]["source_size"] == len(giant)   # recorded, not decoded
                persisted = await store.get_messages(conv)
                assert persisted == inbound                   # raw preserved whole
            finally:
                await pool.close()

        asyncio.run(_run())
