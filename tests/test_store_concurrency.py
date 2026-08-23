from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid

import asyncpg
import httpx
import pytest

MIGRATION_DSN = os.environ.get("TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not MIGRATION_DSN,
    reason="set TEST_DATABASE_URL to run PostgreSQL integration tests",
)


@pytest.fixture(autouse=True)
def _migrated_db():
    async def _apply():
        pool = await asyncpg.create_pool(dsn=MIGRATION_DSN)
        try:
            from context_proxy.db.database import apply_migrations

            await apply_migrations(pool)
        finally:
            await pool.close()

    asyncio.run(_apply())


async def _make_store():
    from context_proxy.conversation.store import PostgresConversationStore

    pool = await asyncpg.create_pool(dsn=MIGRATION_DSN, min_size=2, max_size=8)
    return pool, PostgresConversationStore(pool)


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


async def _clean(pool, conv: str) -> None:
    for statement in (
        """
        DELETE FROM tool_results WHERE message_id IN
            (SELECT id FROM messages WHERE conversation_id = $1::uuid)
        """,
        """
        DELETE FROM tool_calls WHERE message_id IN
            (SELECT id FROM messages WHERE conversation_id = $1::uuid)
        """,
        "DELETE FROM messages WHERE conversation_id = $1::uuid",
    ):
        await pool.execute(statement, conv)


async def _seqs(pool, conv: str) -> list[int]:
    rows = await pool.fetch(
        "SELECT seq FROM messages WHERE conversation_id = $1::uuid ORDER BY seq",
        conv,
    )
    return [r["seq"] for r in rows]


def test_concurrent_identical_full_history_never_duplicates():
    """7.1: two racing reconciles of [A,B,C] end with exactly [A,B,C]."""

    async def _run():
        pool, store = await _make_store()
        try:
            conv = str(uuid.uuid4())
            await _clean(pool, conv)
            history = [_user("A"), _assistant("B"), _user("C")]
            await store.reconcile_history(conv, [_user("A")])

            results = await asyncio.gather(
                store.reconcile_history(conv, history),
                store.reconcile_history(conv, history),
            )

            persisted = await store.get_messages(conv)
            assert persisted == history
            assert await _seqs(pool, conv) == [0, 1, 2]
            # one writer appended the 2-message suffix, the other saw it committed
            assert sorted(len(r) for r in results) == [0, 2]
        finally:
            await pool.close()

    asyncio.run(_run())


def test_concurrent_divergent_histories_never_merge():
    """7.2: [A,B] vs [A,C] -> exactly one wins; final is never [A,B,C]."""

    async def _run():
        from context_proxy.conversation.store import HistoryDivergenceError

        pool, store = await _make_store()
        try:
            conv = str(uuid.uuid4())
            await _clean(pool, conv)
            await store.reconcile_history(conv, [_user("A")])

            outcomes = await asyncio.gather(
                store.reconcile_history(conv, [_user("A"), _assistant("B")]),
                store.reconcile_history(conv, [_user("A"), _user("C")]),
                return_exceptions=True,
            )

            errors = [o for o in outcomes if isinstance(o, HistoryDivergenceError)]
            successes = [o for o in outcomes if not isinstance(o, BaseException)]
            assert len(errors) == 1
            assert len(successes) == 1

            persisted = await store.get_messages(conv)
            assert persisted in (
                [_user("A"), _assistant("B")],
                [_user("A"), _user("C")],
            )
            assert len(persisted) == 2  # never silently merged to 3
            assert await _seqs(pool, conv) == [0, 1]
        finally:
            await pool.close()

    asyncio.run(_run())


def test_concurrent_same_suffix_replay_idempotent():
    """7.3: retry-after-commit replay stays idempotent under contention."""

    async def _run():
        pool, store = await _make_store()
        try:
            conv = str(uuid.uuid4())
            await _clean(pool, conv)
            full = [_user("A"), _assistant("B"), _user("C")]
            await store.reconcile_history(conv, full[:2])

            results = await asyncio.gather(
                store.reconcile_history(conv, full),
                store.reconcile_history(conv, full),
                store.reconcile_history(conv, full),
            )
            assert sorted(len(r) for r in results) == [0, 0, 1]

            persisted = await store.get_messages(conv)
            assert persisted == full
            assert await _seqs(pool, conv) == [0, 1, 2]
        finally:
            await pool.close()

    asyncio.run(_run())


def test_independent_conversations_do_not_block_or_leak():
    """7.4: per-conversation locking; A and B stay isolated."""

    async def _run():
        pool, store = await _make_store()
        try:
            conv_a, conv_b = str(uuid.uuid4()), str(uuid.uuid4())
            await _clean(pool, conv_a)
            await _clean(pool, conv_b)

            async def write_many(conv: str, tag: str) -> None:
                for i in range(10):
                    await store.append_messages(conv, [_user(f"{tag}-{i}")])

            await asyncio.gather(write_many(conv_a, "alpha"), write_many(conv_b, "beta"))

            a = await store.get_messages(conv_a)
            b = await store.get_messages(conv_b)
            assert len(a) == 10 and all("alpha" in m["content"] for m in a)
            assert len(b) == 10 and all("beta" in m["content"] for m in b)
        finally:
            await pool.close()

    asyncio.run(_run())


def test_concurrent_direct_append_no_seq_races():
    """7.5: public append_messages is independently concurrency-safe."""

    async def _run():
        pool, store = await _make_store()
        try:
            conv = str(uuid.uuid4())
            await _clean(pool, conv)

            await asyncio.gather(
                *(store.append_messages(conv, [_user(f"m{i}")]) for i in range(12))
            )

            rows = await pool.fetch(
                """
                SELECT seq, content FROM messages
                WHERE conversation_id = $1::uuid ORDER BY seq
                """,
                conv,
            )
            assert [r["seq"] for r in rows] == list(range(12))  # unique + ordered
            contents = [json.loads(r["content"])["content"] for r in rows]
            assert sorted(contents) == sorted(f"m{i}" for i in range(12))  # none lost
        finally:
            await pool.close()

    asyncio.run(_run())


def test_failed_append_rolls_back_completely_and_retries():
    """§9: DB failure AFTER an insert -> zero partial rows; retry succeeds."""

    async def _run():
        pool, store = await _make_store()
        try:
            conv = str(uuid.uuid4())
            await _clean(pool, conv)
            await store.append_messages(conv, [_user("base")])

            with pytest.raises(asyncpg.CheckViolationError):
                # second message violates the role CHECK after the first row
                # was already inserted inside the same transaction.
                await store.append_messages(
                    conv,
                    [{"role": "user", "content": "will-fail"}, {"role": "wizard"}],
                )

            rows = await pool.fetch(
                "SELECT content FROM messages WHERE conversation_id = $1::uuid ORDER BY seq",
                conv,
            )
            assert len(rows) == 1  # no partial batch persisted
            assert json.loads(rows[0]["content"])["content"] == "base"

            ids = await store.append_messages(conv, [_assistant("recovered")])
            assert len(ids) == 1  # conversation usable again
            assert len(await store.get_messages(conv)) == 2
        finally:
            await pool.close()

    asyncio.run(_run())


def test_orphan_tool_result_emits_structured_warning(caplog):
    """7.6: NULL ref + raw preserved + orphan_tool_result event with ids."""

    async def _run():
        pool, store = await _make_store()
        try:
            conv = str(uuid.uuid4())
            await _clean(pool, conv)
            with caplog.at_level(logging.WARNING, logger="context_proxy.conversation.store"):
                await store.append_messages(
                    conv,
                    [{"role": "tool", "tool_call_id": "call_unknown", "content": "x"}],
                )

            records = [r for r in caplog.records if r.message == "orphan_tool_result"]
            assert len(records) == 1
            assert records[0].conversation_id == conv  # type: ignore[attr-defined]
            assert records[0].tool_call_id == "call_unknown"  # type: ignore[attr-defined]

            row = await pool.fetchrow(
                """
                SELECT tr.tool_call_ref, tr.tool_call_id
                FROM tool_results tr
                JOIN messages m ON m.id = tr.message_id
                WHERE m.conversation_id = $1::uuid
                """,
                conv,
            )
            assert row["tool_call_id"] == "call_unknown"
            assert row["tool_call_ref"] is None
            assert len(await store.get_messages(conv)) == 1  # raw preserved
        finally:
            await pool.close()

    asyncio.run(_run())


def test_duplicate_tool_call_ids_resolve_deterministically():
    """7.7: duplicate call ids attach to the NEWEST call, never arbitrarily."""

    async def _run():
        pool, store = await _make_store()
        try:
            conv = str(uuid.uuid4())
            await _clean(pool, conv)

            def call_msg() -> dict:
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_dup",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                }

            await store.append_messages(conv, [_user("q1"), call_msg()])
            first_call_row = await pool.fetchval(
                """
                SELECT tc.id FROM tool_calls tc
                JOIN messages m ON m.id = tc.message_id
                WHERE m.conversation_id = $1::uuid AND tc.tool_call_id = 'call_dup'
                  AND m.seq = 1
                """,
                conv,
            )
            await store.append_messages(conv, [_user("q2"), call_msg()])
            second_call_row = await pool.fetchval(
                """
                SELECT tc.id FROM tool_calls tc
                JOIN messages m ON m.id = tc.message_id
                WHERE m.conversation_id = $1::uuid AND tc.tool_call_id = 'call_dup'
                  AND m.seq = 3
                """,
                conv,
            )
            assert first_call_row != second_call_row

            await store.append_messages(
                conv, [{"role": "tool", "tool_call_id": "call_dup", "content": "r"}]
            )
            ref = await pool.fetchval(
                """
                SELECT tool_call_ref FROM tool_results
                WHERE tool_call_id = 'call_dup' AND message_id IN
                    (SELECT id FROM messages WHERE conversation_id = $1::uuid)
                """,
                conv,
            )
            assert ref == second_call_row  # documented rule: newest matching call
        finally:
            await pool.close()

    asyncio.run(_run())


def test_api_level_concurrent_identical_full_history():
    """§8: API-level — concurrent identical requests persist idempotently.

    Documented behavior: both inference calls occur (no request coalescing at
    M2); only persistence is guaranteed idempotent.
    """

    async def _run():
        from context_proxy.config import Settings
        from context_proxy.main import create_app

        pool, store = await _make_store()
        try:
            conv = str(uuid.uuid4())
            await _clean(pool, conv)

            upstream_hits = {"n": 0}

            def handler(request: httpx.Request) -> httpx.Response:
                upstream_hits["n"] += 1
                return httpx.Response(
                    200,
                    json={
                        "id": "chatcmpl-x",
                        "object": "chat.completion",
                        "model": "test-model",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "ok"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    },
                )

            settings = Settings(_env_file=None)
            app = create_app(
                settings,
                llm_client=httpx.AsyncClient(
                    base_url=str(settings.inference.base_url),
                    transport=httpx.MockTransport(handler),
                ),
                store=store,
            )
            # No lifespan under ASGITransport: wire the injected store directly.
            app.state.store = store

            transport = httpx.ASGITransport(app=app)
            payload = {
                "model": "m",
                "messages": [{"role": "user", "content": "A"}],
                "conversation_id": conv,
            }
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                responses = await asyncio.gather(
                    client.post("/v1/chat/completions", json=payload),
                    client.post("/v1/chat/completions", json=payload),
                )

            assert all(r.status_code == 200 for r in responses)
            assert len({r.headers["X-Conversation-ID"] for r in responses}) == 1
            assert upstream_hits["n"] == 2  # documented: no inference dedup at M2

            persisted = await store.get_messages(conv)
            assert [(m["role"], m["content"]) for m in persisted] == [
                ("user", "A"),
                ("assistant", "ok"),
            ]
            assert await _seqs(pool, conv) == [0, 1]
        finally:
            await pool.close()

    asyncio.run(_run())


def test_concurrent_assistant_responses_single_continuation(caplog):
    """M2.3 §5–§6: X/Y race -> exactly one assistant persisted, both clients OK."""

    async def _run():
        from context_proxy.config import Settings
        from context_proxy.main import create_app

        pool, store = await _make_store()
        try:
            conv = str(uuid.uuid4())
            await _clean(pool, conv)
            base = [_user("A"), _assistant("B")]
            await store.reconcile_history(conv, base)

            upstream_hits = {"n": 0}

            def handler(request: httpx.Request) -> httpx.Response:
                upstream_hits["n"] += 1
                content = "X" if upstream_hits["n"] == 1 else "Y"
                return httpx.Response(
                    200,
                    json={
                        "id": f"chatcmpl-{content}",
                        "model": "test-model",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": content},
                                "finish_reason": "stop",
                            }
                        ],
                    },
                )

            settings = Settings(_env_file=None)
            app = create_app(
                settings,
                llm_client=httpx.AsyncClient(
                    base_url=str(settings.inference.base_url),
                    transport=httpx.MockTransport(handler),
                ),
                store=store,
            )
            app.state.store = store

            transport = httpx.ASGITransport(app=app)
            payload = {"model": "m", "messages": base, "conversation_id": conv}
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                with caplog.at_level(logging.WARNING, logger="context_proxy.request"):
                    responses = await asyncio.gather(
                        client.post("/v1/chat/completions", json=payload),
                        client.post("/v1/chat/completions", json=payload),
                    )

            # both clients receive their own generated response
            assert all(r.status_code == 200 for r in responses)
            bodies = {r.json()["choices"][0]["message"]["content"] for r in responses}
            assert bodies == {"X", "Y"}

            # persistence: exactly ONE continuation beyond the base turn
            persisted = await store.get_messages(conv)
            assert persisted[:2] == base
            continuations = persisted[2:]
            assert len(continuations) == 1
            assert continuations[0]["role"] == "assistant"
            assert continuations[0]["content"] in {"X", "Y"}
            tail = [m["content"] for m in persisted[-2:]]
            assert not (tail == ["X", "Y"] or tail == ["Y", "X"])
            assert await _seqs(pool, conv) == [0, 1, 2]

            # logging: exactly one conflict, no unexpected failure
            conflicts = [
                r for r in caplog.records if r.message == "assistant_persistence_conflict"
            ]
            failures = [
                r for r in caplog.records if r.message == "assistant_persistence_failed"
            ]
            assert len(conflicts) == 1
            assert failures == []
        finally:
            await pool.close()

    asyncio.run(_run())


def test_streaming_concurrent_assistant_conflicts_keep_streams_intact(caplog):
    """M2.3 §9: losing streaming response still fully reaches its client."""

    async def _run():
        from context_proxy.config import Settings
        from context_proxy.main import create_app

        pool, store = await _make_store()
        try:
            conv = str(uuid.uuid4())
            await _clean(pool, conv)
            base = [_user("A"), _assistant("B")]
            await store.reconcile_history(conv, base)

            upstream_hits = {"n": 0}

            def handler(request: httpx.Request) -> httpx.Response:
                upstream_hits["n"] += 1
                content = "X" if upstream_hits["n"] == 1 else "Y"

                def sse_for(content: str):
                    async def agen():
                        role = (
                            b'data: {"model":"m","choices":[{"index":0,'
                            b'"delta":{"role":"assistant"}}]}\n\n'
                        )
                        delta = (
                            f'data: {{"choices":[{{"index":0,'
                            f'"delta":{{"content":"{content}"}},"finish_reason":null}}]}}\n\n'
                        ).encode()
                        yield role
                        yield delta
                        yield b"data: [DONE]\n\n"

                    return agen()

                return httpx.Response(
                    200,
                    content=sse_for(content),
                    headers={"content-type": "text/event-stream"},
                )

            settings = Settings(_env_file=None)
            app = create_app(
                settings,
                llm_client=httpx.AsyncClient(
                    base_url=str(settings.inference.base_url),
                    transport=httpx.MockTransport(handler),
                ),
                store=store,
            )
            app.state.store = store

            transport = httpx.ASGITransport(app=app)
            payload = {"model": "m", "messages": base, "stream": True, "conversation_id": conv}

            async def consume(client) -> bytes:
                async with client.stream(
                    "POST", "/v1/chat/completions", json=payload
                ) as response:
                    assert response.status_code == 200
                    return b"".join([chunk async for chunk in response.aiter_bytes()])

            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                with caplog.at_level(logging.WARNING, logger="context_proxy.request"):
                    bodies = await asyncio.gather(consume(client), consume(client))

            # both streams delivered completely, each preserving its own
            # upstream response (order-independent: X vs Y scheduling varies)
            assert all(b.endswith(b"data: [DONE]\n\n") for b in bodies)
            assert any(b'"content":"X"' in body for body in bodies)
            assert any(b'"content":"Y"' in body for body in bodies)
            assert {b for b in bodies} and len(bodies) == 2

            persisted = await store.get_messages(conv)
            assert persisted[:2] == base
            continuations = [m for m in persisted[2:] if m["role"] == "assistant"]
            assert len(continuations) == 1  # only one continuation committed
            assert continuations[0]["content"] in {"X", "Y"}
            assert await _seqs(pool, conv) == [0, 1, 2]

            conflicts = [
                r for r in caplog.records if r.message == "assistant_persistence_conflict"
            ]
            assert len(conflicts) == 1
        finally:
            await pool.close()

    asyncio.run(_run())


def test_streaming_conflict_event_carries_context(caplog):
    """M2.3.1 §6: conflict event includes conversation_id and index."""

    async def _run():
        from context_proxy.config import Settings
        from context_proxy.main import create_app

        pool, store = await _make_store()
        try:
            conv = str(uuid.uuid4())
            await _clean(pool, conv)
            base = [_user("A"), _assistant("B")]
            await store.reconcile_history(conv, base)

            upstream_hits = {"n": 0}

            def handler(request: httpx.Request) -> httpx.Response:
                upstream_hits["n"] += 1
                content = "X" if upstream_hits["n"] == 1 else "Y"

                def sse_for(content: str):
                    async def agen():
                        yield (
                            f'data: {{"choices":[{{"delta":{{"content":"{content}"}}}}]}}\n\n'
                        ).encode()
                        yield b"data: [DONE]\n\n"

                    return agen()

                return httpx.Response(
                    200,
                    content=sse_for(content),
                    headers={"content-type": "text/event-stream"},
                )

            settings = Settings(_env_file=None)
            app = create_app(
                settings,
                llm_client=httpx.AsyncClient(
                    base_url=str(settings.inference.base_url),
                    transport=httpx.MockTransport(handler),
                ),
                store=store,
            )
            app.state.store = store

            transport = httpx.ASGITransport(app=app)
            payload = {"model": "m", "messages": base, "stream": True, "conversation_id": conv}

            async def consume(client) -> bytes:
                async with client.stream(
                    "POST", "/v1/chat/completions", json=payload
                ) as response:
                    return b"".join([chunk async for chunk in response.aiter_bytes()])

            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                with caplog.at_level(logging.WARNING, logger="context_proxy.request"):
                    bodies = await asyncio.gather(consume(client), consume(client))

            assert all(b.endswith(b"data: [DONE]\n\n") for b in bodies)
            conflicts = [
                r for r in caplog.records if r.message == "assistant_persistence_conflict"
            ]
            failures = [
                r for r in caplog.records if r.message == "assistant_persistence_failed"
            ]
            assert len(conflicts) == 1
            assert failures == []
            assert conflicts[0].conversation_id == conv  # type: ignore[attr-defined]
            assert isinstance(conflicts[0].index, int)  # type: ignore[attr-defined]

            persisted = await store.get_messages(conv)
            continuations = [m["content"] for m in persisted[2:]]
            assert len(continuations) == 1 and continuations[0] in {"X", "Y"}
        finally:
            await pool.close()

    asyncio.run(_run())


def test_streaming_unexpected_persistence_failure_no_retry(caplog):
    """M2.3.1 §8–§9: RuntimeError at assistant step -> failed event, stream OK,
    exactly one upstream call (no inference/persistence retry)."""

    async def _run():
        from context_proxy.config import Settings
        from context_proxy.main import create_app

        pool, store = await _make_store()

        calls = {"reconcile": 0}
        original_reconcile = store.reconcile_history

        async def flaky_reconcile(conversation_id, messages, metadata=None):
            calls["reconcile"] += 1
            if calls["reconcile"] == 2:  # inbound=1 ok, assistant=2 fails
                raise RuntimeError("simulated db outage")
            return await original_reconcile(conversation_id, messages, metadata)

        store.reconcile_history = flaky_reconcile  # type: ignore[method-assign]

        try:
            conv = str(uuid.uuid4())
            await _clean(pool, conv)

            upstream_hits = {"n": 0}

            def handler(request: httpx.Request) -> httpx.Response:
                upstream_hits["n"] += 1

                async def agen():
                    yield b'data: {"choices":[{"delta":{"content":"Z"}}]}\n\n'
                    yield b"data: [DONE]\n\n"

                return httpx.Response(
                    200,
                    content=agen(),
                    headers={"content-type": "text/event-stream"},
                )

            settings = Settings(_env_file=None)
            app = create_app(
                settings,
                llm_client=httpx.AsyncClient(
                    base_url=str(settings.inference.base_url),
                    transport=httpx.MockTransport(handler),
                ),
                store=store,
            )
            app.state.store = store

            transport = httpx.ASGITransport(app=app)
            payload = {
                "model": "m",
                "messages": [_user("A")],
                "stream": True,
                "conversation_id": conv,
            }

            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                with caplog.at_level(logging.WARNING, logger="context_proxy.request"):
                    response = await client.post("/v1/chat/completions", json=payload)
                    body = b"".join([chunk async for chunk in response.aiter_bytes()])

                assert response.status_code == 200
                assert body.endswith(b"data: [DONE]\n\n")  # complete stream preserved
                assert upstream_hits["n"] == 1  # no inference retry

                failures = [
                    r for r in caplog.records if r.message == "assistant_persistence_failed"
                ]
                conflicts = [
                    r for r in caplog.records if r.message == "assistant_persistence_conflict"
                ]
                assert len(failures) == 1
                assert conflicts == []

                # nothing partial persisted; retry of the request works normally
                assert await _seqs(pool, conv) == [0]
                calls["reconcile"] = -10  # disarm the fault
                response = await client.post("/v1/chat/completions", json=payload)
                assert response.status_code == 200

            persisted = await store.get_messages(conv)
            assert [(m["role"], m["content"]) for m in persisted] == [
                ("user", "A"),
                ("assistant", "Z"),
            ]
        finally:
            await pool.close()

    asyncio.run(_run())
