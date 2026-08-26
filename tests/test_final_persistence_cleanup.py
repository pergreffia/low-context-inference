"""M6-final: persistence exception classification + startup cleanup ownership."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from context_proxy.config import DatabaseSettings
from context_proxy.db.database import Database
from context_proxy.main import create_app
from context_proxy.memory.errors import PersistenceInfrastructureError
from tests.conftest import UPSTREAM, make_settings, upstream_handler

CHAT_OK = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


class _TypedOutageStore:
    async def ensure_conversation(self, conversation_id):
        raise PersistenceInfrastructureError("postgres connection refused")

    async def reconcile_history(self, conversation_id, messages, metadata=None):
        raise PersistenceInfrastructureError("postgres connection refused")


class _ProgrammingBugStore:
    async def ensure_conversation(self, conversation_id):
        return None

    async def reconcile_history(self, conversation_id, messages, metadata=None):
        raise TypeError("'NoneType' object is not subscriptable")


class _AssertionBugStore:
    async def ensure_conversation(self, conversation_id):
        raise AssertionError("invariant broken")

    async def reconcile_history(self, conversation_id, messages, metadata=None):  # pragma: no cover
        raise AssertionError("unreachable")


def _post_with_store(store) -> httpx.Response:
    """Run lifespan so the injected store reaches app.state, then POST."""
    app = create_app(
        make_settings(),
        llm_client=httpx.AsyncClient(base_url=UPSTREAM,
                                     transport=upstream_handler([])),
        store=store,
    )
    with TestClient(app) as running:
        return running.post("/v1/chat/completions", json=CHAT_OK)


class TestPersistenceExceptionClassification:
    def test_typed_outage_degrades_to_passthrough(self):
        response = _post_with_store(_TypedOutageStore())
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "hello"

    def test_programming_type_error_surfaces_as_500(self):
        with pytest.raises(TypeError):
            _post_with_store(_ProgrammingBugStore())

    def test_assertion_bug_surfaces_as_500(self):
        with pytest.raises(AssertionError):
            _post_with_store(_AssertionBugStore())

    def test_timeout_is_treated_as_infrastructure(self):
        class TimeoutStore(_TypedOutageStore):
            async def ensure_conversation(self, conversation_id):
                raise TimeoutError("db timeout")

        response = _post_with_store(TimeoutStore())
        assert response.status_code == 200  # degraded passthrough


class _ClientTracker:
    """Tracks every httpx.AsyncClient created through a module's namespace."""

    def __init__(self, module):
        self.module = module
        self.created: list[httpx.AsyncClient] = []
        self.closed: set[int] = set()
        self._original = module.httpx.AsyncClient

    def install(self, monkeypatch) -> None:
        original = self._original
        created = self.created
        closed = self.closed
        tracker = self

        class Factory:
            def __call__(self, *args, **kwargs):
                client = original(*args, **kwargs)
                created.append(client)

                real_aclose = client.aclose

                async def tracked_aclose():
                    closed.add(id(client))
                    await real_aclose()

                client.aclose = tracked_aclose  # type: ignore[method-assign]
                return client

        monkeypatch.setattr(
            self.module.httpx, "AsyncClient", Factory(), raising=True
        )
        _ = tracker

    @property
    def all_closed(self) -> bool:
        return bool(self.created) and all(id(c) in self.closed for c in self.created)


def _ok_database(settings) -> Database:
    class OkDatabase(Database):
        async def start(self) -> bool:
            self._pool = object()  # pool exists for downstream wiring
            return True

    return OkDatabase(settings)


class TestStartupCleanupOwnership:
    def test_migration_failure_closes_owned_inference_client(self, monkeypatch):
        from context_proxy.db.database import apply_migrations

        main_module = __import__(
            "context_proxy.main", fromlist=["httpx"]
        )
        tracker = _ClientTracker(main_module)
        tracker.install(monkeypatch)
        monkeypatch.setattr(main_module, "apply_migrations", None, raising=False)

        # Patch Database.start to fail AFTER the app created its owned LLM
        # client but BEFORE any other resource is wired.
        async def failing_start(self):  # noqa: ANN001
            raise RuntimeError("migration boom")

        monkeypatch.setattr(Database, "start", failing_start)
        monkeypatch.setattr("context_proxy.main.apply_migrations", apply_migrations)

        settings = make_settings().model_copy(
            update={"database": DatabaseSettings(url="postgresql://ok/ok")}
        )
        app = create_app(settings, store=None)

        with pytest.raises(RuntimeError, match="migration boom"):
            with TestClient(app):
                pass

        # exactly one app-owned client existed (the inference one) and it was
        # closed by the unconditional cleanup.
        assert len(tracker.created) == 1
        assert tracker.all_closed

    def test_memory_init_failure_closes_owned_embed_and_qdrant_clients(
        self, monkeypatch
    ):
        import context_proxy.main as main_module

        tracker = _ClientTracker(main_module)
        tracker.install(monkeypatch)

        class OkDatabase(Database):
            async def start(self) -> bool:
                self._pool = object()
                return True

            async def close(self):  # noqa: ANN001
                pass

        class BoomMemory:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("memory init boom")

        monkeypatch.setattr(main_module, "MemoryService", BoomMemory)

        settings = make_settings().model_copy(
            update={"database": DatabaseSettings(url="postgresql://ok/ok")}
        )
        injected = httpx.AsyncClient(base_url=UPSTREAM)  # injected: not owned
        app = create_app(
            settings,
            llm_client=injected,
            database=OkDatabase(settings),
            store=None,
        )

        with pytest.raises(RuntimeError, match="memory init boom"):
            with TestClient(app):
                pass

        owned = [c for c in tracker.created if c is not injected]
        assert len(owned) == 2                       # embed + qdrant
        assert all(id(c) in tracker.closed for c in owned)
        assert not injected.is_closed                # never touched by the app

    def test_generic_startup_exception_closes_owned_and_keeps_injected(
        self, monkeypatch
    ):
        import context_proxy.main as main_module

        injected_llm = httpx.AsyncClient(base_url=UPSTREAM)  # NOT tracked
        tracker = _ClientTracker(main_module)
        tracker.install(monkeypatch)

        class OkDatabase(Database):
            async def start(self) -> bool:
                self._pool = object()
                return True

        class BoomVectors:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("generic startup boom")

        # explodes AFTER both embed and qdrant clients were app-created
        monkeypatch.setattr(main_module, "QdrantVectorStore", BoomVectors)

        settings = make_settings().model_copy(
            update={"database": DatabaseSettings(url="postgresql://ok/ok")}
        )
        app = create_app(
            settings,
            llm_client=injected_llm,
            database=OkDatabase(settings),
            store=None,
        )

        with pytest.raises(RuntimeError, match="generic startup boom"):
            with TestClient(app):
                pass

        assert len(tracker.created) == 2 and tracker.all_closed
        assert not injected_llm.is_closed
        asyncio.run(injected_llm.aclose())
