from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from context_proxy.api.routes import router
from context_proxy.api.routes_internal import router as internal_router
from context_proxy.config import Settings, load_settings
from context_proxy.conversation.store import PostgresConversationStore
from context_proxy.db.database import Database
from context_proxy.memory.embeddings import OpenAICompatibleEmbeddingProvider
from context_proxy.memory.qdrant import QdrantVectorStore
from context_proxy.memory.service import MemoryService
from context_proxy.providers.llm import OpenAICompatibleLLMProvider

logger = logging.getLogger("context_proxy")


def create_app(
    settings: Settings | None = None,
    *,
    llm_client: httpx.AsyncClient | None = None,
    database: Database | None = None,
    store=None,
    memory_service=None,
    embedding_client: httpx.AsyncClient | None = None,
    qdrant_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    database = database or Database(settings.database)
    owned_clients: list[httpx.AsyncClient] = []

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await database.start()
        app.state.database = database
        if store is not None:
            # Test/injected store wins over the database-backed one.
            app.state.store = store
        elif database.available and database.pool is not None:
            app.state.store = PostgresConversationStore(database.pool)
        else:
            app.state.store = None

        if memory_service is not None:
            app.state.memory = memory_service
        elif database.available and database.pool is not None:
            # Ownership rule (M3 review §2): clients created here are closed on
            # shutdown; injected/external clients are NEVER closed by the app.
            embed_client = embedding_client
            if embed_client is None:
                embed_client = httpx.AsyncClient(
                    base_url=settings.embeddings.base_url,
                    timeout=httpx.Timeout(settings.embeddings.timeout_seconds),
                )
                owned_clients.append(embed_client)
            qdrant_http = qdrant_client
            if qdrant_http is None:
                qdrant_http = httpx.AsyncClient(
                    base_url=settings.qdrant.base_url,
                    timeout=httpx.Timeout(settings.qdrant.timeout_seconds),
                )
                owned_clients.append(qdrant_http)

            embedder = OpenAICompatibleEmbeddingProvider(
                settings.embeddings, client=embed_client
            )
            vectors = QdrantVectorStore(
                settings.qdrant.base_url,
                collection=settings.qdrant.collection,
                client=qdrant_http,
            )
            app.state.memory = MemoryService(
                database.pool,
                embedder,
                vectors,
                retrieval_settings=settings.retrieval,
                max_embed_chars=settings.memory.max_embed_chars,
            )
        else:
            app.state.memory = None
        yield
        for client in owned_clients:
            await client.aclose()
        await database.close()

    app = FastAPI(title="Context Proxy", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.llm = OpenAICompatibleLLMProvider(settings.inference, client=llm_client)
    app.include_router(router)
    app.include_router(internal_router)

    @app.get("/healthz")
    async def healthz():
        db_state = "ok" if database.available else "degraded"
        if database.available:
            try:
                await database.ping()
            except Exception:  # noqa: BLE001
                db_state = "degraded"
        return {"status": "ok", "database": db_state}

    return app


app = create_app()
