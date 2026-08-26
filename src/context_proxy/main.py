from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response
from starlette.responses import JSONResponse

from context_proxy.api.routes import router
from context_proxy.api.routes_internal import router as internal_router
from context_proxy.config import Settings, load_settings
from context_proxy.context.engine import ContextAssemblyEngine
from context_proxy.conversation.projection_store import ProjectionAwareConversationStore
from context_proxy.db.database import Database
from context_proxy.memory.embeddings import OpenAICompatibleEmbeddingProvider
from context_proxy.memory.qdrant import QdrantVectorStore
from context_proxy.memory.service import MemoryService
from context_proxy.observability.logging_setup import configure_logging
from context_proxy.observability.middleware import ObservabilityMiddleware
from context_proxy.observability.ratelimit import RateLimiter
from context_proxy.providers.llm import OpenAICompatibleLLMProvider
from context_proxy.providers.resilience import CircuitBreaker

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
    context_engine: ContextAssemblyEngine | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    database = database or Database(settings.database)
    owned_clients: list[httpx.AsyncClient] = []
    owned_llm_client = llm_client is None
    breaker = CircuitBreaker(
        failure_threshold=settings.resilience.breaker_failure_threshold,
        reset_seconds=settings.resilience.breaker_reset_seconds,
    )
    rate_limiter = RateLimiter(
        requests_per_minute=settings.rate_limit.requests_per_minute,
        burst=settings.rate_limit.burst,
        max_identities=settings.rate_limit.max_identities,
        identity_ttl_seconds=settings.rate_limit.identity_ttl_seconds,
        max_identity_chars=settings.rate_limit.max_identity_chars,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings.server.log_level, log_json=settings.server.log_json)
        app.state.settings = settings
        app.state.breaker = breaker
        started = False
        try:
            await database.start()
            app.state.database = database
            if store is not None:
                app.state.store = store
            elif database.available and database.pool is not None:
                app.state.store = ProjectionAwareConversationStore(database.pool)
            else:
                app.state.store = None

            if memory_service is not None:
                app.state.memory = memory_service
            elif database.available and database.pool is not None:
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
            started = True
            yield
        finally:
            if owned_llm_client and getattr(app.state, "llm", None) is not None:
                try:
                    await app.state.llm.aclose()
                except Exception as exc:
                    logger.warning("inference_client_close_failed", extra={"error": str(exc)})
            for client in owned_clients:
                try:
                    await client.aclose()
                except Exception as exc:
                    logger.warning("client_close_failed", extra={"error": str(exc)})
            try:
                await database.close()
            except Exception as exc:
                logger.warning("database_close_failed", extra={"error": str(exc)})
        _ = started

    docs_kwargs = (
        {}
        if settings.security.mode == "development"
        else {"docs_url": None, "redoc_url": None, "openapi_url": None}
    )
    app = FastAPI(
        title="Context Proxy", version="0.1.0", lifespan=lifespan, **docs_kwargs
    )
    app.state.settings = settings
    app.state.rate_limiter = rate_limiter
    if llm_client is None:
        llm_client = httpx.AsyncClient(
            base_url=settings.inference.base_url,
            timeout=httpx.Timeout(settings.inference.timeout_seconds),
        )
        if settings.inference.api_key:
            llm_client.headers["Authorization"] = f"Bearer {settings.inference.api_key}"
    app.state.llm = OpenAICompatibleLLMProvider(
        settings.inference,
        client=llm_client,
        resilience=settings.resilience,
        breaker=breaker,
    )
    if context_engine is not None:
        app.state.context_engine = context_engine
    elif settings.assembly.enabled:
        app.state.context_engine = ContextAssemblyEngine(
            usable_budget=settings.context.usable_budget_tokens,
            settings=settings.assembly,
            retrieval_settings=settings.retrieval,
        )
    else:
        app.state.context_engine = None
    app.include_router(router)
    app.include_router(internal_router)

    app.add_middleware(
        ObservabilityMiddleware,
        max_body_bytes=settings.server.max_body_bytes,
        rate_limiter=rate_limiter,
        rate_limit_enabled=settings.rate_limit.enabled,
    )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "Internal server error",
                    "type": "internal_error",
                    "param": None,
                    "code": "internal_error",
                }
            },
        )

    @app.get("/metrics")
    async def prometheus_metrics():
        from context_proxy.observability.metrics import REGISTRY

        return Response(
            content=REGISTRY.render(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.get("/readyz")
    async def readyz():
        db_state = "degraded"
        if database.available:
            try:
                await database.ping()
                db_state = "ok"
            except Exception:
                db_state = "degraded"
        return {
            "ready": True,
            "checks": {"database": db_state, "circuit_breaker": breaker.state},
        }

    @app.get("/healthz")
    async def healthz():
        db_state = "ok" if database.available else "degraded"
        if database.available:
            try:
                await database.ping()
            except Exception:
                db_state = "degraded"
        return {"status": "ok", "database": db_state}

    return app


app = create_app()
