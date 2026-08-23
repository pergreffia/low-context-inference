from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from context_proxy.api.routes import router
from context_proxy.config import Settings, load_settings
from context_proxy.db.database import Database
from context_proxy.providers.llm import OpenAICompatibleLLMProvider

logger = logging.getLogger("context_proxy")


def create_app(
    settings: Settings | None = None,
    *,
    llm_client: httpx.AsyncClient | None = None,
    database: Database | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    database = database or Database(settings.database)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await database.start()
        app.state.database = database
        yield
        await database.close()

    app = FastAPI(title="Context Proxy", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.llm = OpenAICompatibleLLMProvider(settings.inference, client=llm_client)
    app.include_router(router)

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
