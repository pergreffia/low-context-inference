"""Internal Memory Service API (master prompt §33).

Typed request/response schemas; local infrastructure surface — not part of
the public OpenAI-compatible contract. All identifiers are validated as UUIDs
so malformed input yields controlled 400s instead of raw database errors.

INTERNAL-ONLY (M5 review §5; post-0876b10 review §2): this router is
administrative. `/index/rebuild` in particular can consume significant
embedding/Qdrant resources. Deployment boundary:

    public ingress  -> /v1/*          (OpenAI-compatible surface)
    private network -> /internal/*    (this router)

The URL prefix alone is NOT a security mechanism. When
SECURITY__INTERNAL_AUTH_TOKEN is configured, every /internal/* request must
present it via the X-Internal-Auth header (401 otherwise). An empty token
keeps unauthenticated local development working — the real boundary remains
the private network.
"""

from __future__ import annotations

import hmac
import uuid
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from context_proxy.api.validation import (
    PayloadValidationError,
    validate_chat_payload,
)
from context_proxy.context.engine import (
    ContextOverflowError as EngineContextOverflowError,
)
from context_proxy.context.engine import separate_current_request
from context_proxy.context.query import extract_retrieval_query
from context_proxy.db.database import Database
from context_proxy.memory.errors import RetrievalError
from context_proxy.memory.models import (
    MemoryCreate,
    RetrievalResponse,
    SupersedeRequest,
)

router = APIRouter(prefix="/internal/v1")

INTERNAL_AUTH_HEADER = "x-internal-auth"


async def require_internal_auth(request: Request) -> None:
    """Configurable application-level gate for the administrative surface.

    Token unset -> allow (local deployment, network provides the boundary).
    Token set   -> constant-time comparison against X-Internal-Auth.
    """
    expected = request.app.state.settings.security.internal_auth_token
    if not expected:
        return
    presented = request.headers.get(INTERNAL_AUTH_HEADER)
    if presented is None or not hmac.compare_digest(presented, expected):
        raise HTTPException(status_code=401, detail="internal endpoint requires authentication")


router.dependencies = [Depends(require_internal_auth)]  # type: ignore[assignment]


class ContextPreviewRequest(BaseModel):
    """Mirror of the chat payload fields the engine plans against."""

    messages: list[dict] = Field(default_factory=list)
    tools: list[dict] | None = None


def _parse_uuid(value: str, what: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise HTTPException(
            status_code=400, detail=f"invalid {what}: {value!r}"
        ) from exc


def _memory(request: Request):
    memory = getattr(request.app.state, "memory", None)
    if memory is None:
        raise HTTPException(status_code=503, detail="memory service unavailable")
    return memory


@router.post("/memories")
async def create_memory(request: Request, spec: MemoryCreate):
    _ = _parse_uuid(spec.conversation_id, "conversation_id")
    if spec.supersedes is not None:
        _ = _parse_uuid(spec.supersedes, "supersedes")
    memory = _memory(request)
    try:
        memory_id = await memory.create_memory(spec)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"id": memory_id, "status": "active"}


@router.post("/memories/{memory_id}/supersede")
async def supersede_memory(memory_id: str, request: Request, body: SupersedeRequest):
    parsed = _parse_uuid(memory_id, "memory_id")
    memory = _memory(request)
    updated = await memory.supersede_memory(parsed, body.status)
    if not updated:
        raise HTTPException(status_code=404, detail="active memory not found")
    return {"id": memory_id, "status": body.status.value}


@router.get("/retrieval", response_model=RetrievalResponse)
async def retrieve(
    request: Request,
    q: str = Query(min_length=1),
    conversation_id: str = Query(alias="conversation_id"),
    limit: int | None = Query(default=None, ge=1),
):
    parsed = _parse_uuid(conversation_id, "conversation_id")
    memory = _memory(request)
    items = await memory.retrieve(q, parsed, limit)
    return RetrievalResponse(query=q, items=items)


@router.post("/conversations/{conversation_id}/index")
async def index_conversation(conversation_id: str, request: Request):
    parsed = _parse_uuid(conversation_id, "conversation_id")
    memory = _memory(request)
    created = await memory.index_completed_turns(parsed)
    return {"chunks_created": created}


@router.post("/index/rebuild")
async def rebuild_index(request: Request, conversation_id: str | None = None, force: bool = False):
    """Rebuild the derived Qdrant index from authoritative PostgreSQL (M5)."""
    memory = _memory(request)
    parsed = _parse_uuid(conversation_id, "conversation_id") if conversation_id else None
    summary = await memory.rebuild_vector_index(parsed, force=force)
    return {"status": "ok", **summary}


def _sanitized_endpoint(url: str) -> dict:
    """Host/port only — NEVER the full URL.

    base_url may embed credentials (`https://user:secret@host/v1`) or carry
    secrets in the query/fragment (`?api_key=...`, `#token`). Diagnostics
    expose just configured/host/port; username, password, path, query and
    fragment are dropped.
    """
    if not url:
        return {"configured": False, "host": None, "port": None}
    parsed = urlsplit(url)
    return {
        "configured": True,
        "host": parsed.hostname,
        "port": parsed.port,
    }


@router.get("/diagnostics")
async def diagnostics(request: Request):
    """Operational snapshot (M5). Never includes secrets or raw content."""
    app_state = request.app.state
    settings = app_state.settings
    database: Database | None = getattr(app_state, "database", None)
    db_ok = False
    if database is not None and database.available:
        try:
            await database.ping()
            db_ok = True
        except Exception:  # noqa: BLE001 - reported, never raised to client
            db_ok = False
    breaker = getattr(app_state, "breaker", None)
    return {
        "database": {
            "available": bool(database and database.available),
            "reachable": db_ok,
            "pool_size": getattr(getattr(database, "pool", None), "get_size", lambda: 0)(),
        },
        "memory_service": app_state.memory is not None,
        "context_engine": {
            "enabled": app_state.context_engine is not None,
            "assembly_enabled": settings.assembly.enabled,
        },
        "resilience": {
            "max_retries": settings.resilience.max_retries,
            "breaker_state": breaker.state if breaker else "unknown",
        },
        "rate_limit": {
            "enabled": settings.rate_limit.enabled,
            "requests_per_minute": settings.rate_limit.requests_per_minute,
            # Live bucket count — bounded by RATE_LIMIT__MAX_IDENTITIES.
            "live_identities": getattr(
                getattr(app_state, "rate_limiter", None), "identity_count", lambda: 0
            )(),
        },
        "inference": _sanitized_endpoint(settings.inference.base_url),
    }


@router.post("/conversations/{conversation_id}/context/preview")
async def context_preview(
    conversation_id: str, request: Request, body: ContextPreviewRequest
):
    """Dry-run the Context Assembly Engine for one conversation (M4 §11.11).

    Read-only: no persistence mutation, no inference call. The response is a
    diagnostic view (ids, scores, tokens, reasons) without raw message
    content, scoped strictly to the requested conversation.
    """
    parsed = _parse_uuid(conversation_id, "conversation_id")
    engine = getattr(request.app.state, "context_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="context engine unavailable")

    try:
        validate_chat_payload({"messages": body.messages, "tools": body.tools})
    except PayloadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    history, current_request = separate_current_request(body.messages)

    retrieved = []
    memory = getattr(request.app.state, "memory", None)
    query = extract_retrieval_query(body.messages)  # same helper as production
    if memory is not None and query:
        try:
            retrieved = await memory.retrieve(query, parsed)
        except RetrievalError:  # expected failure — preview degrades like prod
            retrieved = []

    try:
        plan = engine.build(
            history=history,
            current_request=current_request,
            tools=body.tools,
            retrieved=retrieved,
            conversation_id=str(parsed),
        )
    except EngineContextOverflowError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return plan.debug_view()
