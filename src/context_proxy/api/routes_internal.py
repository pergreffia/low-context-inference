"""Internal Memory Service API (master prompt §33).

Typed request/response schemas; local infrastructure surface — not part of
the public OpenAI-compatible contract.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from context_proxy.memory.models import (
    MemoryCreate,
    RetrievalResponse,
    SupersedeRequest,
)

router = APIRouter(prefix="/internal/v1")


def _memory(request):
    memory = getattr(request.app.state, "memory", None)
    if memory is None:
        raise HTTPException(status_code=503, detail="memory service unavailable")
    return memory


@router.post("/memories")
async def create_memory(request: Request, spec: MemoryCreate):
    memory = _memory(request)
    try:
        memory_id = await memory.create_memory(spec)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"id": memory_id, "status": "active"}


@router.post("/memories/{memory_id}/supersede")
async def supersede_memory(memory_id: str, request: Request, body: SupersedeRequest):
    memory = _memory(request)
    updated = await memory.supersede_memory(memory_id, body.status)
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
    memory = _memory(request)
    items = await memory.retrieve(q, conversation_id, limit)
    return RetrievalResponse(query=q, items=items)


@router.post("/conversations/{conversation_id}/index")
async def index_conversation(conversation_id: str, request: Request):
    memory = _memory(request)
    created = await memory.index_completed_turns(conversation_id)
    return {"chunks_created": created}
