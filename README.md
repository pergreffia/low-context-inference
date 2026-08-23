# Context Proxy

Model-agnostic context management proxy exposing an OpenAI-compatible API. Lets clients like OpenCode run conversations larger than the downstream model context window while keeping the complete raw conversation locally.

See `context-proxy-master-prompt.md` for the full design.

## Status: M4 — Context Assembly Engine

M3 memory service plus a dedicated **Context Assembly Engine** (`context/engine.py`) that decides what context actually reaches the model:

```text
candidate fusion → deduplication → supersession filtering
    → relevance scoring → MMR diversity → budget allocation
    → packing → ContextPlan
```

- **Explicit candidate sources**: system messages, tool definitions, pinned context, the current request, recent raw interaction units, retrieved memories, retrieved chunks — each carrying stable keys, token counts, and score components for deterministic selection.
- **Category precedence under budget pressure**: system + tools > pinned > current request > recent turns (newest-first, one contiguous window) > retrieved blocks (MMR order). Recent raw truth outranks derived memories by design.
- **Deduplication without touching raw state**: a memory restating a recent turn, or a chunk whose stored JSON-lines match the live window, is dropped from the *assembled request only*; persisted history is never rewritten. Identical messages legitimately repeated in raw history survive (no content-based dedup of authoritative data).
- **Supersession + isolation defense-in-depth**: PostgreSQL already returns active-only records; the engine additionally rejects any superseded id passed by the caller and any retrieval block whose `conversation_id` differs from the request's.
- **Deterministic scoring & MMR**: weighted components (semantic/lexical/recency/importance/type — same weights as hybrid retrieval), greedy MMR with `ASSEMBLY__MMR_LAMBDA`, ties always broken by stable key. Identical inputs produce byte-identical plans.
- **Budget guarantees**: the final context never exceeds `usable = model_limit − safety_margin`; tool definitions consume budget; if system + tools + pinned + current request alone cannot fit, the proxy answers OpenAI `context_length_exceeded` (HTTP 400) before any inference call.
- **Diagnostics**: every plan records selected/dropped items with reasons (`duplicate`, `superseded`, `foreign_conversation`, `budget`, `not_selected`), category token accounting, and weights. `POST /internal/v1/conversations/{id}/context/preview` returns a read-only debug view (ids/scores/reasons only, no raw content) scoped to one conversation.
- **Compatibility unchanged**: streaming SSE passthrough, response bodies, headers, usage metadata and M0–M3 behavior are untouched; retrieval outages degrade to raw-recent context automatically.

Configuration: `ASSEMBLY__ENABLED` (default on; off restores the M2 window planner), `ASSEMBLY__MMR_LAMBDA`, `ASSEMBLY__MAX_RETRIEVED_ITEMS`, `ASSEMBLY__RETRIEVED_BUDGET_TOKENS`.

Not yet implemented: pinned-context ingestion source (the engine packs it, but no API populates it yet), summaries as candidate source, compaction.

## Previous milestones

### M3 — Memory Service

M2 foundation plus:

- **Turn chunking**: completed interaction turns are automatically chunked (`conversation_chunks`, idempotent via `(conversation_id, start_seq)`); tool calls stay attached to their results; the trailing turn and system prompts stay raw in the recent window.
- **Memory records**: typed kinds (decision/constraint/fact/task/bug/implementation/…) with importance, source-message traceability, and **supersession** — superseding a decision marks the old record `superseded` (never deleted, only excluded from active retrieval).
- **Hybrid retrieval**: Qdrant semantic leg + PostgreSQL full-text lexical leg, fused with configurable weights (`RETRIEVAL__*`, master prompt §19), conversation-scoped by payload filter and SQL predicate. Embedding/Qdrant outages degrade to lexical-only; nothing is lost because PostgreSQL is the source of truth and the index is rebuildable.
- **Indexing latency**: chunking/embedding/vector-upsert run synchronously after the response is persisted but are bounded by `MEMORY__INDEX_TIMEOUT_SECONDS` (default 10s); timeouts and failures never alter the HTTP response. Background workers can replace this later without interface changes.
- **Internal API**: `/internal/v1/memories` (create/supersede), `/internal/v1/memories/{id}/supersede`, `/internal/v1/retrieval?q=&conversation_id=`, `/internal/v1/conversations/{id}/index`.

Not yet implemented at M3: memory *extraction* from conversations and compaction, diversity/MMR + budget selection + prompt injection of retrieved context.

### M2 — Conversation Management

M1 foundation plus:

- **Raw persistence** (PostgreSQL, source of truth): conversations, messages (verbatim JSONB), tool calls/results with relational integrity. The original conversation is always reconstructable.
- **Conversation identification** (precedence): body `conversation_id` > `X-Conversation-ID` header > configured session header (`CONVERSATION__CLIENT_ID_HEADER`, default `X-Session-ID`) > generated UUID. Explicit ids must be valid UUIDs; session identities are opaque tokens mapped deterministically to a UUID. Stable client identity → same conversation; no identity → fresh conversation per request. The resolved id is echoed via `X-Conversation-ID` and never forwarded upstream.
- **Full-history idempotency**: clients that resend the whole conversation each turn only get the new suffix persisted. Divergent history is rejected with HTTP 409 `history_conflict` before inference — raw history is never rewritten.
- **Concurrency**: all persistence takes a PostgreSQL row lock on the conversation (`SELECT … FOR UPDATE`) inside the transaction that reads/compares/writes history, so concurrent identical requests stay idempotent and divergent ones can never silently merge — across workers, containers, processes. No application-level lock is relied upon.
- **Locks are never held during inference.** The flow is deliberately `lock → reconcile inbound → unlock → inference → lock → reconcile assistant → unlock`: a DB row lock plus a seconds/minutes-long model call would turn PostgreSQL into the serialization mechanism for inference. Assistant persistence is therefore best-effort: if two concurrent requests produce different responses, the first continuation commits and the loser gets an `assistant_persistence_conflict` warning while still receiving its own generated response. Optimistic versioning may be considered later (out of M2 scope).
- **Orphan tool results**: a tool result with an unknown `tool_call_id` keeps `tool_call_ref = NULL`, the raw message stays reconstructable, and an `orphan_tool_result` warning is logged. Duplicate call ids resolve deterministically to the newest matching call within the same conversation.
- PostgreSQL unavailable → passthrough continues in degraded mode (nothing persisted, `/healthz` reports `database: degraded`).
- **Token counting + dynamic budgeting**: heuristic counter (~4 chars/token); `usable = model_limit − safety_margin`; when messages exceed the budget the oldest complete interaction units are dropped (system prompts and the current request are never sacrificed; tool calls stay attached to their results). Impossible requests get OpenAI error `context_length_exceeded` (HTTP 400) instead of being forwarded.

Still ahead (M3+): memory service, hybrid retrieval, compaction, context selection.

## Local development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # adjust inference endpoint
uvicorn context_proxy.main:app --host 0.0.0.0 --port 8080
```

### PostgreSQL availability

- **Local process startup**: PostgreSQL is optional at M1. If unreachable, the proxy logs a warning and serves inference passthrough in degraded mode (`/healthz` reports `database: degraded`). This is intentional graceful degradation, not a weakening of the source-of-truth design.
- **Docker Compose deployment**: PostgreSQL is a hard dependency — the standard Compose stack waits for it to be healthy before starting the proxy.

### Qdrant

Qdrant is defined in Docker Compose as infrastructure reserved for later milestones (vector retrieval, M3). It is **not** a runtime dependency of `context-proxy` in M1: the proxy starts and runs without it. It becomes a runtime dependency when vector retrieval is implemented.

## Model selection

If `INFERENCE__MODEL` is set, it overrides the client's `model` field on chat completions (streaming and non-streaming). If unset, the client's model is forwarded unchanged. `/v1/models` is always passed through as-is.

## Docker

```bash
docker compose up --build
```

## Configuration

All settings via environment variables using `__` nesting (see `.env.example`): `INFERENCE__BASE_URL`, `COMPACT__MODEL`, `DATABASE__URL`, etc. Inference, compact, and embedding endpoints are independently configurable.

## Connect OpenCode

Point any OpenAI-compatible client at the proxy:

```bash
BASE_URL=http://localhost:8080/v1
API_KEY=unused
MODEL=<model served by your inference endpoint>
```

The proxy forwards requests to the configured inference endpoint and passes responses through unchanged (response bodies are never rewritten; hop-by-hop headers are stripped).

## Tests

```bash
pytest
```

PostgreSQL-backed suites (store, concurrency, memory service, engine integration) run when `TEST_DATABASE_URL` is set, e.g.:

```bash
TEST_DATABASE_URL=postgresql://context_proxy:context_proxy@localhost:5433/context_proxy pytest
```
