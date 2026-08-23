# Context Proxy

Model-agnostic context management proxy exposing an OpenAI-compatible API. Lets clients like OpenCode run conversations larger than the downstream model context window while keeping the complete raw conversation locally.

See `context-proxy-master-prompt.md` for the full design.

## Status: M2 — Conversation Management

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

Integration test for PostgreSQL migrations auto-skips when no server is reachable.
