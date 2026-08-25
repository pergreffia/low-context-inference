# Architecture

This document describes the implemented system as of the current `main`.
Every claim is grounded in the code under `src/context_proxy/` and the test
suite under `tests/`.

## Component map

| Component | Code | Role |
|---|---|---|
| HTTP/API layer | `main.py`, `api/routes.py` | FastAPI app, `/v1/*` chat flow, orchestration of persistence/context/inference |
| Internal API | `api/routes_internal.py` | administrative surface under `/internal/v1/*` (memory CRUD, retrieval, indexing, rebuild, diagnostics, context preview) |
| Request validation | `api/validation.py` | structural shape checks (messages/tools/stream/n); semantic meaning deliberately not validated |
| Response/error mapping | `api/responses.py`, `providers/errors.py` | OpenAI-style error envelopes; upstream error passthrough vs normalization |
| Header policy | `providers/headers.py` | hop-by-hop filtering, content-encoding truthfulness for buffered vs streamed bodies |
| Conversation identity | `conversation/identity.py` | precedence: body `conversation_id` > `X-Conversation-ID` header > session header (uuid5-mapped) > generated UUID |
| Conversation store | `conversation/store.py`, `db/database.py`, `db/migrations/` | PostgreSQL source of truth: verbatim message JSONB + relational projections (tool_calls/tool_results/conversation_media), row-lock reconciliation |
| Context planner | `context/planner.py` | M2 window fallback planner + interaction segmentation (`segment_messages`) |
| Context engine | `context/engine.py` | candidate fusion → dedup → supersession/foreign-conversation filters → scoring → MMR → tiered budget allocation → `ContextPlan` |
| Token estimation | `context/tokens.py` | deterministic heuristic (~4 chars/token), multimodal flat costs |
| Memory service | `memory/service.py` | turn chunking (watermark-based), embedding+Qdrant indexing with durable per-chunk markers, hybrid retrieval, memory records, rebuild |
| Embeddings | `memory/embeddings.py` | OpenAI-compatible `/embeddings` client, batched |
| Vector store | `memory/qdrant.py` | minimal Qdrant HTTP client; collection compatibility gate; 404 cache invalidation |
| Inference provider | `providers/llm.py`, `providers/base.py` | passthrough client for any OpenAI-compatible endpoint; buffered and streaming paths |
| Resilience | `providers/resilience.py` | circuit breaker (CLOSED/OPEN/HALF_OPEN) + bounded retries restricted to provably pre-send transport failures |
| Capture/persistence | `capture.py` | byte-exact SSE passthrough with a bounded side-channel that reconstructs the assistant message; overflow-safe |
| Observability | `observability/middleware.py`, `metrics.py`, `logging_setup.py` | request-id correlation, low-cardinality metrics, structured logs with redaction, body-size limits |
| Rate limiter | `observability/ratelimit.py` | in-process token buckets on two dimensions (client/IP + conversation), hard memory cap |

## Request lifecycle (`POST /v1/chat/completions`)

Implemented order in `ObservabilityMiddleware.__call__` then
`api/routes.py::chat_completions`:

```text
client
  ↓
ASGI middleware (observability/rate limit)
  ├─ request-id correlation (X-Request-ID honored or generated)
  ├─ Content-Length / drained-body size check        → 413
  ├─ rate limit admission (client bucket + conversation bucket) → 429
  ↓
JSON parse + structural validation                   → 400
  ↓
conversation id resolution                           → 400 invalid_conversation_id
  ↓
inbound persistence (store present only)
  ├─ ensure conversation row
  ├─ positional reconcile of full inbound history    → 409 history_conflict
  └─ infrastructure failure → degrade to passthrough (store=None)
  ↓
retrieval (engine path, best-effort; RetrievalError → empty)
  ↓
context planning
  ├─ separate_current_request(messages)
  ├─ engine.build(...) or planner.plan_context(...)
  └─ overflow                                        → 400 context_length_exceeded
  ↓
upstream call (provider)
  ├─ streaming branch: PersistingLLMStream wraps SSE passthrough
  └─ buffered branch: parse choices[0].message + metadata
  ↓
assistant persistence (best-effort; conflicts/failures logged, response intact)
  ↓
token accounting (exactly once per upstream response)
  ↓
response to client (byte-exact body/SSE; X-Conversation-ID echoed)
```

Notes that are load-bearing:

- **Locks never span inference.** The store takes a conversation row lock
  (`SELECT … FOR UPDATE`) inside each reconciliation transaction only.
- **Assistant persistence is best-effort by design**: concurrent competing
  continuations log `assistant_persistence_conflict`; every client still gets
  its own generated response.
- **Degradation ladder**: PostgreSQL down → passthrough-only (no history);
  Qdrant/embedding down → lexical-only retrieval; both retrieval legs down →
  raw/recent context only. The upstream provider being down is *not*
  degraded — it surfaces as `502 upstream_unavailable`.

## Data flow and ownership

| Data | Owner | Storage | Derived? |
|---|---|---|---|
| Raw conversations/messages | ConversationStore | PostgreSQL `messages.jsonb` verbatim | no — source of truth |
| Tool-call relational projection | Store (`_persist_tool_parts`) | PostgreSQL `tool_calls` (+0012 `call_type/input/extra`) | yes — rebuildable from raw JSONB |
| Tool results + association | Store | `tool_results` (`tool_call_ref` NULL when orphan) | yes |
| Media metadata registry | Store (`_persist_media_parts`) | `conversation_media` (source data/url, sha256, part_index, source_size) | yes |
| Turn chunks | MemoryService | `conversation_chunks` + watermark `conversations.last_chunked_seq` | derived |
| Vectors | MemoryService via QdrantVectorStore | Qdrant collection `context_proxy` | derived, rebuildable (`/internal/v1/index/rebuild`) |
| Memory records | operator/API | `memory_records` (status active/superseded/…) | authoritative records, but excluded-from-retrieval when superseded |
| Assembled context | Context engine | ephemeral (`ContextPlan`) | derived per request |

Rebuild path: PostgreSQL → chunk/embed/upsert (`rebuild_vector_index`),
optionally scoped to one conversation.

## Trust boundaries

| Boundary | Trusted? | Enforcement |
|---|---|---|
| Public clients calling `/v1/*` | untrusted payload | structural validation only; oversized bodies cut at 413; rate limited |
| `X-Conversation-ID` header | untrusted | must be valid UUID when used explicitly; otherwise ignored as principal |
| Retrieved memories/chunks | **untrusted derived content** | rendered as user-role messages with `[retrieved …]` provenance header; structural guard raises if a derived candidate ever renders as `system` |
| Model-generated assistant content | untrusted | persisted verbatim, never executed or re-interpreted as instructions by the proxy |
| Upstream provider responses | semi-trusted | bodies passed through opaquely; headers filtered; error status preserved/normalized |
| `/internal/v1/*` | administrative | network boundary + optional `X-Internal-Auth` token; production mode refuses to boot without one |
| PostgreSQL / Qdrant | infrastructure | reachable via private network; Compose publishes them loopback-only |
| Client-supplied tools/payloads | untrusted | validated structurally (function/custom/unknown types pass through opaquely) |

## State ownership summary

- **Proxy process owns**: in-flight streams, capture buffers (bounded),
  breaker state, rate-limit buckets (bounded), metrics registries,
  readiness cache in the Qdrant client. All process-local; multi-worker
  deployments keep per-worker instances (documented consequence).
- **PostgreSQL owns**: all authoritative conversation state.
- **Qdrant owns**: nothing authoritative — vectors only, always rebuildable.
- **Client owns**: full-history responsibility for reconciliation semantics;
  may send suffixes or full histories, divergence is rejected rather than merged.

See also: [api.md](api.md) · [security.md](security.md) ·
[memory-and-context.md](memory-and-context.md)
