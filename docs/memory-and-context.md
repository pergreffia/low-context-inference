# Memory and Context

This is the core differentiating behavior: the proxy keeps the complete raw
conversation in PostgreSQL and builds a fresh, budget-fitting provider
request every turn.

## Conversation state

- **Identity**: precedence body `conversation_id` → `X-Conversation-ID` →
  session header (uuid5-derived) → generated UUID. Explicit ids must be
  valid UUIDs.
- **Persistence**: every inbound message is stored verbatim as JSONB
  (`messages.jsonb`) under `(conversation_id, seq)`; the assistant response
  is appended after inference. The original conversation — text, tool calls,
  images, ordering — is always reconstructable exactly (`get_messages`
  returns raw JSONB).
- **Relational projections** exist for queryability only (tool_calls,
  tool_results, conversation_media); they never replace or rewrite raw data.

## History reconciliation

Each request reconciles the client-sent history against stored truth:

1. `ensure_conversation` inserts the conversation row if new.
2. The conversation row is locked `SELECT … FOR UPDATE` inside one
   transaction.
3. Stored history and incoming history are compared **positionally**
   (index by index, exact dict equality).
4. Only the new suffix is inserted. Identical replays are idempotent.
5. Any divergence raises `HistoryDivergenceError` → HTTP `409
   history_conflict`; the database is left untouched.

Guarantees:

- no silent merge, ever;
- locks are held per-write only — never across inference;
- concurrent identical writers persist the suffix exactly once (row lock);
  concurrent *divergent* writers: first commit wins, loser gets `409`.
- assistant persistence after inference repeats the same reconcile with
  `[*inbound_messages, assistant_message]`; a competing continuation that
  loses logs `assistant_persistence_conflict` while its own client still
  receives its response.

Clients must serialize requests per conversation (client-side sequencing);
the server guarantees state consistency, not inference ordering.

## Context construction

Per request:

```text
messages = [system?, developer?, user/assistant/tool …, current_user…]
separate_current_request()  → history | current_request
engine.build(history=…, current_request=…, tools=…, retrieved=…) → ContextPlan
out_payload.messages = plan.messages        # sent upstream
```

`ContextPlan.messages` is what the model sees. It is assembled from raw
stored/inbound dicts plus labeled retrieval blocks — never rewritten prose.

### Candidate sources

| Source | Tier / role |
|---|---|
| system messages | trusted instruction tier |
| developer messages | trusted instruction tier (own units, emitted first when mid-history) |
| tool definitions | counted separately, mandatory alongside instructions |
| pinned context | reserved tier (engine packs it; no ingestion API yet) |
| current request | mandatory atomic candidate, always rendered LAST |
| recent turns | raw interaction units, newest-first contiguous window |
| retrieved memories/chunks | untrusted derived blocks, user-role + `[retrieved …]` provenance header |

Interaction atomicity: a unit starts at a `user` message and spans its
assistant/tool-call/tool-result tail; units are dropped whole. A
developer/system message becomes its own protected unit without closing an
open interaction.

## Budgeting

```
usable_budget = CONTEXT__MODEL_LIMIT_TOKENS − CONTEXT__SAFETY_MARGIN_TOKENS
```

Eviction order under pressure:

1. retrieved blocks (MMR-ordered, capped by `ASSEMBLY__RETRIEVED_BUDGET_TOKENS`,
   max `ASSEMBLY__MAX_RETRIEVED_ITEMS`);
2. oldest recent turns first, as whole atomic units (assistant orphaning is
   impossible — a turn drops together with its answer);
3. never: system, developer, tools, current request.

If mandatory content alone exceeds the budget → `400
context_length_exceeded` before any upstream call.

### Token accounting

Deterministic heuristic (`context/tokens.py`): ≈4 chars/token plus fixed
overheads per message/tool-call. Multimodal parts: text parts counted as
text; `image_url` flat `1024`; unknown part types flat `16` — base64 size
never distorts the budget. Counts are estimates absorbed by the safety
margin; all counts are non-negative.

## Invariants (enforced by tests)

- budget never exceeded: `plan.token_estimate ≤ usable_budget`;
- system/developer survive any pressure; current request always present and last;
- ordinary history evicted before any trusted instruction;
- retrieved content can never render as `system`/`developer` (structural guard);
- conversation isolation: retrieval filters on `conversation_id`, engine
  additionally rejects foreign-conversation candidates;
- duplicates compared positionally — identical texts are not collapsed from
  authoritative history;
- deterministic: identical inputs → identical plans (tie-breaks by stable key);
- raw conversation remains source of truth; assembly never mutates it;
- token accounting happens exactly once per upstream response.

With `ASSEMBLY__ENABLED=false` the simpler window planner provides the same
hard guarantees minus MMR/retrieval packing.

## Memory lifecycle

Records are created through `/internal/v1/memories`:

```json
{"kind": "decision", "content": "use SQLite for tests",
 "conversation_id": "<uuid>", "importance": 0.8,
 "supersedes": "<optional memory id>",
 "source_message_ids": ["<uuid>", "..."]}
```

Kinds: decision, constraint, fact, task, bug, implementation, tool_result,
episode_summary, conversation_summary. Statuses: active, superseded,
resolved, obsolete. Superseding marks the old record non-active — records
are excluded from retrieval, never deleted.

There is **no automatic memory extraction** in the current implementation:
memories are operator/API-created. What IS automatic is turn chunking +
embedding (below).

## Retrieval

Hybrid pipeline (`memory/service.py::retrieve`):

```text
query (text-only parts of latest user message)
  ├─ semantic leg: embed → Qdrant search (conversation-scoped filter)
  └─ lexical leg: PostgreSQL full-text over chunks/memories
fusion by id (union) → active-status filtering (supersession)
→ weighted score = semantic·0.40 + lexical·0.25 + recency·0.15
                   + importance·0.15 + type_priority·0.05
→ sort (-score, id), cut to limit
```

Type priority (fixed): decision .9, constraint .85, task/bug .7, summaries
.6, fact/implementation .5, chunk .4, tool_result .3.

Failure behavior: Qdrant/embedding outage → semantic scores empty
(lexical-only); lexical leg infra failure → `RetrievalError` → engine degrades
to raw/recent context. Programming errors propagate (visible, not masked).

## Indexing

After each persisted response (`MEMORY__AUTO_INDEX=true`):

1. **Chunking**: completed interaction units become `conversation_chunks`
   rows; the trailing (live) unit stays raw. Watermark
   `conversations.last_chunked_seq` makes it incremental and idempotent.
2. **Embedding + upsert**: pending chunks (`vector_indexed_at IS NULL`)
   are embedded (batched, truncated at `MEMORY__MAX_EMBED_CHARS`) and upserted
   into Qdrant; the marker flips only after both succeed — partial failures
   self-heal on the next pass. Whole pass bounded by
   `MEMORY__INDEX_TIMEOUT_SECONDS`.

Recovery: `POST /internal/v1/index/rebuild` rebuilds vectors from
PostgreSQL (optionally one conversation, `force=true` re-chunks).

Provenance: every RetrievedItem carries source ids and (chunks)
`start_seq/end_seq` pointing into authoritative history.

See also: [architecture.md](architecture.md) · [multimodal.md](multimodal.md)
