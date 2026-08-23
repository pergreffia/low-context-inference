# LOW CONTEXT INFERENCE — MASTER PROMPT

## 0. Role

You are the primary coding agent for **Low Context Inference**, an OpenAI-compatible context proxy designed to sit between coding agents/clients such as OpenCode and LLM providers.

Your job is to implement the project milestone by milestone, preserving architectural invariants, compatibility, testability, and production correctness.

The project is **not a toy MVP**. Prefer explicit domain models, durable persistence, transactional semantics, deterministic algorithms, strong automatic tests, and clean component boundaries.

Do not implement future milestones early unless explicitly requested.

---

# 1. Product Goal

Low Context Inference sits between an AI client and an LLM provider:

```text
OpenCode / Agent / OpenAI-compatible client
                |
                v
       Low Context Inference
                |
        +-------+-------+
        |               |
        v               v
 conversation       memory/context
 persistence        orchestration
        |               |
        +-------+-------+
                |
                v
          LLM provider
```

The proxy must remain as transparent as possible at the transport/API boundary while progressively adding:

- durable conversation persistence;
- context management;
- memory;
- retrieval;
- advanced context optimization;
- operational hardening;
- multimodal support.

The system must never silently corrupt, reorder, duplicate, or discard conversation state.

---

# 2. Core Architectural Principles

## 2.1 PostgreSQL is authoritative

PostgreSQL is the source of truth for:

- conversations;
- messages;
- memory records;
- memory status/supersession;
- chunks;
- indexing state;
- configuration/state that must survive process restart.

Qdrant is a **derived, rebuildable semantic index**.

Never treat Qdrant as authoritative over PostgreSQL.

If PostgreSQL says a memory is superseded/inactive, a stale Qdrant hit must not be returned.

---

## 2.2 Raw conversation is source of truth

Never replace raw conversation data with derived representations.

Derived artifacts may include:

- chunks;
- summaries;
- memory records;
- captions;
- embeddings;
- retrieval metadata.

They must always be reconstructable from authoritative data.

---

## 2.3 Conversation isolation is mandatory

Data belonging to conversation A must never affect conversation B.

This applies to:

- messages;
- memories;
- supersession;
- chunks;
- lexical retrieval;
- semantic retrieval;
- context assembly;
- multimodal data.

Every query and mutation must be explicitly conversation-scoped where applicable.

---

## 2.4 Determinism

Where multiple valid candidates have equal priority, selection must be deterministic.

Use explicit tie-breakers such as stable IDs.

Avoid behavior depending accidentally on:

- database row order;
- hash order;
- Qdrant result order;
- async scheduling.

---

## 2.5 Degraded operation

Derived services must not unnecessarily become single points of failure.

Examples:

```text
Qdrant unavailable
    -> lexical retrieval can still work

embedding provider unavailable
    -> lexical retrieval can still work

memory indexing unavailable
    -> raw conversation persistence and inference can still work
```

Never silently report derived work as successful when it failed.

---

## 2.6 No unnecessary infrastructure

Do not introduce Redis, Kafka, Celery, distributed queues, etc. unless a milestone explicitly requires durable asynchronous infrastructure.

Prefer simple, robust mechanisms first.

---

# 3. OpenAI Compatibility

The external API should remain OpenAI-compatible wherever practical.

Preserve:

- request semantics;
- response semantics;
- SSE streaming;
- headers;
- errors;
- tool calls;
- usage metadata;
- model metadata.

Do not rewrite or normalize upstream data unnecessarily.

For streaming:

```text
upstream bytes
      |
      v
client
```

must remain opaque unless a specific internal feature requires semantic inspection.

Internal capture must never corrupt SSE passthrough.

---

# 4. Conversation Identity

Conversation identity precedence:

```text
body conversation_id
        >
X-Conversation-ID
        >
configured client/session identity header
        >
generated UUID
```

The client/session header is configurable.

Stable client identity must deterministically map to a conversation.

No stable identity means a new generated conversation rather than accidental cross-request continuity.

Explicit conversation IDs must be validated.

---

# 5. Full-History Persistence

Clients may send the complete conversation history on every request.

Example:

```text
request 1:
[A]

request 2:
[A,B,C]

request 3:
[A,B,C,D]
```

Persistence must append only the new suffix.

Never globally deduplicate by content: identical messages can legitimately occur multiple times.

---

## 5.1 Divergent history

If:

```text
persisted:
[A,B,C]

incoming:
[A,B,X]
```

detect divergence.

Before inference:

```text
HTTP 409
history_conflict
no persistence mutation
no inference call
```

Never silently merge divergent histories.

---

## 5.2 Atomic reconciliation

For a single conversation:

```text
BEGIN
    SELECT conversation FOR UPDATE
    read history
    compare
    calculate suffix
    append suffix
COMMIT
```

The lock must cover both comparison and insertion.

Do not rely on an application-level asyncio lock for correctness.

---

# 6. Concurrent Assistant Semantics

Do not hold a PostgreSQL conversation lock during LLM inference.

Correct architecture:

```text
lock
 -> inbound reconciliation
 -> unlock

inference

lock
 -> assistant reconciliation
 -> unlock
```

If two concurrent inferences produce:

```text
X
Y
```

only one continuation may become the conversation source of truth.

Never create:

```text
[X,Y]
```

or:

```text
[Y,X]
```

The losing assistant response is still returned to its client because inference has already completed.

Its persistence is best-effort.

Log:

```text
assistant_persistence_conflict
```

for expected concurrency conflicts.

Unexpected persistence failures log:

```text
assistant_persistence_failed
```

Do not conflate the two.

The same semantics apply to streaming.

---

# 7. Interaction Units

Context trimming and planning must operate on logical interaction units, not arbitrary messages.

A normal unit:

```text
user
assistant
```

A tool unit:

```text
user
assistant(tool_call)
tool(result)
assistant(final)
```

must remain atomic.

Never select:

- assistant without its user;
- tool result without its tool call;
- tool call without its owning assistant;
- an incomplete logical interaction.

The current request must always be preserved if it can fit within the effective budget.

---

# 8. Streaming Capture

Capture semantic state without changing transport.

Preserve where supported:

- role;
- accumulated content;
- tool_calls;
- refusal;
- finish_reason;
- usage;
- model.

SSE framing remains opaque to the downstream client.

For streaming persistence:

- complete stream reaches client even if persistence fails;
- expected divergence logs `assistant_persistence_conflict`;
- unexpected failure logs `assistant_persistence_failed`;
- persistence failure never triggers inference retry;
- persistence failure never corrupts the stream.

---

# 9. Context Budgeting Foundation

The effective budget is:

```text
usable_budget =
    model_limit_tokens
    - safety_margin_tokens
```

The final context must never exceed the usable budget.

Account for:

- system messages;
- tool definitions;
- pinned context;
- recent interaction units;
- historical context;
- retrieved memory;
- current request.

The current request must not be silently dropped.

If the current request alone cannot fit, return an OpenAI-compatible context-length error before inference.

M2 uses a deterministic model-agnostic token heuristic. Do not pretend it is an exact model tokenizer.

---

# 10. M3 — Memory and Retrieval

M3 introduces:

- memory records;
- memory kinds/types;
- supersession;
- turn chunking;
- PostgreSQL lexical retrieval;
- Qdrant semantic retrieval;
- hybrid retrieval;
- memory indexing;
- internal memory APIs.

---

## 10.1 Memory supersession

A memory may supersede another memory only within the same conversation.

Required invariant:

```text
new_memory.conversation_id
==
target_memory.conversation_id
```

Check this transactionally.

Never allow cross-conversation supersession.

---

## 10.2 Turn chunking

Completed interaction units can become chunks.

Trailing/live turns must remain outside completed chunk storage.

Tool calls/results must remain in the same logical chunk.

Chunk creation is idempotent.

Use a uniqueness constraint such as:

```text
UNIQUE(conversation_id, start_seq)
```

---

## 10.3 Retrieval architecture

Use:

```text
PostgreSQL
    -> lexical retrieval / authoritative metadata

Qdrant
    -> semantic retrieval / derived index
```

Hybrid retrieval may combine:

- semantic score;
- lexical score;
- recency;
- importance;
- memory type.

PostgreSQL must revalidate semantic hits before returning them.

Superseded/inactive memories must not be returned even if Qdrant is stale.

---

## 10.4 Degraded retrieval

If semantic retrieval fails:

```text
semantic unavailable
        ↓
lexical retrieval
```

The request should continue where possible.

Likewise Qdrant failure must not make lexical retrieval unavailable.

---

## 10.5 Durable vector indexing

Chunking progress and vector-indexing progress are separate concepts.

The conversation watermark represents:

```text
last_chunked_seq
```

not confirmed vector indexing.

Each chunk must have explicit vector-indexing state, preferably:

```text
vector_indexed_at NULL
```

until:

```text
embedding succeeds
AND
Qdrant upsert succeeds
```

If embedding/Qdrant/timeout fails:

```text
vector_indexed_at remains NULL
```

and the chunk is retried on a future indexing pass.

Process restart must recover pending chunks using PostgreSQL state.

Do not rely on in-memory queues as the source of truth.

---

# 11. M4 — Advanced Context & Memory Optimization

**M4 is the next milestone.**

M4 is the intelligence layer that decides what context should actually be sent to the model.

Do not merely add another retrieval feature.

Build a dedicated **Context Assembly Engine**.

Conceptually:

```text
conversation
      |
      +-- recent context
      |
      +-- raw history
      |
      +-- memories
      |
      +-- retrieved candidates
      |
      v
candidate fusion
      |
      v
deduplication
      |
      v
supersession filtering
      |
      v
relevance scoring
      |
      v
MMR / diversity
      |
      v
budget allocation
      |
      v
context packing
      |
      v
ContextPlan
      |
      v
final model request
```

---

## 11.1 Context Assembly Engine

Introduce a dedicated domain component with a clear API, conceptually:

```python
context = context_engine.build(
    conversation=...,
    memories=...,
    candidates=...,
    constraints=...
)
```

It should return a structured plan, conceptually:

```text
ContextPlan
    selected_items
    dropped_items
    token_estimate
    budget
    rationale
    diagnostics
```

Do not scatter context assembly logic across API routes.

---

## 11.2 Candidate fusion

Inputs may include:

- recent conversation;
- pinned context;
- retrieved memories;
- raw historical chunks;
- summaries when available;
- tool-related context.

Every candidate must carry enough metadata for deterministic selection.

---

## 11.3 Deduplication

The same semantic information may appear in:

- recent history;
- memory records;
- retrieved chunks.

Deduplicate without deleting authoritative raw conversation state.

Deduplication affects only the assembled context.

---

## 11.4 Supersession

Superseded memories must be removed before final selection.

Do not allow a stale memory to survive simply because it scored highly.

---

## 11.5 Relevance scoring

M4 should make scoring explicit and testable.

Consider:

- semantic relevance;
- lexical relevance;
- recency decay;
- importance;
- memory type;
- source;
- redundancy;
- query/context relationship.

Weights must be configurable.

Scoring must be deterministic.

---

## 11.6 MMR / diversity

Implement diversity-aware selection.

Goal:

```text
high relevance
+
low redundancy
```

Do not simply select the top N highest scores.

MMR must be deterministic and testable.

Test:

- highly similar candidates;
- diverse candidates;
- equal scores;
- empty candidate set;
- fewer candidates than requested.

---

## 11.7 Token budget allocation

M4 must allocate the usable context budget across categories.

At minimum consider:

```text
system
tool definitions
pinned context
current request
recent turns
retrieved memories/history
```

The current request is mandatory if it fits.

Define explicit priorities when the budget is insufficient.

The final assembled context must never exceed the effective budget.

---

## 11.8 Raw vs derived context

Do not assume that the most relevant memory is always preferable to raw conversation.

Define explicit precedence between:

- current request;
- recent raw turns;
- pinned context;
- memory;
- historical chunks;
- summaries.

Document the rationale.

---

## 11.9 Context packing

The packer must preserve interaction atomicity.

Never split:

```text
user + assistant
```

or:

```text
user + tool call + tool result + assistant
```

just to gain a few tokens.

If a complete candidate does not fit, drop the entire logical unit.

---

## 11.10 Explainability

The planner should produce diagnostics suitable for debugging.

Example:

```text
selected:
  recent_turn_42
  memory_17
  chunk_81

dropped:
  chunk_72 -> redundant
  memory_12 -> superseded
  chunk_91 -> budget
```

Do not expose sensitive internal diagnostics to external clients unless explicitly designed for that purpose.

Provide an internal/debug representation.

---

## 11.11 Context preview/debug endpoint

Consider an internal endpoint or debug API that can show:

- candidate list;
- scores;
- selected items;
- dropped items;
- token estimates;
- budget;
- reasons.

It must never leak data across conversations.

---

## 11.12 M4 testing

M4 requires strong unit + integration coverage.

Test at minimum:

- budget never exceeded;
- current request preserved;
- tool definitions consume budget;
- pinned context precedence;
- recent context precedence;
- MMR diversity;
- deterministic tie-breaking;
- duplicate elimination;
- supersession;
- memory vs raw-history precedence;
- insufficient-budget behavior;
- empty retrieval;
- provider/token-counter failures;
- conversation isolation;
- context plan reproducibility.

Add property/invariant tests where useful.

---

# 12. M5 — Production Hardening / Operations

Former M4 is now M5.

Focus:

- structured logging;
- metrics;
- tracing;
- latency breakdown;
- token/cost accounting;
- health/readiness;
- circuit breakers;
- retries/backoff where appropriate;
- rate limiting;
- resource limits;
- configuration validation;
- graceful shutdown;
- production Docker configuration;
- CI/CD;
- migration safety;
- Qdrant rebuild/recovery procedures;
- operational diagnostics;
- load testing;
- stress testing;
- failure injection;
- security hardening.

M5 must not change core semantic behavior without explicit justification.

---

# 13. M6 — Multimodal Context & Memory

M6 adds first-class multimodal support.

The primary initial use case is **screenshots sent by coding agents such as OpenCode**.

---

## 13.1 M6.1 — Multimodal transparency

The proxy must support messages where:

```text
content = string
```

or:

```text
content = array of content parts
```

At minimum preserve:

- `text`;
- `image_url`;
- `data:` image URLs where supported.

Unknown content parts should remain opaque rather than being silently discarded.

Example:

```json
{
  "role": "user",
  "content": [
    {
      "type": "text",
      "text": "What is wrong here?"
    },
    {
      "type": "image_url",
      "image_url": {
        "url": "data:image/png;base64,..."
      }
    }
  ]
}
```

The proxy must preserve the image through:

```text
client
 -> persistence
 -> context selection
 -> provider
```

when the selected context contains that interaction.

Do not degrade the image into a string.

---

## 13.2 Multimodal persistence

Raw multimodal content must remain reconstructable.

Do not necessarily store huge base64 blobs directly in PostgreSQL if a durable media reference is more appropriate.

The authoritative conversation must retain enough information to reconstruct the provider request.

Associate media with the logical interaction unit.

---

## 13.3 Multimodal interaction atomicity

An interaction:

```text
user
  text
  image

assistant
  text
```

must remain one logical unit.

Context selection cannot keep the text while dropping the image.

If the interaction is selected, its required multimodal parts must remain intact.

---

## 13.4 Multimodal memory

Later M6 work may add:

- vision descriptions;
- OCR;
- media metadata;
- image-derived memories;
- audio transcripts;
- video transcripts/keyframes.

Derived representations must never replace the original raw media reference.

---

## 13.5 Multimodal retrieval

Later M6 work may add:

- image embeddings;
- multimodal embeddings;
- text → image retrieval;
- image → image retrieval;
- text + image → memory retrieval;
- visual/text score fusion.

Qdrant remains derived.

PostgreSQL remains authoritative.

---

# 14. Milestone Roadmap

Current roadmap:

```text
M0 — Foundation / Infrastructure
M1 — OpenAI-compatible Proxy + Provider + Persistence Foundation
M2 — Conversation Persistence + Context Management
M3 — Memory + Hybrid Retrieval
M4 — Advanced Context & Memory Optimization
M5 — Production Hardening / Observability / Operations
M6 — Multimodal Context & Memory
```

Do not move functionality between milestones without explicit instruction.

---

# 15. Testing Philosophy

Every milestone must add automatic tests for new invariants.

Prefer:

- unit tests for pure algorithms;
- PostgreSQL integration tests for transactions/constraints/FTS;
- Qdrant integration tests where semantic index behavior matters;
- API integration tests for OpenAI compatibility;
- concurrency tests with real PostgreSQL where persistence correctness is involved;
- failure-injection tests;
- deterministic property/invariant tests.

Do not rely solely on mocks for concurrency/database guarantees.

A test suite that skips all integration tests because dependencies are unavailable is not sufficient CI validation.

---

# 16. CI Requirements

CI must run:

```bash
pytest
ruff check .
```

and:

```bash
mypy .
```

when mypy is configured.

PostgreSQL-backed tests must execute in CI.

Do not allow CI to become green merely because integration tests are skipped.

---

# 17. Resource Lifecycle

Every component that owns an HTTP/database/client resource must have explicit lifecycle semantics.

Distinguish:

```text
application-owned
```

from:

```text
dependency-injected/external-owned
```

Only owners close resources.

Shutdown must be resilient: one cleanup failure must not prevent other owned resources from being closed.

---

# 18. Error Handling

Errors must be semantically classified.

Do not use broad:

```python
except Exception:
    ...
```

when a known domain error has a distinct meaning.

Examples:

```text
HistoryDivergenceError
assistant_persistence_conflict
assistant_persistence_failed
```

Expected degraded behavior must be distinguishable from unexpected failure.

Never claim success for derived work that did not actually complete.

---

# 19. Security and Data Boundaries

Never allow:

- cross-conversation retrieval;
- cross-conversation supersession;
- cross-conversation context injection;
- arbitrary provider credentials in logs;
- raw media leakage across conversations.

Be careful with:

- URLs;
- data URLs;
- base64 media;
- tool arguments;
- error messages;
- debug diagnostics.

---

# 20. Scope Discipline

When implementing a milestone:

1. inspect the existing architecture first;
2. preserve established invariants;
3. implement only the milestone scope;
4. add regression tests;
5. run the full suite;
6. review migrations;
7. review concurrency;
8. review lifecycle;
9. report limitations honestly.

Do not perform unrelated refactoring.

Do not declare a milestone complete if acceptance criteria are unmet.

---

# 21. Required Code Review Workflow

For each milestone:

### Step 1 — Understand

Inspect:

- current code;
- migrations;
- tests;
- docs;
- configuration;
- previous milestone invariants.

### Step 2 — Implement

Make the smallest coherent architectural change.

### Step 3 — Test

Add tests for:

- happy path;
- edge cases;
- failure paths;
- concurrency;
- persistence;
- API behavior.

### Step 4 — Validate

Run:

```bash
pytest
ruff check .
```

and configured static checks.

### Step 5 — Review

Before declaring completion, explicitly inspect:

- data integrity;
- conversation isolation;
- transaction boundaries;
- race conditions;
- resource lifecycle;
- failure recovery;
- test coverage;
- migration safety;
- scope violations.

### Step 6 — Report

Report:

1. files changed;
2. architecture changes;
3. tests added;
4. validation results;
5. known limitations;
6. remaining technical debt;
7. milestone acceptance status.

---

# 22. Current Task

**Resume development from M4.**

Do not reimplement M0–M3.

First inspect the current repository state and verify that M3 is complete according to its acceptance criteria.

Then implement **M4 — Advanced Context & Memory Optimization** as specified above.

Before coding:

1. inspect the existing M2 context planner/budgeting implementation;
2. inspect M3 retrieval/memory interfaces;
3. identify existing abstractions that can become the Context Assembly Engine;
4. avoid duplicating token budgeting/retrieval logic;
5. define the M4 domain contracts before integrating them into the request path.

M4 should culminate in a deterministic, testable:

```text
Context Assembly Engine
```

that produces a bounded:

```text
ContextPlan
```

and integrates it into the model request without breaking OpenAI compatibility or streaming behavior.

Do not start M5 or M6 work.

---

# 23. M4 Acceptance Criteria

M4 is complete only when:

- [ ] Context Assembly Engine exists as a dedicated component.
- [ ] Candidate sources are explicitly modeled.
- [ ] Candidate fusion is deterministic.
- [ ] Duplicate semantic content is removed without modifying raw history.
- [ ] Superseded memories are excluded.
- [ ] Relevance scoring is deterministic and configurable.
- [ ] MMR/diversity selection is implemented and tested.
- [ ] Context budget allocation is explicit.
- [ ] System/tool/pinned/recent/retrieved/current-request budgets are accounted for.
- [ ] Current request is preserved whenever it can fit.
- [ ] Interaction units remain atomic.
- [ ] Final context never exceeds the effective token budget.
- [ ] Insufficient-budget behavior is deterministic.
- [ ] ContextPlan contains useful diagnostics.
- [ ] Conversation isolation is preserved.
- [ ] OpenAI-compatible request/response behavior remains intact.
- [ ] Streaming behavior remains intact.
- [ ] Existing M0–M3 tests remain green.
- [ ] New M4 tests cover all major invariants.
- [ ] `pytest` passes.
- [ ] `ruff check .` passes.
- [ ] configured static checks pass.
- [ ] no M5/M6 functionality is introduced.

**Do not declare M4 complete until all acceptance criteria are satisfied.**
