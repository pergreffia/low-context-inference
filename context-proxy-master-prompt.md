# MASTER PROMPT — Context Proxy

## 0. Mission

Implement a production-quality, model-agnostic **Context Management Proxy** that exposes an OpenAI-compatible API to clients such as OpenCode.

The proxy must allow conversations substantially larger than the context window of the downstream inference model by:

- keeping the complete raw conversation locally;
- maintaining a protected recent-context window;
- extracting durable memories and decisions;
- storing historical chunks in a hybrid lexical/vector memory system;
- retrieving only context relevant to the current request;
- compacting archived history in batches;
- dynamically building a context that never exceeds the configured model budget;
- independently configuring the inference, compact, and embedding endpoints.

The system must work transparently with OpenCode.

The client must not need to know that a context-management layer exists.

---

# 1. Core Design Principle

The most important invariant of the entire system is:

> **Loss of context exposure must never imply loss of information.**

The raw conversation is the source of truth.

Compaction, summaries, embeddings, extracted memories, rankings, and pinned state are derived data.

Never destroy raw conversation data because it has been compacted.

---

# 2. Non-Goals

Do NOT turn this project into:

- an agent framework;
- a replacement for OpenCode;
- a generic orchestration platform;
- a distributed system;
- a multi-user SaaS;
- a workflow engine;
- a Redis-based architecture without a demonstrated need.

The system is primarily a **local context-management infrastructure**.

---

# 3. High-Level Architecture

```text
                         OpenCode
                            │
                            │ OpenAI-compatible API
                            ▼
                 ┌────────────────────┐
                 │   Context Proxy    │
                 │                    │
                 │ OpenAI-compatible  │
                 └─────────┬──────────┘
                           │
              ┌────────────┼─────────────┐
              │            │             │
              ▼            ▼             ▼
       Conversation      Memory       Providers
          Store          Service
              │            │        ┌──────┴──────┐
              │            │        │             │
              │            ▼     Compact       Embedding
              │          Qdrant   Endpoint       Endpoint
              │
              ▼
          PostgreSQL
```

Inference is external to the core Context Proxy:

```text
Context Proxy ───────────────► Inference Endpoint
```

The inference endpoint may be:

- Claude;
- GPT;
- Bonsai;
- llama.cpp;
- Ollama;
- another OpenAI-compatible service.

The compact endpoint may be a completely different model.

The embedding endpoint may be a third model/service.

---

# 4. Provider Independence

The Context Proxy MUST NOT be coupled to a specific LLM provider.

Define abstractions/interfaces for:

```text
ConversationStore
MemoryStore
VectorStore
EmbeddingProvider
LLMProvider
CompactProvider
```

Reference implementations:

```text
PostgresConversationStore
QdrantVectorStore
OpenAICompatibleEmbeddingProvider
OpenAICompatibleLLMProvider
OpenAICompatibleCompactProvider
```

All provider implementations must be replaceable through configuration.

---

# 5. Deployment Architecture

Provide a Docker Compose deployment with these services:

```text
context-proxy
memory-service
postgres
qdrant
embedding-service
compact-service
```

The embedding and compact services may initially wrap external/local OpenAI-compatible endpoints rather than hosting the models themselves.

The architecture must allow replacing them later.

Do not add Redis, Kafka, Elasticsearch, or other infrastructure unless a concrete requirement is demonstrated.

All services must provide:

- health checks;
- configurable ports;
- restart policies where appropriate;
- persistent volumes for stateful services;
- structured logging.

---

# 6. Context Proxy

The Context Proxy is the public-facing service.

It must expose at minimum:

```text
GET  /v1/models
POST /v1/chat/completions
```

The API must be OpenAI-compatible.

Support:

- streaming;
- non-streaming;
- system messages;
- user messages;
- assistant messages;
- tool calls;
- tool results;
- function calling;
- finish_reason;
- usage;
- model selection;
- OpenAI-compatible errors.

The proxy may transform the input `messages` used for inference.

It must NOT semantically rewrite or reinterpret the inference response.

Streaming responses must be passed through as directly as possible.

---

# 7. Conversation Store

PostgreSQL is the reference source of truth.

At minimum persist:

```text
conversations
messages
tool_calls
tool_results
conversation_chunks
memory_records
summaries
```

The raw message content must remain available.

Each message must have a stable identifier.

The system must support reconstructing the original conversation from persisted records.

---

# 8. Conversation Model

A conversation contains ordered messages.

Messages may be:

```text
system
user
assistant
tool
```

Tool calls must retain:

- tool call ID;
- tool name;
- arguments;
- associated assistant message;
- tool result;
- timestamps.

Never flatten structured tool calls into plain text as the only representation.

---

# 9. Conversation Chunks

Do not create an embedding for every individual message.

Group messages into semantic conversation chunks.

A chunk may contain:

```text
user request
assistant response
tool call
tool result
assistant interpretation
```

Do not separate a tool call from its corresponding result when they belong to the same interaction.

Each chunk must contain:

```text
chunk_id
conversation_id
message_ids
raw_content
summary
token_count
importance
created_at
embedding reference
```

The raw chunk remains recoverable.

---

# 10. Memory Record

Use the following logical structure:

```text
MemoryRecord
├── id
├── conversation_id
├── kind
├── content
├── source_message_ids
├── created_at
├── importance
├── status
├── supersedes
├── superseded_by
├── embedding_id
└── metadata
```

Supported `kind` values:

```text
decision
constraint
fact
task
bug
implementation
tool_result
episode_summary
conversation_summary
```

Supported `status` values:

```text
active
superseded
resolved
obsolete
```

Do not delete obsolete/superseded records from the source-of-truth database.

They must simply be excluded from active retrieval.

---

# 11. Source Traceability

Every derived memory must contain references to its source messages.

Example:

```json
{
  "id": "memory-381",
  "kind": "decision",
  "content": "The domain layer must not depend on GitHub adapters.",
  "source_message_ids": [
    "msg-184",
    "msg-185",
    "msg-186"
  ],
  "status": "active"
}
```

This allows the system to retrieve the original conversation when required.

Never make a summary the only representation of an important fact.

---

# 12. Decision Supersession

Decisions must support supersession.

Example:

```text
D001
"Use SQLite"
status = superseded
superseded_by = D014

D014
"Use PostgreSQL"
status = active
```

Before ranking retrieval candidates:

1. remove obsolete records;
2. remove superseded records;
3. retain only the currently active decision.

A historical decision must never be injected into the final context as if it were still active.

---

# 13. Pinned Memory

The system must support a special class of high-priority memory called `pinned`.

Pinned information may include:

- active architectural decisions;
- explicit constraints;
- durable requirements;
- current project goal;
- critical project state.

Pinned memory is always a candidate for the final context.

It is subject to a configurable token budget.

Example:

```yaml
context:
  pinned_budget_tokens: 2000
```

If pinned memory exceeds its budget, lower-priority records must be demoted to normal retrieval rather than arbitrarily truncating critical records.

---

# 14. Recent Window

The recent conversation window is protected.

Example configuration:

```yaml
context:
  recent_target_tokens: 14000
  recent_min_tokens: 10000
  recent_max_tokens: 18000
```

Recent messages must remain raw.

Do not compact the recent window during normal operation.

Tool calls and tool results must remain logically associated.

Do not split an interaction arbitrarily.

---

# 15. Context Budget

Never use the theoretical model context limit directly.

Configuration:

```yaml
context:
  model_limit_tokens: 32768
  safety_margin_tokens: 2048
```

Effective usable budget:

```text
usable_budget =
    model_limit_tokens
    - safety_margin_tokens
```

For a 32k model:

```text
32768 - 2048 = 30720
```

The budget manager must dynamically allocate this budget.

Example:

```text
system/tools        4k
pinned              2k
recent             14k
retrieved           7k
current request     2k
-----------------------
                    29k
```

If tool definitions grow, retrieval must shrink.

Do not exceed the effective budget.

---

# 16. Context Priority

When the budget is insufficient, use this priority order:

```text
1. system prompt
2. tool definitions
3. current user request
4. recent conversation
5. active pinned state
6. retrieved relevant raw context
7. retrieved historical summaries
8. generic historical summary
```

Never sacrifice the current user request.

Never silently exceed the model budget.

---

# 17. Historical Retrieval

Historical retrieval must be hybrid.

Do NOT rely exclusively on embeddings.

The retrieval pipeline must conceptually be:

```text
Current request
      │
      ├── semantic/vector search
      │
      └── lexical search
              │
              ▼
        candidate fusion
              │
              ▼
      metadata filtering
              │
              ▼
      supersession filtering
              │
              ▼
           ranking
              │
              ▼
       diversity/MMR
              │
              ▼
       budget selection
```

---

# 18. Retrieval Metadata

Retrieval candidates should support filtering by:

```text
conversation_id
kind
status
importance
created_at
source
```

At minimum, retrieval must be restricted to the relevant conversation/session unless explicit cross-conversation memory is implemented.

Do not leak memories between unrelated conversations.

---

# 19. Retrieval Ranking

Initial configurable score:

```text
score =
    0.40 * semantic_similarity
  + 0.25 * lexical_score
  + 0.15 * recency
  + 0.15 * importance
  + 0.05 * type_priority
```

Weights must be configurable.

The ranking implementation must be deterministic for equal inputs where practical.

---

# 20. Retrieval Diversity

Avoid returning many nearly identical chunks.

Example bad result:

```text
chunk-181
chunk-182
chunk-183
chunk-184
```

when all describe the same event.

Prefer diverse evidence:

```text
old decision
old bug
test failure
implementation
tool result
```

Implement MMR or an equivalent diversity-aware selection algorithm.

---

# 21. Raw vs Summary Retrieval

A historical chunk may have both:

```text
raw content
summary
```

If enough budget is available:

```text
retrieve raw
```

If budget is constrained:

```text
retrieve summary
```

This should be automatic.

Raw content has higher information fidelity.

Summary has lower token cost.

---

# 22. Retrieval Confidence

The system should support a minimum retrieval confidence threshold.

If retrieval produces only weak candidates, it is safer to inject less historical context than to inject irrelevant or contradictory information.

Do not fill the context budget merely because free tokens are available.

Relevance is more important than utilization.

---

# 23. Compaction

Compaction must NOT happen on every request.

Use configurable thresholds:

```yaml
compaction:
  trigger_ratio: 0.85
  target_ratio: 0.60
  min_archive_tokens: 10000
```

For a 32k context:

```text
trigger ≈ 27k
target  ≈ 19k
```

The purpose is to create significant headroom, not save a few tokens.

---

# 24. Compaction Strategy

Compaction must operate on archived history, not the protected recent window.

Conceptually:

```text
RAW HISTORY
│
├── protected recent window
│
└── archived history
      │
      ├── chunks
      ├── memories
      └── summaries
```

When sufficient archived material accumulates:

```text
archived chunks
      │
      ▼
compact endpoint
      │
      ▼
structured compact state
```

Do not compact only the oldest few tokens every time the threshold is crossed.

Compaction should be batch-oriented.

---

# 25. Compaction Batch

Example:

```text
archive:
    4k
    4k
    4k

total archived:
    12k
```

Only then invoke the compact endpoint.

Target:

```text
12k raw
   ↓
~1-2k structured state
```

The exact target is configurable.

The system must prevent pathological behavior where many consecutive requests each trigger a compact.

---

# 26. Compaction Output

The compact prompt must explicitly request structured state.

Conceptual output:

```json
{
  "decisions": [],
  "constraints": [],
  "facts": [],
  "open_tasks": [],
  "bugs": [],
  "implementations": [],
  "important_files": [],
  "obsolete_decisions": []
}
```

The compact model must be instructed:

- do not invent information;
- preserve exact identifiers;
- preserve file names;
- preserve class/function names;
- preserve important error messages;
- preserve active constraints;
- identify obsolete decisions;
- preserve unresolved problems;
- preserve important implementation details;
- retain source message references where possible.

Compact output must be token-budgeted.

Default:

```yaml
compact:
  max_output_tokens: 2048
```

---

# 27. Compact Provider

Compact endpoint must be independently configurable from inference.

Example:

```yaml
compact:
  base_url: http://localhost:8001/v1
  api_key: local
  model: bonsai
  max_output_tokens: 2048
```

Inference may simultaneously be:

```yaml
inference:
  base_url: https://provider.example/v1
  api_key: ${API_KEY}
  model: strong-model
```

This configuration must work without code changes.

---

# 28. Embedding Provider

Embedding endpoint must also be independent.

Example:

```yaml
embeddings:
  base_url: http://localhost:8002/v1
  api_key: local
  model: local-embedding-model
```

The memory system must not depend on the embedding model implementation.

---

# 29. Conversation Lifecycle

For every request:

```text
1. receive request
2. identify conversation
3. persist raw incoming messages
4. update recent window
5. archive messages when required
6. create/update conversation chunks
7. update retrieval index
8. check compaction conditions
9. compact archive if required
10. calculate context budget
11. retrieve relevant historical context
12. select final context
13. validate token budget
14. send inference request
15. stream/pass-through response
16. persist raw assistant response
17. update memory metadata
```

Do not reorder these steps unless required by correctness.

---

# 30. Response Handling

Inference responses must be treated as opaque protocol data wherever possible.

For streaming:

```text
Inference
   │
   │ SSE chunks
   ▼
Context Proxy
   │
   │ unchanged/pass-through
   ▼
OpenCode
```

Do not aggregate the entire response unnecessarily.

Do not rewrite tool calls.

Do not convert structured responses into text.

---

# 31. Failure Handling

The system must degrade gracefully.

### Vector database unavailable

Fallback to:

```text
recent + pinned + summary
```

### Embedding service unavailable

Continue using lexical retrieval where possible.

### Compact service unavailable

Do not block requests if a valid working context can still be constructed.

### Memory service unavailable

Fallback to:

```text
recent raw + pinned
```

### Inference endpoint unavailable

Return a valid OpenAI-compatible error.

### Context overflow

Never send an over-budget request.

The proxy must detect this before forwarding.

---

# 32. Persistence and Rebuild

Qdrant must be treated as a derived index.

There must be a mechanism to rebuild vector indexes from PostgreSQL.

The system must remain recoverable if:

```text
Qdrant data is lost
```

PostgreSQL must contain sufficient information to reconstruct derived memory/index state.

---

# 33. API Contracts Between Services

Define explicit internal APIs.

At minimum:

```text
Context Proxy → Memory Service
    store message
    store chunk
    retrieve memories
    retrieve raw chunks
    create memory
    supersede memory

Context Proxy → Compact Service
    compact archive

Memory Service → Embedding Service
    create embeddings
```

Use typed request/response schemas.

Do not pass arbitrary unvalidated dictionaries between services.

Version internal APIs where appropriate.

---

# 34. Configuration

All important behavior must be configurable.

Example:

```yaml
server:
  host: 0.0.0.0
  port: 8080

inference:
  base_url: ...
  api_key: ...
  model: ...

compact:
  base_url: ...
  api_key: ...
  model: ...
  max_output_tokens: 2048

embeddings:
  base_url: ...
  api_key: ...
  model: ...

context:
  model_limit_tokens: 32768
  safety_margin_tokens: 2048
  pinned_budget_tokens: 2000
  recent_target_tokens: 14000
  recent_min_tokens: 10000
  recent_max_tokens: 18000

retrieval:
  semantic_weight: 0.40
  lexical_weight: 0.25
  recency_weight: 0.15
  importance_weight: 0.15
  type_weight: 0.05

compaction:
  trigger_ratio: 0.85
  target_ratio: 0.60
  min_archive_tokens: 10000
```

Never hardcode model-specific limits in application logic.

---

# 35. Observability

Implement structured logging.

Every request should be traceable with a request ID.

Expose metrics/log information for:

- request latency;
- inference latency;
- retrieval latency;
- embedding latency;
- compact latency;
- number of retrieved chunks;
- retrieved tokens;
- final context tokens;
- recent tokens;
- pinned tokens;
- compact invocations;
- compact input tokens;
- compact output tokens;
- fallback events;
- provider errors.

Do not log API keys or sensitive credentials.

---

# 36. Testing

Implement comprehensive automated tests.

## Context tests

Verify:

- recent window remains raw;
- tool call/result remain associated;
- context never exceeds configured budget;
- system prompt is preserved;
- current user request is preserved;
- retrieval shrinks when budget shrinks.

## Memory tests

Verify:

- source message IDs are correct;
- superseded decisions are excluded;
- obsolete memories are excluded;
- duplicate chunks are deduplicated;
- diversity selection works.

## Compaction tests

Explicitly test that:

```text
100 requests
+ small context increments
```

do NOT result in:

```text
100 compact operations
```

Also verify:

```text
raw history before compaction
==
raw history after compaction
```

## API tests

Test:

- OpenAI-compatible request;
- streaming;
- non-streaming;
- tool calls;
- tool results;
- errors;
- usage;
- models endpoint.

## Failure tests

Simulate:

- Qdrant unavailable;
- PostgreSQL unavailable;
- embedding service unavailable;
- compact service unavailable;
- inference service unavailable.

---

# 37. A/B Benchmark

Create a benchmark comparing:

### Baseline

```text
Bonsai + native 32k context
```

### Context Proxy

```text
Bonsai + virtual context
```

Use real or representative coding tasks.

Measure:

- task success rate;
- test pass rate;
- tool-call correctness;
- regressions;
- unnecessary edits;
- latency;
- retrieval calls;
- compact calls;
- inference tokens;
- compact tokens.

The goal is to verify that virtual context does not significantly degrade coding quality.

---

# 38. Acceptance Criteria

The project is complete when:

1. OpenCode can use the proxy without functional client-side changes.
2. The proxy supports streaming and tool calls.
3. Raw conversation data is never lost.
4. Final context never exceeds the configured model budget.
5. Recent messages remain raw.
6. Superseded decisions are never presented as active.
7. Retrieval combines semantic and lexical search.
8. Retrieval applies diversity.
9. Compaction does not execute on every request.
10. Compact and inference can use completely different models.
11. Embeddings can be replaced through configuration.
12. Qdrant can be rebuilt from PostgreSQL.
13. The system degrades gracefully when retrieval or compaction is unavailable.
14. The complete stack runs with Docker Compose.
15. Automated tests verify context budgeting.
16. Automated tests verify OpenAI-compatible behavior.
17. An A/B benchmark exists for native versus virtual context.
18. No provider-specific logic leaks into the Context Manager core.
19. No raw history is destroyed by compaction.
20. The system is usable as a transparent OpenAI-compatible endpoint for OpenCode.

---

# 39. Milestones

Implement incrementally.

## M1 — Foundation

Implement:

- repository structure;
- project configuration;
- Docker Compose skeleton;
- provider abstractions;
- PostgreSQL;
- basic OpenAI-compatible proxy;
- inference passthrough;
- streaming passthrough.

Do not implement memory yet.

All M1 tests must pass before proceeding.

---

## M2 — Conversation Management

Implement:

- conversations;
- messages;
- tool calls/results;
- raw persistence;
- recent window;
- token counting;
- dynamic context budgeting.

Tests must prove the context never exceeds the configured budget.

---

## M3 — Memory Service

Implement:

- Memory Service;
- PostgreSQL memory records;
- Qdrant;
- embeddings;
- conversation chunks;
- hybrid retrieval;
- metadata filtering;
- supersession.

All memory data must be traceable back to raw messages.

---

## M4 — Compaction

Implement:

- archive lifecycle;
- batch compaction;
- structured compact output;
- compact provider;
- memory extraction;
- decision supersession;
- summary management.

Explicitly test that compaction does not happen on every request.

---

## M5 — Advanced Context Selection

Implement:

- configurable scoring;
- hybrid ranking;
- diversity/MMR;
- raw-vs-summary selection;
- dynamic budget allocation;
- retrieval confidence threshold;
- pinned state.

This milestone is critical to final quality.

---

## M6 — Reliability

Implement:

- retries;
- timeouts;
- health checks;
- graceful degradation;
- provider failure handling;
- structured logging;
- metrics;
- persistence/index rebuild.

---

## M7 — Validation

Implement:

- full integration test suite;
- OpenCode compatibility tests;
- A/B benchmark;
- latency measurements;
- memory usage measurements;
- token usage measurements;
- compact frequency measurements.

---

# 40. Development Rules

Follow these rules throughout implementation:

1. Do not skip tests.
2. Do not silently simplify requirements.
3. Do not replace Qdrant/PostgreSQL with in-memory mocks in production code.
4. Mocks are acceptable only in unit tests.
5. Keep service boundaries explicit.
6. Keep provider abstractions independent of concrete implementations.
7. Never destroy raw conversation data.
8. Never exceed the context budget.
9. Never silently discard current user input.
10. Preserve OpenAI-compatible protocol semantics.
11. Preserve tool-call structure.
12. Do not introduce unnecessary infrastructure.
13. Prefer deterministic algorithms where possible.
14. Make thresholds and ranking weights configurable.
15. Document non-obvious design decisions.
16. Add regression tests for every discovered bug.
17. Keep commits small and logically scoped.
18. Run the relevant test suite after each milestone.
19. Do not mark a milestone complete while known test failures remain.
20. Before implementing a major architectural deviation, document the reason and update the design.

---

# 41. Required Deliverables

The implementation must include:

```text
README.md
ARCHITECTURE.md
CONFIGURATION.md
API.md
DEVELOPMENT.md
docker-compose.yml
.env.example
```

The README must explain:

- architecture;
- local deployment;
- configuration;
- connecting OpenCode;
- configuring inference;
- configuring compact;
- configuring embeddings;
- rebuilding Qdrant;
- running tests.

---

# 42. Final Engineering Principle

The Context Proxy is not attempting to make the underlying LLM's context window physically larger.

It creates a **virtual context layer**:

```text
                 Virtual Context
                       │
       ┌───────────────┼────────────────┐
       │               │                │
   recent raw      pinned state    historical retrieval
       │               │                │
       └───────────────┼────────────────┘
                       ▼
                fixed-size context
                       │
                       ▼
                     LLM
```

The model sees only the most relevant information.

The system, however, retains the complete information locally.

The fundamental goal is:

> **Given a model with N tokens of context, construct the highest-value N-token representation of the entire available conversation for the current request, while retaining the complete original conversation outside the model.**
