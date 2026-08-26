# M8 — Release Candidate / Production Hardening

Timestamp: 2026-08-26 20:15 CEST

## Objective

Prepare the post-M7 codebase for release-candidate validation without introducing new core architecture.

M8 focuses on:

- production configuration correctness;
- API/provider compatibility;
- security hardening;
- failure/recovery behavior;
- persistence and deployment verification;
- deterministic CI quality;
- real-provider pre-release validation;
- documentation consistency.

M8 must not become a new feature-development milestone.

## Implemented in this branch

### Client-owned inference model selection

- `INFERENCE__MODEL` has been removed from the inference configuration model.
- Inference endpoint URL, credentials and timeout remain system configuration.
- The client `model` field is forwarded unchanged to the inference provider.
- `/v1/models` remains a passthrough to the configured inference endpoint.
- Embedding model configuration remains static and system-owned.
- Compact model configuration remains separate and system-owned for future compaction work.
- README, `.env.example` and configuration documentation were aligned with this contract.

## Remaining M8 validation gates

### Security

- verify production fail-closed internal authentication;
- verify docs/OpenAPI are disabled in production;
- verify request-ID sanitization and bounded logging;
- verify error responses do not leak internals;
- verify provider URL configuration does not create an unintended SSRF boundary;
- verify conversation isolation and history projection cannot cross sessions;
- verify dependency audit remains clean.

### Reliability

Exercise:

- PostgreSQL unavailable;
- Qdrant unavailable;
- inference provider unavailable;
- embedding provider unavailable;
- provider timeout/disconnect;
- client disconnect during streaming;
- persistence failure;
- transaction rollback;
- process restart.

Expected degraded modes must remain explicit and must not corrupt durable state.

### Persistence / recovery

Verify from a clean environment:

1. migrations from empty PostgreSQL;
2. conversation persistence;
3. restart with existing volumes;
4. Qdrant persistence;
5. rebuildability of derived vector state;
6. backup/restore procedure;
7. recovery after provider/database failure.

### Performance

Validate representative long-running sessions and concurrent traffic, with attention to:

- long history reconciliation;
- context assembly latency;
- capture memory bound;
- PostgreSQL pool usage;
- Qdrant latency;
- rate-limiter lock contention;
- process-local metrics/rate-limit/breaker semantics.

No M8 optimization should weaken correctness invariants.

### CI / release validation

Standard CI remains deterministic:

- Ruff;
- unit tests;
- PostgreSQL integration;
- Docker build/smoke;
- dependency security scan.

Real-provider E2E remains a manual/pre-release gate and must not depend on GPU/Ollama/model behavior in normal CI.

### OpenCode validation

Before RC, run a real OpenCode session covering:

- normal conversation;
- streaming;
- reasoning;
- function/custom tools;
- compaction;
- post-compaction continuation;
- multimodal request;
- multiple inference models exposed by the same provider;
- LCI restart during an existing conversation.

## Acceptance criteria

- [ ] Client selects inference model per request.
- [ ] No `INFERENCE__MODEL` configuration remains documented or required.
- [ ] Embedding URL/model remain static system configuration.
- [ ] `/v1/models` works against the configured provider.
- [ ] Security scan is green.
- [ ] Unit and PostgreSQL integration suites are green.
- [ ] Docker build/smoke is green.
- [ ] Production configuration is fail-closed where required.
- [ ] Failure/recovery behavior is verified.
- [ ] Persistence/restart behavior is verified.
- [ ] OpenCode real-world validation passes.
- [ ] Documentation matches the implementation.

M8 is complete only after the code review and all applicable release gates pass.
