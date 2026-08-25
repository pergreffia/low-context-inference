# Low Context Inference

Model-agnostic context-management proxy exposing an OpenAI-compatible API.
It sits between any OpenAI-compatible client (OpenCode, Codex CLI, IDE
agents, plain `curl`) and any OpenAI-compatible inference backend (Ollama,
vLLM, LM Studio, ...), letting clients run conversations far larger than the
downstream model's context window while keeping the complete raw conversation
in PostgreSQL.

## What problem does it solve

Local/self-hosted models have small context windows and coding agents keep
growing conversations. Naive truncation loses instructions, decisions and
file context. Low Context Inference persists every message verbatim, then assembles a
**fresh, budget-fitting provider request for each turn** from the raw history plus
derived memory — instead of forwarding ever-growing payloads or blindly
truncating them.

## What makes it different from a plain proxy

- **Raw conversation is the source of truth** — stored verbatim in PostgreSQL,
  positionally reconciled, never rewritten; divergent client histories are
  rejected (`409`) instead of silently merged.
- **Context Assembly Engine** — per turn it selects what reaches the model:
  system/developer instructions and the current request are mandatory;
  ordinary history is dropped oldest-first as an atomic window; retrieved
  memories/chunks are packed last under an explicit token budget
  (`model_limit − safety_margin`).
- **First-class `developer` role** — persisted verbatim, protected like
  `system`, never normalized away.
- **Hybrid memory** — completed turns are chunked and embedded into Qdrant;
  typed memory records (decision/constraint/fact/task/bug/...) support
  supersession; retrieval fuses semantic + lexical + recency + importance +
  type signals, scoped strictly to one conversation.
- **Multimodal transparency** — `content` part arrays (`text`, `image_url`)
  pass through persistence → context selection → provider untouched; images
  are flat-costed in the budget and excluded from retrieval queries.
- **Tool transparency** — `function` and `custom` tool calls are validated,
  captured from streamed deltas, persisted (raw + relational projection) and
  replayed byte-exactly.
- **Byte-exact streaming passthrough** — SSE frames reach the client
  untouched; a bounded side-channel captures the assistant message for
  persistence without ever delaying or rewriting the stream.
- **Production posture** — bounded rate limiting (client + conversation
  dimensions), circuit breaker with provably-pre-send-only retries,
  Prometheus metrics, structured JSON logs with secret redaction,
  fail-closed `/internal/*` authentication policy in production mode.

## Architecture at a glance

```text
OpenAI-compatible client
        ↓ HTTPS
FastAPI app (validation · conversation identity · rate limit)
        ↓
PostgreSQL  ← raw conversations (source of truth) + relational projections
        ↓
Context Assembly Engine  ← retrieval ← Qdrant ← embeddings
        ↓
Upstream inference provider (any OpenAI-compatible endpoint)
        ↓
capture + persistence  →  client (byte-exact response/SSE)
```

Details: [docs/architecture.md](docs/architecture.md)

## Main capabilities

| Area | What it does | Docs |
|---|---|---|
| API | `/v1/chat/completions` (+ streaming), `/v1/models`, OpenAI-compatible errors | [api.md](docs/api.md) |
| Context | token budgeting, protected tiers, deterministic eviction | [memory-and-context.md](docs/memory-and-context.md) |
| Memory | chunking, embeddings, hybrid retrieval, supersession | [memory-and-context.md](docs/memory-and-context.md) |
| Multimodal | image parts, media registry, flat budget cost | [multimodal.md](docs/multimodal.md) |
| Tools | function/custom calls, streamed reconstruction, replay | [api.md](docs/api.md) |
| Security | internal auth policy, two-dimension rate limit, redaction | [security.md](docs/security.md) |
| Ops | health/readiness/metrics/diagnostics, degraded modes | [operations.md](docs/operations.md) |

## Quick start

### Run from source (development)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" -c constraints.txt
cp .env.example .env          # point INFERENCE__BASE_URL at your provider
uvicorn context_proxy.main:app --host 127.0.0.1 --port 8080
```

PostgreSQL is optional locally: when unreachable the proxy serves pure
passthrough and reports `database: degraded`.

### Docker Compose

```bash
docker compose up --build
```

This starts `context-proxy` (:8080), `postgres` (loopback-only :5432) and
`qdrant` (loopback-only :6333/6344). Migrations run automatically at proxy
startup. Details: [docs/deployment.md](docs/deployment.md).

### Point a client at it

Any OpenAI-compatible client works:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "m",
    "messages": [{"role": "user", "content": "hi"}],
    "conversation_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7"
  }'
```

Repeat with the same `conversation_id` and the full history builds up server-side —
clients may send just the new user message each turn.

### Configure an upstream provider

The proxy forwards to whatever OpenAI-compatible endpoint you configure:

```bash
# Ollama example
INFERENCE__BASE_URL=http://localhost:11434/v1
INFERENCE__MODEL=MichelRosselli/bonsai-27b:latest   # optional: overrides client model
CONTEXT__MODEL_LIMIT_TOKENS=32768                   # must match the model's real window
```

Full reference: [docs/configuration.md](docs/configuration.md).

## Development & testing quick start

```bash
pip install -e ".[dev]" -c constraints.txt
ruff check .
pytest -q                                            # unit suite (PG tests skip)
TEST_DATABASE_URL=postgresql://context_proxy:context_proxy@localhost:5433/context_proxy pytest -q
```

See [docs/development.md](docs/development.md) and
[docs/testing.md](docs/testing.md).

## Status

The implemented system covers conversation management, context assembly,
hybrid memory, multimodal transparency, tool-call lifecycle, operational
hardening and security boundaries, verified by a deterministic test suite
(~630 tests incl. PostgreSQL integration). Not implemented: distributed
rate limiting/metrics backends, OTel tracing export, automatic memory
extraction, compaction/summarization candidates, vision-derived
descriptions of images.

## Documentation map

| Document | Content |
|---|---|
| [docs/architecture.md](docs/architecture.md) | components, request lifecycle, data flow, trust boundaries |
| [docs/configuration.md](docs/configuration.md) | full configuration reference + example envs |
| [docs/deployment.md](docs/deployment.md) | Docker/Compose, providers, production checklist |
| [docs/operations.md](docs/operations.md) | operator runbook: health, backups, troubleshooting |
| [docs/api.md](docs/api.md) | public API + streaming + errors + internal API |
| [docs/memory-and-context.md](docs/memory-and-context.md) | reconciliation, budgeting, memory, retrieval, invariants |
| [docs/multimodal.md](docs/multimodal.md) | content parts, media registry, budget/retrieval behavior |
| [docs/security.md](docs/security.md) | threat model, auth policy, rate limiting, retry safety, secrets |
| [docs/development.md](docs/development.md) | repo layout, workflows, conventions |
| [docs/testing.md](docs/testing.md) | test strategy, commands, real-provider E2E |
