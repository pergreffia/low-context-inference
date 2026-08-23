# Context Proxy

Model-agnostic context management proxy exposing an OpenAI-compatible API. Lets clients like OpenCode run conversations larger than the downstream model context window while keeping the complete raw conversation locally.

See `context-proxy-master-prompt.md` for the full design.

## Status: M1 — Foundation

Implemented:

- repository structure, project configuration;
- provider abstractions (`ConversationStore`, `MemoryStore`, `VectorStore`, `EmbeddingProvider`, `LLMProvider`, `CompactProvider`);
- PostgreSQL schema + idempotent migrations (source of truth);
- Docker Compose skeleton (context-proxy, postgres, qdrant);
- basic OpenAI-compatible proxy:
  - `GET /v1/models` — passthrough;
  - `POST /v1/chat/completions` — passthrough, streaming (SSE) and non-streaming;
  - `/healthz`.

Not yet implemented (later milestones): conversation persistence logic, memory service, retrieval, compaction.

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
