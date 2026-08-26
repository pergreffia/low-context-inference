# Deployment

## Prerequisites

| Component | Requirement | Notes |
|---|---|---|
| Docker + Docker Compose | v2+ | `docker compose` syntax |
| PostgreSQL | 16 (any ≥ 14 works) | source of truth; Compose uses `postgres:16-alpine` |
| Qdrant | v1.12.x pinned in Compose (`qdrant/qdrant:v1.12.4`) | derived vector index |
| Embedding provider | any OpenAI-compatible `/embeddings` (Ollama serves these too) | needed for semantic retrieval only |
| Upstream inference provider | any OpenAI-compatible endpoint | Ollama, vLLM, LM Studio, ... |

Python 3.12+ only if running from source instead of the image.

## Compose topology

`docker-compose.yml` defines:

| Service | Image / build | Published ports | Internal name |
|---|---|---|---|
| `context-proxy` | built from `Dockerfile` | `${PROXY_PORT:-11435}:8080` (host → container) | `context-proxy` |
| `postgres` | `postgres:16-alpine` | `127.0.0.1:${POSTGRES_PORT:-5432}:5432` (**loopback only**) | `postgres` |
| `qdrant` | `qdrant/qdrant:v1.12.4` | `127.0.0.1:6333:6333`, `127.0.0.1:6344:6344` (**loopback only**) | `qdrant` |

- Container-to-container traffic uses service names
  (`DATABASE__URL=…@postgres:5432`, `QDRANT__BASE_URL=http://qdrant:6333`);
  the loopback-only publishes exist solely for host-side tooling. Remove the
  `ports:` blocks entirely if you do not need host access.
- Volumes: `pgdata` (PostgreSQL), `qdrant_storage` (Qdrant).
- Health checks gate startup: the proxy waits for
  `postgres`/`qdrant` `service_healthy`.
- Restart policy: `unless-stopped` on all services.
- Proxy hardening: `no-new-privileges:true`, `cap_drop: [ALL]`,
  `PYTHONDONTWRITEBYTECODE=1`, `PYTHONUNBUFFERED=1`, non-root user inside the
  image (`USER contextproxy`), image `HEALTHCHECK`.
- `env_file: .env` is loaded by the proxy service.

## Startup

```bash
cp .env.example .env         # then edit: inference endpoint (+ production security)
docker compose up --build -d
```

Migrations run automatically during proxy startup
(`Database.start()` → `apply_migrations`). There is no separate migration
step. A reachable database whose schema cannot be migrated aborts proxy
startup (fail-closed); an unreachable database degrades to passthrough mode
instead — see [operations.md](operations.md#degraded-modes).

Verify:

```bash
curl -s http://localhost:11435/healthz     # {"status":"ok","database":"ok"}
curl -s http://localhost:11435/metrics | head
```

## Connecting a provider

The proxy speaks to **any** OpenAI-compatible endpoint; there is no
provider-specific integration. Set in `.env` (host-run) or inject into the
proxy container for Compose:

### Ollama on the host

```bash
INFERENCE__BASE_URL=http://host.docker.internal:11434/v1   # from inside Compose
INFERENCE__MODEL=MichelRosselli/bonsai-27b:latest          # optional override
CONTEXT__MODEL_LIMIT_TOKENS=32768                          # must match the served model
```

From a source checkout on the same machine simply use
`http://localhost:11434/v1`.

Embeddings for memory retrieval follow the same pattern:

```bash
EMBEDDINGS__BASE_URL=http://host.docker.internal:11434/v1
EMBEDDINGS__MODEL=nomic-embed-text
QDRANT__BASE_URL=http://qdrant:6333                        # already set by Compose
```

Without embeddings/Qdrant the proxy still serves traffic with lexical-only
retrieval or plain recent-history context.

### Other providers (vLLM, LM Studio, cloud gateways)

Same two variables: `INFERENCE__BASE_URL` (+ optional `INFERENCE__API_KEY`
sent as `Bearer`). `/v1/models` responses are passed through verbatim;
response bodies are never rewritten.

## Production checklist

- **Reverse proxy / TLS**: terminate TLS in front of the proxy (nginx,
  Caddy, Traefik). The proxy speaks plain HTTP internally on :8080 (published as
  `${PROXY_PORT:-11435}`) and sets no
  TLS options.
- **Internal endpoints**: put `/internal/*` behind your ingress deny-rule
  AND set:
  ```bash
  SECURITY__MODE=production
  SECURITY__INTERNAL_AUTH_TOKEN=<openssl rand -hex 32>
  ```
  With `production` mode an empty token refuses to start (fail-closed).
  Clients must send `X-Internal-Auth: <token>` for administrative calls.
- **Secrets**: pass via environment/secrets manager; nothing is written to
  disk by the app. Logs are redacted for bearer/api-key patterns.
- **PostgreSQL persistence**: keep the `pgdata` volume; back it up with
  standard `pg_dump`/physical backups — it is the source of truth.
- **Qdrant persistence**: keep `qdrant_storage`; loss is recoverable via
  `POST /internal/v1/index/rebuild` (rebuilds from PostgreSQL).
- **Published port**: Compose publishes the API on host port
  `PROXY_PORT` (default **11435**); the container itself always listens on
  8080. Override with `PROXY_PORT=<port> docker compose … up -d`.
- **Monitoring**: scrape `/metrics` (Prometheus text format); watch request
  counters/latency, breaker gauge, degradation/capture-overflow/rate-limit
  counters. `/metrics` has no application-level authentication — restrict it
  at the ingress/network layer. See [operations.md](operations.md).
- **Workers**: `SERVER__WEB_CONCURRENCY=N` scales uvicorn workers; metrics
  and rate buckets are per-worker (documented consequence).

See also: [configuration.md](configuration.md) · [security.md](security.md)
