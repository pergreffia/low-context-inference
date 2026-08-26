# Configuration

All configuration is environment-based, parsed by
`context_proxy/config.py` (pydantic-settings). Nesting uses `__`
(e.g. `SERVER__MAX_BODY_BYTES`). A `.env` file in the working directory is
honored; see `.env.example` for a complete skeleton.

Namespaces actually implemented: `SERVER`, `DATABASE`, `CONTEXT`,
`ASSEMBLY`, `CONVERSATION`, `INFERENCE`, `QDRANT`, `RETRIEVAL`, `MEMORY`,
`RESILIENCE`, `RATE_LIMIT`, `SECURITY`, `COMPACT`, `EMBEDDINGS`.

There are **no** `OBSERVABILITY__*` or `MEDIA__*` variables — observability
is configured through `SERVER__LOG_*` and metrics are fixed-name;
multimodal costs are code constants.

## SERVER

| Variable | Type | Default | Effect |
|---|---|---|---|
| `SERVER__HOST` | str | `0.0.0.0` | uvicorn bind host (used via CLI/Compose) |
| `SERVER__PORT` | int | `8080` | listen port (process-level; source runs & container internal) |

Compose-only: the PUBLISHED host port is `PROXY_PORT` (default `11435`) —
it maps to the container's fixed internal 8080. `SERVER__PORT` inside the
container stays 8080 regardless.
| `SERVER__LOG_LEVEL` | str | `INFO` | log level |
| `SERVER__LOG_JSON` | bool | `false` | JSON-lines logging when true |
| `SERVER__MAX_BODY_BYTES` | int > 0 | `8388608` (8 MiB) | request-body cap; oversized → `413 request_too_large` |
| `SERVER__MAX_CAPTURE_BYTES` | int > 0 | `2097152` (2 MiB) | streaming assistant-capture bound; exceeding it disables persistence for that response while passthrough continues |
| `SERVER__WEB_CONCURRENCY` | int | `1` | uvicorn workers (CLI flag in image). Metrics/rate buckets/breaker are per-worker |

## DATABASE

| Variable | Type | Default | Effect |
|---|---|---|---|
| `DATABASE__URL` | str | `postgresql://context_proxy:context_proxy@localhost:5432/context_proxy` | asyncpg DSN. Unreachable → degraded passthrough; reachable but migration failure → startup aborts |
| `DATABASE__MIN_POOL_SIZE` | int ≥ 1 | `1` | pool floor |
| `DATABASE__MAX_POOL_SIZE` | int ≥ 1 | `10` | pool ceiling |
| `DATABASE__CONNECT_TIMEOUT_SECONDS` | float > 0 | `3.0` | connect timeout |

Migrations run automatically at startup (`apply_migrations`); there is no
separate migration command.

## CONTEXT

| Variable | Type | Default | Effect |
|---|---|---|---|
| `CONTEXT__MODEL_LIMIT_TOKENS` | int > 0 | `32768` | upstream model window; must match reality |
| `CONTEXT__SAFETY_MARGIN_TOKENS` | int ≥ 0 | `2048` | subtracted from limit; **must be smaller than** the model limit (validated) |
| `CONTEXT__PINNED_BUDGET_TOKENS` | int ≥ 0 | `2000` | reserved headroom used as planner reserve |
| `CONTEXT__RECENT_TARGET_TOKENS` | int > 0 | `14000` | recent-window target size |
| `CONTEXT__RECENT_MIN_TOKENS` | int > 0 | `10000` | recent-window minimum |
| `CONTEXT__RECENT_MAX_TOKENS` | int > 0 | `18000` | recent-window maximum |

Usable budget = `MODEL_LIMIT_TOKENS − SAFETY_MARGIN_TOKENS`.

## ASSEMBLY

| Variable | Type | Default | Effect |
|---|---|---|---|
| `ASSEMBLY__ENABLED` | bool | `true` | false restores the simple window planner |
| `ASSEMBLY__MMR_LAMBDA` | float 0–1 | `0.7` | 1 = pure relevance, 0 = max diversity |
| `ASSEMBLY__MAX_RETRIEVED_ITEMS` | int ≥ 0 | `8` | retrieved-block cap per plan |
| `ASSEMBLY__RETRIEVED_BUDGET_TOKENS` | int ≥ 0 | `4000` | hard cap for retrieved tier |

## CONVERSATION

| Variable | Type | Default | Effect |
|---|---|---|---|
| `CONVERSATION__CLIENT_ID_HEADER` | str | `X-Session-ID` | fallback session header (uuid5-mapped to a stable conversation UUID) |
| `CONVERSATION__MAX_SESSION_IDENTITY_CHARS` | int > 0 | `128` | length cap on session tokens |

Identity precedence: body `conversation_id` → `X-Conversation-ID` header →
session header → generated UUID.

## INFERENCE

| Variable | Type | Default | Effect |
|---|---|---|---|
| `INFERENCE__BASE_URL` | URL | `http://localhost:8000/v1` | any OpenAI-compatible endpoint |
| `INFERENCE__API_KEY` | str | empty | sent as `Authorization: Bearer …` when non-empty |
| `INFERENCE__MODEL` | str | empty | overrides client `model` field when set |
| `INFERENCE__TIMEOUT_SECONDS` | float > 0 | `600.0` | full httpx timeout for inference calls |

## COMPACT and EMBEDDINGS

Independent endpoint settings with the same shape as INFERENCE:

| Variable | COMPACT default | EMBEDDINGS default |
|---|---|---|
| `COMPACT__BASE_URL` / `EMBEDDINGS__BASE_URL` | `http://localhost:8001/v1` | `http://localhost:8002/v1` |
| `COMPACT__API_KEY` / `EMBEDDINGS__API_KEY` | `local` | `local` |
| `COMPACT__MODEL` / `EMBEDDINGS__MODEL` | `compact-model` | `embedding-model` |
| `COMPACT__TIMEOUT_SECONDS` / `EMBEDDINGS__TIMEOUT_SECONDS` | `600.0` | `600.0` |

`COMPACT__*` is reserved for compaction work and is not exercised by any
current flow; `EMBEDDINGS__*` backs chunk/memory embeddings for retrieval
(client built at startup from these settings).

## QDRANT

| Variable | Type | Default | Effect |
|---|---|---|---|
| `QDRANT__BASE_URL` | URL | `http://localhost:6333` | vector store; unreachable → lexical-only retrieval |
| `QDRANT__COLLECTION` | str | `context_proxy` | collection name; created on demand, compatibility-checked on conflict |
| `QDRANT__TIMEOUT_SECONDS` | float > 0 | `5.0` | HTTP timeout |

The collection must be single-vector `{size: <embeddings dim>, distance:
Cosine}`; an incompatible existing collection raises an explicit error and is
never deleted or recreated automatically.

## RETRIEVAL

Weighted ranking: `score = semantic·w + lexical·w + recency·w +
importance·w + type·w`.

| Variable | Type | Default | Range |
|---|---|---|---|
| `RETRIEVAL__SEMANTIC_WEIGHT` | float | `0.40` | free |
| `RETRIEVAL__LEXICAL_WEIGHT` | float | `0.25` | free |
| `RETRIEVAL__RECENCY_WEIGHT` | float | `0.15` | free |
| `RETRIEVAL__IMPORTANCE_WEIGHT` | float | `0.15` | free |
| `RETRIEVAL__TYPE_WEIGHT` | float | `0.05` | free |
| `RETRIEVAL__LIMIT_DEFAULT` | int ≥ 1 | `8` | items returned |
| `RETRIEVAL__CANDIDATE_POOL` | int ≥ 1 | `50` | candidate pool before ranking |

## MEMORY

| Variable | Type | Default | Effect |
|---|---|---|---|
| `MEMORY__AUTO_INDEX` | bool | `true` | index completed turns after each response |
| `MEMORY__MAX_EMBED_CHARS` | int > 0 | `8000` | per-text embedding truncation bound |
| `MEMORY__INDEX_TIMEOUT_SECONDS` | float > 0 | `10.0` | synchronous indexing budget; timeout/failure never affects the response |

## RESILIENCE

| Variable | Type | Default | Effect |
|---|---|---|---|
| `RESILIENCE__MAX_RETRIES` | int ≥ 0 | `2` | retries ONLY on `ConnectError`/`ConnectTimeout`; post-send failures are never retried |
| `RESILIENCE__BACKOFF_BASE_SECONDS` | float ≥ 0 | `0.2` | full-jitter exponential backoff base |
| `RESILIENCE__BACKOFF_MAX_SECONDS` | float ≥ 0 | `2.0` | backoff cap |
| `RESILIENCE__BREAKER_FAILURE_THRESHOLD` | int ≥ 1 | `5` | consecutive pre-send failures that OPEN the breaker |
| `RESILIENCE__BREAKER_RESET_SECONDS` | float > 0 | `30.0` | OPEN→HALF_OPEN probe delay |

Upstream HTTP error responses (4xx/5xx) are answers: never retried, never
counted against the breaker.

## RATE_LIMIT

Two-dimension token bucket: every request consumes one **client/IP** token
and, when `X-Conversation-ID` is present, one **conversation** token. A
request passes only if both admit. Both dimensions share ONE capped table.

| Variable | Type | Default | Effect |
|---|---|---|---|
| `RATE_LIMIT__ENABLED` | bool | `false` | master switch |
| `RATE_LIMIT__REQUESTS_PER_MINUTE` | int > 0 | `120` | refill rate per bucket dimension |
| `RATE_LIMIT__BURST` | int > 0 | `30` | bucket capacity |
| `RATE_LIMIT__MAX_IDENTITIES` | int ≥ 1 | `10000` | live-bucket cap across both dimensions (LRU eviction at capacity) — memory is O(this) |
| `RATE_LIMIT__IDENTITY_TTL_SECONDS` | float > 0 | `3600` | idle buckets expire |
| `RATE_LIMIT__MAX_IDENTITY_CHARS` | int ≥ 1 | `256` | key truncation bound |

Rejected requests get OpenAI-style `429 rate_limit_error` with
`Retry-After`. `/healthz`, `/readyz` and `/metrics` are exempt.
Per-worker scope: with `WEB_CONCURRENCY>1` limits multiply by worker count.

## SECURITY

| Variable | Type | Default | Effect |
|---|---|---|---|
| `SECURITY__MODE` | `development` \| `production` | `development` | deployment policy selector |
| `SECURITY__INTERNAL_AUTH_TOKEN` | str | empty | when set, every `/internal/v1/*` request must send header `X-Internal-Auth: <token>` (constant-time compare); wrong/missing → `401` |

**Fail-closed rule:** `SECURITY__MODE=production` with an empty token is a
configuration error at startup — the proxy refuses to boot rather than expose
administrative endpoints. Keep `development` for local/test deployments.

The token never appears in diagnostics output or logs (diagnostics exposes
only sanitized inference host/port).

## Interactions worth knowing

- `CONTEXT__SAFETY_MARGIN_TOKENS >= CONTEXT__MODEL_LIMIT_TOKENS` → config error.
- `SERVER__MAX_CAPTURE_BYTES` bounds only streamed assistant capture;
  buffered responses are not size-limited by it.
- `ASSEMBLY__ENABLED=false` ignores `ASSEMBLY__MMR_LAMBDA`,
  `ASSEMBLY__MAX_RETRIEVED_ITEMS`, `ASSEMBLY__RETRIEVED_BUDGET_TOKENS`.
- Rate-limit memory is bounded regardless of how many distinct conversation
  ids clients send; eviction/TTL counters appear under
  `context_proxy_rate_limit_identities_evicted_total`.

## Example configurations

### Local development (`.env`)

```bash
INFERENCE__BASE_URL=http://localhost:11434/v1
INFERENCE__MODEL=MichelRosselli/bonsai-27b:latest
DATABASE__URL=postgresql://context_proxy:context_proxy@localhost:5432/context_proxy
QDRANT__BASE_URL=http://localhost:6333
EMBEDDINGS__BASE_URL=http://localhost:8002/v1
EMBEDDINGS__MODEL=nomic-embed-text
CONTEXT__MODEL_LIMIT_TOKENS=32768
# SECURITY__MODE defaults to development: /internal/* open locally
```

### Docker Compose (service names)

```bash
DATABASE__URL=postgresql://context_proxy:context_proxy@postgres:5432/context_proxy
QDRANT__BASE_URL=http://qdrant:6333
INFERENCE__BASE_URL=http://host.docker.internal:11434/v1   # provider on the host
```

(`docker-compose.yml` sets the DATABASE/QDRANT values itself; you mainly add
inference/embedding endpoints.)

### Production-style

```bash
SECURITY__MODE=production
SECURITY__INTERNAL_AUTH_TOKEN=<generate: openssl rand -hex 32>
RATE_LIMIT__ENABLED=true
SERVER__LOG_JSON=true
SERVER__WEB_CONCURRENCY=4
DATABASE__URL=postgresql://user:<password>@db.internal:5432/context_proxy
INFERENCE__BASE_URL=https://llm.internal/v1
INFERENCE__API_KEY=<token>
```

Never commit real credentials; use your secrets manager.
