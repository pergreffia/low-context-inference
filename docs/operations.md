# Operations runbook

## Endpoints overview

| Endpoint | Scope | Purpose |
|---|---|---|
| `GET /healthz` | public | liveness; reports `{"status":"ok","database":"ok"\|"degraded"}` |
| `GET /readyz` | public | readiness; dependency checks + circuit-breaker state |
| `GET /metrics` | public* | Prometheus text format |
| `POST /v1/chat/completions` | public | OpenAI-compatible chat (+ `"stream": true`) |
| `GET /v1/models` | public | passthrough of upstream model list |
| `/internal/v1/*` | internal/administrative | see below |

\* served by the app; put ingress ACLs in front for untrusted networks.

`/healthz`, `/readyz`, `/metrics` are exempt from rate limiting.

### Internal endpoints (administrative)

All require `X-Internal-Auth: <SECURITY__INTERNAL_AUTH_TOKEN>` when that
token is configured; production mode refuses to start without one. The URL
prefix alone is **not** a security control.

| Endpoint | Effect |
|---|---|
| `POST /internal/v1/memories` | create memory record (`kind`, `content`, `conversation_id`, optional `importance`, `supersedes`, `source_message_ids`) |
| `POST /internal/v1/memories/{id}/supersede` | body `{"status": "superseded"\|"resolved"\|"obsolete"}` — active-only retrieval respects it |
| `GET  /internal/v1/retrieval?q=&conversation_id=&limit=` | hybrid retrieval probe |
| `POST /internal/v1/conversations/{id}/index` | chunk+index completed turns for one conversation |
| `POST /internal/v1/index/rebuild?conversation_id=&force=` | rebuild derived Qdrant index from PostgreSQL |
| `GET  /internal/v1/diagnostics` | component snapshot (no secrets) |
| `POST /internal/v1/conversations/{id}/context/preview` | dry-run the Context Assembly Engine; ids/scores/reasons only |

## Startup / shutdown / restart

```bash
docker compose up --build -d     # start (migrations run inside proxy startup)
docker compose restart context-proxy
docker compose down              # stops; volumes persist
```

Source mode: `uvicorn context_proxy.main:app --host … --port …`;
Ctrl-C triggers graceful shutdown which closes app-owned clients in
isolation (injected clients are never touched).

Migrations: automatic at every proxy start; applied set is tracked in
`schema_migrations`; re-running is a no-op.

## Health, readiness, diagnostics

- `GET /healthz` → liveness. `database: degraded` means PostgreSQL is
  unreachable: traffic continues as passthrough-only.
- `GET /readyz` → `{"ready": true, "checks": {"database": "…",
  "circuit_breaker": "closed"}}`.
- `GET /internal/v1/diagnostics` → pool size, memory-service presence,
  engine state, resilience settings, breaker state, rate-limit config +
  live bucket count, inference endpoint as sanitized `{configured, host,
  port}` (never user/password/query/fragment).

## Metrics

Prometheus text at `/metrics`. Series (all low-cardinality; labels are only
method/route-template/status/component/direction/state):

| Series | Meaning |
|---|---|
| `context_proxy_http_requests_total{method,route,status}` | request counts (real statuses only; no fabricated 499) |
| `context_proxy_http_request_duration_seconds{route}` | end-to-end latency histogram |
| `context_proxy_upstream_duration_seconds{route}` | upstream call latency |
| `context_proxy_llm_tokens_total{direction=prompt\|completion\|total}` | token accounting, exactly once per upstream response |
| `context_proxy_degradations_total{component}` | degraded operations (e.g. `upstream_breaker_open`) |
| `context_proxy_rate_limit_rejects_total` | rejected requests (once each) |
| `context_proxy_rate_limit_identities_evicted_total` | buckets dropped via TTL/capacity |
| `context_proxy_assistant_capture_overflow_total` | streamed responses whose capture was disabled (passthrough unaffected) |
| `context_proxy_http_streams_aborted_total` | streams started then aborted |
| `context_proxy_client_disconnects_total` | abandoned requests mid-body |
| `context_proxy_circuit_state` | gauge: current breaker state |

Route labels are normalized templates (`/v1/chat/completions`,
`/internal/v1/*`, `other`) — never raw paths or ids.

## Logs

Human-readable by default; `SERVER__LOG_JSON=true` for JSON lines. Every
request logs one completion record with correlation id and stage timings
(`inbound_persistence_seconds`, `context_assembly_seconds`). A redaction
filter masks sensitive keys/values in log payloads. Key events:

| Event | Meaning |
|---|---|
| `postgres_unavailable` | DB unreachable → degraded passthrough |
| `inbound_persistence_failed` | infra failure during inbound reconcile → degrade for that request |
| `assistant_persistence_conflict` | concurrent continuation lost the race (response still delivered) |
| `assistant_persistence_failed` | infra failure persisting assistant (response still delivered) |
| `assistant_persistence_skipped_capture_overflow` | stream exceeded capture bound; nothing persisted |
| `assistant_stream_incomplete_persistence_skipped` | upstream died mid-stream |
| `upstream_connect_failed` / `upstream_transport_failed_no_retry` | transport failure classes (retryable vs post-send) |
| `circuit_state_changed` | breaker transitions |
| `rate_limited` (scope, retry_after), `request_too_large`, `client_disconnected_during_body` | rejection paths |
| `vector_search_unavailable`, `embedding_unavailable`, `memory_index_timeout` | retrieval/index degradation |
| `indexing_pass_completed`, `turns_indexed` | indexing progress |
| `orphan_tool_result` | tool result with no matching call in the conversation |

## Backup / restore

**PostgreSQL is the source of truth** — back it up:

```bash
docker exec <pg-container> pg_dump -U context_proxy context_proxy > backup.sql
# restore
cat backup.sql | docker exec -i <pg-container> psql -U context_proxy context_proxy
```

Qdrant data is derived: restore service availability and trigger
`POST /internal/v1/index/rebuild` (optionally per conversation). Qdrant
volume snapshots work too but are not required for correctness.

## Provider failure behavior

1. Connect failures/timeouts before send: retried up to
   `RESILIENCE__MAX_RETRIES` with jittered backoff; each failure increments
   breaker consecutive-failure count.
2. Threshold reached → breaker OPEN: calls fail fast with
   `502 {"code":"upstream_unavailable"}` without touching the network;
   `context_proxy_degradations_total{component="upstream_breaker_open"}`
   increments.
3. After `RESILIENCE__BREAKER_RESET_SECONDS` → HALF_OPEN: exactly one probe
   proceeds; success CLOSES the breaker, failure re-opens it.
4. Post-send failures (read timeout, write error, protocol error): single
   attempt, surfaced as the same 502 contract, do NOT trip the breaker.
5. Upstream HTTP errors (4xx/5xx) are answers: passed through verbatim when
   JSON; never retried.

Client-visible contract for an unavailable provider:
`502 api_error upstream_unavailable`, message
`"upstream inference endpoint is unavailable"` — no internal detail leaks.
Recovery requires no operator action once the provider answers again; the
same process resumes serving (covered by regression tests).

## Degraded modes

| Failure | Behavior |
|---|---|
| PostgreSQL unreachable | pure passthrough; nothing persisted; `history_conflict` protection inactive; health shows `degraded` |
| Migration failure with reachable DB | proxy aborts startup (fail-closed schema policy) |
| Embedding/Qdrant outage | lexical-only retrieval |
| Both retrieval legs down | recent-history context only |
| Memory indexing timeout/failure | logged, response unaffected |
| Stream exceeds capture bound | client stream completes byte-exact; assistant not persisted; overflow counter incremented |

## Common failures & troubleshooting

| Symptom | Likely cause / action |
|---|---|
| `502 upstream_unavailable` on every call | provider down or wrong `INFERENCE__BASE_URL`; check provider, breaker gauge; breaker self-recovers after reset window |
| `400 history_conflict` | client resent a divergent history for that conversation; fix client transcript or use a new conversation id |
| `400 context_length_exceeded` | mandatory content (system/tools/current) alone exceeds budget; raise `CONTEXT__MODEL_LIMIT_TOKENS` to match the real model or shrink tools |
| `413` on upload | body > `SERVER__MAX_BODY_BYTES`; raise limit or shrink payload |
| `409`/`422` unexpected shapes | structural validation; see [api.md](api.md#errors) |
| No memories retrieved | embeddings/Qdrant down (check `vector_search_unavailable` logs) or collection mismatch (diagnostics; rebuild index) |
| Assistant turns missing from DB after huge streams | capture overflow by design (`SERVER__MAX_CAPTURE_BYTES`); raise bound if persistence matters for those responses |
| Everything slow, DB fine | check `upstream_duration` vs total latency; indexing timeouts appear in logs |

## Database maintenance

Standard PostgreSQL practice applies. Migrations live in
`src/context_proxy/db/migrations/NNNN_*.sql`, applied in lexical order at
startup. To verify applied set:

```bash
docker exec <pg-container> psql -U context_proxy -d context_proxy \
  -c "SELECT name FROM schema_migrations ORDER BY name"
```

## Qdrant maintenance

- Collection: single-vector `{size: <embedding dim>, distance: Cosine}`,
  name from `QDRANT__COLLECTION`. On 409-create conflicts the store verifies
  compatibility explicitly and refuses incompatible collections (never
  deletes).
- Full rebuild from truth: `POST /internal/v1/index/rebuild`
  (add `?conversation_id=` to scope; `force=true` re-chunks).

See also: [deployment.md](deployment.md) · [security.md](security.md)
