# Security

## Threat model

| Actor / input | Trust | Mitigations |
|---|---|---|
| Public client on `/v1/*` | untrusted | structural validation, body-size cap (413), rate limiting, controlled error envelopes |
| Request payload (messages/tools) | untrusted | shape-validated only; forwarded verbatim to the provider; never interpreted as proxy instructions |
| Conversation ids / session headers | untrusted | UUID validation; truncation bounds; not the sole rate-limit principal |
| Retrieved memories/chunks | untrusted derived content | rendered as user-role messages with `[retrieved …]` provenance header; engine guard rejects derived content rendering as `system`/`developer` |
| Model-generated assistant output | untrusted | stored verbatim; never re-executed or promoted to instruction tier |
| Upstream provider | semi-trusted network peer | opaque body passthrough; header filtering; generic transport-failure messages |
| `/internal/v1/*` callers | administrative only | private network + configurable token + production fail-closed policy |
| PostgreSQL / Qdrant | infrastructure | loopback-only publication in Compose; service-name traffic inside the Compose network |

## Internal API security

```bash
SECURITY__MODE=production
SECURITY__INTERNAL_AUTH_TOKEN=$(openssl rand -hex 32)
```

Behavior:

| Mode | Token configured | Result |
|---|---|---|
| `development` (default) | no | `/internal/*` open — for local/test only |
| `development` | yes | token enforced (`X-Internal-Auth`, constant-time compare; wrong → 401) |
| `production` | **no** | **configuration error at startup** — fail-closed, the proxy refuses to run |
| `production` | yes | token enforced on every internal call |

The URL prefix `/internal` is explicitly NOT a security mechanism. The
token never appears in diagnostics output, logs, or metrics. Public
`/v1/*` endpoints are unaffected by this policy.

## Network security

Compose topology (see [deployment.md](deployment.md)):

- `context-proxy` publishes `${SERVER__PORT:-8080}` on all interfaces
  (intended to sit behind a reverse proxy/TLS terminator);
- `postgres` publishes `127.0.0.1:5432` and `qdrant` `127.0.0.1:6333/6344`
  — **loopback only**; container-to-container traffic uses service names on
  the default Compose network;
- remove the `ports:` blocks if host access is unnecessary.

Container hardening: non-root image user, `security_opt:
[no-new-privileges:true]`, `cap_drop: [ALL]` on the proxy; pinned images
(`postgres:16-alpine`, `qdrant/qdrant:v1.12.4`).

## Rate limiting

Two-dimension admission (in-process, single-instance scope):

```text
request admitted ⇔ client/IP bucket has a token
              AND conversation bucket has a token (when X-Conversation-ID present)
```

- rotating `X-Conversation-ID` cannot mint quota — the client bucket
  aggregates all of one host's traffic;
- conversation bucket preserves per-conversation isolation for well-behaved
  multi-conversation clients;
- a rejected attempt still consumes the passing dimension's token (closing
  alternating-identity hammering);
- bounded memory: ONE table capped at `RATE_LIMIT__MAX_IDENTITIES` across
  both dimensions, LRU eviction at capacity, TTL expiry of idle buckets,
  key truncation at `RATE_LIMIT__MAX_IDENTITY_CHARS`. Worst-case memory is
  O(max_identities) regardless of client behavior;
- rejections: OpenAI-style `429 rate_limit_error` with positive integer
  `Retry-After`; metric increments exactly once per rejected request;
- `/healthz`, `/readyz`, `/metrics` exempt;
- per-worker scope with multiple uvicorn workers (limits multiply).

## Retry safety

Only failures that are **provably pre-send** are retried:

```python
RETRYABLE_TRANSPORT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout)
```

Rationale: chat completions are POSTs that trigger billable, side-effectful
inference. If a failure occurs after the request may have been delivered
(`ReadTimeout`, `WriteError`, `RemoteProtocolError`) a retry could execute a
second inference. Those failures therefore get exactly one attempt. HTTP
error responses (400/429/5xx) are answers, never retried. Streaming: retry
only before the stream opens; once bytes flow there is no resend.
Circuit-breaker accounting counts only the provably pre-send class.

Regression anchor in the suite: provider receives the POST once, then the
response read times out → no second POST is ever sent.

## Secrets handling

- configuration exclusively via environment variables; nothing persisted;
- **log redaction**: structured logging masks sensitive keys/values
  (`api_key`, `authorization`, bearer patterns, tokens, secrets, ...)
  including nested structures;
- **diagnostics redaction**: `/internal/v1/diagnostics` reports the inference
  endpoint as `{configured, host, port}` only — username/password/query/
  fragment/paths of configured URLs never appear;
- upstream `set-cookie` never forwarded; `Authorization` to the upstream is
  set once from config, never logged;
- errors never contain DSNs, paths, stack traces, exception text (generic
  `internal_error` bodies; details stay in server logs under redaction).

## Header filtering

Forwarded responses strip (per RFC 9110 §7.6.1 plus policy):

`connection` + any headers it names as connection-tokens, `keep-alive`,
`proxy-authenticate`, `proxy-authorization`, `te`, `trailer(s)`,
`transfer-encoding`, `upgrade`, `content-length` (recomputed), `set-cookie`,
and `content-encoding` on buffered responses (bodies arrive decompressed;
streamed passthrough keeps it truthfully). Missing `content-type` gets an
application/json default.

## Error leakage controls

Client-visible errors are limited to:

- OpenAI-shaped validation/conflict messages generated by the proxy;
- verbatim upstream JSON error bodies (the upstream's own words);
- fixed generic strings for transport/internal failures
  (`"upstream inference endpoint is unavailable"`,
  `"Internal server error"`).

Verified by regression tests asserting absence of filesystem paths, DSNs,
passwords, API keys, tracebacks and internal hostnames.

## Supply chain

- `constraints.txt` pins the full dependency tree; CI and the Docker image
  install with `-c constraints.txt` (reproducible builds);
- CI job `security-scan` runs `pip-audit --strict -r <frozen requirements>`
  and fails on known CVEs (policy: all findings block; add dated
  `--ignore-vuln` entries only with justification);
- lint (`ruff`) and tests gate every push.

## Implemented vs deployment responsibilities

Implemented in the app/image/Compose (this repo): everything above plus
non-root user, healthchecks, capability drop, pinned base images.

Deployment responsibilities (operator): TLS termination, ingress ACLs for
`/internal/*` and `/metrics`, secret management, database backups, network
firewalling beyond Compose defaults.

See also: [configuration.md](configuration.md) · [deployment.md](deployment.md)
