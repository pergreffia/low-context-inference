# Testing

## Principles

- **Deterministic first**: no real LLM in the standard suite; providers are
  `httpx.MockTransport` fakes; clocks are injectable; concurrency
  synchronizes on events/row locks/`pg_locks`, never sleeps.
- **PostgreSQL suites** gate on `TEST_DATABASE_URL` and run migrations from
  scratch (CI runs them against a clean instance; they must not silently
  skip there).
- Every bug fix lands with a regression test that fails without it.

## Commands

```bash
ruff check .                                   # lint (E/F/I/UP/B, line 100)
pytest -q                                      # fast: unit + non-PG integration
TEST_DATABASE_URL=postgresql://context_proxy:context_proxy@localhost:5433/context_proxy pytest -q   # full
```

Security/supply-chain:

```bash
python -m pip freeze | grep -v "^-e " | grep -v "^context-proxy" > /tmp/audit.txt
python -m pip_audit --strict -r /tmp/audit.txt
```

Docker smoke (build + healthz/metrics):

```bash
docker build -t context-proxy:dev .
docker run -d --name cpx-smoke -p 8080:8080 context-proxy:dev && \
  curl -sf http://localhost:8080/healthz && docker rm -f cpx-smoke
```

## Suite map

| File(s) | Area |
|---|---|
| `test_proxy.py`, `test_payload.py` | core chat flow, passthrough semantics |
| `test_streaming.py`, `test_streaming_edges.py` | SSE byte-exactness, fragmentation, mid-stream failure, cancellation/close-once, overflow+tools, no-DONE EOF contract |
| `test_capture_overflow.py` | capture bounds: exact N / N+1 / multi-chunk / huge streams, metric accounting |
| `test_conversation_flow.py`, `test_history_sync.py`* | reconciliation, idempotence, divergence, degraded store |
| `test_store.py`*, `test_store_concurrency.py`*, `test_same_session_contract.py`* | PostgreSQL store, row-lock reconciliation contracts |
| `test_concurrency_deterministic.py`* | pg_locks-driven isolation/divergence/tool-call cross-conversation isolation, custom tool projection, orphan results |
| `test_migrations.py`* | clean-DB migration run, exactly-once application |
| `test_engine_*.py`, `test_planner.py`, `test_context_engine.py`, `test_context_budget_matrix.py` | assembly engine + budget guarantees/determinism/multimodal costs |
| `test_memory_service.py`*, `test_memory_providers.py` | chunking, indexing markers, hybrid retrieval, provider failures |
| `test_multimodal_matrix.py` (+ PG section*) | part shapes, media registry, retrieval text-extraction, giant data URLs |
| `test_tool_lifecycle.py`, `test_final_fixes.py` | function/custom tool lifecycle, streamed reconstruction, replay |
| `test_developer_contract.py`, `test_m6_integration.py`* | developer role persistence/replay/protection |
| `test_retry_semantics.py`, `test_provider_recovery.py`, `test_upstream_errors.py`, `test_m5_resilience.py` | retry classification (no post-send resend), breaker states, same-proxy recovery, error mapping |
| `test_rate_limiter_bounds.py` | bounded buckets, two-dimension admission, rotation bypass regression |
| `test_security_regressions.py` | parsing abuse, leakage sweep, header matrix, internal auth, compose hygiene |
| `test_observability_contract.py`, `test_m5_ops.py` | endpoints, counters, label cardinality, breaker gauges |
| `test_header_policy.py`, `test_identity.py`, `test_internal_api.py`, `test_engine_api.py`, `test_compose_config.py` | focused units for policies |

\* requires `TEST_DATABASE_URL`.

## What runs where

| Environment | Runs |
|---|---|
| local dev (no env vars) | unit + fake-provider integration |
| CI job `unit` | same as above |
| CI job `integration-postgres` | full suite against service PostgreSQL (`TEST_DATABASE_URL`) |
| CI job `security-scan` | pip-audit over frozen requirements |
| CI job `docker-build` | image build + healthz/metrics smoke |
| manual | real-provider E2E script (below) |

## Real-provider E2E (optional, never part of standard CI)

`scripts/e2e_real_provider.sh` validates against a live OpenAI-compatible
endpoint (Ollama). It boots a controllable TCP forwarder so provider-failure/
recovery is deterministic against the SAME proxy process, checks PostgreSQL
persistence, developer-role storage, streaming, tools (SKIP if the model
chooses not to call one), concurrency, and reports aggregate PASS/FAIL with
an unconditional cleanup trap.

Verified example with Ollama serving Bonsai:

```bash
ollama pull MichelRosselli/bonsai-27b:latest
BASE_MODEL=MichelRosselli/bonsai-27b:latest ./scripts/e2e_real_provider.sh
```

Requirements: Docker container `cpx-test-postgres` reachable as
`localhost:5433` (see development.md), Ollama on `:11434`,
`.venv` present at repo root. Exit code = number of failed required checks;
model-dependent outcomes are reported as `SKIP`, never false `PASS`.

Authoritative coverage for capture-overflow, developer role, tools and
failure/recovery lives in the deterministic suites — the script is
complementary live validation.

## Test-support infrastructure worth knowing

- `tests/conftest.py`: `upstream_handler`, `make_settings`,
  `client_for_handler` (inject store), fixed CHAT/SSE fixtures;
- fake stores (`FakeConversationStore`, `RecordingStore`) mirror positional
  reconcile + divergence semantics of the real store;
- breaker/rate-limiter accept injected monotonic clocks;
- metrics assertions parse the Prometheus text registry (`REGISTRY.reset()`
  for isolation);
- PG tests create isolated UUID conversations per test; migrations apply via
  autouse fixture.

See also: [development.md](development.md) · [operations.md](operations.md)
