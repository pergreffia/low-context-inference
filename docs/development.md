# Development

## Repository structure

```text
src/context_proxy/
├── main.py                 # create_app, lifespan wiring, exception handler
├── config.py               # all settings (pydantic-settings)
├── capture.py              # bounded streaming assistant capture
├── api/
│   ├── routes.py           # /v1/* chat flow orchestration
│   ├── routes_internal.py  # /internal/v1/* administrative surface
│   ├── validation.py       # structural payload validation
│   └── responses.py        # OpenAI envelopes, streaming/passthrough helpers
├── conversation/
│   ├── identity.py         # conversation id resolution
│   └── store.py            # PostgreSQL source of truth + projections
├── context/
│   ├── engine.py           # Context Assembly Engine
│   ├── planner.py          # segmentation + fallback window planner
│   ├── tokens.py           # deterministic token estimation
│   ├── candidates.py / scoring.py / mmr.py / query.py
├── memory/
│   ├── service.py          # chunking, indexing, hybrid retrieval, rebuild
│   ├── embeddings.py qdrant.py models.py errors.py
├── providers/
│   ├── llm.py base.py      # OpenAI-compatible passthrough client
│   ├── resilience.py       # breaker + retries
│   ├── headers.py errors.py
├── observability/
│   ├── middleware.py       # ASGI: ids, limits, rate limiting, metrics
│   ├── metrics.py ratelimit.py logging_setup.py
└── db/
    ├── database.py         # pool + startup migrations
    └── migrations/NNNN_*.sql
tests/                      # unit + integration suites (see testing.md)
scripts/e2e_real_provider.sh  # optional live-provider E2E
docs/
```

Python **3.12+**. Build backend: hatchling; packaging via `pyproject.toml`.
Runtime deps: fastapi, uvicorn[standard], httpx, asyncpg, pydantic,
pydantic-settings. Dev extras: pytest, pytest-asyncio, pip-audit.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]" -c constraints.txt
cp .env.example .env        # adjust at least INFERENCE__BASE_URL
```

`-c constraints.txt` is mandatory for reproducible installs matching CI.

### Local dependencies

- **PostgreSQL** (`DATABASE__URL`, default localhost:5432). Optional locally:
  unreachable → degraded passthrough. A local test container pattern:

  ```bash
  docker run -d --name cpx-test-postgres -p 5433:5432 \
    -e POSTGRES_USER=context_proxy -e POSTGRES_PASSWORD=context_proxy \
    -e POSTGRES_DB=context_proxy postgres:16-alpine
  export TEST_DATABASE_URL=postgresql://context_proxy:context_proxy@localhost:5433/context_proxy
  ```

- **Qdrant**: `docker run -p 6333:6333 qdrant/qdrant:v1.12.4`. Optional;
  retrieval degrades to lexical without it.
- **Migrations** apply automatically at app startup and in PG-backed tests.

### Running

```bash
uvicorn context_proxy.main:app --host 127.0.0.1 --port 8080 --reload
```

## Lint & tests

```bash
ruff check .            # line-length 100, rules E/F/I/UP/B
pytest -q               # fast suite; PG tests auto-skip
TEST_DATABASE_URL=… pytest -q   # full incl. PostgreSQL integration
```

Details and strategy: [testing.md](testing.md).

## Conventions

### Adding a feature

1. Read the surrounding module first; match existing patterns.
2. Add/extend a regression test that would fail without the change.
3. Keep the public OpenAI-compatible contract untouched unless the task says
   otherwise; bodies are never rewritten.
4. Degrade gracefully: expected infrastructure failures are typed
   (PersistenceInfrastructureError / RetrievalError / VectorStoreError) and
   logged as warnings — programming errors propagate.

### Adding configuration

1. Extend the relevant `BaseModel` in `config.py` (or add one to `Settings`)
   with type, default and validation constraints.
2. Document it in `.env.example` and `docs/configuration.md`.
3. Wire it where consumed; never read `os.environ` outside pydantic-settings.

### Migrations

- append `NNNN_description.sql` to `db/migrations/` (lexical order applied,
  tracked in `schema_migrations`, idempotency assumed);
- raw message JSONB schema is frozen: projections only;
- update `tests/test_migrations.py` expectations when adding files;
- migrations must succeed on a clean database (CI runs the full suite
  against an empty instance).

### Tests

- deterministic by construction: inject clocks, no sleeps, no live LLMs;
- route-level tests use `tests/conftest.py`
  (`client_for_handler`/MockTransport); fake stores mirror reconcile
  semantics;
- PostgreSQL suites gate on `TEST_DATABASE_URL`;
- concurrency tests synchronize on real state (row locks/pg_locks/events),
  never timing guesses.

### API changes

- `/v1/*` is compatibility surface: additive only, errors stay OpenAI-shaped;
- `/internal/v1/*` is administrative: keep auth dependency intact;
- document changes in [api.md](api.md).

## Dependency updates

```bash
pip install -U <package>
pip freeze | grep -v "^-e " > constraints.txt   # keep editable out
ruff check . && pytest -q && python -m pip_audit --strict -r <(pip freeze | grep -v "^-e ")
# then commit the new lock together with any required code fix
```

Never loosen a pin to make a failing test pass.

See also: [architecture.md](architecture.md) · [configuration.md](configuration.md)
