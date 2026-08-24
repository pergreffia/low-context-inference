# Code Review Report — Low Context Inference (M0–M6)

> Baseline: `701d5cd` + CI polish `b6948fa`
> Scope: review complessiva di bug, punti deboli, miglioramenti, test gaps.
> **Nessun fix applicato** — solo report.

---

## 🔴 BUG

### B1 — Assistant orfano con instruction mid-turn
**File**: `context/planner.py::segment_messages`, `context/engine.py::_fuse_candidates`

Sequenza valida `[user q, developer note, assistant a]` segmenta in:

```text
[turn(user q)]  [sys(developer)]  [prefill(assistant a)]
```

Sotto pressione budget il trim droppa le unità oldest non-system in position
order → `turn(user)` viene eliminato **prima** del `prefill(assistant)`,
lasciando un assistant orfano senza il suo user. Viola §7 interaction
atomicity su un edge valido (developer inserito a metà interazione).

**Fix direction**: prefill droppabile solo se nessun turn precedente è stato
droppato; oppure developer mid-turn aggregato all'unità corrente.

---

### B2 — Traceback persi nei JSON logs
**File**: `observability/logging_setup.py::JsonFormatter.format`

`JsonFormatter.format()` non serializza `exc_info`. Con
`SERVER__LOG_JSON=true` ogni `logger.error(..., exc_info=True)`
(es. `assistant_persistence_programming_error`) emette l'evento **senza
stack trace**. Diagnostica production compromessa proprio dove serve di più.

**Fix direction**: aggiungere `exc_info`/`exec_text` al payload JSON quando
presente sul record.

---

### B3 — Metriche inconsistenti su mid-stream failure
**File**: `observability/middleware.py::__call__`

Eccezione dopo `http.response.start` (stream interrotto a metà) →
`finish(500)` conta lo status 500, ma il client ha già ricevuto 200 + stream
troncato. Status counter e realtà lato client divergono.

**Fix direction**: flag `response_started` in `send_wrapper`; su eccezione con
response iniziata contare uno status dedicato (es. 499/“stream_aborted”) invece
di 500.

---

### B4 — Capture buffer illimitato
**File**: `capture.py::AssistantCapture.feed`

L'intero upstream stream viene accumulato in un bytearray per ricostruire il
messaggio. Risposte enormi (legittime o abusive) → memoria O(n) per connessione,
senza cap.

**Fix direction**: soglia configurabile oltre la quale la persistenza viene
disabilitata per quella risposta; passthrough resta intatto.

---

## 🟠 DEBOLEZZE

| # | Titolo | Dettagli |
|---|--------|----------|
| W1 | Retrieved→user: residuo teorico | Blocco retrieved precede turn user di history: modelli confondibili con turn storici legittimi. Mitigazione attuale = provenance header convenzionale. Futuro: metadata/provider field dedicati. |
| W2 | Canonical identity vuota per content scalare | `content_texts()` ritorna `[]` per content numerico/bool. Coerente solo post-validation (ora bloccati); difesa in profondità assente nel layer canonical. |
| W3 | Redaction limitata ai Bearer | `redact_text` maschera solo `Bearer …`. `api_key=…`, `token: …`, query-string credentials passano integri nei message text. Estendere alle stesse field-name rules degli extras. |
| W4 | Error leak su 500 | `error_body_response` fallback usa `str(exc)` dell'eccezione originale: possibili path/DSN parziali nel corpo risposta. Whitelist messaggi per classi interne. |
| W5 | `/metrics` e `/readyz` pubblici | Information disclosure lieve (volumi, status, degradazioni). Documentare bind interno come per `/internal/*` o proteggere. |
| W6 | Explicit-id: cap 64 hard-coded + UUID non normalizzato | Session ha config, explicit no. Inoltre `uuid.UUID(raw)` accetta forme braced/urn → header echo ≠ forma PG. Normalizzare `str(uuid.UUID(raw))` e configurare il cap. |
| W7 | `ensure_collection` ad ogni upsert Qdrant | Roundtrip extra per chunk/memoria in indexing loop e rebuild. Cache flag locale dopo primo successo. |
| W8 | Embedding senza batching | `_safe_embed` e rebuild inviano 1 testo per richiesta HTTP. Il provider supporta già batch (`embed(list)`): usalo in indexing/rebuild. |
| W9 | Rebuild sincrono in HTTP handler | `POST /index/rebuild` inline su indici grandi supera gateway timeout, nessun progress/cancel. Serve job async o scope chunked. |
| W10 | Histogram buckets ≤30s | Latenze LLM tipiche 5–180s cadono oltre l'ultimo bucket → percentuali inutilizzabili (`_sum` resta preciso). Buckets estesi/configurabili. |
| W11 | Logging sincrono nell'event loop | Handler console/file bloccanti sotto carico. `QueueHandler`+`QueueListener` in produzione. |
| W12 | Dipendenze non pinnate | `pyproject.toml` lower-bounds only → build immagine non riproducibile. Introdurre lock file / constraints in CI e Docker. |
| W13 | `readyz` sempre `ready:true` | Coerente col degraded-passthrough design ma ambiguo per orchestratori che vogliono gated traffic. Aggiungere modalità strict (config-driven). |
| W14 | Nessuna retention policy | conversations/messages/media/chunks crescono indefinitamente: TTL/cleanup/archiviazione mancanti. |

---

## 🟡 MIGLIORAMENTI (P3)

- **M1** — `store.get_messages`/reconcile caricano tutta la history: finestratura SQL quando supera il contesto utile.
- **M2** — `RateLimiter` usa `threading.Lock` nell'event loop: O(1), ma `asyncio.Lock` sarebbe idiomatico.
- **M3** — `Registry.reset()` esposto pubblicamente: rinominare `_reset_for_tests`.
- **M4** — `identity`: uniformare cap explicit-id alla config (vedi W6).
- **M5** — Preview endpoint: costo retrieval non throttlato (internal-only documentato).
- **M6** — `chunk_canonical_text`: riga non-JSON → canonical dell'intero raw silenziosamente; aggiungere contatore diagnostico.
- **M7** — `with_retries` jitter su `random` globale: iniettare RNG per test deterministici.
- **M8** — Cross-check config: `pinned_budget + reserved` vs usable non validato (harmless per `min()`, ma esplicito).
- **M9** — Test migrazioni: drop-lists manuali duplicate → estrarre helper condiviso `drop_all_objects(pool)` (recidiva già verificata una volta).
- **M10** — `record_tokens` ingoia eccezioni senza log: aggiungere debug log (accounting non deve rompere le request, ma non deve essere invisibile).

---

## 🧪 TEST GAPS

| Area | Mancanza |
|---|---|
| Capture | preservation test esplicito per unknown delta-fields; refusal-only responses |
| Validation | `tool_choice` malformato → test di NON-reject (passthrough intenzionale) |
| Streaming | contract `n>1` via streaming (oggi coperto solo non-streaming) |
| Breaker | HALF_OPEN→OPEN con concorrente durante la probe (integration-level) |
| Retrieval | shape malformate embedding (`data` array mancante) → typed error |
| Media registry | replay con shift dei part_index (nuova immagine stessa posizione) |
| Budget | property test combinato developer + multimodal + retrieved |
| Compose | boot reale dello stack (attualmente check config-level) |

---

## 🔒 SICUREZZA — sintesi

- Trust boundary retrieved→`role=user`: solida strutturalmente (guard engine attivo).
- Residui: W3 redaction, W4 error leak, W5 endpoint exposure.
- Rate-limit bypass via identity rotation: **documentato by design**.
- Nessuna auth sul proxy pubblico: assunzione "rete interna" — da dichiarare
  esplicitamente in testa al README.

---

## ⚙️ SCALA — sintesi

Single-instance by design (rate limit, breaker, metrics process-local).
Punti che diventano prioritari oltre ~10k conversazioni attive o context >128k:

- full-history rewrite O(n) per request;
- rebuild O(n) sincrono;
- capture memory O(response);
- assenza retention (W14).

---

## VERDETTO

Architettura M0–M6 sana. Gli invariant core reggono:

- isolation tra conversazioni ✓
- atomicità delle interaction unit ✓
- token accounting exactly-once ✓
- budget mai superato ✓
- multimodal transparency ✓
- trust boundary retrieved≠system ✓

**4 bug concreti** (B1–B4), **14 debolezze**, **10 miglioramenti**, **8 test gaps**.
Nessuno blocca il merge. Priorità consigliata se si apre un ciclo di fix:
**B2 → B4 → B1 → W3 → W12**, poi i rimanenti per batch.
