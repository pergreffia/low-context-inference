# API

Base URL: `http://<host>:8080`. The proxy is OpenAI-compatible: clients are
standard SDKs/agents pointed at the proxy base URL. Request/response bodies
are never rewritten (except documented error normalization).

## Public endpoints

| Method | Path | Behavior |
|---|---|---|
| `POST` | `/v1/chat/completions` | chat, streaming and buffered |
| `GET` | `/v1/models` | verbatim passthrough from the upstream provider |
| `GET` | `/healthz`, `/readyz`, `/metrics` | operational; see [operations.md](operations.md) |

## Chat completions

### Request shape

```json
{
  "model": "any-string-or-mapped-model",
  "messages": [ {"role": "...", "content": "..."}, ... ],
  "stream": false,
  "tools": [],
  "conversation_id": "optional-valid-uuid",
  "n": 1
}
```

Supported fields and semantics:

| Field | Supported | Notes |
|---|---|---|
| `model` | yes | forwarded unchanged; overridden by `INFERENCE__MODEL` when configured |
| `messages` | yes, required | array of message objects; roles below |
| `stream` | yes (bool) | SSE when true |
| `tools` / `tool_choice` | yes | passed through after structural validation |
| `conversation_id` | yes | valid UUID; stripped before forwarding upstream; echoed back via `X-Conversation-ID` |
| `n` | only `n=1` | other values → `400` (`only n=1 is supported by this proxy`) |
| `temperature`, `max_tokens`, etc. | untouched | opaque passthrough — not validated or modified |

### Message roles

- **system** — trusted instruction tier.
- **developer** — first-class trusted tier: persisted verbatim with
  `role == "developer"` (never normalized to `system`), protected under
  budget pressure like system messages.
- **user** — ordinary request content; may be a string or a multimodal part
  array (see [multimodal.md](multimodal.md)).
- **assistant** — model output; may carry `tool_calls`; replayed positionally
  on later turns.
- **tool** — tool results; carries `tool_call_id`; association to its call is
  per-conversation.

### Tool calls

Tool definitions support `type: "function"` (`function.name`,
`function.parameters`) and OpenAI's `type: "custom"` (`custom.name`);
unknown tool types pass through opaquely. Assistant `tool_calls` accept the
same two types plus unknown types. Lifecycle:

```text
client sends tools → upstream emits tool_calls → validated → captured
(streamed deltas reconstructed: id/type/name/arguments/input)
→ persisted (raw JSONB + relational projection) → client replays
history including assistant.tool_calls + role:"tool" result
```

Replay preserves call structure exactly (id, type, function/custom payload,
extra transport fields).

### Conversation handling

Identity precedence: body `conversation_id` > `X-Conversation-ID` header >
session header (`CONVERSATION__CLIENT_ID_HEADER`, default `X-Session-ID`;
deterministically uuid5-mapped) > generated UUID.

Clients SHOULD send the full conversation each turn (positional
reconciliation persists only the new suffix) or just the new user turn —
both work because the proxy assembles context server-side from stored
history. A resent history that diverges from stored truth at any position is
rejected:

```json
409 {"error": {"code": "history_conflict", ...}}
```

### Streaming

`"stream": true` returns `text/event-stream` with byte-exact passthrough of
upstream frames — framing, chunking and content are never rewritten. The
proxy reconstructs the assistant message from deltas in a bounded side
channel for persistence only.

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"m","stream":true,
       "messages":[{"role":"user","content":"count to five"}],
       "conversation_id":"7c9e6679-7425-40de-944b-e07fc1f90ae7"}'
```

Frames follow the standard format:

```text
data: {"id":"...","choices":[{"delta":{"content":"one"}}]}

data: [DONE]
```

If the stream exceeds `SERVER__MAX_CAPTURE_BYTES`, persistence of that
assistant message is skipped but the client stream still completes fully.
## Errors

All errors use the OpenAI envelope:

```json
{"error": {"message": "...", "type": "...", "param": null, "code": "..."}}
```

| Status | `code` / `type` | Cause |
|---|---|---|
| 400 | `invalid_request_body` | non-object/malformed JSON, invalid shapes |
| 400 | `invalid_conversation_id` | explicit conversation id not a valid UUID |
| 400 | `context_length_exceeded` | mandatory context alone exceeds usable budget |
| 401 | — (FastAPI detail) | `/internal/*` without valid `X-Internal-Auth` when token configured |
| 409 | `history_conflict` | inbound history diverges from stored truth |
| 413 | `request_too_large` | body over `SERVER__MAX_BODY_BYTES` |
| 422 | FastAPI validation | malformed `/internal/*` bodies |
| 429 | `rate_limit_exceeded` (`rate_limit_error`) | bucket exhausted; see `Retry-After` |
| 502 | `upstream_unavailable` (`api_error`) | transport failure/breaker open; generic message by design |
| 4xx/5xx (passthrough) | upstream's own body | upstream answered an error: JSON bodies forwarded verbatim, non-JSON rewritten to `upstream_error` with safe headers |
| 500 | `internal_error` | unexpected proxy error: generic body, details only in logs |

No stack traces, paths, DSNs, credentials or exception text ever appear in
client-visible errors.

## Headers

Request:

| Header | Effect |
|---|---|
| `X-Conversation-ID` | explicit conversation identity (valid UUID required); also selects the rate-limit conversation bucket |
| `X-Session-ID` (configurable) | stable opaque session token → derived conversation |
| `X-Request-ID` | honored if present, else generated and echoed on the response |
| `X-Internal-Auth` | internal endpoints only; required when `SECURITY__INTERNAL_AUTH_TOKEN` is set |

Response:

| Header | Effect |
|---|---|
| `X-Conversation-ID` | resolved conversation for this request |
| `X-Request-ID` | correlation id (upstream value wins when present) |
| `Retry-After` | seconds, present on 429 |

Hop-by-hop headers (`connection`, declared connection-tokens, `keep-alive`,
`transfer-encoding`, `te`, `trailer`, `upgrade`, `proxy-*`), `set-cookie`
and upstream `content-length` are never forwarded. Buffered responses drop
`content-encoding` (bodies arrive decompressed); streamed responses keep it
(raw bytes). Missing `content-type` gets a default.

## Internal API

`/internal/v1/*` — administrative surface (memory management, indexing,
rebuild, retrieval probe, diagnostics, context preview). Full endpoint table:
[operations.md](operations.md#internal-endpoints-administrative).

Security model:

- intended for a private network; the prefix is NOT authentication;
- when `SECURITY__INTERNAL_AUTH_TOKEN` is set every call needs
  `X-Internal-Auth`;
- `SECURITY__MODE=production` refuses to start without that token
  (fail-closed). See [security.md](security.md) and
  [configuration.md](configuration.md).

See also: [memory-and-context.md](memory-and-context.md) ·
[multimodal.md](multimodal.md)
