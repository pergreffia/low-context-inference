# Multimodal support

Scope: OpenAI-style content-part arrays on chat messages — `text` and
`image_url` parts, including `data:` URLs. There is no video/audio support
and no server-side image processing: images are never fetched, decoded or
described by the proxy.

## Content transparency contract

`content` may be a string or an array of parts. The array is preserved
verbatim through every stage:

```text
client payload → validation → PostgreSQL (raw JSONB) → context selection
→ provider request      [byte-identical part arrays everywhere]
```

Arrays are never stringified; unknown part types are not dropped.

### Example request

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "vision-model",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "what does this screenshot show?"},
        {"type": "image_url",
         "image_url": {"url": "https://example.com/shot.png"}},
        {"type": "image_url",
         "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="}}
      ]
    }],
    "conversation_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7"
  }'
```

Supported shapes (all verified by tests):

| Shape | Behavior |
|---|---|
| single/multiple `image_url` parts | forwarded verbatim |
| `http(s)://` remote URLs | verbatim; never fetched by the proxy |
| `data:image/...;base64,...` URLs | verbatim; size affects only the stored row, not the token budget |
| text + image mixed | both kept in order |
| unknown part types (`type: "acme_hologram", ...`) | opaque passthrough, flat token cost |
| empty content list `[]` | valid, forwarded as-is |

Any role may carry parts — including `developer` (a developer message with
an image stays a trusted instruction with role `developer`).

## Persistence and the media registry

Raw content lives in `messages.jsonb` (source of truth). Additionally each
`image_url` part is registered in `conversation_media` (migration `0009`):

| Column | Meaning |
|---|---|
| `part_index` | position inside the content array (0-based) |
| `kind` | `image_url` |
| `source` | `data` for data URLs, `url` for remote |
| `media_hash` | sha256 of the source reference string (deterministic) |
| `source_size` | **length of the reference string as sent** — not decoded media bytes |

Registry writes are idempotent per `(message_id, part_index)`; replays do
not duplicate rows. Unknown/non-image parts are intentionally not registered.
Multi-megabyte data URLs persist raw with bounded metadata handling.

## Context budgeting

Token estimation (`context/tokens.py`) treats parts as:

| Part type | Cost |
|---|---|
| `text` | normal character estimate |
| `image_url` | flat `1024` tokens regardless of payload size |
| anything else | flat `16` tokens |

Consequences: a 5 MB data URL costs the same ~1k tokens as a thumbnail;
multimodal turns participate in selection atomically (a user turn with an
image drops together with its assistant answer, never split).

## Retrieval behavior

Only **text** feeds retrieval. The retrieval query is built from the latest
user message's `text` parts joined by spaces:

```json
{"role":"user","content":[
  {"type":"text","text":"compare these two"},
  {"type":"image_url","image_url":{"url":"data:image/png;base64,…"}}
]}
→ query = "compare these two"
```

Image URLs (remote or data), their query strings and unknown parts never
leak into lexical or semantic queries. An image-only message yields an empty
query (no retrieval).

## Limits that actually exist

- body size: `SERVER__MAX_BODY_BYTES` (default 8 MiB) applies to the whole
  request including base64 payloads;
- budget: images cost flat 1024 tokens each (see above);
- no transcoding/resizing: what you send upstream is exactly what the model gets;
- persistence stores full data URLs — plan storage accordingly for large images.

See also: [memory-and-context.md](memory-and-context.md) · [api.md](api.md)
