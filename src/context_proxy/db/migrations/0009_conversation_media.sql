-- M6 §13.2: registry of multimodal parts associated with their message
-- (logical interaction unit). Raw content stays verbatim in messages.jsonb —
-- this table is a queryable index for diagnostics and future M6 retrieval;
-- the provider request remains fully reconstructable from messages alone.
--
-- source: 'data' (data: URL payload) or 'url' (remote reference)
-- media_hash: sha256 of the URL string (dedup/diagnostics; never the bytes)

CREATE TABLE IF NOT EXISTS conversation_media (
    id              BIGSERIAL PRIMARY KEY,
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    message_id      UUID NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    part_index      INTEGER NOT NULL CHECK (part_index >= 0),
    kind            TEXT NOT NULL,
    source          TEXT NOT NULL CHECK (source IN ('data', 'url')),
    media_hash      TEXT NOT NULL,
    byte_size       INTEGER NOT NULL CHECK (byte_size >= 0),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT uq_media_message_part UNIQUE (message_id, part_index)
);

CREATE INDEX IF NOT EXISTS idx_media_conversation
    ON conversation_media (conversation_id);
CREATE INDEX IF NOT EXISTS idx_media_message
    ON conversation_media (message_id);
