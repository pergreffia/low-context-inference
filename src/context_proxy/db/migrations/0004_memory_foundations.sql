-- M3: memory service foundations.
-- start_seq makes turn chunking idempotent (one chunk per conversation/turn).
-- Generated tsvector columns power lexical (full-text) retrieval; Qdrant is
-- only the semantic leg and stays a rebuildable derived index.

ALTER TABLE conversation_chunks
    ADD COLUMN IF NOT EXISTS start_seq BIGINT;

-- Full unique constraint (NULL start_seq never conflicts): arbiter for the
-- idempotent turn-chunking upsert.
CREATE UNIQUE INDEX IF NOT EXISTS uq_chunks_conversation_start_seq
    ON conversation_chunks (conversation_id, start_seq);

ALTER TABLE conversation_chunks
    ADD COLUMN IF NOT EXISTS ts tsvector
    GENERATED ALWAYS AS (to_tsvector('simple', coalesce(raw_content, ''))) STORED;

CREATE INDEX IF NOT EXISTS idx_chunks_ts ON conversation_chunks USING GIN (ts);

ALTER TABLE memory_records
    ADD COLUMN IF NOT EXISTS ts tsvector
    GENERATED ALWAYS AS (to_tsvector('simple', coalesce(content, ''))) STORED;

CREATE INDEX IF NOT EXISTS idx_memory_ts ON memory_records USING GIN (ts);
