-- M3 final review: durable vector indexing state.
-- The conversation watermark tracks POSTGRESQL CHUNKING progress only
-- (renamed for clarity); Qdrant completion is tracked per-chunk so partial
-- embedding/upsert failures stay visible and retryable.

ALTER TABLE conversations
    RENAME COLUMN last_indexed_seq TO last_chunked_seq;

ALTER TABLE conversation_chunks
    ADD COLUMN IF NOT EXISTS vector_indexed_at TIMESTAMPTZ;
