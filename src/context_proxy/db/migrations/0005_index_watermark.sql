-- M3 review fix: incremental indexing watermark. Only messages newer than
-- last_indexed_seq are scanned on each indexing pass; per-turn uniqueness is
-- still enforced by uq_chunks_conversation_start_seq.

ALTER TABLE conversations
    ADD COLUMN IF NOT EXISTS last_indexed_seq BIGINT;
