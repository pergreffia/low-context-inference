-- M4 final review §3: make the chunk span invariant explicit.
-- 0007 backfills end_seq from stored message ids; every chunk covers at least
-- one message, so after backfill no NULL may legitimately remain. Enforce:
--   end_seq NOT NULL
--   end_seq >= start_seq
-- A failure here signals orphaned chunk rows (deleted messages), i.e. real
-- data corruption that must surface instead of being papered over.

UPDATE conversation_chunks c
SET end_seq = sub.max_seq
FROM (
    SELECT c2.id, MAX(m.seq) AS max_seq
    FROM conversation_chunks c2
    JOIN messages m ON m.id = ANY(c2.message_ids)
    GROUP BY c2.id
) AS sub
WHERE c.id = sub.id AND c.end_seq IS NULL;

ALTER TABLE conversation_chunks ALTER COLUMN end_seq SET NOT NULL;

ALTER TABLE conversation_chunks
    ADD CONSTRAINT ck_chunks_end_gte_start CHECK (end_seq >= start_seq);
