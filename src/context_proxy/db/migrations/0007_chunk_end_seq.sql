-- M4 review: structural chunk identity — explicit end of the authoritative
-- message seq range covered by the chunk. Backfilled from the stored message
-- ids so existing rows carry the same identity guarantees as new ones.
ALTER TABLE conversation_chunks ADD COLUMN IF NOT EXISTS end_seq INTEGER;

UPDATE conversation_chunks c
SET end_seq = sub.max_seq
FROM (
    SELECT c2.id, MAX(m.seq) AS max_seq
    FROM conversation_chunks c2
    JOIN messages m ON m.id = ANY(c2.message_ids)
    GROUP BY c2.id
) AS sub
WHERE c.id = sub.id AND c.end_seq IS NULL;
