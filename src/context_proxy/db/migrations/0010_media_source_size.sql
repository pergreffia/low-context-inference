-- M6 review §5: remove the misleading byte_size semantics.
--
-- The stored value is the length of the SOURCE REFERENCE string (the data:
-- URL or remote URL as sent by the client), never the decoded media bytes —
-- remote media is deliberately never fetched during persistence.
-- Renamed to source_size so the schema cannot suggest otherwise.

ALTER TABLE conversation_media RENAME COLUMN byte_size TO source_size;
