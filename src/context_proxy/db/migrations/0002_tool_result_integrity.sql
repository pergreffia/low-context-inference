-- Tool-call/result relational integrity (M1.1).
--
-- Rationale for not using a direct FK on tool_call_id (text): the OpenAI
-- protocol treats tool call IDs as opaque strings. Some providers and local
-- runtimes do NOT guarantee global uniqueness of these identifiers, so a
-- database-level relationship keyed on the text value alone would be unsound.
-- The canonical relational link is therefore the surrogate key:
--
--   tool_results.tool_call_ref -> tool_calls.id  (set at persistence time, M2)
--
-- NULL is allowed because M1 performs no persistence; once message persistence
-- lands (M2) the application MUST populate tool_call_ref on insert.
-- UNIQUE (message_id, tool_call_id) additionally prevents duplicate results
-- for the same call within a single tool message.

ALTER TABLE tool_results
    ADD COLUMN tool_call_ref UUID REFERENCES tool_calls (id);

CREATE INDEX idx_tool_results_call_ref ON tool_results (tool_call_ref);

ALTER TABLE tool_results
    ADD CONSTRAINT uq_tool_results_message_call UNIQUE (message_id, tool_call_id);
