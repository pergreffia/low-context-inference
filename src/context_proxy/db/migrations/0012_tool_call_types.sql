-- 0012: relational projection support for custom tool calls (post-0876b10
-- review §6). The raw message in messages.jsonb remains the source of truth
-- and is NOT modified; this only lets the queryable tool_calls table carry
-- custom-type calls without destructive normalization into function shape.
--
--   function calls: call_type='function', name, arguments (JSONB)
--   custom calls:   call_type='custom',    name, input    (JSONB)
--   extra fields:   preserved verbatim in `extra` (everything except
--                   id/type/function/custom)

ALTER TABLE tool_calls ADD COLUMN call_type TEXT NOT NULL DEFAULT 'function';
ALTER TABLE tool_calls ADD COLUMN input JSONB;
ALTER TABLE tool_calls ADD COLUMN extra JSONB NOT NULL DEFAULT '{}'::jsonb;

-- arguments is only meaningful for function calls; custom calls carry input.
ALTER TABLE tool_calls ALTER COLUMN arguments DROP NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'tool_calls_type_shape'
    ) THEN
        ALTER TABLE tool_calls ADD CONSTRAINT tool_calls_type_shape CHECK (
            (call_type = 'function' AND arguments IS NOT NULL)
            OR (call_type = 'custom' AND input IS NOT NULL)
            OR call_type NOT IN ('function', 'custom')
        );
    END IF;
END $$;
