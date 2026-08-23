-- M2.1: response-level semantic state (finish_reason, model, usage) captured
-- from inference responses is stored alongside the raw message. Transport
-- framing (SSE boundaries) is never stored.

ALTER TABLE messages
    ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb;
