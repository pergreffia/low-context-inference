-- M6 final: `developer` is a first-class trusted instruction role in the
-- OpenAI-compatible contract. It must persist like system/user/assistant/tool
-- and be protected by the context budget (never dropped before ordinary
-- history/retrieval).

ALTER TABLE messages DROP CONSTRAINT messages_role_check;
ALTER TABLE messages ADD CONSTRAINT messages_role_check
    CHECK (role IN ('system', 'developer', 'user', 'assistant', 'tool'));
