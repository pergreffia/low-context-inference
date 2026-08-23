-- Initial schema (M1). Covers the persistence surface from master prompt §7.
-- Raw content is never destroyed; derived tables reference raw message ids.

CREATE TABLE conversations (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE messages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id),
    seq             BIGINT NOT NULL,
    role            TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
    content         JSONB NOT NULL, -- full raw OpenAI-format message
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (conversation_id, seq)
);

CREATE INDEX idx_messages_conversation ON messages (conversation_id, seq);

CREATE TABLE tool_calls (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id      UUID NOT NULL REFERENCES messages(id), -- assistant message owning the call
    tool_call_id    TEXT NOT NULL,
    name            TEXT NOT NULL,
    arguments       JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (message_id, tool_call_id)
);

CREATE TABLE tool_results (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id      UUID NOT NULL REFERENCES messages(id), -- tool message holding the result
    tool_call_id    TEXT NOT NULL,
    content         JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_tool_results_call ON tool_results (tool_call_id);

CREATE TABLE conversation_chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id),
    message_ids     UUID[] NOT NULL,
    raw_content     TEXT NOT NULL,
    summary         TEXT,
    token_count     INTEGER NOT NULL DEFAULT 0,
    importance      REAL NOT NULL DEFAULT 0.0,
    embedding_ref   TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_chunks_conversation ON conversation_chunks (conversation_id);

CREATE TABLE memory_records (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id    UUID NOT NULL REFERENCES conversations(id),
    kind               TEXT NOT NULL CHECK (kind IN (
                           'decision', 'constraint', 'fact', 'task', 'bug',
                           'implementation', 'tool_result', 'episode_summary',
                           'conversation_summary'
                       )),
    status             TEXT NOT NULL DEFAULT 'active' CHECK (status IN (
                           'active', 'superseded', 'resolved', 'obsolete'
                       )),
    content            TEXT NOT NULL,
    source_message_ids UUID[] NOT NULL DEFAULT '{}',
    importance         REAL NOT NULL DEFAULT 0.0,
    supersedes         UUID REFERENCES memory_records(id),
    superseded_by      UUID REFERENCES memory_records(id),
    embedding_id       TEXT,
    metadata           JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_memory_conversation_kind_status
    ON memory_records (conversation_id, kind, status);

CREATE TABLE summaries (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id    UUID NOT NULL REFERENCES conversations(id),
    kind               TEXT NOT NULL CHECK (kind IN ('episode_summary', 'conversation_summary')),
    content            TEXT NOT NULL,
    token_count        INTEGER NOT NULL DEFAULT 0,
    source_message_ids UUID[] NOT NULL DEFAULT '{}',
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
