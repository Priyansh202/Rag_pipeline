CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('pdf', 'web')),
    title TEXT NOT NULL,
    pages INTEGER,
    url TEXT,
    chunks INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'ready',
    extra JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS documents_kind_idx ON documents (kind);
