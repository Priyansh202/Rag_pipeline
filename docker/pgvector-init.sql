CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('pdf', 'web')),
    title TEXT NOT NULL,
    page INTEGER,
    url TEXT,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    content TEXT NOT NULL,
    embedding vector(384) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_source_type_idx ON chunks (source_type);
CREATE INDEX IF NOT EXISTS chunks_source_id_idx ON chunks (source_id);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
