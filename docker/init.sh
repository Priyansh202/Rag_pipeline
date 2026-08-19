#!/bin/bash
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-SQL
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
SQL

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres <<-SQL
SELECT 'CREATE DATABASE rag_vectors'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'rag_vectors')\gexec
SQL

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname rag_vectors <<-SQL
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
SQL
