from __future__ import annotations

from functools import lru_cache

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.config import get_settings

_INIT_SQL = """
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

CREATE TABLE IF NOT EXISTS chunk_texts (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('pdf', 'web')),
    title TEXT NOT NULL,
    page INTEGER,
    url TEXT,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    content TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunk_texts_source_type_idx ON chunk_texts (source_type);
CREATE INDEX IF NOT EXISTS chunk_texts_source_id_idx ON chunk_texts (source_id);
"""


@lru_cache
def app_pool() -> ConnectionPool:
    settings = get_settings()
    pool = ConnectionPool(
        conninfo=settings.app_dsn,
        min_size=1,
        max_size=8,
        kwargs={"row_factory": dict_row, "autocommit": True},
        open=False,
    )
    pool.open(wait=True)
    return pool


def init_app_schema() -> None:
    with app_pool().connection() as conn:
        conn.execute(_INIT_SQL)
