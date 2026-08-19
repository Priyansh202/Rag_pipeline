from __future__ import annotations

from functools import lru_cache

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.config import get_settings


def _configure(conn) -> None:
    register_vector(conn)


@lru_cache
def vector_pool() -> ConnectionPool:
    settings = get_settings()
    pool = ConnectionPool(
        conninfo=settings.vector_dsn,
        min_size=1,
        max_size=8,
        configure=_configure,
        kwargs={"row_factory": dict_row, "autocommit": True},
        open=False,
    )
    pool.open(wait=True)
    return pool


def _current_embedding_dim(conn) -> int | None:
    exists = conn.execute(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'chunks'
        """
    ).fetchone()
    if not exists:
        return None
    row = conn.execute(
        """
        SELECT atttypmod
        FROM pg_attribute
        WHERE attrelid = 'chunks'::regclass
          AND attname = 'embedding'
          AND NOT attisdropped
        """
    ).fetchone()
    if not row or row["atttypmod"] in (None, -1):
        return None
    return int(row["atttypmod"])


def init_vector_schema() -> None:
    settings = get_settings()
    dim = settings.embedding_dim
    with psycopg.connect(settings.vector_dsn, autocommit=True, row_factory=dict_row) as conn:
        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except Exception as exc:
            probe = conn.execute("SELECT 1 FROM pg_type WHERE typname = 'vector'").fetchone()
            if not probe:
                raise RuntimeError(
                    "Postgres is reachable but the pgvector extension is missing. "
                    "On Railway, deploy the 'Postgres with pgvector' template "
                    "(https://railway.com/deploy/postgres-with-pgvector-engine), "
                    "not the default Postgres plugin."
                ) from exc
        current = _current_embedding_dim(conn)
        if current is not None and current != dim:
            conn.execute("DROP TABLE IF EXISTS chunks")
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                source_type TEXT NOT NULL CHECK (source_type IN ('pdf', 'web')),
                title TEXT NOT NULL,
                page INTEGER,
                url TEXT,
                chunk_index INTEGER NOT NULL DEFAULT 0,
                content TEXT NOT NULL,
                embedding vector({dim}) NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS chunks_source_type_idx ON chunks (source_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS chunks_source_id_idx ON chunks (source_id)")
        try:
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
                ON chunks USING hnsw (embedding vector_cosine_ops)
                """
            )
        except Exception:
            pass
    vector_pool.cache_clear()
