"""Postgres-backed chunk text catalog for full-corpus BM25 keyword search."""

from __future__ import annotations

from typing import Literal

from app.db.postgres import app_pool
from app.retrieval.bmb25 import bm25_scores, bm25_top_indices

SourceType = Literal["pdf", "web"]

ChunkRow = dict[str, object]


def _row_from_record(record: tuple) -> tuple:
    """(id, source_id, source_type, title, page, url, chunk_index, content)."""
    chunk_id, source_id, kind, title, page, url, chunk_index, content = record[:8]
    return (chunk_id, source_id, kind, title, page, url, chunk_index, content)


def upsert_chunk_rows(rows: list[tuple]) -> None:
    if not rows:
        return
    payload = [_row_from_record(row) for row in rows]
    sql = """
            INSERT INTO chunk_texts (
                id, source_id, source_type, title, page, url, chunk_index, content
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                title = EXCLUDED.title,
                page = EXCLUDED.page,
                url = EXCLUDED.url,
                chunk_index = EXCLUDED.chunk_index,
                content = EXCLUDED.content
            """
    with app_pool().connection() as conn:
        for record in payload:
            conn.execute(sql, record)


def delete_by_source(kind: SourceType, source_id: str) -> int:
    with app_pool().connection() as conn:
        deleted = conn.execute(
            """
            DELETE FROM chunk_texts
            WHERE source_type = %s AND source_id = %s
            RETURNING id
            """,
            (kind, source_id),
        ).fetchall()
    return len(deleted)


def delete_ids(ids: list[str]) -> int:
    if not ids:
        return 0
    with app_pool().connection() as conn:
        deleted = conn.execute(
            "DELETE FROM chunk_texts WHERE id = ANY(%s) RETURNING id",
            (ids,),
        ).fetchall()
    return len(deleted)


def count_for_kind(kind: SourceType) -> int:
    with app_pool().connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM chunk_texts WHERE source_type = %s",
            (kind,),
        ).fetchone()
    return int(row["n"]) if row else 0


def count_for_source(kind: SourceType, source_id: str) -> int:
    with app_pool().connection() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS n
            FROM chunk_texts
            WHERE source_type = %s AND source_id = %s
            """,
            (kind, source_id),
        ).fetchone()
    return int(row["n"]) if row else 0


def fetch_all_for_kind(kind: SourceType) -> list[ChunkRow]:
    with app_pool().connection() as conn:
        rows = conn.execute(
            """
            SELECT id, source_id, source_type, title, page, url, chunk_index, content
            FROM chunk_texts
            WHERE source_type = %s
            ORDER BY source_id, chunk_index
            """,
            (kind,),
        ).fetchall()
    return [dict(row) for row in rows]


def top_bm25_for_kind(
    kind: SourceType,
    query: str,
    k: int,
    *,
    k1: float,
    b: float,
    corpus: list[ChunkRow] | None = None,
) -> list[ChunkRow]:
    """Return top-k chunks for *kind* scored over the full corpus."""
    rows = corpus if corpus is not None else fetch_all_for_kind(kind)
    if not rows or k <= 0:
        return []

    scores = bm25_scores(
        query,
        [str(row["content"]) for row in rows],
        k1=k1,
        b=b,
    )
    ranked = bm25_top_indices(scores, k)
    hits: list[ChunkRow] = []
    for index in ranked:
        if scores[index] <= 0:
            continue
        hit = dict(rows[index])
        hit["bm25_score"] = scores[index]
        hits.append(hit)
    return hits
