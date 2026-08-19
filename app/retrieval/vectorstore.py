"""pgvector-backed retrieval with PDF/web isolation for RBAC."""

from __future__ import annotations

from typing import Literal
from uuid import uuid4

from langchain_core.documents import Document

from app.config import get_settings
from app.db.pgvector_store import vector_pool
from app.retrieval.hashing import chunk_text_hash
from app.retrieval.embeddings import get_embeddings
from app.retrieval.bmb25 import bm25_scores

SourceType = Literal["pdf", "web"]


def _chunk_id(source_id: str, content_hash: str) -> str:
    return f"{source_id}:{content_hash}"


def chunk_hashes_for_source(kind: SourceType, source_id: str) -> set[str]:
    with vector_pool().connection() as conn:
        rows = conn.execute(
            """
            SELECT id, content
            FROM chunks
            WHERE source_type = %s AND source_id = %s
            """,
            (kind, source_id),
        ).fetchall()
    prefix = f"{source_id}:"
    hashes: set[str] = set()
    for row in rows:
        chunk_id = row["id"]
        if chunk_id.startswith(prefix) and len(chunk_id) > len(prefix):
            suffix = chunk_id[len(prefix) :]
            if len(suffix) == 64:
                hashes.add(suffix)
                continue
        hashes.add(chunk_text_hash(row["content"]))
    return hashes


def add_documents(kind: SourceType, documents: list[Document]) -> int:
    if not documents:
        return 0
    vectors = get_embeddings().embed_documents([doc.page_content for doc in documents])
    with vector_pool().connection() as conn:
        with conn.cursor() as cur:
            for doc, vector in zip(documents, vectors, strict=True):
                meta = doc.metadata or {}
                source_id = str(meta.get("source_id") or uuid4().hex)
                content_hash = meta.get("content_hash") or chunk_text_hash(doc.page_content)
                chunk_index = int(meta.get("chunk_index") or 0)
                page = meta.get("page")
                chunk_id = _chunk_id(source_id, content_hash)
                cur.execute(
                    """
                    INSERT INTO chunks (
                        id, source_id, source_type, title, page, url,
                        chunk_index, content, embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        title = EXCLUDED.title,
                        page = EXCLUDED.page,
                        chunk_index = EXCLUDED.chunk_index,
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding
                    """,
                    (
                        chunk_id,
                        source_id,
                        kind,
                        meta.get("title") or "Untitled",
                        page,
                        meta.get("url"),
                        chunk_index,
                        doc.page_content,
                        vector,
                    ),
                )
    return len(documents)


def sync_documents(kind: SourceType, source_id: str, documents: list[Document]) -> dict[str, int]:
    """Keep unchanged chunk vectors, embed only new/changed text, drop removed chunks."""
    if not documents:
        delete_by_source_id(kind, source_id)
        return {"added": 0, "removed": 0, "reused": 0, "total": 0}

    new_by_hash: dict[str, Document] = {}
    for doc in documents:
        meta = doc.metadata or {}
        content_hash = meta.get("content_hash") or chunk_text_hash(doc.page_content)
        meta["content_hash"] = content_hash
        doc.metadata = meta
        new_by_hash[content_hash] = doc

    with vector_pool().connection() as conn:
        rows = conn.execute(
            """
            SELECT id, content
            FROM chunks
            WHERE source_type = %s AND source_id = %s
            """,
            (kind, source_id),
        ).fetchall()

        existing_by_hash: dict[str, str] = {}
        prefix = f"{source_id}:"
        for row in rows:
            chunk_id = row["id"]
            if chunk_id.startswith(prefix) and len(chunk_id) > len(prefix):
                suffix = chunk_id[len(prefix) :]
                if len(suffix) == 64:
                    existing_by_hash[suffix] = chunk_id
                    continue
            existing_by_hash[chunk_text_hash(row["content"])] = chunk_id

        existing_hashes = set(existing_by_hash.keys())
        new_hashes = set(new_by_hash.keys())
        to_remove = existing_hashes - new_hashes
        to_add = new_hashes - existing_hashes
        reused = existing_hashes & new_hashes

        for content_hash in to_remove:
            conn.execute("DELETE FROM chunks WHERE id = %s", (existing_by_hash[content_hash],))

        for content_hash in reused:
            doc = new_by_hash[content_hash]
            meta = doc.metadata or {}
            conn.execute(
                """
                UPDATE chunks
                SET title = %s, page = %s, chunk_index = %s, content = %s
                WHERE id = %s
                """,
                (
                    meta.get("title") or "Untitled",
                    meta.get("page"),
                    int(meta.get("chunk_index") or 0),
                    doc.page_content,
                    existing_by_hash[content_hash],
                ),
            )

    docs_to_add = [new_by_hash[content_hash] for content_hash in to_add]
    if docs_to_add:
        vectors = get_embeddings().embed_documents([doc.page_content for doc in docs_to_add])
        with vector_pool().connection() as conn:
            with conn.cursor() as cur:
                for doc, vector in zip(docs_to_add, vectors, strict=True):
                    meta = doc.metadata or {}
                    content_hash = meta["content_hash"]
                    cur.execute(
                        """
                        INSERT INTO chunks (
                            id, source_id, source_type, title, page, url,
                            chunk_index, content, embedding
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO NOTHING
                        """,
                        (
                            _chunk_id(source_id, content_hash),
                            source_id,
                            kind,
                            meta.get("title") or "Untitled",
                            meta.get("page"),
                            meta.get("url"),
                            int(meta.get("chunk_index") or 0),
                            doc.page_content,
                            vector,
                        ),
                    )

    return {
        "added": len(to_add),
        "removed": len(to_remove),
        "reused": len(reused),
        "total": len(new_hashes),
    }


def similarity_search(kind: SourceType, query: str, k: int | None = None) -> list[tuple[Document, float]]:
    settings = get_settings()
    k = k or settings.retrieval_k
    rows = _vector_candidates(kind, query, k)
    hits: list[tuple[Document, float]] = []
    for row in rows:
        hits.append(
            (
                Document(
                    page_content=row["content"],
                    metadata={
                        "source_type": kind,
                        "source_id": row["source_id"],
                        "title": row["title"],
                        "page": row["page"],
                        "url": row["url"],
                        "chunk_index": row["chunk_index"],
                    },
                ),
                float(row["distance"]),
            )
        )
    return hits


def _vector_candidates(kind: SourceType, query: str, k: int) -> list[dict]:
    with vector_pool().connection() as conn:
        count_row = conn.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE source_type = %s",
            (kind,),
        ).fetchone()
        if not count_row or int(count_row["n"]) == 0:
            return []
        query_vector = get_embeddings().embed_query(query)
        rows = conn.execute(
            """
            SELECT content, title, page, url, source_id, chunk_index,
                   (embedding <=> %s::vector) AS distance
            FROM chunks
            WHERE source_type = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_vector, kind, query_vector, k),
        ).fetchall()
    return rows


def hybrid_search(kind: SourceType, query: str, k: int | None = None) -> list[tuple[Document, float]]:
    """Hybrid retrieval: pgvector top candidates + BM25 keyword scoring.

    We compute BM25 over the candidate set (not the full corpus) for speed.
    """
    settings = get_settings()
    k = k or settings.retrieval_k
    candidate_k = max(k, settings.retrieval_candidate_k)

    vector_rows = _vector_candidates(kind, query, candidate_k)
    if not vector_rows:
        return []

    # vector distance -> similarity (smaller distance => higher similarity)
    vector_sims: list[float] = []
    for row in vector_rows:
        dist = float(row["distance"])
        vector_sims.append(1.0 / (1.0 + dist))

    bm25_doc_texts = [row["content"] for row in vector_rows]
    bm25 = bm25_scores(
        query,
        bm25_doc_texts,
        k1=settings.hybrid_bm25_k1,
        b=settings.hybrid_bm25_b,
    )

    # Normalize both signals to [0,1] so alpha works consistently.
    def _norm(xs: list[float]) -> list[float]:
        if not xs:
            return []
        mn, mx = min(xs), max(xs)
        if mx - mn < 1e-12:
            return [0.5 for _ in xs]
        return [(x - mn) / (mx - mn) for x in xs]

    v_norm = _norm(vector_sims)
    b_norm = _norm(bm25)

    alpha = float(settings.hybrid_alpha)
    combined = [alpha * v + (1.0 - alpha) * b for v, b in zip(v_norm, b_norm, strict=True)]

    # Top-k combined.
    top_idx = sorted(range(len(combined)), key=lambda i: combined[i], reverse=True)[:k]
    hits: list[tuple[Document, float]] = []
    for i in top_idx:
        row = vector_rows[i]
        hits.append(
            (
                Document(
                    page_content=row["content"],
                    metadata={
                        "source_type": kind,
                        "source_id": row["source_id"],
                        "title": row["title"],
                        "page": row["page"],
                        "url": row["url"],
                        "chunk_index": row["chunk_index"],
                    },
                ),
                float(combined[i]),
            )
        )
    return hits


def delete_by_source_id(kind: SourceType, source_id: str) -> int:
    with vector_pool().connection() as conn:
        row = conn.execute(
            """
            DELETE FROM chunks
            WHERE source_type = %s AND source_id = %s
            RETURNING id
            """,
            (kind, source_id),
        ).fetchall()
    return len(row)


def reset_cache() -> None:
    get_embeddings.cache_clear()
