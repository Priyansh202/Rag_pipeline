"""Vector retrieval with PDF/web isolation for RBAC (Pinecone or pgvector)."""

from __future__ import annotations

from typing import Literal
from uuid import uuid4

from langchain_core.documents import Document

from app.config import get_settings
from app.retrieval.hashing import chunk_text_hash
from app.retrieval.embeddings import get_embeddings
from app.retrieval.bmb25 import bm25_scores

SourceType = Literal["pdf", "web"]


def _uses_pinecone() -> bool:
    return get_settings().uses_pinecone


def _chunk_id(source_id: str, content_hash: str) -> str:
    return f"{source_id}:{content_hash}"


def _hash_from_chunk_id(source_id: str, chunk_id: str, content: str = "") -> str:
    prefix = f"{source_id}:"
    if chunk_id.startswith(prefix) and len(chunk_id) > len(prefix):
        suffix = chunk_id[len(prefix) :]
        if len(suffix) == 64:
            return suffix
    return chunk_text_hash(content)


def chunk_hashes_for_source(kind: SourceType, source_id: str) -> set[str]:
    if _uses_pinecone():
        from app.db import pinecone_store

        prefix = f"{source_id}:"
        hashes: set[str] = set()
        for chunk_id in pinecone_store.list_chunk_ids(source_id):
            if chunk_id.startswith(prefix):
                suffix = chunk_id[len(prefix) :]
                if len(suffix) == 64:
                    hashes.add(suffix)
        return hashes

    from app.db.pgvector_store import vector_pool

    with vector_pool().connection() as conn:
        rows = conn.execute(
            """
            SELECT id, content
            FROM chunks
            WHERE source_type = %s AND source_id = %s
            """,
            (kind, source_id),
        ).fetchall()
    hashes: set[str] = set()
    for row in rows:
        hashes.add(_hash_from_chunk_id(source_id, row["id"], row["content"]))
    return hashes


def count_chunks_for_source(kind: SourceType, source_id: str) -> int:
    if _uses_pinecone():
        from app.db import pinecone_store

        # Only need "exists or not" to decide whether to reindex.
        # Listing all chunk ids can be extremely slow for large PDFs.
        return 1 if pinecone_store.has_any_vectors_for_source(source_id) else 0

    from app.db.pgvector_store import vector_pool

    with vector_pool().connection() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS n
            FROM chunks
            WHERE source_type = %s AND source_id = %s
            """,
            (kind, source_id),
        ).fetchone()
    return int(row["n"]) if row else 0


def indexed_counts() -> dict[tuple[str, str], int]:
    from app.retrieval.registry import list_sources

    catalog = list_sources()
    counts: dict[tuple[str, str], int] = {}
    for record in catalog.pdfs:
        counts[("pdf", record.id)] = count_chunks_for_source("pdf", record.id)
    for record in catalog.websites:
        counts[("web", record.id)] = count_chunks_for_source("web", record.id)
    return counts


def _pinecone_upsert_rows(kind: SourceType, rows: list[tuple]) -> None:
    from app.db import pinecone_store

    vectors = []
    for chunk_id, source_id, _kind, title, page, url, chunk_index, content, vector in rows:
        vectors.append(
            {
                "id": chunk_id,
                "values": vector,
                "metadata": pinecone_store._chunk_metadata(
                    kind=kind,
                    source_id=source_id,
                    title=title or "Untitled",
                    page=page,
                    url=url,
                    chunk_index=chunk_index,
                    content=content,
                ),
            }
        )
    pinecone_store.upsert_vectors(vectors)


def _pg_insert_chunks(rows: list[tuple]) -> None:
    from app.db.pgvector_store import vector_pool

    if not rows:
        return
    sql = """
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
            """
    with vector_pool().connection() as conn:
        for record in rows:
            conn.execute(sql, record)


def _insert_chunks(kind: SourceType, rows: list[tuple]) -> None:
    if not rows:
        return
    if _uses_pinecone():
        _pinecone_upsert_rows(kind, rows)
    else:
        _pg_insert_chunks(rows)
    from app.db import chunk_catalog

    chunk_catalog.upsert_chunk_rows(rows)


def add_documents(kind: SourceType, documents: list[Document]) -> int:
    if not documents:
        return 0
    vectors = get_embeddings().embed_documents([doc.page_content for doc in documents])
    rows: list[tuple] = []
    for doc, vector in zip(documents, vectors, strict=True):
        meta = doc.metadata or {}
        source_id = str(meta.get("source_id") or uuid4().hex)
        content_hash = meta.get("content_hash") or chunk_text_hash(doc.page_content)
        chunk_index = int(meta.get("chunk_index") or 0)
        page = meta.get("page")
        rows.append(
            (
                _chunk_id(source_id, content_hash),
                source_id,
                kind,
                meta.get("title") or "Untitled",
                page,
                meta.get("url"),
                chunk_index,
                doc.page_content,
                vector,
            )
        )
    _insert_chunks(kind, rows)
    return len(documents)


def _existing_by_hash(kind: SourceType, source_id: str) -> dict[str, str]:
    if _uses_pinecone():
        from app.db import pinecone_store

        existing: dict[str, str] = {}
        for chunk_id in pinecone_store.list_chunk_ids(source_id):
            existing[_hash_from_chunk_id(source_id, chunk_id)] = chunk_id
        return existing

    from app.db.pgvector_store import vector_pool

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
    for row in rows:
        existing_by_hash[_hash_from_chunk_id(source_id, row["id"], row["content"])] = row["id"]
    return existing_by_hash


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

    existing_by_hash = _existing_by_hash(kind, source_id)
    existing_hashes = set(existing_by_hash.keys())
    new_hashes = set(new_by_hash.keys())
    to_remove = existing_hashes - new_hashes
    to_add = new_hashes - existing_hashes
    reused = existing_hashes & new_hashes

    if _uses_pinecone():
        from app.db import pinecone_store

        if to_remove:
            remove_ids = [existing_by_hash[h] for h in to_remove]
            pinecone_store.delete_ids(remove_ids)
            from app.db import chunk_catalog

            chunk_catalog.delete_ids(remove_ids)
        if reused:
            reused_ids = [existing_by_hash[h] for h in reused]
            fetched = pinecone_store.fetch_vectors(reused_ids)
            reuse_rows: list[tuple] = []
            extra_add: set[str] = set()
            for content_hash in reused:
                doc = new_by_hash[content_hash]
                meta = doc.metadata or {}
                chunk_id = existing_by_hash[content_hash]
                vector = fetched.get(chunk_id)
                if not vector:
                    extra_add.add(content_hash)
                    continue
                reuse_rows.append(
                    (
                        chunk_id,
                        source_id,
                        kind,
                        meta.get("title") or "Untitled",
                        meta.get("page"),
                        meta.get("url"),
                        int(meta.get("chunk_index") or 0),
                        doc.page_content,
                        vector,
                    )
                )
            _pinecone_upsert_rows(kind, reuse_rows)
            if reuse_rows:
                from app.db import chunk_catalog

                chunk_catalog.upsert_chunk_rows(reuse_rows)
            to_add = to_add | extra_add
    else:
        from app.db.pgvector_store import vector_pool

        with vector_pool().connection() as conn:
            reuse_rows: list[tuple] = []
            for content_hash in to_remove:
                conn.execute("DELETE FROM chunks WHERE id = %s", (existing_by_hash[content_hash],))
            for content_hash in reused:
                doc = new_by_hash[content_hash]
                meta = doc.metadata or {}
                chunk_id = existing_by_hash[content_hash]
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
                        chunk_id,
                    ),
                )
                reuse_rows.append(
                    (
                        chunk_id,
                        source_id,
                        kind,
                        meta.get("title") or "Untitled",
                        meta.get("page"),
                        meta.get("url"),
                        int(meta.get("chunk_index") or 0),
                        doc.page_content,
                    )
                )
            if to_remove:
                from app.db import chunk_catalog

                chunk_catalog.delete_ids([existing_by_hash[h] for h in to_remove])
            if reuse_rows:
                from app.db import chunk_catalog

                chunk_catalog.upsert_chunk_rows(reuse_rows)

    docs_to_add = [new_by_hash[content_hash] for content_hash in to_add]
    if docs_to_add:
        vectors = get_embeddings().embed_documents([doc.page_content for doc in docs_to_add])
        rows: list[tuple] = []
        for doc, vector in zip(docs_to_add, vectors, strict=True):
            meta = doc.metadata or {}
            content_hash = meta["content_hash"]
            rows.append(
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
                )
            )
        if _uses_pinecone():
            _pinecone_upsert_rows(kind, rows)
        else:
            from app.db.pgvector_store import vector_pool

            sql = """
                    INSERT INTO chunks (
                        id, source_id, source_type, title, page, url,
                        chunk_index, content, embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                    """
            with vector_pool().connection() as conn:
                for record in rows:
                    conn.execute(sql, record)
        from app.db import chunk_catalog

        chunk_catalog.upsert_chunk_rows(rows)

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
    if _uses_pinecone():
        from app.db import pinecone_store

        stats = pinecone_store.pinecone_index().describe_index_stats()
        total = int(getattr(stats, "total_vector_count", 0) or 0)
        if total == 0:
            return []
        query_vector = get_embeddings().embed_query(query)
        return pinecone_store.query_by_source_type(kind, query_vector, k)

    from app.db.pgvector_store import vector_pool

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
            SELECT id, content, title, page, url, source_id, chunk_index,
                   (embedding <=> %s::vector) AS distance
            FROM chunks
            WHERE source_type = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_vector, kind, query_vector, k),
        ).fetchall()
    return rows


def _fetch_corpus_from_pgvector(kind: SourceType) -> list[dict]:
    from app.db.pgvector_store import vector_pool

    with vector_pool().connection() as conn:
        rows = conn.execute(
            """
            SELECT id, source_id, source_type, title, page, url, chunk_index, content
            FROM chunks
            WHERE source_type = %s
            ORDER BY source_id, chunk_index
            """,
            (kind,),
        ).fetchall()
    return [dict(row) for row in rows]


def _corpus_for_kind(kind: SourceType) -> list[dict]:
    from app.db import chunk_catalog

    rows = chunk_catalog.fetch_all_for_kind(kind)
    if rows:
        return rows
    if not _uses_pinecone():
        return _fetch_corpus_from_pgvector(kind)
    return []


def _bm25_candidates(kind: SourceType, query: str, k: int) -> list[dict]:
    settings = get_settings()
    corpus = _corpus_for_kind(kind)
    if not corpus:
        return []
    from app.db import chunk_catalog

    return chunk_catalog.top_bm25_for_kind(
        kind,
        query,
        k,
        k1=settings.hybrid_bm25_k1,
        b=settings.hybrid_bm25_b,
        corpus=corpus,
    )


def _row_chunk_id(row: dict) -> str:
    chunk_id = row.get("id")
    if chunk_id:
        return str(chunk_id)
    source_id = str(row.get("source_id") or "")
    return _chunk_id(source_id, chunk_text_hash(str(row.get("content") or "")))


def hybrid_search(kind: SourceType, query: str, k: int | None = None) -> list[tuple[Document, float]]:
    """Hybrid retrieval: vector candidates + full-corpus BM25, merged and re-ranked."""
    settings = get_settings()
    k = k or settings.retrieval_k
    candidate_k = max(k, settings.retrieval_candidate_k)

    corpus = _corpus_for_kind(kind)
    vector_rows = _vector_candidates(kind, query, candidate_k)
    bm25_rows = _bm25_candidates(kind, query, candidate_k) if corpus else []

    if not vector_rows and not bm25_rows:
        return []

    corpus_scores: dict[str, float] = {}
    if corpus:
        scores = bm25_scores(
            query,
            [str(row["content"]) for row in corpus],
            k1=settings.hybrid_bm25_k1,
            b=settings.hybrid_bm25_b,
        )
        corpus_scores = {
            str(corpus[i]["id"]): scores[i]
            for i in range(len(corpus))
        }

    by_id: dict[str, dict] = {}
    vector_sims: dict[str, float] = {}
    for row in vector_rows:
        chunk_id = _row_chunk_id(row)
        by_id[chunk_id] = row
        dist = float(row["distance"])
        vector_sims[chunk_id] = 1.0 / (1.0 + dist)

    for row in bm25_rows:
        chunk_id = _row_chunk_id(row)
        by_id.setdefault(chunk_id, row)

    merge_ids = set(vector_sims.keys()) | { _row_chunk_id(row) for row in bm25_rows }

    def _norm(xs: list[float]) -> list[float]:
        if not xs:
            return []
        mn, mx = min(xs), max(xs)
        if mx - mn < 1e-12:
            return [0.5 for _ in xs]
        return [(x - mn) / (mx - mn) for x in xs]

    ordered_ids = list(merge_ids)
    v_raw = [vector_sims.get(chunk_id, 0.0) for chunk_id in ordered_ids]
    b_raw = [corpus_scores.get(chunk_id, 0.0) for chunk_id in ordered_ids]
    v_norm = _norm(v_raw)
    b_norm = _norm(b_raw)
    alpha = float(settings.hybrid_alpha)
    combined = [alpha * v + (1.0 - alpha) * b for v, b in zip(v_norm, b_norm, strict=True)]

    from app.retrieval.chunk_quality import is_low_content_chunk, quality_bonus

    adjusted = [
        combined[i] + quality_bonus(str(by_id[ordered_ids[i]].get("content") or ""))
        for i in range(len(combined))
    ]
    ranked = sorted(range(len(adjusted)), key=lambda i: adjusted[i], reverse=True)
    preferred = [
        i
        for i in ranked
        if not is_low_content_chunk(str(by_id[ordered_ids[i]].get("content") or ""))
    ]
    fallback = [i for i in ranked if i not in preferred]
    top_idx = (preferred + fallback)[:k]
    hits: list[tuple[Document, float]] = []
    for i in top_idx:
        chunk_id = ordered_ids[i]
        row = by_id[chunk_id]
        hits.append(
            (
                Document(
                    page_content=str(row.get("content") or ""),
                    metadata={
                        "source_type": kind,
                        "source_id": row.get("source_id"),
                        "title": row.get("title"),
                        "page": row.get("page"),
                        "url": row.get("url"),
                        "chunk_index": row.get("chunk_index"),
                    },
                ),
                float(adjusted[i]),
            )
        )
    return hits


def delete_by_source_id(kind: SourceType, source_id: str) -> int:
    from app.db import chunk_catalog

    chunk_catalog.delete_by_source(kind, source_id)
    if _uses_pinecone():
        from app.db import pinecone_store

        return pinecone_store.delete_by_source(kind, source_id)

    from app.db.pgvector_store import vector_pool

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
    if _uses_pinecone():
        from app.db import pinecone_store

        pinecone_store.reset_pinecone_cache()
