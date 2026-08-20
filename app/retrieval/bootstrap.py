"""Restore catalog + vectors from disk after restarts or schema changes."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import REGISTRY_PATH, get_settings
from app.models.schemas import DocumentRecord
from app.retrieval.pdf_processor import extract_pdf_documents
from app.retrieval.registry import add_record, document_count, list_sources
from app.retrieval.vectorstore import add_documents, count_chunks_for_source, indexed_counts


def migrate_local_registry() -> int:
    if document_count() > 0 or not REGISTRY_PATH.exists():
        return 0
    try:
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0

    migrated = 0
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    for row in data.get("pdfs") or []:
        path = Path((row.get("extra") or {}).get("path") or "")
        if not path.exists():
            continue
        source_id = row["id"]
        title = row.get("title") or path.name
        pages, documents = extract_pdf_documents(path, source_id, title)
        chunks = add_documents("pdf", documents)
        add_record(
            "pdf",
            DocumentRecord(
                id=source_id,
                title=title,
                kind="pdf",
                pages=pages,
                chunks=chunks,
                extra=row.get("extra") or {},
            ),
        )
        migrated += 1

    for row in data.get("websites") or []:
        cache = Path((row.get("extra") or {}).get("cache_path") or "")
        if not cache.exists():
            continue
        raw = cache.read_text(encoding="utf-8")
        parts = raw.split("\n\n", 1)
        body = parts[1] if len(parts) == 2 else raw
        source_id = row["id"]
        title = row.get("title") or "Untitled page"
        url = row.get("url")
        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "source_type": "web",
                    "source_id": source_id,
                    "title": title,
                    "url": url,
                    "chunk_index": index,
                },
            )
            for index, chunk in enumerate(splitter.split_text(body))
        ]
        chunks = add_documents("web", documents)
        add_record(
            "web",
            DocumentRecord(
                id=source_id,
                title=title,
                kind="web",
                url=url,
                chunks=chunks,
                extra=row.get("extra") or {},
            ),
        )
        migrated += 1
    return migrated


def _indexed_counts() -> dict[tuple[str, str], int]:
    return indexed_counts()


def _reindex_pdf(record: DocumentRecord) -> int:
    path = Path((record.extra or {}).get("path") or "")
    if not path.exists():
        return 0
    _, documents = extract_pdf_documents(path, record.id, record.title)
    return add_documents("pdf", documents)


def _reindex_web(record: DocumentRecord) -> int:
    cache = Path((record.extra or {}).get("cache_path") or "")
    if not cache.exists():
        return 0
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    raw = cache.read_text(encoding="utf-8")
    parts = raw.split("\n\n", 1)
    body = parts[1] if len(parts) == 2 else raw
    documents = [
        Document(
            page_content=chunk,
            metadata={
                "source_type": "web",
                "source_id": record.id,
                "title": record.title,
                "url": record.url,
                "chunk_index": index,
            },
        )
        for index, chunk in enumerate(splitter.split_text(body))
    ]
    return add_documents("web", documents)


def reindex_missing_vectors() -> int:
    """Rebuild vectors for any catalog row that lost its vector index chunks."""
    catalog = list_sources()
    if not catalog.pdfs and not catalog.websites:
        return 0

    indexed = _indexed_counts()
    restored = 0
    for record in catalog.pdfs:
        if indexed.get(("pdf", record.id), 0) > 0:
            continue
        chunks = _reindex_pdf(record)
        if chunks:
            add_record(
                "pdf",
                record.model_copy(update={"chunks": chunks}),
            )
            restored += 1

    for record in catalog.websites:
        if indexed.get(("web", record.id), 0) > 0:
            continue
        chunks = _reindex_web(record)
        if chunks:
            add_record(
                "web",
                record.model_copy(update={"chunks": chunks}),
            )
            restored += 1
    return restored


def _backfill_chunk_texts_from_pgvector() -> int:
    from app.db import chunk_catalog
    from app.db.pgvector_store import vector_pool

    with vector_pool().connection() as conn:
        rows = conn.execute(
            """
            SELECT id, source_id, source_type, title, page, url, chunk_index, content
            FROM chunks
            ORDER BY source_type, source_id, chunk_index
            """
        ).fetchall()
    if not rows:
        return 0
    payload = [
        (
            row["id"],
            row["source_id"],
            row["source_type"],
            row["title"],
            row["page"],
            row["url"],
            row["chunk_index"],
            row["content"],
        )
        for row in rows
    ]
    chunk_catalog.upsert_chunk_rows(payload)
    return len(payload)


def _backfill_chunk_texts_from_pinecone() -> int:
    from app.db import chunk_catalog, pinecone_store

    catalog = list_sources()
    restored = 0
    for record in catalog.pdfs:
        if chunk_catalog.count_for_source("pdf", record.id) > 0:
            continue
        if count_chunks_for_source("pdf", record.id) <= 0:
            continue
        ids = pinecone_store.list_chunk_ids(record.id)
        if not ids:
            continue
        records = pinecone_store.fetch_chunk_records(ids)
        payload = [
            (
                row["id"],
                row["source_id"],
                "pdf",
                row["title"],
                row["page"],
                row["url"],
                row["chunk_index"],
                row["content"],
            )
            for row in records
            if row.get("content")
        ]
        if payload:
            chunk_catalog.upsert_chunk_rows(payload)
            restored += len(payload)

    for record in catalog.websites:
        if chunk_catalog.count_for_source("web", record.id) > 0:
            continue
        if count_chunks_for_source("web", record.id) <= 0:
            continue
        ids = pinecone_store.list_chunk_ids(record.id)
        if not ids:
            continue
        records = pinecone_store.fetch_chunk_records(ids)
        payload = [
            (
                row["id"],
                row["source_id"],
                "web",
                row["title"],
                row["page"],
                row["url"],
                row["chunk_index"],
                row["content"],
            )
            for row in records
            if row.get("content")
        ]
        if payload:
            chunk_catalog.upsert_chunk_rows(payload)
            restored += len(payload)
    return restored


def backfill_chunk_texts() -> int:
    """Populate chunk_texts for BM25 when vectors exist but text catalog is empty."""
    from app.config import get_settings
    from app.db import chunk_catalog

    settings = get_settings()
    if settings.uses_pgvector:
        if chunk_catalog.count_for_kind("pdf") or chunk_catalog.count_for_kind("web"):
            return 0
        return _backfill_chunk_texts_from_pgvector()

    if not settings.uses_pinecone:
        return 0
    return _backfill_chunk_texts_from_pinecone()
