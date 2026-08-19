"""Chunk-level deduplication and document update detection."""

from __future__ import annotations

from langchain_core.documents import Document

from app.config import get_settings
from app.models.schemas import DocumentRecord
from app.retrieval.hashing import chunk_text_hash
from app.retrieval.registry import list_sources
from app.retrieval.vectorstore import chunk_hashes_for_source


def attach_chunk_hashes(documents: list[Document]) -> list[Document]:
    for doc in documents:
        meta = dict(doc.metadata or {})
        meta["content_hash"] = chunk_text_hash(doc.page_content)
        doc.metadata = meta
    return documents


def chunk_hash_set(documents: list[Document]) -> set[str]:
    return {doc.metadata["content_hash"] for doc in documents if doc.metadata.get("content_hash")}


def overlap_score(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    shared = len(left & right)
    return shared / min(len(left), len(right))


def find_pdf_update_candidate(chunk_hashes: set[str]) -> DocumentRecord | None:
    """Pick an existing PDF if most of its chunks still match the new upload."""
    if not chunk_hashes:
        return None
    settings = get_settings()
    threshold = settings.pdf_update_overlap_ratio
    best: DocumentRecord | None = None
    best_score = 0.0
    for record in list_sources().pdfs:
        existing = chunk_hashes_for_source("pdf", record.id)
        if not existing:
            continue
        score = overlap_score(chunk_hashes, existing)
        if score >= threshold and score > best_score:
            best_score = score
            best = record
    return best
