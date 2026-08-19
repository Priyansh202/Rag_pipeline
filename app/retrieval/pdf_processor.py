from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import PDF_DIR, get_settings
from app.models.schemas import DocumentRecord
from app.retrieval.dedup import attach_chunk_hashes, chunk_hash_set, find_pdf_update_candidate
from app.retrieval.registry import add_record, find_pdf_by_content_hash
from app.retrieval.vectorstore import add_documents, sync_documents
from app.security import sanitize_filename


def _splitter() -> RecursiveCharacterTextSplitter:
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def ingest_pdf(file_bytes: bytes, original_name: str) -> DocumentRecord:
    filename = sanitize_filename(original_name)
    content_hash = hashlib.sha256(file_bytes).hexdigest()

    existing = find_pdf_by_content_hash(content_hash)
    if existing and (existing.extra or {}).get("path") and Path(existing.extra["path"]).exists():
        return existing

    temp_id = uuid4().hex
    temp_path = PDF_DIR / f"_tmp_{temp_id}_{filename}"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(file_bytes)

    try:
        _, documents = extract_pdf_documents(temp_path, temp_id, filename)
        attach_chunk_hashes(documents)
        new_hashes = chunk_hash_set(documents)
        pages = _page_count(temp_path)

        update_target = find_pdf_update_candidate(new_hashes)
        if update_target:
            source_id = update_target.id
            old_path = (update_target.extra or {}).get("path")
            dest_path = PDF_DIR / f"{source_id}_{filename}"
            dest_path.write_bytes(file_bytes)
            if old_path and Path(old_path) != dest_path:
                Path(old_path).unlink(missing_ok=True)
            sync_stats = sync_documents("pdf", source_id, documents)
            temp_path.unlink(missing_ok=True)
            record = DocumentRecord(
                id=source_id,
                title=filename,
                kind="pdf",
                pages=pages,
                chunks=sync_stats["total"],
                status="ready",
                extra={
                    "path": str(dest_path),
                    "content_hash": content_hash,
                    "sync": sync_stats,
                    "updated_from": update_target.title,
                },
            )
            return add_record("pdf", record)

        source_id = uuid4().hex
        dest_path = PDF_DIR / f"{source_id}_{filename}"
        temp_path.replace(dest_path)
        for doc in documents:
            doc.metadata["source_id"] = source_id
            doc.metadata["title"] = filename
        chunk_count = add_documents("pdf", documents)
        record = DocumentRecord(
            id=source_id,
            title=filename,
            kind="pdf",
            pages=pages,
            chunks=chunk_count,
            status="ready",
            extra={"path": str(dest_path), "content_hash": content_hash},
        )
        return add_record("pdf", record)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _page_count(path: Path) -> int:
    with pymupdf.open(path) as pdf:
        return pdf.page_count


def extract_pdf_documents(path: Path, source_id: str, title: str) -> tuple[int, list[Document]]:
    """Page-aware chunking so every citation can name a real page number."""
    splitter = _splitter()
    documents: list[Document] = []
    with pymupdf.open(path) as pdf:
        page_count = pdf.page_count
        for index, page in enumerate(pdf):
            text = page.get_text("text") or ""
            text = " ".join(text.split())
            if len(text) < 40:
                continue
            for chunk_index, chunk in enumerate(splitter.split_text(text)):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source_type": "pdf",
                            "source_id": source_id,
                            "title": title,
                            "page": index + 1,
                            "chunk_index": chunk_index,
                        },
                    )
                )
    if not documents:
        raise ValueError("No extractable text found in this PDF.")
    return page_count, documents
