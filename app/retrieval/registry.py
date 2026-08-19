"""Document catalog stored in Postgres (the app database)."""

from __future__ import annotations

import json

from app.db.postgres import app_pool
from app.models.schemas import DocumentRecord, RegistryResponse


def _row_to_record(row: dict) -> DocumentRecord:
    extra = row.get("extra") or {}
    if isinstance(extra, str):
        extra = json.loads(extra)
    return DocumentRecord(
        id=row["id"],
        title=row["title"],
        kind=row["kind"],
        pages=row.get("pages"),
        url=row.get("url"),
        chunks=row.get("chunks") or 0,
        status=row.get("status") or "ready",
        extra=extra,
    )


def list_sources() -> RegistryResponse:
    with app_pool().connection() as conn:
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY created_at ASC"
        ).fetchall()
    pdfs = [_row_to_record(row) for row in rows if row["kind"] == "pdf"]
    websites = [_row_to_record(row) for row in rows if row["kind"] == "web"]
    return RegistryResponse(pdfs=pdfs, websites=websites)


def add_record(kind: str, record: DocumentRecord) -> DocumentRecord:
    payload = record.model_dump()
    payload["kind"] = kind if kind in {"pdf", "web"} else record.kind
    with app_pool().connection() as conn:
        conn.execute(
            """
            INSERT INTO documents (id, kind, title, pages, url, chunks, status, extra)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (id) DO UPDATE SET
                title = EXCLUDED.title,
                pages = EXCLUDED.pages,
                url = EXCLUDED.url,
                chunks = EXCLUDED.chunks,
                status = EXCLUDED.status,
                extra = EXCLUDED.extra
            """,
            (
                payload["id"],
                payload["kind"],
                payload["title"],
                payload["pages"],
                payload["url"],
                payload["chunks"],
                payload["status"],
                json.dumps(payload.get("extra") or {}),
            ),
        )
    return record


def get_record(kind: str, doc_id: str) -> DocumentRecord | None:
    with app_pool().connection() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE id = %s AND kind = %s",
            (doc_id, "pdf" if kind == "pdf" else "web"),
        ).fetchone()
    return _row_to_record(row) if row else None


def _extra_hash_query(kind: str, json_field: str, value: str) -> DocumentRecord | None:
    with app_pool().connection() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM documents
            WHERE kind = %s
              AND extra->>%s = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            ("pdf" if kind == "pdf" else "web", json_field, value),
        ).fetchone()
    return _row_to_record(row) if row else None


def find_pdf_by_content_hash(content_hash: str) -> DocumentRecord | None:
    return _extra_hash_query("pdf", "content_hash", content_hash)


def find_web_by_url_hash(url_hash: str) -> DocumentRecord | None:
    return _extra_hash_query("web", "url_hash", url_hash)


def remove_record(kind: str, doc_id: str) -> DocumentRecord | None:
    record = get_record(kind, doc_id)
    if record is None:
        return None
    with app_pool().connection() as conn:
        conn.execute(
            "DELETE FROM documents WHERE id = %s AND kind = %s",
            (doc_id, "pdf" if kind == "pdf" else "web"),
        )
    return record


def document_count() -> int:
    with app_pool().connection() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()
    return int(row["n"] if row else 0)
