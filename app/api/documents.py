from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.auth.dependencies import CurrentUser, get_current_user, require_pdf_access
from app.config import get_settings
from app.models.schemas import DocumentRecord, MessageResponse
from app.retrieval.pdf_processor import ingest_pdf
from app.retrieval.registry import get_record, list_sources, remove_record
from app.retrieval.vectorstore import delete_by_source_id

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=list[DocumentRecord])
def list_pdfs(user: CurrentUser = Depends(require_pdf_access)) -> list[DocumentRecord]:
    _ = user
    return list_sources().pdfs


@router.post("", response_model=DocumentRecord, status_code=status.HTTP_201_CREATED)
async def upload_pdf(
    file: UploadFile = File(...),
    user: CurrentUser = Depends(require_pdf_access),
) -> DocumentRecord:
    _ = user
    settings = get_settings()
    filename = file.filename or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    payload = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(payload) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.max_upload_mb} MB upload limit.",
        )
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        return ingest_pdf(payload, filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/{doc_id}", response_model=MessageResponse)
def delete_pdf(doc_id: str, user: CurrentUser = Depends(require_pdf_access)) -> MessageResponse:
    _ = user
    record = get_record("pdf", doc_id)
    if record is None:
        raise HTTPException(status_code=404, detail="PDF not found.")
    delete_by_source_id("pdf", doc_id)
    path = (record.extra or {}).get("path")
    if path:
        Path(path).unlink(missing_ok=True)
    remove_record("pdf", doc_id)
    return MessageResponse(message="PDF removed from disk and the index.", id=doc_id)


@router.get("/all/visible")
def visible_sources(user: CurrentUser = Depends(get_current_user)) -> dict:
    """Return only the source types the caller's role may see."""
    catalog = list_sources()
    from app.auth.rbac import can_access_pdfs, can_access_web

    return {
        "pdfs": catalog.pdfs if can_access_pdfs(user.role) else [],
        "websites": catalog.websites if can_access_web(user.role) else [],
        "role": user.role,
    }
