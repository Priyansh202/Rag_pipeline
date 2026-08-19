from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth.dependencies import CurrentUser, require_web_access
from app.models.schemas import DocumentRecord, MessageResponse, WebsiteCreate
from app.retrieval.registry import get_record, list_sources, remove_record
from app.retrieval.vectorstore import delete_by_source_id
from app.retrieval.web_scraper import ingest_website

router = APIRouter(prefix="/websites", tags=["websites"])


@router.get("", response_model=list[DocumentRecord])
def list_sites(user: CurrentUser = Depends(require_web_access)) -> list[DocumentRecord]:
    _ = user
    return list_sources().websites


@router.post("", response_model=DocumentRecord, status_code=status.HTTP_201_CREATED)
def add_site(
    body: WebsiteCreate,
    user: CurrentUser = Depends(require_web_access),
) -> DocumentRecord:
    _ = user
    try:
        return ingest_website(str(body.url))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.delete("/{doc_id}", response_model=MessageResponse)
def delete_site(doc_id: str, user: CurrentUser = Depends(require_web_access)) -> MessageResponse:
    _ = user
    record = get_record("web", doc_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Website not found.")
    delete_by_source_id("web", doc_id)
    cache_path = (record.extra or {}).get("cache_path")
    if cache_path:
        Path(cache_path).unlink(missing_ok=True)
    remove_record("web", doc_id)
    return MessageResponse(message="Website removed from cache and the index.", id=doc_id)
