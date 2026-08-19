from langchain_core.tools import tool

from app.models.schemas import Citation
from app.retrieval.vectorstore import hybrid_search


def search_pdfs(query: str, k: int = 5) -> list[Citation]:
    hits = hybrid_search("pdf", query, k=k)
    citations: list[Citation] = []
    for doc, score in hits:
        meta = doc.metadata
        page = meta.get("page")
        citations.append(
            Citation(
                source_type="pdf",
                title=meta.get("title", "Untitled PDF"),
                locator=f"p. {page}" if page else "unknown page",
                snippet=doc.page_content[:300],
                score=float(score),
                page=page,
            )
        )
    return citations


@tool("pdf_search")
def pdf_search(query: str) -> str:
    """Search indexed PDF documents. Use for policies, handbooks, and uploaded files.
    Returns passages with PDF title and page number."""
    citations = search_pdfs(query)
    if not citations:
        return "No PDF content is indexed yet. Upload a PDF first."
    lines = []
    for item in citations:
        lines.append(f"[{item.title} | {item.locator}] {item.snippet}")
    return "\n\n".join(lines)
