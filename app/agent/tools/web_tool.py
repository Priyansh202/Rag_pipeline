from langchain_core.tools import tool

from app.config import get_settings
from app.models.schemas import Citation
from app.retrieval.vectorstore import hybrid_search


def search_web(query: str, k: int | None = None) -> list[Citation]:
    k = k or get_settings().retrieval_k
    hits = hybrid_search("web", query, k=k)
    citations: list[Citation] = []
    for doc, score in hits:
        meta = doc.metadata
        url = meta.get("url")
        citations.append(
            Citation(
                source_type="web",
                title=meta.get("title", "Untitled page"),
                locator=url or "unknown url",
                snippet=doc.page_content,
                score=float(score),
                url=url,
            )
        )
    return citations


@tool("web_search")
def web_search(query: str) -> str:
    """Search indexed website content. Use for public pages that were added by URL.
    Returns passages with source URL and snippet."""
    citations = search_web(query)
    if not citations:
        return "No website content is indexed yet. Add a URL first."
    lines = []
    for item in citations:
        lines.append(f"[{item.title} | {item.locator}] {item.snippet}")
    return "\n\n".join(lines)
