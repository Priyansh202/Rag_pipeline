from langchain_core.documents import Document

from app.retrieval import vectorstore


def test_hybrid_search_merges_vector_and_full_corpus_bm25(monkeypatch):
    corpus = [
        {
            "id": "doc:toc",
            "source_id": "doc",
            "source_type": "pdf",
            "title": "OM",
            "page": 2,
            "url": None,
            "chunk_index": 0,
            "content": "Table of Contents Executive Summary Investment Highlights",
        },
        {
            "id": "doc:body",
            "source_id": "doc",
            "source_type": "pdf",
            "title": "OM",
            "page": 5,
            "url": None,
            "chunk_index": 1,
            "content": "Alta Merita is a 120-unit multifamily property located in Austin.",
        },
    ]
    vector_rows = [
        {
            "id": "doc:toc",
            "source_id": "doc",
            "title": "OM",
            "page": 2,
            "url": None,
            "chunk_index": 0,
            "content": corpus[0]["content"],
            "distance": 0.05,
        }
    ]

    monkeypatch.setattr(vectorstore, "_corpus_for_kind", lambda kind: corpus)
    monkeypatch.setattr(vectorstore, "_vector_candidates", lambda kind, query, k: vector_rows)

    hits = vectorstore.hybrid_search("pdf", "Alta Merita multifamily units", k=2)
    assert len(hits) == 2
    pages = {hit[0].metadata["page"] for hit in hits}
    assert 5 in pages
    assert all(isinstance(hit[0], Document) for hit in hits)
