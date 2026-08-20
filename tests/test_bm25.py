from app.db.chunk_catalog import top_bm25_for_kind
from app.retrieval.bmb25 import bm25_scores, bm25_top_indices


def test_bm25_prefers_keyword_overlap():
    query = "remote work policy"
    docs = [
        "This chunk talks about office hours and benefits.",
        "Remote work policy allows employees to work remotely up to three days per week.",
        "Completely unrelated content about travel expenses.",
    ]
    scores = bm25_scores(query, docs)
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]


def test_bm25_top_indices_returns_highest_scores():
    scores = [0.1, 2.5, 0.0, 1.2]
    assert bm25_top_indices(scores, 2) == [1, 3]


def test_full_corpus_bm25_finds_keyword_match_outside_vector_top_k():
    query = "Alta Merita multifamily units"
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
        {
            "id": "doc:other",
            "source_id": "doc",
            "source_type": "pdf",
            "title": "OM",
            "page": 40,
            "url": None,
            "chunk_index": 2,
            "content": "Financial assumptions and market overview for the submarket.",
        },
    ]
    hits = top_bm25_for_kind("pdf", query, 2, k1=1.5, b=0.75, corpus=corpus)
    assert hits
    assert hits[0]["id"] == "doc:body"
    assert hits[0]["bm25_score"] > 0
