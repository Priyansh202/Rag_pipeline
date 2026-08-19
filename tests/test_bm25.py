from app.retrieval.bmb25 import bm25_scores


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

