from langchain_core.documents import Document

from app.retrieval.dedup import attach_chunk_hashes, chunk_hash_set, overlap_score


def test_overlap_score_favors_updates_not_new_docs():
    old = {"aaa", "bbb", "ccc"}
    minor_edit = {"aaa", "bbb", "ccc", "ddd"}
    unrelated = {"xxx", "yyy", "zzz"}
    assert overlap_score(old, minor_edit) > 0.7
    assert overlap_score(old, unrelated) == 0.0


def test_chunk_hashes_are_stable_for_same_text():
    docs = attach_chunk_hashes(
        [
            Document(page_content="Remote work is three days per week.", metadata={}),
            Document(page_content="Parental leave is 16 weeks.", metadata={}),
        ]
    )
    first = chunk_hash_set(docs)
    again = chunk_hash_set(
        attach_chunk_hashes(
            [
                Document(page_content="Remote work is three days per week.", metadata={}),
                Document(page_content="Parental leave is 16 weeks.", metadata={}),
            ]
        )
    )
    assert first == again
