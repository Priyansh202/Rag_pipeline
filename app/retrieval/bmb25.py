"""Simple BM25 scorer for hybrid search re-ranking.

This is intentionally lightweight (no external search engine). We compute IDF
over the candidate set retrieved by pgvector, then score each candidate.
"""

from __future__ import annotations

import math
import re
from collections import Counter


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "as",
    "at",
    "by",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "it",
    "this",
    "that",
    "these",
    "those",
    "your",
    "you",
    "i",
    "we",
    "they",
}


def tokenize(text: str) -> list[str]:
    tokens = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    return [t for t in tokens if len(t) > 2 and t not in _STOPWORDS]


def bm25_scores(
    query: str,
    documents: list[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """Return BM25 score for each document (tokenized on demand)."""
    query_tokens = tokenize(query)
    if not query_tokens or not documents:
        return [0.0 for _ in documents]

    doc_tokens = [tokenize(doc) for doc in documents]
    doc_lens = [len(toks) for toks in doc_tokens]
    avgdl = sum(doc_lens) / max(len(doc_lens), 1)
    if avgdl <= 0:
        avgdl = 1.0

    # document frequency over the candidate set
    N = len(doc_tokens)
    dfs = Counter()
    for toks in doc_tokens:
        for term in set(toks):
            dfs[term] += 1

    scores: list[float] = []
    for toks, dlen in zip(doc_tokens, doc_lens, strict=True):
        tf = Counter(toks)
        score = 0.0
        for term in query_tokens:
            if term not in tf:
                continue
            df = dfs.get(term, 0)
            # BM25+ style IDF with smoothing
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            numer = tf[term] * (k1 + 1.0)
            denom = tf[term] + k1 * (1.0 - b + b * (dlen / avgdl))
            score += idf * (numer / max(denom, 1e-12))
        scores.append(score)
    return scores

