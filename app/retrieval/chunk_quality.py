"""Downrank table-of-contents and heading-only PDF chunks."""

from __future__ import annotations

_TOC_MARKERS = (
    "the offering",
    "the property",
    "competitive positioning",
    "the financials",
)


def is_low_content_chunk(text: str) -> bool:
    """True for index/TOC pages that mention the title but have almost no facts."""
    t = " ".join((text or "").split())
    if len(t) < 160:
        return True
    lowered = t.lower()
    marker_hits = sum(1 for marker in _TOC_MARKERS if marker in lowered)
    sentence_endings = t.count(".") + t.count("?") + t.count("!")
    if marker_hits >= 2 and sentence_endings < 3:
        return True
    letters = [c for c in t if c.isalpha()]
    if letters:
        upper_ratio = sum(c.isupper() for c in letters) / len(letters)
        if upper_ratio > 0.55 and len(t) < 500:
            return True
    return False


def quality_bonus(text: str) -> float:
    """Small score bump for longer narrative chunks; penalty for TOC-like text."""
    if is_low_content_chunk(text):
        return -0.5
    n = len((text or "").strip())
    return min(n / 1000.0, 1.0) * 0.12
