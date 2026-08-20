"""Short-term chat memory for follow-up questions."""

from __future__ import annotations

from app.models.schemas import ChatTurn


def normalize_history(
    history: list[ChatTurn] | None, *, max_turns: int = 8, max_chars: int = 1500
) -> list[dict[str, str]]:
    if not history:
        return []
    cleaned: list[dict[str, str]] = []
    for turn in history[-max_turns:]:
        text = (turn.content or "").strip()
        if turn.role == "assistant":
            text = text.split("\n\nTools used:")[0].split("Tools used:")[0].strip()
        text = " ".join(text.split())
        if not text:
            continue
        cleaned.append({"role": turn.role, "content": text[:max_chars]})
    return cleaned


def search_query_from_history(question: str, history: list[dict[str, str]]) -> str:
    """Blend recent user questions into the retrieval query so follow-ups still match sources."""
    prior = [item["content"] for item in history if item["role"] == "user"][-3:]
    if not prior:
        return question
    return " ".join([*prior, question])[:2000]
