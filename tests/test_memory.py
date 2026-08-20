from app.agent.memory import normalize_history, search_query_from_history
from app.models.schemas import ChatTurn


def test_search_query_includes_prior_user_turns():
    history = [
        {"role": "user", "content": "What is Alta Merita?"},
        {"role": "assistant", "content": "A 260-unit apartment community."},
    ]
    query = search_query_from_history("How many units?", history)
    assert "Alta Merita" in query
    assert "How many units?" in query


def test_normalize_history_keeps_last_turns_and_strips_meta():
    turns = [
        ChatTurn(role="user", content="first"),
        ChatTurn(role="assistant", content="ok\n\nTools used: `pdf_search`"),
        ChatTurn(role="user", content="second"),
    ]
    cleaned = normalize_history(turns, max_turns=4)
    assert cleaned[-1]["content"] == "second"
    assert cleaned[1]["content"] == "ok"
