"""Persist UI auth + chat locally so refresh does not wipe sources/history."""

from __future__ import annotations

import json
from pathlib import Path

SESSION_PATH = Path(__file__).resolve().parent.parent / "data" / "ui_session.json"


def load_session() -> dict:
    if not SESSION_PATH.exists():
        return {}
    try:
        return json.loads(SESSION_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_session(
    *,
    token: str | None,
    username: str | None,
    role: str | None,
    allowed: list[str] | None,
    messages: list[dict] | None,
) -> None:
    SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "token": token,
        "username": username,
        "role": role,
        "allowed": allowed or [],
        "messages": messages or [],
    }
    SESSION_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def clear_session() -> None:
    if SESSION_PATH.exists():
        SESSION_PATH.unlink(missing_ok=True)
