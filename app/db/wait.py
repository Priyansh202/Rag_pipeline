from __future__ import annotations

import time

import psycopg

from app.config import get_settings


def _ready(dsn: str) -> bool:
    try:
        with psycopg.connect(dsn, connect_timeout=2) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


def _admin_dsn() -> str:
    settings = get_settings()
    return (
        f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/postgres"
    )


def ensure_vector_database() -> None:
    settings = get_settings()
    if settings.uses_shared_database:
        return
    if _ready(settings.vector_dsn):
        return
    with psycopg.connect(_admin_dsn(), autocommit=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (settings.pgvector_db,),
        ).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{settings.pgvector_db}"')


def wait_for_databases(timeout_s: float = 90.0) -> None:
    settings = get_settings()
    deadline = time.time() + timeout_s
    last_error = "databases not reachable"
    while time.time() < deadline:
        app_ok = _ready(settings.app_dsn)
        if app_ok:
            try:
                ensure_vector_database()
            except Exception as exc:  # noqa: BLE001
                last_error = f"could not create vector database: {exc}"
                time.sleep(1.5)
                continue
        vec_ok = _ready(settings.vector_dsn)
        if app_ok and vec_ok:
            return
        missing = []
        if not app_ok:
            missing.append("postgres")
        if not vec_ok:
            missing.append("pgvector")
        last_error = "waiting for " + ", ".join(missing)
        time.sleep(1.5)
    hint = (
        "Set DATABASE_URL to the Railway pgvector service, or start local databases "
        "with `docker compose up -d`."
        if settings.uses_shared_database
        else "Start them with `docker compose up -d` from the project root."
    )
    raise RuntimeError(f"{last_error}. {hint}")
