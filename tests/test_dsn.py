from app.config import Settings, normalize_postgres_dsn


def test_normalize_postgres_scheme():
    assert (
        normalize_postgres_dsn("postgres://u:p@h:5432/db")
        == "postgresql://u:p@h:5432/db"
    )
    assert (
        normalize_postgres_dsn("postgresql://u:p@h:5432/db")
        == "postgresql://u:p@h:5432/db"
    )


def test_database_url_shares_app_and_vector_dsn(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://u:p@railway-host:5432/railway")
    settings = Settings()
    assert settings.uses_shared_database
    assert settings.app_dsn == "postgresql://u:p@railway-host:5432/railway"
    assert settings.vector_dsn == settings.app_dsn
