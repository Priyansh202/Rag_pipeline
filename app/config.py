from functools import lru_cache
import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def normalize_postgres_dsn(url: str) -> str:
    """Railway (and Heroku) often emit postgres://; psycopg wants postgresql://."""
    url = (url or "").strip()
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://") :]
    return url


def resolve_data_dir() -> Path:
    raw = os.environ.get("DATA_DIR") or os.environ.get("RAILWAY_VOLUME_MOUNT_PATH")
    return Path(raw) if raw else PROJECT_ROOT / "data"


DATA_DIR = resolve_data_dir()
PDF_DIR = DATA_DIR / "pdfs"
WEB_CACHE_DIR = DATA_DIR / "web_cache"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
REGISTRY_PATH = DATA_DIR / "registry.json"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    jwt_secret: str = "change-me-in-any-non-demo-environment-32b"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 120

    api_host: str = "127.0.0.1"
    api_port: int = 8000

    llm_provider: str = "auto"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    ollama_base_url: str = "http://127.0.0.1:11434/v1"
    ollama_model: str = "llama3.2"

    openrouter_api_key: str = ""
    openrouter_model: str = "openai/gpt-4o-mini"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_site_url: str = "http://127.0.0.1:8501"
    openrouter_app_name: str = "Secure Multi-Source QA Agent"

    scraperapi_key: str = ""

    # Railway / hosted Postgres. When set, catalog + vectors share one database.
    database_url: str = ""
    database_url_private: str = ""

    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5433
    postgres_user: str = "rag"
    postgres_password: str = "rag"
    postgres_db: str = "rag_app"

    pgvector_host: str = "127.0.0.1"
    pgvector_port: int = 5433
    pgvector_user: str = "rag"
    pgvector_password: str = "rag"
    pgvector_db: str = "rag_vectors"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 5
    retrieval_candidate_k: int = 20

    # Hybrid search: vector + BM25 keyword scoring over the vector candidate set.
    hybrid_enabled: bool = True
    hybrid_alpha: float = 0.6  # weight on vector similarity in the final score
    hybrid_bm25_k1: float = 1.5
    hybrid_bm25_b: float = 0.75
    pdf_update_overlap_ratio: float = 0.55

    max_query_chars: int = 2000
    max_upload_mb: int = 80

    @property
    def shared_dsn(self) -> str:
        raw = self.database_url_private or self.database_url
        return normalize_postgres_dsn(raw)

    @property
    def uses_shared_database(self) -> bool:
        return bool(self.shared_dsn)

    @property
    def app_dsn(self) -> str:
        if self.shared_dsn:
            return self.shared_dsn
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def vector_dsn(self) -> str:
        if self.shared_dsn:
            return self.shared_dsn
        return (
            f"postgresql://{self.pgvector_user}:{self.pgvector_password}"
            f"@{self.pgvector_host}:{self.pgvector_port}/{self.pgvector_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


def ensure_data_dirs() -> None:
    for path in (PDF_DIR, WEB_CACHE_DIR, VECTORSTORE_DIR / "pdf", VECTORSTORE_DIR / "web"):
        path.mkdir(parents=True, exist_ok=True)
    if not REGISTRY_PATH.exists():
        REGISTRY_PATH.write_text('{"pdfs": [], "websites": []}', encoding="utf-8")
