from contextlib import asynccontextmanager

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agent.llm import resolve_llm
from app.api import auth, documents, query, websites
from app.auth.rbac import ROLE_TOOLS
from app.config import ensure_data_dirs, get_settings
from app.db.postgres import init_app_schema
from app.db.wait import wait_for_databases
from app.retrieval.bootstrap import backfill_chunk_texts, migrate_local_registry, reindex_missing_vectors


def _init_vector_store() -> None:
    settings = get_settings()
    if settings.uses_pinecone:
        from app.db.pinecone_store import init_pinecone_index

        init_pinecone_index()
        return
    from app.db.pgvector_store import init_vector_schema

    init_vector_schema()


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_data_dirs()
    wait_for_databases()
    init_app_schema()
    _init_vector_store()
    try:
        migrate_local_registry()
        if os.environ.get("SKIP_STARTUP_REINDEX", "0") != "1":
            reindex_missing_vectors()
        backfill_chunk_texts()
    except Exception as exc:
        print(f"Startup index rebuild skipped: {exc}")
    yield


app = FastAPI(
    title="Secure Multi-Source QA Agent",
    description="JWT + RBAC gated PDF/web retrieval agent.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(websites.router)
app.include_router(query.router)


@app.get("/health")
def health() -> dict:
    from app.config import get_settings

    settings = get_settings()
    llm = resolve_llm()
    return {
        "status": "ok",
        "llm_provider": llm.provider,
        "llm_model": llm.model,
        "vector_backend": "pinecone" if settings.uses_pinecone else "pgvector",
        "embedding_provider": "voyage" if settings.use_voyage_embeddings else "local",
        "embedding_model": settings.voyage_model if settings.use_voyage_embeddings else settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
        "catalog_database": settings.uses_shared_database,
        "roles": {role.value: sorted(tools) for role, tools in ROLE_TOOLS.items()},
    }
