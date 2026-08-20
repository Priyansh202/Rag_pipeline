from app.config import get_settings
from app.db.pgvector_store import init_vector_schema, vector_pool
from app.db.postgres import init_app_schema
from app.db.wait import wait_for_databases
from app.retrieval.bootstrap import backfill_chunk_texts, migrate_local_registry, reindex_missing_vectors
from app.retrieval.registry import document_count


def main() -> None:
    settings = get_settings()
    print("APP", settings.app_dsn)
    print("VEC", settings.vector_dsn)
    wait_for_databases()
    init_app_schema()
    init_vector_schema()
    print("migrated", migrate_local_registry())
    print("reindexed", reindex_missing_vectors())
    print("chunk_texts", backfill_chunk_texts())
    print("docs", document_count())
    with vector_pool().connection() as conn:
        print("chunks", conn.execute("SELECT source_type, COUNT(*) AS n FROM chunks GROUP BY source_type").fetchall())
        print("ext", conn.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'").fetchone())


if __name__ == "__main__":
    main()
