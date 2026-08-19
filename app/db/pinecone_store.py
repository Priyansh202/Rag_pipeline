"""Pinecone serverless index for chunk embeddings."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.config import get_settings


def _chunk_metadata(
    *,
    kind: str,
    source_id: str,
    title: str,
    page: int | None,
    url: str | None,
    chunk_index: int,
    content: str,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "source_type": kind,
        "source_id": source_id,
        "title": title,
        "chunk_index": chunk_index,
        "content": content[:8000],
    }
    if page is not None:
        meta["page"] = page
    if url:
        meta["url"] = url
    return meta


@lru_cache
def pinecone_client():
    from pinecone import Pinecone

    settings = get_settings()
    return Pinecone(api_key=settings.pinecone_api_key)


@lru_cache
def pinecone_index():
    return pinecone_client().Index(get_settings().pinecone_index_name)


def init_pinecone_index() -> None:
    from pinecone import ServerlessSpec

    settings = get_settings()
    client = pinecone_client()
    names = set(client.list_indexes().names())
    if settings.pinecone_index_name not in names:
        client.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )


def upsert_vectors(vectors: list[dict[str, Any]]) -> None:
    if not vectors:
        return
    index = pinecone_index()
    batch_size = 100
    for start in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[start : start + batch_size])


def fetch_vectors(ids: list[str]) -> dict[str, list[float]]:
    if not ids:
        return {}
    index = pinecone_index()
    fetched = index.fetch(ids=ids)
    out: dict[str, list[float]] = {}
    for chunk_id, record in (fetched.vectors or {}).items():
        out[chunk_id] = list(record.values)
    return out


def list_chunk_ids(source_id: str) -> list[str]:
    index = pinecone_index()
    prefix = f"{source_id}:"
    ids: list[str] = []
    for batch in index.list(prefix=prefix, limit=100):
        ids.extend(batch)
    return ids


def delete_ids(ids: list[str]) -> int:
    if not ids:
        return 0
    index = pinecone_index()
    batch_size = 100
    for start in range(0, len(ids), batch_size):
        index.delete(ids=ids[start : start + batch_size])
    return len(ids)


def delete_by_source(kind: str, source_id: str) -> int:
    index = pinecone_index()
    ids = list_chunk_ids(source_id)
    if ids:
        delete_ids(ids)
        return len(ids)
    try:
        index.delete(filter={"source_type": {"$eq": kind}, "source_id": {"$eq": source_id}})
    except Exception:
        pass
    return 0


def query_by_source_type(kind: str, query_vector: list[float], top_k: int) -> list[dict[str, Any]]:
    index = pinecone_index()
    response = index.query(
        vector=query_vector,
        top_k=top_k,
        filter={"source_type": {"$eq": kind}},
        include_metadata=True,
    )
    rows: list[dict[str, Any]] = []
    for match in response.matches or []:
        meta = match.metadata or {}
        score = float(match.score or 0.0)
        rows.append(
            {
                "content": meta.get("content") or "",
                "title": meta.get("title") or "Untitled",
                "page": meta.get("page"),
                "url": meta.get("url"),
                "source_id": meta.get("source_id") or "",
                "chunk_index": int(meta.get("chunk_index") or 0),
                "distance": 1.0 - score,
            }
        )
    return rows


def count_by_source_type(kind: str) -> int:
    stats = pinecone_index().describe_index_stats()
    namespaces = stats.namespaces or {}
    default = namespaces.get("") or namespaces.get("__default__")
    if default is None:
        return 0
    # Pinecone stats are not split by source_type; use list only when needed.
    _ = kind
    return int(getattr(default, "vector_count", 0) or 0)


def reset_pinecone_cache() -> None:
    pinecone_client.cache_clear()
    pinecone_index.cache_clear()
