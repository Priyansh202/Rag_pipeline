from __future__ import annotations

import time
from typing import Any

from functools import lru_cache

import httpx
from langchain_core.embeddings import Embeddings

from app.config import get_settings

VOYAGE_EMBED_URL = "https://api.voyageai.com/v1/embeddings"


class VoyageEmbeddings(Embeddings):
    """Voyage API embeddings with batching and 429 backoff for free-tier limits."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        batch_size: int,
        batch_delay_s: float,
        max_retries: int,
        timeout_s: float,
    ):
        self.api_key = api_key
        self.model = model
        self.batch_size = max(1, batch_size)
        self.batch_delay_s = max(0.0, batch_delay_s)
        self.max_retries = max(1, max_retries)
        self.timeout_s = timeout_s

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, input_type="document")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], input_type="query")[0]

    def _embed(self, texts: list[str], *, input_type: str) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            out.extend(self._embed_batch(batch, input_type=input_type))
            if start + self.batch_size < len(texts) and self.batch_delay_s > 0:
                time.sleep(self.batch_delay_s)
        return out

    def _embed_batch(self, texts: list[str], *, input_type: str) -> list[list[float]]:
        payload: dict[str, Any] = {
            "input": texts,
            "model": self.model,
            "input_type": input_type,
            "truncation": True,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error = "Voyage embedding request failed"
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    response = client.post(VOYAGE_EMBED_URL, headers=headers, json=payload)
            except httpx.HTTPError as exc:
                last_error = str(exc)
                self._sleep_backoff(attempt, retry_after=None)
                continue

            if response.status_code == 200:
                data = response.json()
                embeddings = data.get("data") or []
                if len(embeddings) != len(texts):
                    raise RuntimeError(
                        f"Voyage returned {len(embeddings)} embeddings for {len(texts)} inputs"
                    )
                return [list(item["embedding"]) for item in embeddings]

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                last_error = response.text or "rate limited (429)"
                self._sleep_backoff(attempt, retry_after=retry_after)
                continue

            if response.status_code >= 500:
                last_error = response.text or f"server error {response.status_code}"
                self._sleep_backoff(attempt, retry_after=response.headers.get("Retry-After"))
                continue

            raise RuntimeError(
                f"Voyage embedding failed ({response.status_code}): {response.text[:500]}"
            )

        raise RuntimeError(
            f"Voyage embedding failed after {self.max_retries} attempts: {last_error}"
        )

    def _sleep_backoff(self, attempt: int, *, retry_after: str | None) -> None:
        if retry_after:
            try:
                time.sleep(max(float(retry_after), 0.5))
                return
            except ValueError:
                pass
        # 1s, 2s, 4s, 8s, ... capped at 30s
        time.sleep(min(30.0, 1.0 * (2**attempt)))


class SentenceTransformerEmbeddings(Embeddings):
    """Local CPU embeddings (high memory — not for Railway free tier)."""

    def __init__(self, model_name: str, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return (
            self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=self.batch_size,
            ).tolist()
        )

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True, show_progress_bar=False).tolist()


@lru_cache
def get_embeddings() -> Embeddings:
    settings = get_settings()
    if settings.use_voyage_embeddings:
        return VoyageEmbeddings(
            api_key=settings.voyage_api_key,
            model=settings.voyage_model,
            batch_size=settings.voyage_embed_batch_size,
            batch_delay_s=settings.voyage_batch_delay_s,
            max_retries=settings.voyage_max_retries,
            timeout_s=settings.voyage_timeout_s,
        )
    return SentenceTransformerEmbeddings(settings.embedding_model, settings.embedding_batch_size)
