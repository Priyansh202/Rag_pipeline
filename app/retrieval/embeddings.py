from functools import lru_cache

from langchain_core.embeddings import Embeddings

from app.config import get_settings


class SentenceTransformerEmbeddings(Embeddings):
    """Thin wrapper so we do not depend on the sunset langchain-community package."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True, show_progress_bar=False).tolist()


@lru_cache
def get_embeddings() -> Embeddings:
    settings = get_settings()
    return SentenceTransformerEmbeddings(settings.embedding_model)
