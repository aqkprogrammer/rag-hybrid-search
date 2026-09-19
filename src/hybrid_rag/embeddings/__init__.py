"""Embedding providers."""

from __future__ import annotations

from hybrid_rag.config import Settings
from hybrid_rag.embeddings.base import Embedder, l2_normalize
from hybrid_rag.embeddings.hashing import HashingEmbedder

__all__ = ["Embedder", "HashingEmbedder", "build_embedder", "l2_normalize"]


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "hashing":
        return HashingEmbedder(dim=settings.hashing_dim)
    if settings.embedding_provider == "sentence-transformers":
        from hybrid_rag.embeddings.sentence_transformers import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(
            settings.embedding_model, batch_size=settings.embedding_batch_size
        )
    if settings.embedding_provider == "openai":
        from hybrid_rag.embeddings.openai_embedder import OpenAIEmbedder

        assert settings.openai_api_key is not None  # guaranteed by Settings validation
        return OpenAIEmbedder(
            api_key=settings.openai_api_key.get_secret_value(),
            model_name=settings.openai_embedding_model,
            base_url=settings.openai_base_url,
        )
    raise ValueError(f"unknown embedding provider {settings.embedding_provider!r}")
