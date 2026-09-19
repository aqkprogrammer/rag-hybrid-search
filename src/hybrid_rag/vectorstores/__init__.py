"""Vector store backends."""

from __future__ import annotations

from hybrid_rag.config import Settings
from hybrid_rag.vectorstores.base import VectorStore
from hybrid_rag.vectorstores.memory import InMemoryVectorStore

__all__ = ["InMemoryVectorStore", "VectorStore", "build_vector_store"]


def build_vector_store(settings: Settings, dim: int, embedder_slug: str) -> VectorStore:
    # One collection per embedding model: switching models never mixes incompatible vectors.
    collection = f"{settings.collection_prefix}-{embedder_slug}"[:63].strip("-_.")
    if settings.vector_store == "memory":
        return InMemoryVectorStore(dim=dim)
    if settings.vector_store == "chroma":
        from hybrid_rag.vectorstores.chroma import ChromaVectorStore

        return ChromaVectorStore(settings.chroma_path, collection)
    if settings.vector_store == "qdrant":
        from hybrid_rag.vectorstores.qdrant import QdrantVectorStore

        return QdrantVectorStore(
            collection=collection,
            dim=dim,
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
            path=None if settings.qdrant_url else settings.qdrant_path,
        )
    raise ValueError(f"unknown vector store {settings.vector_store!r}")
