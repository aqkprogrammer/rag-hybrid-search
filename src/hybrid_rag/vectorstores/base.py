"""Vector store interface. Implementations: Chroma (embedded default), Qdrant, in-memory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from hybrid_rag.schemas import Chunk, MetadataFilters


def scalar_metadata(chunk: Chunk) -> dict[str, str | int | float | bool]:
    """Vector stores only need (and some only accept) flat scalar metadata for filtering."""
    return {
        k: v
        for k, v in chunk.metadata.items()
        if isinstance(v, str | int | float | bool) and not isinstance(v, bytes)
    }


class VectorStore(ABC):
    backend: str

    @abstractmethod
    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray) -> None: ...

    @abstractmethod
    def query(
        self, embedding: np.ndarray, k: int, filters: MetadataFilters | None = None
    ) -> list[tuple[str, float]]:
        """Return up to `k` ``(chunk_id, cosine_similarity)`` pairs, best first."""

    @abstractmethod
    def delete_document(self, doc_id: str) -> None: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def reset(self) -> None:
        """Drop every vector (used by `reindex`)."""

    def info(self) -> dict[str, Any]:
        return {"backend": self.backend, "count": self.count()}

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources (optional)."""
