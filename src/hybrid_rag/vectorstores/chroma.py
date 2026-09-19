"""ChromaDB (embedded, persistent) vector store - the zero-infrastructure default."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hybrid_rag.schemas import Chunk, MetadataFilters
from hybrid_rag.vectorstores.base import VectorStore, scalar_metadata

_MAX_BATCH = 4000


def to_chroma_where(filters: MetadataFilters | None) -> dict[str, Any] | None:
    if not filters:
        return None
    clauses: list[dict[str, Any]] = [
        {k: {"$in": v} if isinstance(v, list) else {"$eq": v}} for k, v in filters.items()
    ]
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class ChromaVectorStore(VectorStore):
    backend = "chroma"

    def __init__(self, path: Path, collection: str) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection_name = collection
        self._collection = self._get_collection()

    def _get_collection(self) -> Any:
        return self._client.get_or_create_collection(
            self.collection_name,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
        )

    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        for start in range(0, len(chunks), _MAX_BATCH):
            batch = chunks[start : start + _MAX_BATCH]
            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings[start : start + _MAX_BATCH].astype(np.float32).tolist(),
                metadatas=[scalar_metadata(c) for c in batch],
            )

    def query(
        self, embedding: np.ndarray, k: int, filters: MetadataFilters | None = None
    ) -> list[tuple[str, float]]:
        n = self._collection.count()
        if n == 0:
            return []
        res = self._collection.query(
            query_embeddings=[embedding.astype(np.float32).tolist()],
            n_results=min(k, n),
            where=to_chroma_where(filters),
            include=["distances"],
        )
        ids = res["ids"][0]
        distances = (res.get("distances") or [[]])[0]
        # Chroma returns cosine *distance*; convert back to similarity.
        return [(cid, 1.0 - float(d)) for cid, d in zip(ids, distances, strict=True)]

    def delete_document(self, doc_id: str) -> None:
        self._collection.delete(where={"doc_id": {"$eq": doc_id}})

    def count(self) -> int:
        return int(self._collection.count())

    def reset(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._get_collection()

    def info(self) -> dict[str, Any]:
        return {**super().info(), "collection": self.collection_name}
