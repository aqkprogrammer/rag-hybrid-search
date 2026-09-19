"""Qdrant vector store (server via docker-compose, or embedded local/in-memory mode)."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

from hybrid_rag.schemas import Chunk, MetadataFilters
from hybrid_rag.vectorstores.base import VectorStore, scalar_metadata

_NAMESPACE = uuid.UUID("7b0e8a52-3c1d-4f4e-9d2a-6f1c2b9e0a11")
_BATCH = 256


def to_qdrant_filter(filters: MetadataFilters | None) -> models.Filter | None:
    if not filters:
        return None
    must: list[models.Condition] = []
    for key, value in filters.items():
        match: models.MatchAny | models.MatchValue = (
            models.MatchAny(any=value)
            if isinstance(value, list)
            else models.MatchValue(value=value)  # type: ignore[arg-type]
        )
        must.append(models.FieldCondition(key=key, match=match))
    return models.Filter(must=must)


class QdrantVectorStore(VectorStore):
    backend = "qdrant"

    def __init__(
        self,
        collection: str,
        dim: int,
        url: str | None = None,
        api_key: str | None = None,
        path: Path | None = None,
    ) -> None:
        if url:
            self._client = QdrantClient(url=url, api_key=api_key, timeout=30)
            self.location = url
        elif path is not None:
            path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(path))
            self.location = str(path)
        else:
            self._client = QdrantClient(location=":memory:")
            self.location = ":memory:"
        self._remote = bool(url)
        self.collection_name = collection
        self.dim = dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self._client.collection_exists(self.collection_name):
            self._client.create_collection(
                self.collection_name,
                vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
            )
            if self._remote:  # payload indexes only exist on a Qdrant server
                for field in ("doc_id", "source", "doc_type", "department"):
                    self._client.create_payload_index(
                        self.collection_name, field, models.PayloadSchemaType.KEYWORD
                    )

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(_NAMESPACE, chunk_id))

    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        for start in range(0, len(chunks), _BATCH):
            batch = chunks[start : start + _BATCH]
            vectors = embeddings[start : start + _BATCH].astype(np.float32)
            self._client.upsert(
                self.collection_name,
                points=[
                    models.PointStruct(
                        id=self._point_id(c.chunk_id),
                        vector=v.tolist(),
                        payload={**scalar_metadata(c), "chunk_id": c.chunk_id},
                    )
                    for c, v in zip(batch, vectors, strict=True)
                ],
                wait=True,
            )

    def query(
        self, embedding: np.ndarray, k: int, filters: MetadataFilters | None = None
    ) -> list[tuple[str, float]]:
        res = self._client.query_points(
            self.collection_name,
            query=embedding.astype(np.float32).tolist(),
            limit=k,
            query_filter=to_qdrant_filter(filters),
            with_payload=["chunk_id"],
        )
        return [
            (str(p.payload["chunk_id"]), float(p.score))
            for p in res.points
            if p.payload is not None
        ]

    def delete_document(self, doc_id: str) -> None:
        self._client.delete(
            self.collection_name,
            points_selector=models.FilterSelector(filter=to_qdrant_filter({"doc_id": doc_id})),  # type: ignore[arg-type]
            wait=True,
        )

    def count(self) -> int:
        return int(self._client.count(self.collection_name, exact=True).count)

    def reset(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._ensure_collection()

    def info(self) -> dict[str, Any]:
        return {**super().info(), "collection": self.collection_name, "location": self.location}

    def close(self) -> None:
        self._client.close()
