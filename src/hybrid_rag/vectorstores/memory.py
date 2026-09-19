"""Brute-force in-memory vector store (tests, tiny corpora, zero-dependency demos)."""

from __future__ import annotations

import threading

import numpy as np

from hybrid_rag.schemas import Chunk, MetadataFilters
from hybrid_rag.storage.filters import matches_filters
from hybrid_rag.vectorstores.base import VectorStore, scalar_metadata


class InMemoryVectorStore(VectorStore):
    backend = "memory"

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._lock = threading.Lock()
        self._ids: list[str] = []
        self._metas: list[dict[str, str | int | float | bool]] = []
        self._matrix = np.zeros((0, dim), dtype=np.float32)

    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        with self._lock:
            incoming = {c.chunk_id for c in chunks}
            keep = [i for i, cid in enumerate(self._ids) if cid not in incoming]
            self._ids = [self._ids[i] for i in keep] + [c.chunk_id for c in chunks]
            self._metas = [self._metas[i] for i in keep] + [scalar_metadata(c) for c in chunks]
            self._matrix = np.vstack([self._matrix[keep], embeddings.astype(np.float32)])

    def query(
        self, embedding: np.ndarray, k: int, filters: MetadataFilters | None = None
    ) -> list[tuple[str, float]]:
        with self._lock:
            if not self._ids:
                return []
            sims = self._matrix @ embedding.astype(np.float32)
            order = np.argsort(-sims, kind="stable")
            out: list[tuple[str, float]] = []
            for i in order:
                if matches_filters(self._metas[i], filters):
                    out.append((self._ids[i], float(sims[i])))
                    if len(out) >= k:
                        break
            return out

    def delete_document(self, doc_id: str) -> None:
        with self._lock:
            keep = [i for i, m in enumerate(self._metas) if m.get("doc_id") != doc_id]
            self._ids = [self._ids[i] for i in keep]
            self._metas = [self._metas[i] for i in keep]
            self._matrix = self._matrix[keep]

    def count(self) -> int:
        return len(self._ids)

    def reset(self) -> None:
        with self._lock:
            self._ids, self._metas = [], []
            self._matrix = np.zeros((0, self.dim), dtype=np.float32)
