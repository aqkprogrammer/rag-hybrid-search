"""Embedding provider interface."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    """Maps text to L2-normalised float32 vectors (cosine similarity == dot product)."""

    #: Human-readable model identifier, e.g. ``BAAI/bge-small-en-v1.5``.
    model_name: str
    dim: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed passages. Returns an array of shape ``(len(texts), dim)``."""

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a search query. Override for asymmetric models that use query prefixes."""
        return self.embed_documents([text])[0]

    @property
    def slug(self) -> str:
        """Filesystem/collection-safe identifier so different models never share an index."""
        base = re.sub(r"[^a-zA-Z0-9]+", "-", self.model_name).strip("-").lower()
        return f"{base}-{self.dim}"[:60].strip("-")


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)
