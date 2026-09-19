"""Local sentence-transformers embedder (install the `ml` extra)."""

from __future__ import annotations

from typing import Any

import numpy as np

from hybrid_rag.embeddings.base import Embedder, l2_normalize

# Asymmetric retrieval models expect an instruction prefix on queries (not on passages).
_QUERY_PREFIXES: dict[str, str] = {
    "bge-small-en": "Represent this sentence for searching relevant passages: ",
    "bge-base-en": "Represent this sentence for searching relevant passages: ",
    "bge-large-en": "Represent this sentence for searching relevant passages: ",
    "e5-": "query: ",
}


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str, batch_size: int = 64, device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "EMBEDDING_PROVIDER=sentence-transformers requires the 'ml' extra: "
                "uv sync --extra ml"
            ) from exc
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: Any = SentenceTransformer(model_name, device=device)
        get_dim = getattr(self._model, "get_embedding_dimension", None)
        if get_dim is None:  # sentence-transformers < 5
            get_dim = self._model.get_sentence_embedding_dimension
        dim = get_dim()
        self.dim = int(dim) if dim else int(self.embed_documents(["probe"]).shape[1])
        self._query_prefix = next(
            (p for key, p in _QUERY_PREFIXES.items() if key in model_name.lower()), ""
        )
        self._doc_prefix = "passage: " if "e5-" in model_name.lower() else ""

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(
            [self._doc_prefix + t for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return l2_normalize(vecs)

    def embed_query(self, text: str) -> np.ndarray:
        vec = self._model.encode(
            [self._query_prefix + text], normalize_embeddings=True, convert_to_numpy=True
        )
        return l2_normalize(vec)[0]
