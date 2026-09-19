"""OpenAI embeddings (e.g. text-embedding-3-small)."""

from __future__ import annotations

import numpy as np
from openai import OpenAI

from hybrid_rag.embeddings.base import Embedder, l2_normalize

_KNOWN_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder(Embedder):
    def __init__(
        self,
        api_key: str,
        model_name: str = "text-embedding-3-small",
        base_url: str | None = None,
        batch_size: int = 128,
        timeout_s: float = 60.0,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)
        self.dim = _KNOWN_DIMS.get(model_name) or int(self.embed_documents(["probe"]).shape[1])

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [t.replace("\n", " ") or " " for t in texts[start : start + self.batch_size]]
            resp = self._client.embeddings.create(model=self.model_name, input=batch)
            rows.extend(item.embedding for item in sorted(resp.data, key=lambda d: d.index))
        return l2_normalize(np.asarray(rows, dtype=np.float32))
