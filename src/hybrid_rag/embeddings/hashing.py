"""Deterministic, dependency-free feature-hashing embedder.

This is the offline default so the whole system (API, tests, demo, CI) runs without API keys or
model downloads. It is *lexical*, not semantic: it hashes stemmed unigrams, bigrams and
character trigrams into a fixed-size signed vector with sub-linear TF weighting. It captures
morphology and phrase overlap, which makes it a reasonable stand-in, but it will not know that
"PTO" and "vacation" are related - use `sentence-transformers` or `openai` for real deployments.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from itertools import pairwise

import numpy as np

from hybrid_rag.embeddings.base import Embedder, l2_normalize
from hybrid_rag.text import analyze


def _bucket(feature: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return value % dim, (1.0 if (value >> 63) & 1 else -1.0)


class HashingEmbedder(Embedder):
    def __init__(self, dim: int = 768) -> None:
        self.dim = dim
        self.model_name = "hashing-v1"

    def _features(self, text: str) -> dict[str, float]:
        terms = analyze(text)
        feats: defaultdict[str, float] = defaultdict(float)
        for t in terms:
            feats[f"w:{t}"] += 1.0
            padded = f"#{t}#"
            if len(t) > 3:
                for i in range(len(padded) - 2):
                    feats[f"c:{padded[i : i + 3]}"] += 0.25
        for a, b in pairwise(terms):
            feats[f"b:{a}_{b}"] += 0.75
        return feats

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, weight in self._features(text).items():
                idx, sign = _bucket(feat, self.dim)
                out[row, idx] += sign * (1.0 + math.log(weight)) if weight >= 1 else sign * weight
        return l2_normalize(out)
