"""Second-stage rerankers.

* `CrossEncoderReranker` - a sentence-transformers cross-encoder (default
  ``cross-encoder/ms-marco-MiniLM-L-6-v2``) that reads query and passage *jointly*; far more
  precise than bi-encoder similarity, but too slow to run over the whole corpus - hence
  retrieve-then-rerank.
* `LexicalReranker` - deterministic, dependency-free stand-in used offline/in CI. It scores
  IDF-weighted query-term coverage, phrase (bigram) matches, heading matches and term proximity.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter
from itertools import pairwise
from typing import Any

from hybrid_rag.schemas import Chunk
from hybrid_rag.text import analyze


class Reranker(ABC):
    name: str

    @abstractmethod
    def score(self, query: str, chunks: list[Chunk]) -> list[float]:
        """Relevance score per chunk (higher is better), aligned with `chunks`."""


class LexicalReranker(Reranker):
    name = "lexical"

    def score(self, query: str, chunks: list[Chunk]) -> list[float]:
        q_terms = list(dict.fromkeys(analyze(query)))
        if not q_terms or not chunks:
            return [0.0] * len(chunks)
        q_bigrams = set(pairwise(q_terms))
        docs = [analyze(c.text) for c in chunks]
        heads = [
            set(analyze(" ".join([str(c.metadata.get("title", "")), *c.heading_path])))
            for c in chunks
        ]
        # IDF over the candidate set: terms present in every candidate carry little signal.
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        idf = {t: math.log(1.0 + (n - df[t] + 0.5) / (df[t] + 0.5)) + 0.1 for t in q_terms}
        total = sum(idf.values())

        scores: list[float] = []
        for terms, head in zip(docs, heads, strict=True):
            present = set(terms)
            coverage = sum(idf[t] for t in q_terms if t in present) / total
            heading = sum(idf[t] for t in q_terms if t in head) / total
            doc_bigrams = set(pairwise(terms))
            phrase = len(q_bigrams & doc_bigrams) / len(q_bigrams) if q_bigrams else 0.0
            proximity = _proximity(terms, set(q_terms))
            tf = Counter(terms)
            density = sum(min(tf[t], 3) for t in q_terms) / (3 * len(q_terms))
            scores.append(
                0.45 * coverage + 0.15 * heading + 0.15 * phrase + 0.15 * proximity + 0.10 * density
            )
        return scores


def _proximity(terms: list[str], query: set[str], window: int = 12) -> float:
    """Max fraction of distinct query terms that co-occur within a sliding window."""
    if not terms or not query:
        return 0.0
    best = 0
    positions = [i for i, t in enumerate(terms) if t in query]
    for i, start in enumerate(positions):
        seen = set()
        for pos in positions[i:]:
            if pos - start >= window:
                break
            seen.add(terms[pos])
        best = max(best, len(seen))
    return best / len(query)


class CrossEncoderReranker(Reranker):
    name = "cross-encoder"

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError(
                "RERANKER=cross-encoder requires the 'ml' extra: uv sync --extra ml"
            ) from exc
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: Any = CrossEncoder(model_name, max_length=512)

    def score(self, query: str, chunks: list[Chunk]) -> list[float]:
        if not chunks:
            return []
        pairs = [(query, c.contextual_text) for c in chunks]
        logits = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        # Sigmoid maps ms-marco logits to a readable (0, 1) relevance score; order is unchanged.
        return [1.0 / (1.0 + math.exp(-float(x))) for x in logits]


class NoopReranker(Reranker):
    name = "none"

    def score(self, query: str, chunks: list[Chunk]) -> list[float]:
        return [0.0] * len(chunks)
