"""Retrieval: fusion, reranking and the hybrid retriever."""

from __future__ import annotations

from hybrid_rag.config import Settings
from hybrid_rag.retrieval.fusion import reciprocal_rank_fusion
from hybrid_rag.retrieval.rerankers import (
    CrossEncoderReranker,
    LexicalReranker,
    NoopReranker,
    Reranker,
)
from hybrid_rag.retrieval.retriever import HybridRetriever

__all__ = [
    "CrossEncoderReranker",
    "HybridRetriever",
    "LexicalReranker",
    "NoopReranker",
    "Reranker",
    "build_reranker",
    "reciprocal_rank_fusion",
]


def build_reranker(settings: Settings) -> Reranker:
    if settings.reranker == "cross-encoder":
        return CrossEncoderReranker(settings.reranker_model)
    if settings.reranker == "lexical":
        return LexicalReranker()
    return NoopReranker()
