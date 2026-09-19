"""Hybrid retrieval pipeline: dense + BM25 -> RRF fusion -> cross-encoder rerank -> top-k."""

from __future__ import annotations

import time
from typing import Any

from hybrid_rag.config import RetrievalMode, Settings
from hybrid_rag.embeddings import Embedder
from hybrid_rag.logging_setup import get_logger
from hybrid_rag.retrieval.fusion import reciprocal_rank_fusion
from hybrid_rag.retrieval.rerankers import Reranker
from hybrid_rag.schemas import (
    MetadataFilters,
    RetrievalDebug,
    RetrievalResult,
    RetrievedChunk,
    StageScores,
)
from hybrid_rag.storage import BM25Index, DocStore
from hybrid_rag.vectorstores import VectorStore

log = get_logger(__name__)


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        docstore: DocStore,
        bm25: BM25Index,
        vector_store: VectorStore,
        embedder: Embedder,
        reranker: Reranker,
    ) -> None:
        self.settings = settings
        self.docstore = docstore
        self.bm25 = bm25
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        *,
        mode: RetrievalMode | None = None,
        top_k: int | None = None,
        rerank: bool | None = None,
        filters: MetadataFilters | None = None,
    ) -> RetrievalResult:
        s = self.settings
        mode = mode or s.retrieval_mode
        top_k = top_k or s.final_top_k
        use_rerank = (self.reranker.name != "none") if rerank is None else rerank
        use_rerank = use_rerank and self.reranker.name != "none"
        timings: dict[str, float] = {}
        t_total = time.perf_counter()

        dense: list[tuple[str, float]] = []
        if mode in ("hybrid", "dense"):
            t = time.perf_counter()
            qvec = self.embedder.embed_query(query)
            timings["embed_query"] = _ms(t)
            t = time.perf_counter()
            dense = self.vector_store.query(qvec, s.dense_top_k, filters)
            timings["dense"] = _ms(t)

        sparse: list[tuple[str, float]] = []
        if mode in ("hybrid", "bm25"):
            t = time.perf_counter()
            sparse = self.bm25.search(query, s.bm25_top_k, filters)
            timings["bm25"] = _ms(t)

        stages: dict[str, StageScores] = {}
        for rank, (cid, score) in enumerate(dense, start=1):
            st = stages.setdefault(cid, StageScores())
            st.dense_rank, st.dense_score = rank, round(score, 6)
        for rank, (cid, score) in enumerate(sparse, start=1):
            st = stages.setdefault(cid, StageScores())
            st.bm25_rank, st.bm25_score = rank, round(score, 6)

        t = time.perf_counter()
        if mode == "hybrid":
            fused = reciprocal_rank_fusion(
                [[c for c, _ in dense], [c for c, _ in sparse]],
                k=s.rrf_k,
                weights=[s.dense_weight, s.bm25_weight],
            )
            for rank, (cid, score) in enumerate(fused, start=1):
                stages[cid].rrf_rank, stages[cid].rrf_score = rank, round(score, 6)
        else:
            fused = dense if mode == "dense" else sparse
        timings["fusion"] = _ms(t)

        pool_size = max(s.rerank_candidates, top_k) if use_rerank else top_k
        pool_ids = [cid for cid, _ in fused[: pool_size * 2]]  # headroom for stale ids
        chunk_map = self.docstore.get_chunks(pool_ids)
        stale = [cid for cid in pool_ids if cid not in chunk_map]
        if stale:
            log.warning("retrieval_stale_ids", count=len(stale))
        pool = [(cid, score) for cid, score in fused if cid in chunk_map][:pool_size]

        ordered: list[tuple[str, float]] = pool
        if use_rerank and pool:
            t = time.perf_counter()
            rr_scores = self.reranker.score(query, [chunk_map[cid] for cid, _ in pool])
            timings["rerank"] = _ms(t)
            ranked = sorted(
                zip([cid for cid, _ in pool], rr_scores, strict=True),
                key=lambda x: -x[1],
            )
            for rank, (cid, score) in enumerate(ranked, start=1):
                stages[cid].rerank_rank, stages[cid].rerank_score = rank, round(score, 6)
            ordered = ranked

        results = [
            RetrievedChunk(chunk=chunk_map[cid], score=round(score, 6), stages=stages[cid])
            for cid, score in ordered[:top_k]
        ]
        timings["total"] = _ms(t_total)

        selected = {r.chunk.chunk_id for r in results}
        candidates: list[dict[str, Any]] = []
        for cid, _ in ordered:
            chunk = chunk_map[cid]
            candidates.append(
                {
                    "chunk_id": cid,
                    "title": chunk.metadata.get("title", ""),
                    "heading_path": chunk.breadcrumb,
                    "selected": cid in selected,
                    **stages[cid].model_dump(),
                }
            )

        debug = RetrievalDebug(
            mode=mode,
            reranker=self.reranker.name if use_rerank else "none",
            embedding_model=self.embedder.model_name,
            dense_candidates=len(dense),
            bm25_candidates=len(sparse),
            fused_candidates=len(fused),
            timings_ms=timings,
            candidates=candidates,
        )
        log.info(
            "retrieval_done",
            mode=mode,
            rerank=use_rerank,
            dense=len(dense),
            bm25=len(sparse),
            returned=len(results),
            ms=timings["total"],
        )
        return RetrievalResult(query=query, results=results, debug=debug)
