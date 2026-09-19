"""In-memory BM25 index over the chunk store.

Implements Okapi BM25 with the Lucene IDF variant ``log(1 + (N - df + 0.5) / (df + 0.5))``,
which - unlike the classic formula - never goes negative, so tiny or highly repetitive corpora
still rank sensibly. Postings live in a sparse inverted index, so query cost is proportional to
the postings of the query terms rather than the corpus size.

The index is derived state: it is rebuilt from the SQLite doc store on startup and whenever the
store's revision counter changes (including writes made by another process, e.g. the CLI).
Rebuilding takes milliseconds for thousands of chunks; for millions, swap this for a persistent
inverted index (OpenSearch, Tantivy, Postgres FTS) behind the same `search()` signature.
"""

from __future__ import annotations

import math
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hybrid_rag.logging_setup import get_logger
from hybrid_rag.schemas import Chunk, MetadataFilters
from hybrid_rag.storage.docstore import DocStore
from hybrid_rag.storage.filters import matches_filters
from hybrid_rag.text import analyze

log = get_logger(__name__)


@dataclass(frozen=True)
class _Snapshot:
    """Immutable index state; swapped atomically so readers never see a half-built index."""

    ids: list[str] = field(default_factory=list)
    metas: list[dict[str, Any]] = field(default_factory=list)
    postings: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    doc_len: np.ndarray = field(default_factory=lambda: np.zeros(0))
    avgdl: float = 0.0
    revision: int = -1


class BM25Index:
    def __init__(self, docstore: DocStore, k1: float = 1.2, b: float = 0.75) -> None:
        self.docstore = docstore
        self.k1 = k1
        self.b = b
        self._lock = threading.Lock()
        self._snap = _Snapshot()

    @property
    def size(self) -> int:
        return len(self._snap.ids)

    @property
    def vocabulary_size(self) -> int:
        return len(self._snap.postings)

    def stats(self) -> dict[str, int]:
        snap = self._ensure_fresh()
        return {
            "chunks": len(snap.ids),
            "vocabulary": len(snap.postings),
            "revision": snap.revision,
        }

    def rebuild(self) -> None:
        with self._lock:
            self._snap = self._build(self.docstore.iter_chunks(), self.docstore.revision())

    def _ensure_fresh(self) -> _Snapshot:
        revision = self.docstore.revision()
        if revision != self._snap.revision:
            with self._lock:
                if revision != self._snap.revision:
                    self._snap = self._build(self.docstore.iter_chunks(), revision)
        return self._snap

    @staticmethod
    def _build(chunks: list[Chunk], revision: int) -> _Snapshot:
        raw: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths: list[int] = []
        for i, chunk in enumerate(chunks):
            terms = analyze(chunk.contextual_text)
            lengths.append(len(terms))
            for term, tf in Counter(terms).items():
                raw[term].append((i, tf))
        postings = {
            term: (
                np.fromiter((p[0] for p in plist), dtype=np.int64, count=len(plist)),
                np.fromiter((p[1] for p in plist), dtype=np.float64, count=len(plist)),
            )
            for term, plist in raw.items()
        }
        doc_len = np.asarray(lengths, dtype=np.float64)
        log.info("bm25_index_built", chunks=len(chunks), terms=len(postings), revision=revision)
        return _Snapshot(
            ids=[c.chunk_id for c in chunks],
            metas=[c.metadata for c in chunks],
            postings=postings,
            doc_len=doc_len,
            avgdl=float(doc_len.mean()) if lengths else 0.0,
            revision=revision,
        )

    @staticmethod
    def _idf(snap: _Snapshot, term: str) -> float:
        entry = snap.postings.get(term)
        df = 0 if entry is None else len(entry[0])
        n = len(snap.ids)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def _score_all(self, snap: _Snapshot, query: str) -> np.ndarray:
        scores = np.zeros(len(snap.ids), dtype=np.float64)
        if not snap.ids:
            return scores
        norm = self.k1 * (1.0 - self.b + self.b * snap.doc_len / max(snap.avgdl, 1e-9))
        for term, qtf in Counter(analyze(query)).items():
            entry = snap.postings.get(term)
            if entry is None:
                continue
            idx, tf = entry
            scores[idx] += qtf * self._idf(snap, term) * tf * (self.k1 + 1.0) / (tf + norm[idx])
        return scores

    def search(
        self, query: str, k: int, filters: MetadataFilters | None = None
    ) -> list[tuple[str, float]]:
        """Top-`k` `(chunk_id, score)` pairs with a strictly positive BM25 score."""
        snap = self._ensure_fresh()
        if not snap.ids or k <= 0:
            return []
        scores = self._score_all(snap, query)
        if filters:
            mask = np.fromiter(
                (matches_filters(m, filters) for m in snap.metas), dtype=bool, count=len(snap.ids)
            )
            scores = np.where(mask, scores, 0.0)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(snap.ids[i], float(scores[i])) for i in top if scores[i] > 0]
