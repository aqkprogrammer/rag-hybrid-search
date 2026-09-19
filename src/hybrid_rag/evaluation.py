"""Retrieval evaluation: hit@k and MRR for dense vs BM25 vs hybrid vs hybrid+rerank.

Relevance is defined robustly to chunking changes: a retrieved chunk is relevant when it comes
from the expected source document *and* contains at least one of the expected answer phrases.
The evaluation builds an isolated, in-memory index from the corpus so it never touches the
application's data directory.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hybrid_rag.config import RetrievalMode, Settings
from hybrid_rag.container import Container
from hybrid_rag.llm import MockLLMProvider
from hybrid_rag.schemas import Chunk


@dataclass(frozen=True)
class EvalItem:
    question: str
    source: str
    answer_contains: list[str]


@dataclass
class ConfigResult:
    name: str
    hits: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    per_question: list[int | None] = field(default_factory=list)  # 1-based rank or None
    latency_ms: float = 0.0


CONFIGS: list[tuple[str, RetrievalMode, bool]] = [
    ("dense", "dense", False),
    ("bm25", "bm25", False),
    ("hybrid (RRF)", "hybrid", False),
    ("hybrid + rerank", "hybrid", True),
]


def load_eval_set(path: Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            row: dict[str, Any] = json.loads(line)
            items.append(
                EvalItem(row["question"], row["source"], [str(x) for x in row["answer_contains"]])
            )
    return items


def is_relevant(chunk: Chunk, item: EvalItem) -> bool:
    if chunk.metadata.get("source") != item.source:
        return False
    # Normalise whitespace: source documents are hard-wrapped, phrases in the eval set are not.
    text = " ".join(chunk.text.split()).lower()
    return any(" ".join(phrase.split()).lower() in text for phrase in item.answer_contains)


def evaluate(
    settings: Settings,
    corpus_dir: Path,
    eval_path: Path,
    ks: tuple[int, ...] = (1, 3, 5),
    depth: int = 10,
) -> tuple[list[ConfigResult], dict[str, Any]]:
    items = load_eval_set(eval_path)
    with tempfile.TemporaryDirectory(prefix="rag-eval-") as tmp:
        eval_settings = settings.model_copy(
            update={"data_dir": Path(tmp), "vector_store": "memory", "auto_seed_dir": None}
        )
        container = Container.build(eval_settings, llm=MockLLMProvider())
        try:
            ingest = container.ingestion.ingest_directory(corpus_dir)
            container.bm25.rebuild()
            results: list[ConfigResult] = []
            for name, mode, rerank in CONFIGS:
                if rerank and container.reranker.name == "none":
                    continue
                res = ConfigResult(name=f"{name} ({container.reranker.name})" if rerank else name)
                reciprocal = 0.0
                total_ms = 0.0
                for item in items:
                    out = container.retriever.retrieve(
                        item.question, mode=mode, top_k=depth, rerank=rerank
                    )
                    total_ms += out.debug.timings_ms.get("total", 0.0)
                    rank = next(
                        (i for i, r in enumerate(out.results, 1) if is_relevant(r.chunk, item)),
                        None,
                    )
                    res.per_question.append(rank)
                    reciprocal += 1.0 / rank if rank else 0.0
                n = max(len(items), 1)
                res.hits = {k: sum(1 for r in res.per_question if r and r <= k) / n for k in ks}
                res.mrr = reciprocal / n
                res.latency_ms = total_ms / n
                results.append(res)
            meta = {
                "questions": len(items),
                "documents": sum(1 for r in ingest if r.status in ("created", "updated")),
                "chunks": container.docstore.count_chunks(),
                "embedding_model": container.embedder.model_name,
                "reranker": container.reranker.name,
                "depth": depth,
            }
        finally:
            container.docstore.close()
    return results, meta


def format_markdown(results: list[ConfigResult], meta: dict[str, Any]) -> str:
    ks = sorted(results[0].hits) if results else []
    header = (
        "| Retriever | "
        + " | ".join(f"Hit@{k}" for k in ks)
        + f" | MRR@{meta['depth']} | Latency (ms) |"
    )
    sep = "|---|" + "---:|" * (len(ks) + 2)
    rows = [
        f"| {r.name} | "
        + " | ".join(f"{r.hits[k]:.2f}" for k in ks)
        + f" | {r.mrr:.3f} | {r.latency_ms:.1f} |"
        for r in results
    ]
    caption = (
        f"{meta['questions']} questions over {meta['documents']} documents "
        f"({meta['chunks']} chunks); embeddings: `{meta['embedding_model']}`, "
        f"reranker: `{meta['reranker']}`."
    )
    return "\n".join([header, sep, *rows, "", caption])
