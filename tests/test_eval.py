from pathlib import Path

from hybrid_rag.config import Settings
from hybrid_rag.evaluation import evaluate, format_markdown

ROOT = Path(__file__).resolve().parent.parent


def test_eval_runs_all_configs(settings: Settings) -> None:
    results, meta = evaluate(settings, ROOT / "sample_docs", ROOT / "evals/retrieval_qa.jsonl")
    assert [r.name for r in results] == [
        "dense",
        "bm25",
        "hybrid (RRF)",
        "hybrid + rerank (lexical)",
    ]
    assert meta["questions"] >= 20
    for r in results:
        assert 0.0 <= r.mrr <= 1.0
        assert r.hits[1] <= r.hits[3] <= r.hits[5]
        assert r.hits[5] >= 0.7  # sanity floor for the offline configuration
    table = format_markdown(results, meta)
    assert "| hybrid (RRF) |" in table
