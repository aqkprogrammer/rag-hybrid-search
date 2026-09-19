import pytest

from hybrid_rag.container import Container
from hybrid_rag.retrieval.rerankers import LexicalReranker


def _sources(result) -> list[str]:  # type: ignore[no-untyped-def]
    return [r.chunk.metadata["source"] for r in result.results]


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_modes_find_the_right_document(seeded: Container, mode: str) -> None:
    result = seeded.retriever.retrieve(
        "How many PTO days can be carried over?", mode=mode, top_k=3, rerank=False
    )
    assert _sources(result)[0] == "hr/pto-and-leave-policy.md"
    assert result.debug.mode == mode


def test_hybrid_debug_contains_every_stage(seeded: Container) -> None:
    result = seeded.retriever.retrieve("rollback deployment command", top_k=3, rerank=True)
    top = result.results[0]
    assert top.chunk.metadata["source"] == "engineering/deployment-and-rollback-runbook.md"
    st = top.stages
    assert st.rrf_rank is not None and st.rerank_rank == 1
    assert st.dense_rank is not None or st.bm25_rank is not None
    dbg = result.debug
    assert dbg.reranker == "lexical"
    assert {"dense", "bm25", "fusion", "rerank", "total"} <= set(dbg.timings_ms)
    assert sum(c["selected"] for c in dbg.candidates) == 3
    assert len(dbg.candidates) >= 3


def test_metadata_filters_apply_to_both_retrievers(seeded: Container) -> None:
    result = seeded.retriever.retrieve(
        "how many days", top_k=10, filters={"department": "Engineering"}
    )
    assert result.results
    assert {r.chunk.metadata["department"] for r in result.results} == {"Engineering"}
    none = seeded.retriever.retrieve("anything", filters={"department": "Nope"})
    assert none.results == []


def test_top_k_is_respected(seeded: Container) -> None:
    assert len(seeded.retriever.retrieve("policy", top_k=2).results) == 2


def test_empty_index_returns_nothing(container: Container) -> None:
    assert container.retriever.retrieve("anything").results == []


def test_lexical_reranker_prefers_covering_passage(seeded: Container) -> None:
    chunks = seeded.docstore.iter_chunks()
    by_text = {c.chunk_id: c for c in chunks}
    target = next(c for c in chunks if "Premium economy" in c.text)
    other = next(c for c in chunks if "parental leave" in c.text.lower())
    scores = LexicalReranker().score(
        "when can I book premium economy flights", [by_text[other.chunk_id], target]
    )
    assert scores[1] > scores[0]
