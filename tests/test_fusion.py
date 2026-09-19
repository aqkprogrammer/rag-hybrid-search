import pytest

from hybrid_rag.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_scores_and_order() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]], k=60))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"] == pytest.approx(1 / 61)
    order = [x for x, _ in reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]], k=60)]
    assert order[0] == "b"  # good in both lists beats best in one
    assert set(order) == {"a", "b", "c", "d"}


def test_rrf_weights_and_tiebreak() -> None:
    order = [x for x, _ in reciprocal_rank_fusion([["a"], ["b"]], k=60, weights=[1.0, 2.0])]
    assert order == ["b", "a"]
    order = [x for x, _ in reciprocal_rank_fusion([["a"], ["b"]], k=60)]
    assert order == ["a", "b"]  # tie -> first seen


def test_rrf_validates_weights() -> None:
    with pytest.raises(ValueError, match="align"):
        reciprocal_rank_fusion([["a"]], weights=[1.0, 2.0])
