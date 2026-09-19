"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009).

RRF combines ranked lists using only ranks, never raw scores - which is exactly what we need
when fusing cosine similarities (bounded, ~0.2-0.9) with BM25 scores (unbounded, corpus
dependent). ``score(d) = sum_i w_i / (k + rank_i(d))`` with 1-based ranks; k=60 dampens the
influence of top ranks so a document that is decent in *both* lists beats one that is great
in only one.
"""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must align with rankings")
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    order = 0
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
            if item not in first_seen:
                first_seen[item] = order
                order += 1
    # Deterministic tie-break: earlier first appearance wins.
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
