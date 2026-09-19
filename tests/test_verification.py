import pytest

from hybrid_rag.embeddings import HashingEmbedder
from hybrid_rag.llm import MockLLMProvider
from hybrid_rag.schemas import Chunk, RetrievedChunk, StageScores
from hybrid_rag.verification import CitationVerifier, parse_claims


def _passage(i: int, text: str) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=f"d-{i}",
        doc_id="d",
        index=i,
        text=text,
        token_count=10,
        metadata={"title": "Policy"},
    )
    return RetrievedChunk(chunk=chunk, score=1.0, stages=StageScores())


PASSAGES = [
    _passage(0, "Employees may carry over up to 5 unused PTO days into the next calendar year."),
    _passage(1, "Hotel costs are reimbursed up to $350 per night in high-cost cities."),
]


@pytest.fixture
def verifier() -> CitationVerifier:
    return CitationVerifier(HashingEmbedder(dim=512), threshold=0.55)


def test_parse_claims_handles_both_citation_styles() -> None:
    claims = parse_claims("You can carry over 5 days [1]. Hotels are capped at $350. [2]")
    assert [c.citations for c in claims] == [[1], [2]]
    assert claims[1].text == "Hotels are capped at $350."
    multi = parse_claims("Both apply [1, 2][3].")
    assert multi[0].citations == [1, 2, 3]


async def test_supported_answer_scores_high(verifier: CitationVerifier) -> None:
    answer = (
        "Employees may carry over up to 5 unused PTO days into the next calendar year [1]. "
        "Hotel costs are reimbursed up to $350 per night in high-cost cities [2]."
    )
    report, cleaned = await verifier.verify(answer, PASSAGES)
    assert report.total_claims == 2
    assert report.supported_claims == 2
    assert report.groundedness > 0.9
    assert report.unsupported_citations == []
    assert cleaned == answer


async def test_hallucinated_number_and_wrong_citation_are_flagged(
    verifier: CitationVerifier,
) -> None:
    answer = (
        "Employees may carry over up to 12 unused PTO days into the next year [1]. "
        "Hotel costs are reimbursed up to $350 per night in high-cost cities [1]. "
        "Parental leave is 16 weeks [7]."
    )
    report, cleaned = await verifier.verify(answer, PASSAGES)
    statuses = [c.status for c in report.claims]
    assert statuses == ["unsupported", "unsupported", "unsupported"]
    first = report.claims[0].citations[0]
    assert first.numbers_ok is False
    assert report.invalid_citations == [7]
    assert report.unsupported_citations == [1]
    assert report.groundedness < 0.5
    assert "[1]" not in cleaned and "[7]" not in cleaned
    assert cleaned.startswith(
        "Employees may carry over up to 12 unused PTO days into the next year."
    )


async def test_multi_citation_union_support(verifier: CitationVerifier) -> None:
    answer = (
        "Employees may carry over up to 5 unused PTO days and hotel costs are reimbursed "
        "up to $350 per night [1][2]."
    )
    report, _ = await verifier.verify(answer, PASSAGES)
    assert report.claims[0].status == "supported"


async def test_uncited_claims_and_connectives(verifier: CitationVerifier) -> None:
    answer = "Here is what I found:\nThe office dog is named Biscuit and loves long walks outside."
    report, _ = await verifier.verify(answer, PASSAGES)
    assert [c.status for c in report.claims] == ["uncited"]
    assert report.groundedness == 0.0


async def test_llm_judge_overrides(verifier: CitationVerifier) -> None:
    judge = CitationVerifier(HashingEmbedder(dim=512), judge=MockLLMProvider())
    answer = "Employees may carry over up to 5 unused PTO days into the next calendar year [1]."
    report, _ = await judge.verify(answer, PASSAGES)
    check = report.claims[0].citations[0]
    assert check.judge is True and check.supported
    assert "llm-judge" in report.method
