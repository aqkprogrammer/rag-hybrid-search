from types import SimpleNamespace
from typing import Any

from hybrid_rag.generation.prompts import (
    REFUSAL_ANSWER,
    SYSTEM_PROMPT,
    build_user_prompt,
    is_refusal,
    parse_user_prompt,
)
from hybrid_rag.llm import MockLLMProvider
from hybrid_rag.llm.anthropic_provider import REFUSAL_TEXT, AnthropicProvider
from hybrid_rag.schemas import Chunk, RetrievedChunk, StageScores


def _rc(i: int, text: str, heading: list[str]) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=f"c{i}",
        doc_id="d",
        index=i,
        text=text,
        heading_path=heading,
        token_count=5,
        metadata={"title": "Handbook"},
    )
    return RetrievedChunk(chunk=chunk, score=1.0, stages=StageScores())


PASSAGES = [
    _rc(0, "Engineers receive an on-call stipend of $500 per week.\nPages must be acked.", ["Pay"]),
    _rc(1, "Laptops are refreshed every 3 years.", ["Laptops"]),
]


def test_prompt_roundtrip() -> None:
    prompt = build_user_prompt("How much is the stipend?", PASSAGES)
    assert "[1] Source: Handbook | Section: Pay" in prompt
    question, passages = parse_user_prompt(prompt)
    assert question == "How much is the stipend?"
    assert [p.index for p in passages] == [1, 2]
    assert passages[1].text == "Laptops are refreshed every 3 years."
    assert "untrusted" in SYSTEM_PROMPT


async def test_mock_answers_with_citations() -> None:
    llm = MockLLMProvider()
    answer = await llm.complete(
        SYSTEM_PROMPT, build_user_prompt("on-call stipend per week", PASSAGES)
    )
    assert answer == "Engineers receive an on-call stipend of $500 per week [1]."


async def test_mock_refuses_without_relevant_context() -> None:
    llm = MockLLMProvider()
    answer = await llm.complete(SYSTEM_PROMPT, build_user_prompt("stock ticker symbol?", PASSAGES))
    assert answer == REFUSAL_ANSWER
    assert is_refusal(answer)


async def test_mock_stream_reassembles() -> None:
    llm = MockLLMProvider()
    prompt = build_user_prompt("laptops refreshed", PASSAGES)
    parts = [p async for p in llm.stream(SYSTEM_PROMPT, prompt)]
    assert len(parts) > 1
    assert "".join(parts) == await llm.complete(SYSTEM_PROMPT, prompt)


class _FakeMessages:
    def __init__(self, stop_reason: str) -> None:
        self.stop_reason = stop_reason
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text="Answer [1]."),
            ],
        )


async def test_anthropic_provider_request_shape_and_refusal() -> None:
    provider = AnthropicProvider(api_key="test", model="claude-sonnet-5", effort="low")
    fake = _FakeMessages("end_turn")
    provider._client = SimpleNamespace(messages=fake)  # type: ignore[assignment]
    assert await provider.complete("sys", "prompt") == "Answer [1]."
    call = fake.calls[0]
    assert call["model"] == "claude-sonnet-5"
    assert call["system"] == "sys"
    assert call["output_config"] == {"effort": "low"}
    assert "temperature" not in call  # sampling params are rejected by current models

    refusing = _FakeMessages("refusal")
    provider._client = SimpleNamespace(messages=refusing)  # type: ignore[assignment]
    assert await provider.complete("sys", "prompt") == REFUSAL_TEXT
