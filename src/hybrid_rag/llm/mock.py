"""Deterministic offline LLM used for demos, tests and CI (no API key, no network).

It behaves like a well-instructed extractive model: it parses the numbered passages out of the
grounded prompt, selects the sentences that best overlap the question, and emits them with
inline citations - or refuses when nothing relevant is present. It also answers the verifier's
LLM-judge prompt with a lexical support check. Output is fully deterministic.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator

from hybrid_rag.generation.prompts import REFUSAL_ANSWER, parse_user_prompt
from hybrid_rag.llm.base import LLMProvider
from hybrid_rag.text import analyze, split_sentences, strip_markdown, unwrap_lines

_TABLE_RULE_RE = re.compile(r"^[\s|:-]+$")


class MockLLMProvider(LLMProvider):
    name = "mock"

    def __init__(self, stream_delay_s: float = 0.0, max_sentences: int = 3) -> None:
        self.model = "mock-extractive"
        self.stream_delay_s = stream_delay_s
        self.max_sentences = max_sentences

    async def complete(self, system: str, prompt: str, max_tokens: int | None = None) -> str:
        if "CLAIM:" in prompt and "PASSAGE:" in prompt:
            return self._judge(prompt)
        return self._answer(prompt)

    async def stream(self, system: str, prompt: str) -> AsyncIterator[str]:
        text = await self.complete(system, prompt)
        for piece in re.findall(r"\S+\s*|\s+", text):
            if self.stream_delay_s:
                await asyncio.sleep(self.stream_delay_s)
            yield piece

    # -- internals ----------------------------------------------------------------------------
    def _answer(self, prompt: str) -> str:
        question, passages = parse_user_prompt(prompt)
        q_terms = set(analyze(question))
        if not q_terms or not passages:
            return REFUSAL_ANSWER

        scored: list[tuple[float, int, int, str]] = []
        for p in passages:
            for pos, raw in enumerate(_candidate_sentences(p.text)):
                terms = set(analyze(raw))
                overlap = len(q_terms & terms)
                if overlap == 0 or len(terms) < 3:
                    continue
                ratio = overlap / len(q_terms)
                # Prefer higher-ranked passages and earlier sentences on ties.
                score = ratio + 0.02 / p.index - 0.001 * pos
                scored.append((score, p.index, pos, raw))

        min_overlap = 0.5 if len(q_terms) <= 3 else 0.4
        good = [s for s in scored if s[0] >= min_overlap]
        if not good:
            return REFUSAL_ANSWER
        good.sort(key=lambda s: -s[0])

        picked: list[tuple[int, int, str]] = []
        per_passage: dict[int, int] = {}
        for _, idx, pos, sentence in good:
            if per_passage.get(idx, 0) >= 2:
                continue
            per_passage[idx] = per_passage.get(idx, 0) + 1
            picked.append((idx, pos, sentence))
            if len(picked) >= self.max_sentences:
                break
        # Keep the best sentence first, then group the rest in document order for readability.
        head, rest = picked[0], sorted(picked[1:], key=lambda x: (x[0], x[1]))
        return " ".join(_cite(sentence, idx) for idx, _, sentence in [head, *rest])

    @staticmethod
    def _judge(prompt: str) -> str:
        claim = prompt.split("CLAIM:", 1)[1].split("PASSAGE:", 1)[0]
        passage = prompt.split("PASSAGE:", 1)[1]
        c_terms = set(analyze(re.sub(r"\[\d+\]", "", claim)))
        p_terms = set(analyze(passage))
        if not c_terms:
            return "NOT_SUPPORTED"
        return "SUPPORTED" if len(c_terms & p_terms) / len(c_terms) >= 0.7 else "NOT_SUPPORTED"


def _candidate_sentences(text: str) -> list[str]:
    out: list[str] = []
    in_fence = False
    for line in unwrap_lines(text):
        if line.strip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence or _TABLE_RULE_RE.match(line) or not line.strip():
            continue
        clean = strip_markdown(line)
        out.extend(s for s in split_sentences(clean) if len(s) > 12)
    return out


def _cite(sentence: str, index: int) -> str:
    sentence = sentence.rstrip()
    if sentence.endswith(":"):
        sentence = sentence[:-1]
    if sentence and sentence[-1] in ".!?":
        return f"{sentence[:-1]} [{index}]{sentence[-1]}"
    return f"{sentence} [{index}]."
