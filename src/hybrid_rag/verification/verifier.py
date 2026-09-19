"""Post-generation citation verification.

For every sentence ("claim") in the answer we check that each passage it cites actually
supports it, using three independent signals:

* **lexical support** - share of the claim's content terms (stemmed, stop-words removed) that
  occur in the cited passage (recall-oriented: the claim must be *covered* by the passage);
* **semantic support** - max cosine similarity between the claim and any sentence of the passage
  (plus the passage as a whole), using the configured embedding model;
* **numeric guard** - every number in the claim (days, %, amounts, versions) must appear in the
  cited passage. Hallucinated numbers are the most damaging RAG failure in policy documents,
  and neither lexical nor embedding similarity reliably catches "20 days" vs "25 days".

``support = w * lexical + (1 - w) * semantic`` (halved if the numeric guard fails); a citation is
supported when ``support >= threshold`` and the numeric guard passes. Optionally an LLM judge
gives the final verdict. Claims citing several passages are also checked against the union of
those passages, since a sentence may legitimately combine facts from two sources.

Outputs: per-citation checks, per-claim status/score, an answer-level groundedness score, and a
cleaned answer with unsupported / out-of-range citation markers removed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from hybrid_rag.embeddings import Embedder
from hybrid_rag.generation.prompts import (
    JUDGE_SYSTEM_PROMPT,
    build_judge_prompt,
    is_refusal,
)
from hybrid_rag.llm.base import LLMError, LLMProvider
from hybrid_rag.logging_setup import get_logger
from hybrid_rag.schemas import (
    CitationCheck,
    ClaimStatus,
    ClaimVerification,
    RetrievedChunk,
    VerificationReport,
)
from hybrid_rag.text import (
    analyze,
    citation_regex,
    numbers,
    split_sentences,
    strip_markdown,
    unwrap_lines,
)

log = get_logger(__name__)

_LEADING_CITES_RE = re.compile(r"^((?:\s*\[\d+(?:\s*,\s*\d+)*\])+)\s*")


@dataclass
class ParsedClaim:
    text: str  # sentence with citation markers removed
    raw: str  # sentence as it appears in the answer (incl. markers)
    start: int  # span in the original answer
    end: int
    citations: list[int] = field(default_factory=list)


def parse_claims(answer: str) -> list[ParsedClaim]:
    """Split an answer into sentences and attach citation markers to the right sentence.

    Handles both "... 20 days [1]." and "... 20 days. [1]" styles: markers that *start* a
    sentence are re-attached to the previous one.
    """
    pieces: list[tuple[int, int]] = []
    cursor = 0
    for sentence in split_sentences(answer):
        start = answer.find(sentence, cursor)
        if start < 0:  # should not happen; fall back to sequential placement
            start = cursor
        end = start + len(sentence)
        pieces.append((start, end))
        cursor = end

    claims: list[ParsedClaim] = []
    for start, end in pieces:
        raw = answer[start:end]
        lead = _LEADING_CITES_RE.match(raw)
        if lead and claims:
            prev = claims[-1]
            prev.end = start + lead.end(1)
            prev.raw = answer[prev.start : prev.end]
            prev.citations.extend(_indices(lead.group(1)))
            start += lead.end()
            raw = answer[start:end]
            if not raw.strip():
                continue
        text = strip_markdown(citation_regex().sub("", raw)).strip()
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        claims.append(
            ParsedClaim(text=text, raw=raw, start=start, end=end, citations=_indices(raw))
        )
    for c in claims:
        c.citations = list(dict.fromkeys(c.citations))
    return claims


def _indices(text: str) -> list[int]:
    out: list[int] = []
    for group in citation_regex().findall(text):
        out.extend(int(x) for x in group.split(","))
    return out


def _is_claim(c: ParsedClaim) -> bool:
    """Connective text ("Here are the steps:") is not a factual claim."""
    if c.citations:
        return True
    words = c.text.split()
    return len(words) > 6 and not c.text.rstrip().endswith(":") and not is_refusal(c.text)


class CitationVerifier:
    def __init__(
        self,
        embedder: Embedder,
        threshold: float = 0.55,
        lexical_weight: float = 0.5,
        judge: LLMProvider | None = None,
    ) -> None:
        self.embedder = embedder
        self.threshold = threshold
        self.lexical_weight = lexical_weight
        self.judge = judge

    @property
    def method(self) -> str:
        base = f"lexical({self.lexical_weight:.2f})+embedding({self.embedder.model_name})+numeric"
        return f"{base}+llm-judge({self.judge.model})" if self.judge else base

    async def verify(
        self, answer: str, passages: list[RetrievedChunk], strip_unsupported: bool = True
    ) -> tuple[VerificationReport, str]:
        claims = [c for c in parse_claims(answer) if _is_claim(c)]
        n = len(passages)
        passage_texts = [p.chunk.contextual_text for p in passages]
        passage_sents = [
            [s for line in unwrap_lines(t) for s in split_sentences(strip_markdown(line))] or [t]
            for t in passage_texts
        ]

        # Embed everything we need in one batch: claims, passages, passage sentences.
        flat_sents = [s for sents in passage_sents for s in sents]
        texts = [c.text for c in claims] + passage_texts + flat_sents
        vecs = self.embedder.embed_documents(texts) if texts else np.zeros((0, 1))
        claim_vecs = vecs[: len(claims)]
        passage_vecs = vecs[len(claims) : len(claims) + n]
        sent_vecs = vecs[len(claims) + n :]
        offsets = np.cumsum([0, *[len(s) for s in passage_sents]])

        results: list[ClaimVerification] = []
        remove: dict[int, set[int]] = {}  # claim position -> citation indexes to strip
        for ci, claim in enumerate(claims):
            c_terms = set(analyze(claim.text))
            c_nums = numbers(claim.text)
            checks: list[CitationCheck] = []
            for idx in claim.citations:
                if not 1 <= idx <= n:
                    checks.append(
                        CitationCheck(
                            index=idx,
                            valid=False,
                            supported=False,
                            score=0.0,
                            lexical=0.0,
                            semantic=0.0,
                            numbers_ok=False,
                        )
                    )
                    continue
                p = idx - 1
                p_terms = set(analyze(passage_texts[p]))
                lexical = len(c_terms & p_terms) / len(c_terms) if c_terms else 0.0
                sims = sent_vecs[offsets[p] : offsets[p + 1]] @ claim_vecs[ci]
                best = int(np.argmax(sims)) if len(sims) else 0
                semantic = float(
                    max(sims.max() if len(sims) else 0.0, passage_vecs[p] @ claim_vecs[ci])
                )
                semantic = max(0.0, min(1.0, semantic))
                numbers_ok = c_nums <= numbers(passage_texts[p])
                score = self.lexical_weight * lexical + (1 - self.lexical_weight) * semantic
                if not numbers_ok:
                    score *= 0.5
                checks.append(
                    CitationCheck(
                        index=idx,
                        valid=True,
                        supported=numbers_ok and score >= self.threshold,
                        score=round(score, 4),
                        lexical=round(lexical, 4),
                        semantic=round(semantic, 4),
                        numbers_ok=numbers_ok,
                        evidence=passage_sents[p][best] if passage_sents[p] else None,
                    )
                )

            valid = [c for c in checks if c.valid]
            # A sentence may combine facts from several cited passages: check the union too.
            union_supported = False
            union_score = 0.0
            if len(valid) > 1:
                union_text = "\n".join(passage_texts[c.index - 1] for c in valid)
                u_terms = set(analyze(union_text))
                u_lex = len(c_terms & u_terms) / len(c_terms) if c_terms else 0.0
                u_sem = max(c.semantic for c in valid)
                u_nums = c_nums <= numbers(union_text)
                union_score = self.lexical_weight * u_lex + (1 - self.lexical_weight) * u_sem
                union_score *= 1.0 if u_nums else 0.5
                union_supported = u_nums and union_score >= self.threshold
                if union_supported:
                    for c in valid:
                        if not c.supported and c.lexical >= 0.3:
                            c.supported = True

            if self.judge is not None:
                await self._apply_judge(claim.text, valid, passage_texts)

            supported = [c for c in valid if c.supported]
            status: ClaimStatus
            if not claim.citations:
                status, score = "uncited", 0.0
            elif valid and len(supported) == len(claim.citations):
                status = "supported"
                score = max([c.score for c in supported] + [union_score])
            elif supported:
                status = "partially_supported"
                score = max([c.score for c in supported] + [union_score])
                score *= len(supported) / len(claim.citations)
            else:
                status = "unsupported"
                score = max([c.score for c in valid] + [0.0])
            remove[ci] = {c.index for c in checks if not c.supported}
            results.append(
                ClaimVerification(
                    claim=claim.text,
                    status=status,
                    score=round(min(score, 1.0), 4),
                    citations=checks,
                )
            )

        total = len(results)
        n_supported = sum(1 for r in results if r.status == "supported")
        report = VerificationReport(
            groundedness=round(sum(r.score for r in results) / total, 4) if total else 0.0,
            supported_ratio=round(n_supported / total, 4) if total else 0.0,
            total_claims=total,
            supported_claims=n_supported,
            unsupported_citations=sorted(
                {c.index for r in results for c in r.citations if c.valid and not c.supported}
            ),
            invalid_citations=sorted(
                {c.index for r in results for c in r.citations if not c.valid}
            ),
            claims=results,
            method=self.method,
        )
        cleaned = _strip_citations(answer, claims, remove) if strip_unsupported else answer
        return report, cleaned

    async def _apply_judge(
        self, claim: str, checks: list[CitationCheck], passage_texts: list[str]
    ) -> None:
        assert self.judge is not None
        for c in checks:
            try:
                verdict = await self.judge.complete(
                    JUDGE_SYSTEM_PROMPT,
                    build_judge_prompt(claim, passage_texts[c.index - 1]),
                    max_tokens=1024,
                )
            except LLMError as exc:
                log.warning("llm_judge_failed", error=str(exc))
                return
            c.judge = verdict.strip().upper().startswith("SUPPORTED")
            c.supported = c.judge and c.numbers_ok


def _strip_citations(answer: str, claims: list[ParsedClaim], remove: dict[int, set[int]]) -> str:
    """Rewrite citation markers inside each claim's span, dropping unsupported indexes."""
    out: list[str] = []
    cursor = 0
    for ci, claim in enumerate(claims):
        drop = remove.get(ci, set())
        out.append(answer[cursor : claim.start])
        segment = answer[claim.start : claim.end]
        if drop:

            def repl(m: re.Match[str], drop: set[int] = drop) -> str:
                keep = [x.strip() for x in m.group(1).split(",") if int(x) not in drop]
                return f"[{', '.join(keep)}]" if keep else ""

            segment = citation_regex().sub(repl, segment)
            segment = re.sub(r"[ \t]+([.,;:!?])", r"\1", segment)
            segment = re.sub(r"[ \t]{2,}", " ", segment)
        out.append(segment)
        cursor = claim.end
    out.append(answer[cursor:])
    return "".join(out)
