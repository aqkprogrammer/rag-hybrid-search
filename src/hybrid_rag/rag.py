"""RAG orchestration: retrieve -> grounded generation -> citation verification."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from hybrid_rag.config import Settings
from hybrid_rag.generation.prompts import (
    REFUSAL_ANSWER,
    SYSTEM_PROMPT,
    build_user_prompt,
    is_refusal,
)
from hybrid_rag.llm.base import LLMError, LLMProvider
from hybrid_rag.logging_setup import get_logger
from hybrid_rag.retrieval import HybridRetriever
from hybrid_rag.schemas import (
    Citation,
    QueryRequest,
    QueryResponse,
    RetrievalResult,
    VerificationReport,
)
from hybrid_rag.text import citation_regex
from hybrid_rag.verification import CitationVerifier

log = get_logger(__name__)


def build_citations(retrieval: RetrievalResult, answer: str) -> list[Citation]:
    cited = {int(x) for g in citation_regex().findall(answer) for x in g.split(",")}
    return [
        Citation(
            index=i,
            chunk_id=r.chunk.chunk_id,
            doc_id=r.chunk.doc_id,
            title=str(r.chunk.metadata.get("title", "")),
            source=str(r.chunk.metadata.get("source", "")),
            heading_path=r.chunk.heading_path,
            page=r.chunk.page,
            snippet=r.chunk.text,
            score=r.score,
            cited=i in cited,
        )
        for i, r in enumerate(retrieval.results, start=1)
    ]


class RAGService:
    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        llm: LLMProvider,
        verifier: CitationVerifier,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.llm = llm
        self.verifier = verifier

    async def _retrieve(self, req: QueryRequest) -> RetrievalResult:
        return await asyncio.to_thread(
            self.retriever.retrieve,
            req.question,
            mode=req.mode,
            top_k=req.top_k,
            rerank=req.rerank,
            filters=req.filters,
        )

    async def _verify(
        self, req: QueryRequest, answer: str, retrieval: RetrievalResult, refused: bool
    ) -> tuple[VerificationReport | None, str]:
        should_verify = self.settings.verify_citations if req.verify is None else req.verify
        if refused or not should_verify:
            return None, answer
        return await self.verifier.verify(
            answer,
            retrieval.results,
            strip_unsupported=self.settings.strip_unsupported_citations,
        )

    async def answer(self, req: QueryRequest) -> QueryResponse:
        t0 = time.perf_counter()
        retrieval = await self._retrieve(req)
        if retrieval.results:
            answer = await self.llm.complete(
                SYSTEM_PROMPT, build_user_prompt(req.question, retrieval.results)
            )
        else:
            answer = REFUSAL_ANSWER
        answer = answer.strip()
        refused = is_refusal(answer)
        verification, verified = await self._verify(req, answer, retrieval, refused)
        latency = round((time.perf_counter() - t0) * 1000, 2)
        log.info(
            "query_answered",
            refused=refused,
            groundedness=verification.groundedness if verification else None,
            ms=latency,
        )
        return QueryResponse(
            question=req.question,
            answer=answer,
            verified_answer=verified,
            refused=refused,
            citations=build_citations(retrieval, answer),
            verification=verification,
            retrieval=retrieval.debug if req.include_debug else None,
            provider=self.llm.name,
            model=self.llm.model,
            latency_ms=latency,
        )

    async def answer_stream(self, req: QueryRequest) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Yield ``(event, payload)`` pairs: retrieval, token*, verification, done | error."""
        t0 = time.perf_counter()
        retrieval = await self._retrieve(req)
        yield (
            "retrieval",
            {
                "citations": [c.model_dump() for c in build_citations(retrieval, "")],
                "retrieval": retrieval.debug.model_dump() if req.include_debug else None,
            },
        )

        parts: list[str] = []
        if not retrieval.results:
            parts.append(REFUSAL_ANSWER)
            yield "token", {"text": REFUSAL_ANSWER}
        else:
            try:
                async for delta in self.llm.stream(
                    SYSTEM_PROMPT, build_user_prompt(req.question, retrieval.results)
                ):
                    parts.append(delta)
                    yield "token", {"text": delta}
            except LLMError as exc:
                log.error("llm_stream_failed", error=str(exc))
                yield "error", {"message": str(exc)}
                return

        answer = "".join(parts).strip()
        refused = is_refusal(answer)
        verification, verified = await self._verify(req, answer, retrieval, refused)
        yield (
            "verification",
            {
                "answer": answer,
                "verified_answer": verified,
                "verification": verification.model_dump() if verification else None,
                "citations": [c.model_dump() for c in build_citations(retrieval, answer)],
            },
        )
        latency = round((time.perf_counter() - t0) * 1000, 2)
        log.info("query_streamed", refused=refused, ms=latency)
        yield (
            "done",
            {
                "refused": refused,
                "latency_ms": latency,
                "provider": self.llm.name,
                "model": self.llm.model,
            },
        )
