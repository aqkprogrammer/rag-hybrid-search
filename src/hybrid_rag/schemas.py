"""Domain models and API schemas (pydantic v2)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from hybrid_rag.config import RetrievalMode

FilterValue = str | int | float | bool | list[str]
MetadataFilters = dict[str, FilterValue]

# Metadata keys that are copied onto every chunk and can be used in retrieval filters.
FILTERABLE_KEYS: tuple[str, ...] = (
    "doc_id",
    "source",
    "title",
    "doc_type",
    "department",
    "owner",
    "classification",
)


# --------------------------------------------------------------------------------------------
# Documents & chunks
# --------------------------------------------------------------------------------------------
class Section(BaseModel):
    """A structural unit produced by a loader: text under one heading path (or one PDF page)."""

    heading_path: list[str] = Field(default_factory=list)
    text: str
    page: int | None = None


class LoadedDocument(BaseModel):
    source: str
    title: str
    doc_type: Literal["markdown", "pdf", "text", "html"]
    sections: list[Section]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(s.text for s in self.sections)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    index: int
    text: str
    heading_path: list[str] = Field(default_factory=list)
    token_count: int
    page: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def breadcrumb(self) -> str:
        return " > ".join(self.heading_path)

    @property
    def contextual_text(self) -> str:
        """Text that is embedded / BM25-indexed: title + heading breadcrumb + body."""
        title = str(self.metadata.get("title", ""))
        header = " > ".join([p for p in [title, *self.heading_path] if p])
        return f"{header}\n\n{self.text}" if header else self.text


class DocumentRecord(BaseModel):
    doc_id: str
    source: str
    title: str
    doc_type: str
    content_hash: str
    num_chunks: int
    size_bytes: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class IngestResult(BaseModel):
    doc_id: str | None
    source: str
    status: Literal["created", "updated", "unchanged", "duplicate", "failed"]
    num_chunks: int = 0
    duplicate_of: str | None = None
    error: str | None = None


class DocumentDetail(DocumentRecord):
    chunks: list[Chunk] | None = None


class TextIngestRequest(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    text: str = Field(min_length=1)
    source: str | None = Field(default=None, description="Stable identifier; defaults to title")
    format: Literal["markdown", "text"] = "markdown"
    metadata: dict[str, str] = Field(default_factory=dict)


# --------------------------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------------------------
class StageScores(BaseModel):
    """Per-stage debug information for one candidate chunk."""

    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_rank: int | None = None
    rrf_score: float | None = None
    rerank_rank: int | None = None
    rerank_score: float | None = None


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float = Field(description="Score of the final stage that ordered this result")
    stages: StageScores


class RetrievalDebug(BaseModel):
    mode: RetrievalMode
    reranker: str
    embedding_model: str
    dense_candidates: int
    bm25_candidates: int
    fused_candidates: int
    timings_ms: dict[str, float]
    candidates: list[dict[str, Any]] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    query: str
    results: list[RetrievedChunk]
    debug: RetrievalDebug


class RetrievalOptions(BaseModel):
    mode: RetrievalMode | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    rerank: bool | None = None
    filters: MetadataFilters = Field(default_factory=dict)

    @field_validator("filters")
    @classmethod
    def _validate_filters(cls, v: MetadataFilters) -> MetadataFilters:
        unknown = set(v) - set(FILTERABLE_KEYS)
        if unknown:
            raise ValueError(
                f"unsupported filter keys {sorted(unknown)}; allowed: {FILTERABLE_KEYS}"
            )
        return v


class SearchRequest(RetrievalOptions):
    query: str = Field(min_length=1, max_length=2000)


class SearchHit(BaseModel):
    rank: int
    chunk_id: str
    doc_id: str
    title: str
    source: str
    heading_path: list[str]
    page: int | None
    text: str
    score: float
    stages: StageScores


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]
    debug: RetrievalDebug


# --------------------------------------------------------------------------------------------
# Generation & verification
# --------------------------------------------------------------------------------------------
class QueryRequest(RetrievalOptions):
    question: str = Field(min_length=1, max_length=2000)
    verify: bool | None = Field(default=None, description="Override VERIFY_CITATIONS")
    include_debug: bool = True


class Citation(BaseModel):
    index: int = Field(description="1-based passage number used in the answer, e.g. [1]")
    chunk_id: str
    doc_id: str
    title: str
    source: str
    heading_path: list[str]
    page: int | None
    snippet: str
    score: float
    cited: bool = Field(default=False, description="Whether the answer references this passage")


class CitationCheck(BaseModel):
    index: int
    valid: bool = Field(description="False if the answer cites a passage number that doesn't exist")
    supported: bool
    score: float
    lexical: float
    semantic: float
    numbers_ok: bool
    judge: bool | None = None
    evidence: str | None = Field(default=None, description="Best-matching sentence in the passage")


ClaimStatus = Literal["supported", "partially_supported", "unsupported", "uncited"]


class ClaimVerification(BaseModel):
    claim: str
    status: ClaimStatus
    score: float
    citations: list[CitationCheck]


class VerificationReport(BaseModel):
    groundedness: float = Field(description="Mean per-claim support score in [0, 1]")
    supported_ratio: float
    total_claims: int
    supported_claims: int
    unsupported_citations: list[int]
    invalid_citations: list[int]
    claims: list[ClaimVerification]
    method: str


class QueryResponse(BaseModel):
    question: str
    answer: str = Field(description="Raw model answer")
    verified_answer: str = Field(description="Answer with unsupported citations removed")
    refused: bool
    citations: list[Citation]
    verification: VerificationReport | None
    retrieval: RetrievalDebug | None
    provider: str
    model: str
    latency_ms: float


class VerifyRequest(BaseModel):
    """Verify an externally produced answer against stored chunks (numbered in list order)."""

    answer: str = Field(min_length=1, max_length=20000)
    chunk_ids: list[str] = Field(min_length=1, max_length=50)


class VerifyResponse(BaseModel):
    verified_answer: str
    verification: VerificationReport


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    environment: str
    providers: dict[str, str]
    documents: int
    chunks: int
    vector_store: dict[str, Any]
    bm25: dict[str, Any]
