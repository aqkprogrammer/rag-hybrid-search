"""HTTP routes."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse

from hybrid_rag import __version__
from hybrid_rag.api.sse import SSE_HEADERS, sse_event
from hybrid_rag.container import Container
from hybrid_rag.ingestion import SUPPORTED_EXTENSIONS
from hybrid_rag.logging_setup import get_logger
from hybrid_rag.schemas import (
    DocumentDetail,
    DocumentRecord,
    HealthResponse,
    IngestResult,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
    SearchHit,
    SearchRequest,
    SearchResponse,
    StageScores,
    TextIngestRequest,
    VerifyRequest,
    VerifyResponse,
)

log = get_logger(__name__)
router = APIRouter()


def get_container(request: Request) -> Container:
    return request.app.state.container


ContainerDep = Annotated[Container, Depends(get_container)]


# --------------------------------------------------------------------------------------------
# Health & config
# --------------------------------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health(c: ContainerDep) -> HealthResponse:
    status_: Literal["ok", "degraded"] = "ok"
    try:
        vs_info = await asyncio.to_thread(c.vector_store.info)
    except Exception as exc:  # report, don't crash, so orchestrators see "degraded"
        vs_info = {"backend": c.vector_store.backend, "error": str(exc)}
        status_ = "degraded"
    return HealthResponse(
        status=status_,
        version=__version__,
        environment=c.settings.environment,
        providers={
            "llm": f"{c.llm.name}:{c.llm.model}",
            "embeddings": c.embedder.model_name,
            "reranker": c.reranker.name,
            "vector_store": c.vector_store.backend,
        },
        documents=c.docstore.count_documents(),
        chunks=c.docstore.count_chunks(),
        vector_store=vs_info,
        bm25=await asyncio.to_thread(c.bm25.stats),
    )


@router.get("/api/config", tags=["system"])
async def public_config(c: ContainerDep) -> dict[str, Any]:
    """Non-secret settings and filter facets used by the web UI."""
    s = c.settings
    docs = c.docstore.list_documents()
    facets: dict[str, list[str]] = {"department": [], "doc_type": []}
    for d in docs:
        for key in facets:
            value = d.metadata.get(key) if key != "doc_type" else d.doc_type
            if isinstance(value, str) and value not in facets[key]:
                facets[key].append(value)
    return {
        "retrieval_mode": s.retrieval_mode,
        "final_top_k": s.final_top_k,
        "reranker": c.reranker.name,
        "verify_citations": s.verify_citations,
        "llm": {"provider": c.llm.name, "model": c.llm.model},
        "embedding_model": c.embedder.model_name,
        "vector_store": c.vector_store.backend,
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
        "max_upload_mb": s.max_upload_mb,
        "facets": {k: sorted(v) for k, v in facets.items()},
    }


# --------------------------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------------------------
@router.get("/api/documents", response_model=list[DocumentRecord], tags=["documents"])
async def list_documents(c: ContainerDep) -> list[DocumentRecord]:
    return c.docstore.list_documents()


@router.get("/api/documents/{doc_id}", response_model=DocumentDetail, tags=["documents"])
async def get_document(
    doc_id: str, c: ContainerDep, include_chunks: bool = Query(default=False)
) -> DocumentDetail:
    record = c.docstore.get_document(doc_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"document '{doc_id}' not found")
    chunks = c.docstore.chunks_for_document(doc_id) if include_chunks else None
    return DocumentDetail(**record.model_dump(), chunks=chunks)


@router.post(
    "/api/documents",
    response_model=list[IngestResult],
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
)
async def upload_documents(
    c: ContainerDep,
    files: Annotated[list[UploadFile], File(description="Markdown, PDF, TXT or HTML files")],
    department: Annotated[str | None, Form()] = None,
    owner: Annotated[str | None, Form()] = None,
    classification: Annotated[str | None, Form()] = None,
) -> list[IngestResult]:
    limit = c.settings.max_upload_mb * 1024 * 1024
    extra = {"department": department, "owner": owner, "classification": classification}
    results: list[IngestResult] = []
    for upload in files:
        name = upload.filename or "upload"
        data = await upload.read(limit + 1)
        if len(data) > limit:
            results.append(
                IngestResult(
                    doc_id=None,
                    source=name,
                    status="failed",
                    error=f"file exceeds MAX_UPLOAD_MB={c.settings.max_upload_mb}",
                )
            )
            continue
        results.append(await asyncio.to_thread(c.ingestion.ingest_bytes, name, data, extra))
    return results


@router.post(
    "/api/documents/text",
    response_model=IngestResult,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
)
async def ingest_text(body: TextIngestRequest, c: ContainerDep) -> IngestResult:
    ext = ".md" if body.format == "markdown" else ".txt"
    source = body.source or body.title
    if not source.lower().endswith(ext):
        source = f"{source}{ext}"
    meta: dict[str, Any] = {**body.metadata, "title": body.title}
    return await asyncio.to_thread(
        c.ingestion.ingest_bytes, source, body.text.encode("utf-8"), meta
    )


@router.delete(
    "/api/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["documents"]
)
async def delete_document(doc_id: str, c: ContainerDep) -> None:
    deleted = await asyncio.to_thread(c.ingestion.delete, doc_id)
    if not deleted:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"document '{doc_id}' not found")


# --------------------------------------------------------------------------------------------
# Retrieval & QA
# --------------------------------------------------------------------------------------------
@router.post("/api/search", response_model=SearchResponse, tags=["qa"])
async def search(body: SearchRequest, c: ContainerDep) -> SearchResponse:
    result = await asyncio.to_thread(
        c.retriever.retrieve,
        body.query,
        mode=body.mode,
        top_k=body.top_k,
        rerank=body.rerank,
        filters=body.filters,
    )
    hits = [
        SearchHit(
            rank=i,
            chunk_id=r.chunk.chunk_id,
            doc_id=r.chunk.doc_id,
            title=str(r.chunk.metadata.get("title", "")),
            source=str(r.chunk.metadata.get("source", "")),
            heading_path=r.chunk.heading_path,
            page=r.chunk.page,
            text=r.chunk.text,
            score=r.score,
            stages=r.stages,
        )
        for i, r in enumerate(result.results, start=1)
    ]
    return SearchResponse(query=body.query, hits=hits, debug=result.debug)


@router.post("/api/query", response_model=QueryResponse, tags=["qa"])
async def query(body: QueryRequest, c: ContainerDep) -> QueryResponse:
    return await c.rag.answer(body)


@router.post(
    "/api/query/stream",
    tags=["qa"],
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def query_stream(body: QueryRequest, request: Request, c: ContainerDep) -> StreamingResponse:
    """Server-Sent Events: `retrieval`, `token`*, `verification`, `done` (or `error`)."""

    async def events() -> AsyncIterator[str]:
        try:
            async for event, payload in c.rag.answer_stream(body):
                if await request.is_disconnected():
                    log.info("client_disconnected")
                    return
                yield sse_event(event, payload)
        except Exception as exc:  # never break the stream framing with a raw 500
            log.exception("stream_failed")
            yield sse_event("error", {"message": f"internal error: {type(exc).__name__}"})

    return StreamingResponse(events(), media_type="text/event-stream", headers=SSE_HEADERS)


@router.post("/api/verify", response_model=VerifyResponse, tags=["qa"])
async def verify(body: VerifyRequest, c: ContainerDep) -> VerifyResponse:
    """Check the citations of any answer (e.g. from another system) against stored chunks.

    `[1]` refers to `chunk_ids[0]`, `[2]` to `chunk_ids[1]`, and so on.
    """
    found = c.docstore.get_chunks(body.chunk_ids)
    missing = [cid for cid in body.chunk_ids if cid not in found]
    if missing:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown chunk ids: {missing}")
    passages = [
        RetrievedChunk(chunk=found[cid], score=0.0, stages=StageScores()) for cid in body.chunk_ids
    ]
    report, cleaned = await c.verifier.verify(
        body.answer, passages, strip_unsupported=c.settings.strip_unsupported_citations
    )
    return VerifyResponse(verified_answer=cleaned, verification=report)
