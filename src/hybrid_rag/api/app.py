"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from hybrid_rag import __version__
from hybrid_rag.api.routes import router
from hybrid_rag.config import Settings, get_settings
from hybrid_rag.container import Container
from hybrid_rag.llm import LLMError
from hybrid_rag.logging_setup import configure_logging, get_logger

log = get_logger(__name__)
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    resolved: Settings = settings or (container.settings if container else get_settings())
    settings = resolved
    configure_logging(settings.log_level, settings.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        c: Container = (
            container
            if container is not None
            else await asyncio.to_thread(Container.build, resolved)
        )
        await asyncio.to_thread(c.startup)
        app.state.container = c
        try:
            yield
        finally:
            if container is None:
                await c.aclose()

    app = FastAPI(
        title="Hybrid RAG",
        version=__version__,
        description=(
            "Retrieval-augmented QA over internal documents with hybrid retrieval "
            "(dense + BM25 + RRF + cross-encoder reranking) and verified citations."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        structlog.contextvars.bind_contextvars(request_id=request_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.unbind_contextvars("request_id")
        response.headers["x-request-id"] = request_id
        if not request.url.path.startswith("/static"):
            log.info(
                "http_request",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                ms=round((time.perf_counter() - start) * 1000, 2),
            )
        return response

    @app.exception_handler(LLMError)
    async def llm_error_handler(_: Request, exc: LLMError) -> JSONResponse:
        log.error("llm_error", error=str(exc))
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    app.include_router(router)

    if WEB_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")

    return app


def app_factory() -> FastAPI:
    """Entry point for `uvicorn hybrid_rag.api.app:app_factory --factory`."""
    return create_app()
