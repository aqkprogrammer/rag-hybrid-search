"""Command-line interface: `hybrid-rag --help`."""

from __future__ import annotations

import asyncio
import json
import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import httpx
import typer

from hybrid_rag.config import get_settings
from hybrid_rag.container import Container
from hybrid_rag.ingestion.loaders import SUPPORTED_EXTENSIONS
from hybrid_rag.logging_setup import configure_logging


class Mode(StrEnum):
    hybrid = "hybrid"
    dense = "dense"
    bm25 = "bm25"


app = typer.Typer(
    name="hybrid-rag",
    help="Hybrid-search RAG over internal documents.",
    no_args_is_help=True,
    add_completion=False,
)


def _container() -> Container:

    settings = get_settings()
    configure_logging("WARNING", settings.log_format)
    container = Container.build(settings)
    container.bm25.rebuild()
    return container


@app.command()
def serve(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
) -> None:
    """Run the API server (and web UI) with uvicorn."""
    import uvicorn

    uvicorn.run(
        "hybrid_rag.api.app:app_factory",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_config=None,
    )


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(exists=True, help="File or directory to ingest")],
    recursive: bool = True,
) -> None:
    """Ingest a file or directory directly into the local index (no server needed)."""
    c = _container()
    if path.is_dir():
        results = c.ingestion.ingest_directory(path, recursive=recursive)
    else:
        results = [c.ingestion.ingest_path(path)]
    for r in results:
        extra = f" (duplicate of {r.duplicate_of})" if r.duplicate_of else ""
        extra += f" error: {r.error}" if r.error else ""
        typer.echo(f"{r.status:>9}  {r.num_chunks:>3} chunks  {r.source}{extra}")
    typer.echo(f"\n{c.docstore.count_documents()} documents / {c.docstore.count_chunks()} chunks")
    if any(r.status == "failed" for r in results):
        raise typer.Exit(code=1)


@app.command()
def seed(
    directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)] = Path("sample_docs"),
    url: str = typer.Option("http://127.0.0.1:8000", envvar="RAG_API_URL"),
) -> None:
    """Upload a directory of documents to a *running* server via the HTTP API."""
    files = sorted(
        p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        typer.echo(f"no supported files in {directory}", err=True)
        raise typer.Exit(code=1)
    failed = 0
    with httpx.Client(base_url=url, timeout=120) as client:
        for p in files:
            name = p.relative_to(directory).as_posix()
            resp = client.post(
                "/api/documents",
                files=[("files", (name, p.read_bytes(), "application/octet-stream"))],
            )
            if resp.status_code >= 400:
                typer.echo(f"   error  {name}: HTTP {resp.status_code} {resp.text}", err=True)
                failed += 1
                continue
            r = resp.json()[0]
            failed += r["status"] == "failed"
            typer.echo(f"{r['status']:>9}  {r['num_chunks']:>3} chunks  {r['source']}")
    if failed:
        raise typer.Exit(code=1)


@app.command()
def search(
    query: str,
    mode: Annotated[Mode | None, typer.Option(help="Retrieval mode")] = None,
    top_k: int = 5,
    rerank: bool = typer.Option(None, "--rerank/--no-rerank"),
) -> None:
    """Run retrieval only and print per-stage scores."""
    c = _container()
    result = c.retriever.retrieve(
        query, mode=mode.value if mode else None, top_k=top_k, rerank=rerank
    )
    for i, r in enumerate(result.results, 1):
        st = r.stages
        typer.echo(
            f"{i}. [{r.score:.3f}] {r.chunk.metadata.get('title')} > {r.chunk.breadcrumb}\n"
            f"   dense#{st.dense_rank} bm25#{st.bm25_rank} rrf#{st.rrf_rank} "
            f"rerank#{st.rerank_rank}\n   {r.chunk.text[:160].replace(chr(10), ' ')}..."
        )
    typer.echo(json.dumps(result.debug.timings_ms))


@app.command()
def ask(question: str, as_json: bool = typer.Option(False, "--json")) -> None:
    """Answer a question end-to-end (retrieve, generate, verify)."""
    from hybrid_rag.schemas import QueryRequest

    c = _container()
    resp = asyncio.run(c.rag.answer(QueryRequest(question=question, include_debug=False)))
    if as_json:
        typer.echo(resp.model_dump_json(indent=2))
        return
    typer.echo(resp.verified_answer + "\n")
    for cit in resp.citations:
        if cit.cited:
            typer.echo(f"  [{cit.index}] {cit.title} > {' > '.join(cit.heading_path)}")
    if resp.verification:
        v = resp.verification
        typer.echo(
            f"\ngroundedness={v.groundedness:.2f} supported={v.supported_claims}/{v.total_claims}"
        )


@app.command()
def reindex() -> None:
    """Re-embed every stored chunk (after changing the embedding model or vector store)."""
    c = _container()
    n = c.ingestion.reindex()
    typer.echo(f"re-embedded {n} chunks into {c.vector_store.backend}")


@app.command("eval")
def run_eval(
    corpus: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path("sample_docs"),
    qa: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "evals/retrieval_qa.jsonl"
    ),
    output: Annotated[Path | None, typer.Option(help="Write JSON results here")] = None,
) -> None:
    """Compare dense vs BM25 vs hybrid vs hybrid+rerank (hit@k, MRR)."""
    from hybrid_rag.evaluation import evaluate, format_markdown

    settings = get_settings()
    configure_logging("WARNING", settings.log_format)
    results, meta = evaluate(settings, corpus, qa)
    typer.echo(format_markdown(results, meta))
    if output:
        output.write_text(
            json.dumps(
                {
                    "meta": meta,
                    "results": [
                        {
                            "name": r.name,
                            "hits": r.hits,
                            "mrr": r.mrr,
                            "latency_ms": r.latency_ms,
                            "ranks": r.per_question,
                        }
                        for r in results
                    ],
                },
                indent=2,
            )
        )


def main() -> None:  # pragma: no cover
    sys.exit(app())


if __name__ == "__main__":  # pragma: no cover
    main()
