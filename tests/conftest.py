from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hybrid_rag.api.app import create_app
from hybrid_rag.config import Settings
from hybrid_rag.container import Container

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DOCS = ROOT / "sample_docs"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        environment="test",
        data_dir=tmp_path / "data",
        vector_store="memory",
        embedding_provider="hashing",
        reranker="lexical",
        llm_provider="mock",
        mock_stream_delay_s=0.0,
        tokenizer="simple",
        log_level="WARNING",
    )


@pytest.fixture
def container(settings: Settings) -> Iterator[Container]:
    c = Container.build(settings)
    yield c
    c.docstore.close()


@pytest.fixture
def seeded(container: Container) -> Container:
    results = container.ingestion.ingest_directory(SAMPLE_DOCS)
    assert all(r.status == "created" for r in results), results
    return container


@pytest.fixture
def client(seeded: Container) -> Iterator[TestClient]:
    app = create_app(container=seeded)
    with TestClient(app) as tc:
        yield tc
