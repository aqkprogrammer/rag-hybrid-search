from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from hybrid_rag.schemas import Chunk, DocumentRecord
from hybrid_rag.storage import BM25Index, DocStore
from hybrid_rag.vectorstores.base import VectorStore
from hybrid_rag.vectorstores.chroma import ChromaVectorStore
from hybrid_rag.vectorstores.memory import InMemoryVectorStore
from hybrid_rag.vectorstores.qdrant import QdrantVectorStore


def _chunk(doc: str, i: int, text: str, dept: str = "HR") -> Chunk:
    return Chunk(
        chunk_id=f"{doc}-{i:04d}",
        doc_id=doc,
        index=i,
        text=text,
        token_count=len(text.split()),
        metadata={"doc_id": doc, "department": dept, "title": doc},
    )


def _record(doc: str, n: int) -> DocumentRecord:
    now = datetime.now(UTC)
    return DocumentRecord(
        doc_id=doc,
        source=f"{doc}.md",
        title=doc,
        doc_type="markdown",
        content_hash=f"hash-{doc}",
        num_chunks=n,
        size_bytes=10,
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def store(tmp_path: Path) -> DocStore:
    s = DocStore(tmp_path / "db.sqlite3")
    s.upsert_document(
        _record("hr", 2),
        [
            _chunk("hr", 0, "Employees accrue vacation days every month."),
            _chunk("hr", 1, "Parental leave lasts sixteen weeks."),
        ],
    )
    s.upsert_document(
        _record("eng", 1),
        [_chunk("eng", 0, "Roll back a deployment with the launchpad command.", "Engineering")],
    )
    return s


def test_docstore_crud_and_revision(store: DocStore) -> None:
    assert store.count_documents() == 2
    assert store.count_chunks() == 3
    rev = store.revision()
    assert store.get_by_source("hr.md").doc_id == "hr"  # type: ignore[union-attr]
    assert store.get_by_hash("hash-eng").doc_id == "eng"  # type: ignore[union-attr]
    assert [c.index for c in store.chunks_for_document("hr")] == [0, 1]
    assert store.delete_document("hr") is True
    assert store.delete_document("hr") is False
    assert store.count_chunks() == 1  # cascade
    assert store.revision() == rev + 1


def test_bm25_ranks_filters_and_tracks_revisions(store: DocStore) -> None:
    index = BM25Index(store)
    hits = index.search("how do I roll back a deployment", k=3)
    assert hits[0][0] == "eng-0000"
    assert all(score > 0 for _, score in hits)
    assert index.search("roll back deployment", k=3, filters={"department": "HR"}) == []
    assert index.search("the and of", k=3) == []  # only stop-words

    store.upsert_document(_record("sec", 1), [_chunk("sec", 0, "Rotate keys every 90 days.")])
    assert index.search("rotate keys", k=1)[0][0] == "sec-0000"  # rebuilt automatically


def test_bm25_single_document_scores_positive(tmp_path: Path) -> None:
    s = DocStore(tmp_path / "one.sqlite3")
    s.upsert_document(_record("a", 1), [_chunk("a", 0, "vacation policy")])
    assert BM25Index(s).search("vacation", k=5)[0][1] > 0


@pytest.fixture(params=["memory", "chroma", "qdrant"])
def vector_store(request: pytest.FixtureRequest, tmp_path: Path) -> VectorStore:
    if request.param == "memory":
        return InMemoryVectorStore(dim=4)
    if request.param == "chroma":
        return ChromaVectorStore(tmp_path / "chroma", "test-collection")
    return QdrantVectorStore("test-collection", dim=4)


def test_vector_store_contract(vector_store: VectorStore) -> None:
    chunks = [
        _chunk("a", 0, "x"),
        _chunk("a", 1, "y"),
        _chunk("b", 0, "z", dept="Engineering"),
    ]
    vecs = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0.9, 0.1, 0, 0]], dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    vector_store.upsert(chunks, vecs)
    assert vector_store.count() == 3

    q = np.array([1, 0, 0, 0], dtype=np.float32)
    hits = vector_store.query(q, k=2)
    assert [h[0] for h in hits] == ["a-0000", "b-0000"]
    assert hits[0][1] == pytest.approx(1.0, abs=1e-3)

    filtered = vector_store.query(q, k=3, filters={"department": "Engineering"})
    assert [h[0] for h in filtered] == ["b-0000"]
    any_of = vector_store.query(q, k=3, filters={"doc_id": ["b"]})
    assert [h[0] for h in any_of] == ["b-0000"]

    vector_store.upsert(chunks[:1], vecs[:1])  # idempotent upsert
    assert vector_store.count() == 3
    vector_store.delete_document("a")
    assert vector_store.count() == 1
    vector_store.reset()
    assert vector_store.count() == 0
    vector_store.close()
