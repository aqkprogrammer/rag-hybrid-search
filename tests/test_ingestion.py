from hybrid_rag.container import Container
from hybrid_rag.ingestion.service import sanitize_source

MD = b"---\ndepartment: HR\n---\n# Leave\n\n## Vacation\n\nEmployees get 21 days of PTO per year.\n"


def test_create_unchanged_update_duplicate_delete(container: Container) -> None:
    ing = container.ingestion
    first = ing.ingest_bytes("hr/leave.md", MD)
    assert first.status == "created"
    assert first.num_chunks == 1
    assert container.vector_store.count() == 1

    assert ing.ingest_bytes("hr/leave.md", MD).status == "unchanged"

    updated = ing.ingest_bytes("hr/leave.md", MD.replace(b"21 days", b"25 days"))
    assert updated.status == "updated"
    assert updated.doc_id == first.doc_id  # stable id
    chunks = container.docstore.chunks_for_document(first.doc_id)  # type: ignore[arg-type]
    assert "25 days" in chunks[0].text
    assert container.vector_store.count() == 1  # old vectors replaced

    dup = ing.ingest_bytes("copy-of-leave.md", MD.replace(b"21 days", b"25 days"))
    assert dup.status == "duplicate"
    assert dup.duplicate_of == "hr/leave.md"
    assert container.docstore.count_documents() == 1

    assert ing.delete(first.doc_id) is True  # type: ignore[arg-type]
    assert ing.delete("missing") is False
    assert container.docstore.count_chunks() == 0
    assert container.vector_store.count() == 0


def test_failed_ingest_reports_error(container: Container) -> None:
    res = container.ingestion.ingest_bytes("notes.docx", b"binary")
    assert res.status == "failed"
    assert "unsupported" in (res.error or "")


def test_reindex_rebuilds_vectors(seeded: Container) -> None:
    n = seeded.docstore.count_chunks()
    seeded.vector_store.reset()
    assert seeded.ingestion.reindex() == n
    assert seeded.vector_store.count() == n


def test_startup_resyncs_vector_store(seeded: Container) -> None:
    seeded.vector_store.reset()
    seeded.startup()
    assert seeded.vector_store.count() == seeded.docstore.count_chunks()


def test_sanitize_source() -> None:
    assert sanitize_source("../../etc/passwd") == "etc/passwd"
    assert sanitize_source("C:\\docs\\a b.md") == "C_/docs/a b.md"
    assert sanitize_source("") == "document"
