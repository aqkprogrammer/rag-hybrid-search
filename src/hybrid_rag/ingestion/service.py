"""Ingestion pipeline: load -> hash/dedupe -> chunk -> embed -> index.

Idempotency rules (keyed on the document *source*, e.g. its relative path or upload name):

* same source, same content hash & metadata  -> ``unchanged`` (no work done)
* same source, different content              -> ``updated`` (old chunks/vectors replaced)
* new source, content identical to another doc -> ``duplicate`` (skipped, points at original)
* otherwise                                    -> ``created``

`doc_id` is derived from the source, so re-ingesting a file keeps its id stable.
"""

from __future__ import annotations

import hashlib
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from hybrid_rag.embeddings import Embedder
from hybrid_rag.ingestion.chunker import StructureAwareChunker
from hybrid_rag.ingestion.loaders import SUPPORTED_EXTENSIONS, load_document
from hybrid_rag.logging_setup import get_logger
from hybrid_rag.schemas import Chunk, DocumentRecord, IngestResult, LoadedDocument
from hybrid_rag.storage import DocStore
from hybrid_rag.vectorstores import VectorStore

log = get_logger(__name__)


def make_doc_id(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def content_hash(doc: LoadedDocument) -> str:
    normalized = re.sub(r"\s+", " ", doc.full_text).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def sanitize_source(name: str) -> str:
    """Keep a readable, path-traversal-free identifier (``hr/pto-policy.md``)."""
    parts = [p for p in re.split(r"[\\/]+", name) if p not in ("", ".", "..")]
    clean = "/".join(re.sub(r"[^\w.\- ]+", "_", p).strip() for p in parts)
    return clean or "document"


class IngestionService:
    def __init__(
        self,
        docstore: DocStore,
        vector_store: VectorStore,
        embedder: Embedder,
        chunker: StructureAwareChunker,
        batch_size: int = 64,
    ) -> None:
        self.docstore = docstore
        self.vector_store = vector_store
        self.embedder = embedder
        self.chunker = chunker
        self.batch_size = batch_size
        # Serialise writes: embedding is the slow part and stores are not transactional
        # across each other, so concurrent re-ingests of the same source must not interleave.
        self._write_lock = threading.Lock()

    @property
    def chunking_fingerprint(self) -> str:
        c = self.chunker
        return f"{c.counter.name}:{c.chunk_size}/{c.overlap}/{c.min_tokens}"

    # -- public API ---------------------------------------------------------------------------
    def ingest_bytes(
        self, filename: str, data: bytes, metadata: dict[str, Any] | None = None
    ) -> IngestResult:
        source = sanitize_source(filename)
        try:
            doc = load_document(source, data, metadata)
        except ValueError as exc:
            log.warning("ingest_failed", source=source, error=str(exc))
            return IngestResult(doc_id=None, source=source, status="failed", error=str(exc))
        return self.ingest_loaded(doc, size_bytes=len(data))

    def ingest_loaded(self, doc: LoadedDocument, size_bytes: int | None = None) -> IngestResult:
        doc_id = make_doc_id(doc.source)
        digest = content_hash(doc)
        doc_meta = {**doc.metadata, "chunking": self.chunking_fingerprint}

        with self._write_lock:
            existing = self.docstore.get_by_source(doc.source)
            if existing and existing.content_hash == digest and existing.metadata == doc_meta:
                return IngestResult(
                    doc_id=doc_id,
                    source=doc.source,
                    status="unchanged",
                    num_chunks=existing.num_chunks,
                )
            twin = self.docstore.get_by_hash(digest)
            if twin and twin.source != doc.source:
                log.info("ingest_duplicate", source=doc.source, duplicate_of=twin.source)
                return IngestResult(
                    doc_id=twin.doc_id,
                    source=doc.source,
                    status="duplicate",
                    duplicate_of=twin.source,
                    num_chunks=twin.num_chunks,
                )

            chunks = self.chunker.chunk(doc, doc_id)
            embeddings = self.embedder.embed_documents([c.contextual_text for c in chunks])
            # Vectors first, then the system of record (which bumps the revision and so
            # triggers the BM25 rebuild). A crash in between leaves only orphan vectors,
            # which retrieval ignores because they are absent from the doc store.
            self.vector_store.delete_document(doc_id)
            self.vector_store.upsert(chunks, embeddings)
            now = datetime.now(UTC)
            record = DocumentRecord(
                doc_id=doc_id,
                source=doc.source,
                title=doc.title,
                doc_type=doc.doc_type,
                content_hash=digest,
                num_chunks=len(chunks),
                size_bytes=size_bytes if size_bytes is not None else len(doc.full_text.encode()),
                metadata=doc_meta,
                created_at=existing.created_at if existing else now,
                updated_at=now,
            )
            self.docstore.upsert_document(record, chunks)

        status: Literal["created", "updated"] = "updated" if existing else "created"
        log.info("ingest_ok", source=doc.source, doc_id=doc_id, status=status, chunks=len(chunks))
        return IngestResult(doc_id=doc_id, source=doc.source, status=status, num_chunks=len(chunks))

    def ingest_path(self, path: Path, root: Path | None = None) -> IngestResult:
        source = path.relative_to(root).as_posix() if root else path.name
        return self.ingest_bytes(source, path.read_bytes())

    def ingest_directory(self, directory: Path, recursive: bool = True) -> list[IngestResult]:
        if not directory.is_dir():
            raise NotADirectoryError(str(directory))
        pattern = "**/*" if recursive else "*"
        files = sorted(
            p
            for p in directory.glob(pattern)
            if p.is_file()
            and p.suffix.lower() in SUPPORTED_EXTENSIONS
            and not any(part.startswith(".") for part in p.relative_to(directory).parts)
        )
        return [self.ingest_path(p, root=directory) for p in files]

    def delete(self, doc_id: str) -> bool:
        with self._write_lock:
            if self.docstore.get_document(doc_id) is None:
                return False
            self.vector_store.delete_document(doc_id)
            return self.docstore.delete_document(doc_id)

    def reindex(self) -> int:
        """Re-embed every chunk from the doc store (after switching embedding model/store)."""
        with self._write_lock:
            chunks = self.docstore.iter_chunks()
            self.vector_store.reset()
            for start in range(0, len(chunks), self.batch_size * 8):
                batch: list[Chunk] = chunks[start : start + self.batch_size * 8]
                vecs = self.embedder.embed_documents([c.contextual_text for c in batch])
                self.vector_store.upsert(batch, vecs)
        log.info("reindex_done", chunks=len(chunks))
        return len(chunks)
