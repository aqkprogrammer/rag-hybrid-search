"""SQLite document/chunk store - the system of record.

The vector store and BM25 index are *derived* indexes: both can be rebuilt from this store
(BM25 is rebuilt in-memory on startup; `hybrid-rag reindex` re-embeds everything). A monotonic
`revision` counter lets other components cheaply detect that the corpus changed.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hybrid_rag.schemas import Chunk, DocumentRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id        TEXT PRIMARY KEY,
    source        TEXT NOT NULL UNIQUE,
    title         TEXT NOT NULL,
    doc_type      TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    num_chunks    INTEGER NOT NULL,
    size_bytes    INTEGER NOT NULL,
    metadata      TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    idx          INTEGER NOT NULL,
    text         TEXT NOT NULL,
    heading_path TEXT NOT NULL DEFAULT '[]',
    token_count  INTEGER NOT NULL,
    page         INTEGER,
    metadata     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id, idx);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
INSERT OR IGNORE INTO meta(key, value) VALUES ('revision', '0');
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


class DocStore:
    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    # -- revision -----------------------------------------------------------------------------
    def revision(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key='revision'").fetchone()
        return int(row["value"])

    @staticmethod
    def _bump(conn: sqlite3.Connection) -> None:
        conn.execute("UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key='revision'")

    # -- documents ----------------------------------------------------------------------------
    def upsert_document(self, record: DocumentRecord, chunks: list[Chunk]) -> None:
        """Atomically replace a document and all of its chunks."""
        with self._tx() as conn:
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (record.doc_id,))
            conn.execute(
                "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    record.doc_id,
                    record.source,
                    record.title,
                    record.doc_type,
                    record.content_hash,
                    record.num_chunks,
                    record.size_bytes,
                    json.dumps(record.metadata, default=str),
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                ),
            )
            conn.executemany(
                "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?)",
                [
                    (
                        c.chunk_id,
                        c.doc_id,
                        c.index,
                        c.text,
                        json.dumps(c.heading_path),
                        c.token_count,
                        c.page,
                        json.dumps(c.metadata, default=str),
                    )
                    for c in chunks
                ],
            )
            self._bump(conn)

    def delete_document(self, doc_id: str) -> bool:
        with self._tx() as conn:
            cur = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            deleted = cur.rowcount > 0
            if deleted:
                self._bump(conn)
        return deleted

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        return self._row_to_doc(row) if row else None

    def get_by_source(self, source: str) -> DocumentRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM documents WHERE source = ?", (source,)
            ).fetchone()
        return self._row_to_doc(row) if row else None

    def get_by_hash(self, content_hash: str) -> DocumentRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM documents WHERE content_hash = ? LIMIT 1", (content_hash,)
            ).fetchone()
        return self._row_to_doc(row) if row else None

    def list_documents(self) -> list[DocumentRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM documents ORDER BY title COLLATE NOCASE"
            ).fetchall()
        return [self._row_to_doc(r) for r in rows]

    def count_documents(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    # -- chunks -------------------------------------------------------------------------------
    def count_chunks(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def get_chunks(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        out: dict[str, Chunk] = {}
        with self._lock:
            for start in range(0, len(chunk_ids), 500):
                batch = chunk_ids[start : start + 500]
                marks = ",".join("?" * len(batch))
                rows = self._conn.execute(
                    f"SELECT * FROM chunks WHERE chunk_id IN ({marks})", batch
                ).fetchall()
                out.update({r["chunk_id"]: self._row_to_chunk(r) for r in rows})
        return out

    def chunks_for_document(self, doc_id: str) -> list[Chunk]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM chunks WHERE doc_id = ? ORDER BY idx", (doc_id,)
            ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def iter_chunks(self) -> list[Chunk]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM chunks ORDER BY doc_id, idx").fetchall()
        return [self._row_to_chunk(r) for r in rows]

    # -- helpers ------------------------------------------------------------------------------
    @staticmethod
    def _row_to_doc(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            doc_id=row["doc_id"],
            source=row["source"],
            title=row["title"],
            doc_type=row["doc_type"],
            content_hash=row["content_hash"],
            num_chunks=row["num_chunks"],
            size_bytes=row["size_bytes"],
            metadata=json.loads(row["metadata"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        meta: dict[str, Any] = json.loads(row["metadata"])
        return Chunk(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            index=row["idx"],
            text=row["text"],
            heading_path=json.loads(row["heading_path"]),
            token_count=row["token_count"],
            page=row["page"],
            metadata=meta,
        )
