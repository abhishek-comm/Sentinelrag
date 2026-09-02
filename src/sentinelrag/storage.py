import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .config import settings


def _db_path() -> str:
    return settings.database_url.removeprefix("sqlite:///")


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    path = Path(_db_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def initialize() -> None:
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    with connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, filename TEXT NOT NULL, page_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY, document_id TEXT NOT NULL, page INTEGER NOT NULL,
                ordinal INTEGER NOT NULL, text TEXT NOT NULL, safe_text TEXT NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(id)
            );
            CREATE TABLE IF NOT EXISTS traces (
                id TEXT PRIMARY KEY, question TEXT NOT NULL, status TEXT NOT NULL,
                confidence REAL NOT NULL, answer TEXT NOT NULL, citations_json TEXT NOT NULL,
                latency_ms INTEGER NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT NOT NULL, rating TEXT NOT NULL,
                comment TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
            """
        )
        _add_column_if_missing(conn, "documents", "content_hash", "TEXT")
        _add_column_if_missing(conn, "chunks", "embedding_json", "TEXT")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash)")
        _backfill_embeddings(conn)


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _backfill_embeddings(conn: sqlite3.Connection) -> None:
    """Upgrade chunks created before vector retrieval was introduced."""
    rows = conn.execute("SELECT id, safe_text FROM chunks WHERE embedding_json IS NULL").fetchall()
    if not rows:
        return
    from .embeddings import local_embed

    conn.executemany("UPDATE chunks SET embedding_json = ? WHERE id = ?",
                     [(json.dumps(local_embed(row["safe_text"])), row["id"]) for row in rows])


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


def save_trace(payload: dict) -> None:
    with connection() as conn:
        conn.execute(
            "INSERT INTO traces VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (payload["id"], payload["question"], payload["status"], payload["confidence"],
             payload["answer"], json.dumps(payload["citations"]), payload["latency_ms"], utcnow()),
        )
