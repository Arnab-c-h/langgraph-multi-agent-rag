"""Single-process SQLite persistence for local RAG metadata and checkpoints."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar
from uuid import UUID

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from rag_app.contracts import SourceType
from rag_app.ingestion.models import ChunkRecord, IngestionJob, SourceRecord

_T = TypeVar("_T")
_SCHEMA_VERSION = 2
_SCHEMA_V1_STATEMENTS = (
    "CREATE TABLE IF NOT EXISTS sources ("
    "id TEXT PRIMARY KEY, source_type TEXT NOT NULL, canonical_uri TEXT NOT NULL, title TEXT NOT NULL, "
    "content_hash TEXT NOT NULL, metadata TEXT NOT NULL, provenance TEXT NOT NULL, created_at TEXT NOT NULL, "
    "updated_at TEXT NOT NULL, UNIQUE(source_type, canonical_uri))",
    "CREATE TABLE IF NOT EXISTS chunks ("
    "id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE, "
    "chunk_index INTEGER NOT NULL, content TEXT NOT NULL, content_hash TEXT NOT NULL, metadata TEXT NOT NULL, "
    "provenance TEXT NOT NULL, embedding TEXT, UNIQUE(source_id, chunk_index))",
    "CREATE TABLE IF NOT EXISTS ingestion_jobs ("
    "id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE, status TEXT NOT NULL, "
    "error_code TEXT, metadata TEXT NOT NULL, provenance TEXT NOT NULL, created_at TEXT NOT NULL, completed_at TEXT)",
    "CREATE TABLE IF NOT EXISTS threads ("
    "id TEXT PRIMARY KEY, title TEXT NOT NULL, metadata TEXT NOT NULL, created_at TEXT NOT NULL, "
    "updated_at TEXT NOT NULL)",
    "CREATE INDEX IF NOT EXISTS chunks_source_idx ON chunks(source_id, chunk_index)",
    "CREATE INDEX IF NOT EXISTS ingestion_jobs_source_idx ON ingestion_jobs(source_id, created_at)",
)
_SCHEMA_V2_STATEMENTS = (
    "CREATE TABLE ingestion_jobs_v2 ("
    "id TEXT PRIMARY KEY, source_id TEXT NOT NULL, status TEXT NOT NULL, error_code TEXT, metadata TEXT NOT NULL, "
    "provenance TEXT NOT NULL, created_at TEXT NOT NULL, completed_at TEXT)",
    "INSERT INTO ingestion_jobs_v2 SELECT id, source_id, status, error_code, metadata, provenance, created_at, "
    "completed_at FROM ingestion_jobs",
    "DROP TABLE ingestion_jobs",
    "ALTER TABLE ingestion_jobs_v2 RENAME TO ingestion_jobs",
    "CREATE INDEX ingestion_jobs_source_idx ON ingestion_jobs(source_id, created_at)",
)


class SqliteDatabase:
    """One local SQLite connection with serialized transactional writes."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir.expanduser().resolve()
        self.path = self.data_dir / "rag.sqlite3"
        self._connection: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Create the data directory, configure SQLite, and apply schema migrations."""
        if self._connection is not None:
            return
        self.data_dir.mkdir(parents=True, exist_ok=True)
        connection = await aiosqlite.connect(self.path)
        connection.row_factory = aiosqlite.Row
        self._connection = connection
        await connection.execute("PRAGMA foreign_keys = ON")
        await connection.execute("PRAGMA journal_mode = WAL")
        await connection.execute("PRAGMA busy_timeout = 30000")
        await connection.commit()
        await self._migrate()

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def __aenter__(self) -> SqliteDatabase:
        await self.initialize()
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.close()

    async def write(self, operation: Callable[[aiosqlite.Connection], Awaitable[_T]]) -> _T:
        """Run a write atomically so concurrent append/replace operations cannot interleave."""
        connection = self._require_connection()
        async with self._lock:
            try:
                await connection.execute("BEGIN IMMEDIATE")
                result = await operation(connection)
                await connection.commit()
            except BaseException:
                if connection.in_transaction:
                    try:
                        await connection.rollback()
                    except BaseException:
                        pass
                raise
            return result

    async def fetchone(self, query: str, parameters: tuple[object, ...] = ()) -> aiosqlite.Row | None:
        return await self._read_one(query, parameters)

    async def fetchall(self, query: str, parameters: tuple[object, ...] = ()) -> list[aiosqlite.Row]:
        connection = self._require_connection()
        async with self._lock:
            cursor = await connection.execute(query, parameters)
            try:
                return list(await cursor.fetchall())
            finally:
                await cursor.close()

    async def scalar(self, query: str, parameters: tuple[object, ...] = ()) -> Any:
        row = await self._read_one(query, parameters)
        return row[0] if row is not None else None

    async def _read_one(self, query: str, parameters: tuple[object, ...]) -> aiosqlite.Row | None:
        connection = self._require_connection()
        async with self._lock:
            cursor = await connection.execute(query, parameters)
            try:
                return await cursor.fetchone()
            finally:
                await cursor.close()

    async def _migrate(self) -> None:
        async def apply(connection: aiosqlite.Connection) -> None:
            await connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            cursor = await connection.execute("SELECT version FROM schema_migrations")
            try:
                applied = {int(row[0]) for row in await cursor.fetchall()}
            finally:
                await cursor.close()
            migrations = ((1, _SCHEMA_V1_STATEMENTS), (2, _SCHEMA_V2_STATEMENTS))
            for version, statements in migrations:
                if version in applied:
                    continue
                for statement in statements:
                    await connection.execute(statement)
                await connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (version, _timestamp()),
                )

        await self.write(apply)

    def _require_connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError("SqliteDatabase.initialize() must be called before use")
        return self._connection


class SqliteSourceRepository:
    """Authoritative local source, chunk, and ingestion-job records."""

    def __init__(self, database: SqliteDatabase) -> None:
        self._database = database

    async def find_by_identity(self, source_type: SourceType, canonical_uri: str) -> SourceRecord | None:
        row = await self._database.fetchone(
            "SELECT id, source_type, canonical_uri, title, content_hash, metadata, provenance, created_at, updated_at "
            "FROM sources WHERE source_type = ? AND canonical_uri = ?",
            (source_type.value, canonical_uri),
        )
        return _source_from_row(row) if row is not None else None

    async def replace(self, source: SourceRecord, chunks: list[ChunkRecord]) -> None:
        _validate_embeddings([chunk.embedding for chunk in chunks])

        async def operation(connection: aiosqlite.Connection) -> None:
            cursor = await connection.execute(
                "SELECT id, created_at FROM sources WHERE source_type = ? AND canonical_uri = ?",
                (source.source_type.value, source.canonical_uri),
            )
            try:
                existing = await cursor.fetchone()
            finally:
                await cursor.close()
            actual_source_id = str(existing[0]) if existing is not None else str(source.id)
            if existing is None:
                await connection.execute(
                    "INSERT INTO sources (id, source_type, canonical_uri, title, content_hash, metadata, provenance, "
                    "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    _source_values(source, actual_source_id),
                )
            else:
                await connection.execute(
                    "UPDATE sources SET title = ?, content_hash = ?, metadata = ?, provenance = ?, updated_at = ? "
                    "WHERE id = ?",
                    (
                        source.title,
                        source.content_hash,
                        _json(source.metadata),
                        _json(source.provenance),
                        _timestamp(source.updated_at),
                        actual_source_id,
                    ),
                )
            await connection.execute("DELETE FROM chunks WHERE source_id = ?", (actual_source_id,))
            for chunk in chunks:
                await connection.execute(
                    "INSERT INTO chunks (id, source_id, chunk_index, content, content_hash, metadata, provenance, "
                    "embedding) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(chunk.id),
                        actual_source_id,
                        chunk.chunk_index,
                        chunk.content,
                        chunk.content_hash,
                        _json(chunk.metadata),
                        _json(chunk.provenance),
                        None,
                    ),
                )

        await self._database.write(operation)

    async def list_sources(self) -> list[SourceRecord]:
        rows = await self._database.fetchall(
            "SELECT id, source_type, canonical_uri, title, content_hash, metadata, provenance, created_at, updated_at "
            "FROM sources ORDER BY updated_at DESC, id ASC"
        )
        return [_source_from_row(row) for row in rows]

    async def list_chunks(self, source_id: UUID) -> list[ChunkRecord]:
        rows = await self._database.fetchall(
            "SELECT id, source_id, chunk_index, content, content_hash, metadata, provenance, embedding "
            "FROM chunks WHERE source_id = ? ORDER BY chunk_index ASC, id ASC",
            (str(source_id),),
        )
        return [_chunk_from_row(row) for row in rows]

    async def get_chunks_by_ids(self, chunk_ids: list[UUID]) -> list[ChunkRecord]:
        """Load only requested chunks, preserving first-requested ID order."""
        ordered_ids = list(dict.fromkeys(chunk_ids))
        if not ordered_ids:
            return []
        placeholders = ", ".join("?" for _chunk_id in ordered_ids)
        rows = await self._database.fetchall(
            "SELECT id, source_id, chunk_index, content, content_hash, metadata, provenance, embedding "
            f"FROM chunks WHERE id IN ({placeholders})",
            tuple(str(chunk_id) for chunk_id in ordered_ids),
        )
        chunks_by_id = {chunk.id: chunk for chunk in (_chunk_from_row(row) for row in rows)}
        return [chunks_by_id[chunk_id] for chunk_id in ordered_ids if chunk_id in chunks_by_id]

    async def delete(self, source_id: UUID) -> None:
        await self._database.write(
            lambda connection: _execute(connection, "DELETE FROM sources WHERE id = ?", (str(source_id),))
        )

    async def start_job(self, job: IngestionJob) -> None:
        async def operation(connection: aiosqlite.Connection) -> None:
            await connection.execute(
                "INSERT INTO ingestion_jobs (id, source_id, status, error_code, metadata, provenance, created_at, "
                "completed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET status = excluded.status, error_code = excluded.error_code, "
                "metadata = excluded.metadata, provenance = excluded.provenance, completed_at = excluded.completed_at",
                _job_values(job),
            )

        await self._database.write(operation)

    async def complete_job(self, job: IngestionJob) -> None:
        async def operation(connection: aiosqlite.Connection) -> None:
            await connection.execute(
                "UPDATE ingestion_jobs SET status = ?, error_code = ?, metadata = ?, provenance = ?, completed_at = ? "
                "WHERE id = ?",
                (
                    job.status,
                    job.error_code,
                    _json(job.metadata),
                    _json(job.provenance),
                    _timestamp(job.completed_at) if job.completed_at is not None else _timestamp(),
                    str(job.id),
                ),
            )

        await self._database.write(operation)


@asynccontextmanager
async def sqlite_checkpointer(data_dir: Path) -> AsyncIterator[AsyncSqliteSaver]:
    """Open a checkpoint saver whose lifecycle closes its SQLite connection."""
    root = data_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(root / "checkpoints.sqlite3")) as saver:
        yield saver


async def _execute(connection: aiosqlite.Connection, query: str, parameters: tuple[object, ...]) -> None:
    cursor = await connection.execute(query, parameters)
    await cursor.close()


def _validate_embeddings(embeddings: list[list[float] | None]) -> None:
    if any(embedding is not None and len(embedding) != 768 for embedding in embeddings):
        raise ValueError("All embeddings must have exactly 768 dimensions")


def _source_values(source: SourceRecord, source_id: str) -> tuple[object, ...]:
    return (
        source_id,
        source.source_type.value,
        source.canonical_uri,
        source.title,
        source.content_hash,
        _json(source.metadata),
        _json(source.provenance),
        _timestamp(source.created_at),
        _timestamp(source.updated_at),
    )


def _job_values(job: IngestionJob) -> tuple[object, ...]:
    return (
        str(job.id),
        str(job.source_id),
        job.status,
        job.error_code,
        _json(job.metadata),
        _json(job.provenance),
        _timestamp(job.created_at),
        _timestamp(job.completed_at) if job.completed_at is not None else None,
    )


def _source_from_row(row: aiosqlite.Row) -> SourceRecord:
    return SourceRecord(
        id=row["id"],
        source_type=row["source_type"],
        canonical_uri=row["canonical_uri"],
        title=row["title"],
        content_hash=row["content_hash"],
        metadata=_json_object(row["metadata"]),
        provenance=_json_object(row["provenance"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _chunk_from_row(row: aiosqlite.Row) -> ChunkRecord:
    embedding = _json_vector(row["embedding"]) if row["embedding"] is not None else None
    return ChunkRecord(
        id=row["id"],
        source_id=row["source_id"],
        chunk_index=row["chunk_index"],
        content=row["content"],
        content_hash=row["content_hash"],
        metadata=_json_object(row["metadata"]),
        provenance=_json_object(row["provenance"]),
        embedding=embedding,
    )


def _timestamp(value: datetime | None = None) -> str:
    return (value or datetime.now(UTC)).astimezone(UTC).isoformat()


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_object(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Stored JSON must be an object")
    return parsed


def _json_vector(value: str) -> list[float]:
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not all(isinstance(item, (int, float)) for item in parsed):
        raise ValueError("Stored embedding must be a numeric array")
    return [float(item) for item in parsed]
