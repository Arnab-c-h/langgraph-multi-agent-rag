from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from uuid import uuid4

import pytest

from rag_app.config import Settings
from rag_app.contracts import ReasoningMode, SourceType
from rag_app.ingestion.models import ChunkRecord, IngestionJob, SourceRecord
from rag_app.threads import ThreadTurn


def _source(*, source_id: object | None = None) -> SourceRecord:
    return SourceRecord(
        id=source_id or uuid4(),
        source_type=SourceType.WEB,
        canonical_uri="https://example.com/docs",
        title="Example documentation",
        content_hash="a" * 64,
        metadata={"language": "en"},
        provenance={"fetched_from": "https://example.com/docs"},
    )


def _chunk(source_id: object, index: int = 0) -> ChunkRecord:
    return ChunkRecord(
        id=uuid4(),
        source_id=source_id,  # type: ignore[arg-type]
        chunk_index=index,
        content=f"Evidence {index}",
        content_hash=f"{index:x}" * 64,
        metadata={"page": index + 1},
        provenance={"section": f"Section {index + 1}"},
    )


def test_rag_data_dir_resolves_without_creating_the_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configured = tmp_path / "nested" / "rag-data"
    monkeypatch.setenv("RAG_DATA_DIR", str(configured))

    settings = Settings(app_profile="test")

    assert settings.rag_data_dir == configured.resolve()
    assert not configured.exists()


def test_blank_rag_data_dir_uses_the_platform_default_without_creating_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("RAG_DATA_DIR", "   ")
    monkeypatch.setattr("rag_app.config.user_data_path", lambda *_args, **_kwargs: tmp_path / "platform-default")

    settings = Settings(app_profile="test")

    assert settings.rag_data_dir == (tmp_path / "platform-default").resolve()
    assert not settings.rag_data_dir.exists()


@pytest.mark.asyncio
async def test_sqlite_initialization_creates_versioned_schema_and_required_pragmas(tmp_path: Path) -> None:
    from rag_app.storage.sqlite import SqliteDatabase

    database = SqliteDatabase(tmp_path)
    await database.initialize()
    try:
        assert database.path == tmp_path / "rag.sqlite3"
        assert (await database.scalar("PRAGMA foreign_keys")) == 1
        assert (await database.scalar("PRAGMA journal_mode")).lower() == "wal"
        assert (await database.scalar("PRAGMA busy_timeout")) == 30_000
        assert await database.scalar("SELECT version FROM schema_migrations") == 1
        tables = await database.fetchall(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        )
        assert {row[0] for row in tables} >= {"schema_migrations", "sources", "chunks", "ingestion_jobs", "threads"}
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_sqlite_initialization_recovers_after_an_interrupted_schema_migration(tmp_path: Path) -> None:
    """A restart must finish schema v1 after its first DDL statement was committed alone."""
    database_path = tmp_path / "rag.sqlite3"
    with sqlite3.connect(database_path) as interrupted:
        interrupted.execute(
            "CREATE TABLE sources ("
            "id TEXT PRIMARY KEY, source_type TEXT NOT NULL, canonical_uri TEXT NOT NULL, title TEXT NOT NULL, "
            "content_hash TEXT NOT NULL, metadata TEXT NOT NULL, provenance TEXT NOT NULL, created_at TEXT NOT NULL, "
            "updated_at TEXT NOT NULL, UNIQUE(source_type, canonical_uri))"
        )

    from rag_app.storage.sqlite import SqliteDatabase

    database = SqliteDatabase(tmp_path)
    await database.initialize()
    try:
        assert await database.scalar("SELECT version FROM schema_migrations WHERE version = 1") == 1
        tables = await database.fetchall("SELECT name FROM sqlite_master WHERE type = 'table'")
        assert {row[0] for row in tables} >= {"schema_migrations", "sources", "chunks", "ingestion_jobs", "threads"}
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_sqlite_migrates_v1_jobs_to_allow_failures_before_source_activation(tmp_path: Path) -> None:
    import sqlite3

    from rag_app.storage.sqlite import SqliteDatabase, SqliteSourceRepository

    path = tmp_path / "rag.sqlite3"
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);"
            "INSERT INTO schema_migrations VALUES (1, '2026-08-18T00:00:00Z');"
            "CREATE TABLE sources (id TEXT PRIMARY KEY, source_type TEXT NOT NULL, canonical_uri TEXT NOT NULL, "
            "title TEXT NOT NULL, content_hash TEXT NOT NULL, metadata TEXT NOT NULL, provenance TEXT NOT NULL, "
            "created_at TEXT NOT NULL, updated_at TEXT NOT NULL, UNIQUE(source_type, canonical_uri));"
            "CREATE TABLE ingestion_jobs (id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id) "
            "ON DELETE CASCADE, status TEXT NOT NULL, error_code TEXT, metadata TEXT NOT NULL, provenance TEXT "
            "NOT NULL, created_at TEXT NOT NULL, completed_at TEXT);"
            "CREATE INDEX ingestion_jobs_source_idx ON ingestion_jobs(source_id, created_at);"
        )
        connection.commit()
    finally:
        connection.close()

    database = SqliteDatabase(tmp_path)
    await database.initialize()
    try:
        job = IngestionJob(source_id=uuid4(), status="started")
        await SqliteSourceRepository(database).start_job(job)

        assert await database.scalar("SELECT status FROM ingestion_jobs WHERE id = ?", (str(job.id),)) == "started"
        versions = await database.fetchall("SELECT version FROM schema_migrations ORDER BY version")
        assert [row[0] for row in versions] == [1, 2]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_sqlite_write_rolls_back_and_remains_usable_after_a_commit_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from rag_app.storage.sqlite import SqliteDatabase

    database = SqliteDatabase(tmp_path)
    await database.initialize()
    try:
        connection = database._connection
        assert connection is not None
        original_commit = connection.commit

        async def fail_commit_once() -> None:
            monkeypatch.setattr(connection, "commit", original_commit)
            raise RuntimeError("simulated commit failure")

        async def create_aborted_table(active_connection: object) -> None:
            await active_connection.execute("CREATE TABLE aborted_write (value TEXT)")  # type: ignore[union-attr]

        async def create_committed_table(active_connection: object) -> None:
            await active_connection.execute("CREATE TABLE committed_write (value TEXT)")  # type: ignore[union-attr]

        monkeypatch.setattr(connection, "commit", fail_commit_once)
        with pytest.raises(RuntimeError, match="simulated commit failure"):
            await database.write(create_aborted_table)

        await database.write(create_committed_table)
        assert await database.scalar("SELECT name FROM sqlite_master WHERE name = 'aborted_write'") is None
        assert (
            await database.scalar("SELECT name FROM sqlite_master WHERE name = 'committed_write'")
            == "committed_write"
        )
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_sqlite_source_repository_replaces_identity_rows_and_cascades_chunks(tmp_path: Path) -> None:
    from rag_app.storage.sqlite import SqliteDatabase, SqliteSourceRepository

    database = SqliteDatabase(tmp_path)
    await database.initialize()
    try:
        repository = SqliteSourceRepository(database)
        source = _source()
        original_chunk = _chunk(source.id)
        await repository.replace(source, [original_chunk])

        replacement = _source(source_id=uuid4())
        replacement.title = "Replacement title"
        replacement_chunk = _chunk(replacement.id, 1)
        await repository.replace(replacement, [replacement_chunk])

        found = await repository.find_by_identity(SourceType.WEB, "https://example.com/docs")
        assert found is not None
        assert found.id == source.id
        assert found.title == "Replacement title"
        assert [chunk.chunk_index for chunk in await repository.list_chunks(found.id)] == [1]

        await repository.delete(found.id)
        assert await repository.list_chunks(found.id) == []
        assert await database.scalar("SELECT COUNT(*) FROM chunks") == 0
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_sqlite_source_repository_persists_job_lifecycle_and_validates_embeddings(tmp_path: Path) -> None:
    from rag_app.storage.sqlite import SqliteDatabase, SqliteSourceRepository

    database = SqliteDatabase(tmp_path)
    await database.initialize()
    try:
        repository = SqliteSourceRepository(database)
        source = _source()
        await repository.replace(source, [])
        job = IngestionJob(source_id=source.id, status="started", metadata={"phase": "load"})
        await repository.start_job(job)
        await repository.complete_job(job.model_copy(update={"status": "completed", "metadata": {"phase": "done"}}))

        row = await database.fetchone(
            "SELECT status, metadata, completed_at FROM ingestion_jobs WHERE id = ?", (str(job.id),)
        )
        assert row[0] == "completed"
        assert json.loads(row[1]) == {"phase": "done"}
        assert row[2] is not None
        with pytest.raises(ValueError, match="768"):
            await repository.replace(source, [_chunk(source.id).model_copy(update={"embedding": [0.0] * 767})])
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_sqlite_thread_store_redacts_bounds_and_serializes_concurrent_appends(tmp_path: Path) -> None:
    from rag_app.storage.sqlite import SqliteDatabase
    from rag_app.threads import SqliteThreadStore

    database = SqliteDatabase(tmp_path)
    await database.initialize()
    try:
        store = SqliteThreadStore(database)
        thread_id = uuid4()

        await asyncio.gather(
            *(
                store.append(
                    thread_id,
                    ThreadTurn(
                        question="Authorization: Bearer private-value" if index == 0 else f"question {index}",
                        answer="postgresql://user:" + "private@host/database" if index == 0 else f"answer {index}",
                        mode=ReasoningMode.STANDARD,
                        run_id=uuid4(),
                    ),
                )
                for index in range(12)
            )
        )

        turns = await store.load(thread_id)
        assert len(turns) == 10
        assert all("private-value" not in turn.question and "private@" not in turn.answer for turn in turns)
        assert {turn.question for turn in turns} >= {"question 2", "question 11"}
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_sqlite_thread_store_persists_turns_across_database_restart(tmp_path: Path) -> None:
    from rag_app.storage.sqlite import SqliteDatabase
    from rag_app.threads import SqliteThreadStore

    thread_id = uuid4()
    turn = ThreadTurn(question="persist me", answer="grounded answer", mode=ReasoningMode.REACT, run_id=uuid4())
    database = SqliteDatabase(tmp_path)
    await database.initialize()
    try:
        await SqliteThreadStore(database).append(thread_id, turn)
    finally:
        await database.close()

    restarted = SqliteDatabase(tmp_path)
    await restarted.initialize()
    try:
        assert await SqliteThreadStore(restarted).load(thread_id) == [turn]
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_sqlite_a2a_store_persists_tasks_after_engine_restart(tmp_path: Path) -> None:
    from a2a.server.context import ServerCallContext
    from a2a.types import Message, Part, Role, Task, TaskStatus

    from rag_app.a2a.store import create_sqlite_task_store, dispose_task_store_engine

    task_id = str(uuid4())
    context = ServerCallContext()
    store, engine = create_sqlite_task_store(tmp_path)
    try:
        await store.save(
            Task(id=task_id, status=TaskStatus(), history=[Message(role=Role.ROLE_USER, parts=[Part(text="Hello")])]),
            context,
        )
    finally:
        await dispose_task_store_engine(engine)

    restarted_store, restarted_engine = create_sqlite_task_store(tmp_path)
    try:
        recovered = await restarted_store.get(task_id, context)
    finally:
        await dispose_task_store_engine(restarted_engine)

    assert recovered is not None
    assert recovered.id == task_id


@pytest.mark.asyncio
async def test_sqlite_checkpoint_factory_recovers_state_after_restart(tmp_path: Path) -> None:
    from rag_app.storage.sqlite import sqlite_checkpointer

    config = {"configurable": {"thread_id": f"{uuid4()}:standard", "checkpoint_ns": ""}}
    checkpoint = {
        "v": 1,
        "id": str(uuid4()),
        "ts": "2026-08-18T00:00:00+00:00",
        "channel_values": {"answer": "saved"},
        "channel_versions": {},
        "versions_seen": {},
    }
    metadata = {"source": "loop", "step": 1, "parents": {}}
    async with sqlite_checkpointer(tmp_path) as saver:
        await saver.aput(config, checkpoint, metadata, {})

    async with sqlite_checkpointer(tmp_path) as restarted_saver:
        recovered = await restarted_saver.aget_tuple(config)

    assert recovered is not None
    assert recovered.checkpoint["channel_values"] == {"answer": "saved"}
