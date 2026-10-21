from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from rag_app import composition
from rag_app.config import Settings
from rag_app.contracts import ReasoningMode


class FakeEmbeddings:
    dimensions = 768

    def __init__(self, _api_key: str) -> None:
        pass

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * 767 for _ in texts]

    def embed_query(self, _text: str) -> list[float]:
        return [1.0] + [0.0] * 767

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


@pytest.mark.asyncio
async def test_local_runtime_initializes_all_nine_graphs_and_ingestion_facades_from_local_data_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(composition, "GeminiEmbeddingAdapter", FakeEmbeddings)
    monkeypatch.setattr(composition.GeminiEvidenceGrader, "from_settings", lambda _settings: object())
    monkeypatch.setattr(composition.GeminiWorkflowModel, "from_settings", lambda _settings: object())
    settings = Settings(
        app_profile="full",
        rag_data_dir=tmp_path,
        gemini_api_key="test-gemini",
        admin_token="test-admin",
    )

    async with composition.compose_runtime(settings) as services:
        assert services.readiness == {
            "sqlite": True,
            "chroma": True,
            "checkpoints": True,
            "sources": True,
            "runner": True,
            "a2a_store": True,
        }
        assert all(services.runner._registry.graph_for(mode) is not None for mode in ReasoningMode)  # type: ignore[attr-defined]
        assert all(
            callable(getattr(services.sources, name))
            for name in ("ingest_pdf", "ingest_web", "ingest_github", "reindex", "delete")
        )
        database = services.sources._repository._sqlite._database  # type: ignore[attr-defined]
        assert database._connection is not None
        assert (tmp_path / "rag.sqlite3").exists()
        assert (tmp_path / "chroma").is_dir()
        assert (tmp_path / "checkpoints.sqlite3").exists()

    assert database._connection is None
    (tmp_path / "checkpoints.sqlite3").rename(tmp_path / "closed-checkpoints.sqlite3")
    (tmp_path / "rag.sqlite3").rename(tmp_path / "closed-rag.sqlite3")


def test_local_runtime_credentials_require_only_gemini_and_admin() -> None:
    Settings(
        app_profile="full",
        gemini_api_key="test-gemini",
        admin_token="test-admin",
    ).validate_runtime_credentials()


@pytest.mark.asyncio
async def test_local_runtime_closes_checkpoint_then_a2a_then_sqlite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    original_checkpointer = composition.sqlite_checkpointer
    original_dispose = composition.dispose_task_store_engine
    original_database_close = composition.SqliteDatabase.close

    @asynccontextmanager
    async def tracked_checkpointer(data_dir: Path) -> Any:
        async with original_checkpointer(data_dir) as saver:
            yield saver
        events.append("checkpoints")

    async def tracked_dispose(engine: Any) -> None:
        await original_dispose(engine)
        events.append("a2a_store")

    async def tracked_database_close(database: Any) -> None:
        await original_database_close(database)
        events.append("sqlite")

    monkeypatch.setattr(composition, "sqlite_checkpointer", tracked_checkpointer)
    monkeypatch.setattr(composition, "dispose_task_store_engine", tracked_dispose)
    monkeypatch.setattr(composition.SqliteDatabase, "close", tracked_database_close)
    monkeypatch.setattr(composition, "GeminiEmbeddingAdapter", FakeEmbeddings)
    monkeypatch.setattr(composition.GeminiEvidenceGrader, "from_settings", lambda _settings: object())
    monkeypatch.setattr(composition.GeminiWorkflowModel, "from_settings", lambda _settings: object())
    settings = Settings(
        app_profile="full",
        rag_data_dir=tmp_path,
        gemini_api_key="test-gemini",
        admin_token="test-admin",
    )

    async with composition.compose_runtime(settings):
        assert events == []

    assert events == ["checkpoints", "a2a_store", "sqlite"]


@pytest.mark.asyncio
async def test_local_runtime_attempts_later_closes_when_checkpoint_close_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    checkpoint_error = RuntimeError("checkpoint close failed")
    original_checkpointer = composition.sqlite_checkpointer
    original_dispose = composition.dispose_task_store_engine
    original_database_close = composition.SqliteDatabase.close

    @asynccontextmanager
    async def failing_checkpointer(data_dir: Path) -> Any:
        async with original_checkpointer(data_dir) as saver:
            yield saver
        events.append("checkpoints")
        raise checkpoint_error

    async def tracked_dispose(engine: Any) -> None:
        await original_dispose(engine)
        events.append("a2a_store")

    async def tracked_database_close(database: Any) -> None:
        await original_database_close(database)
        events.append("sqlite")

    monkeypatch.setattr(composition, "sqlite_checkpointer", failing_checkpointer)
    monkeypatch.setattr(composition, "dispose_task_store_engine", tracked_dispose)
    monkeypatch.setattr(composition.SqliteDatabase, "close", tracked_database_close)
    monkeypatch.setattr(composition, "GeminiEmbeddingAdapter", FakeEmbeddings)
    monkeypatch.setattr(composition.GeminiEvidenceGrader, "from_settings", lambda _settings: object())
    monkeypatch.setattr(composition.GeminiWorkflowModel, "from_settings", lambda _settings: object())
    settings = Settings(
        app_profile="full",
        rag_data_dir=tmp_path,
        gemini_api_key="test-gemini",
        admin_token="test-admin",
    )

    with pytest.raises(RuntimeError, match="checkpoint close failed") as exc_info:
        async with composition.compose_runtime(settings):
            pass

    assert exc_info.value is checkpoint_error
    assert events == ["checkpoints", "a2a_store", "sqlite"]


@pytest.mark.asyncio
async def test_local_runtime_attempts_sqlite_close_when_a2a_disposal_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    a2a_error = RuntimeError("A2A disposal failed")
    original_checkpointer = composition.sqlite_checkpointer
    original_dispose = composition.dispose_task_store_engine
    original_database_close = composition.SqliteDatabase.close

    @asynccontextmanager
    async def tracked_checkpointer(data_dir: Path) -> Any:
        async with original_checkpointer(data_dir) as saver:
            yield saver
        events.append("checkpoints")

    async def failing_dispose(engine: Any) -> None:
        await original_dispose(engine)
        events.append("a2a_store")
        raise a2a_error

    async def tracked_database_close(database: Any) -> None:
        await original_database_close(database)
        events.append("sqlite")

    monkeypatch.setattr(composition, "sqlite_checkpointer", tracked_checkpointer)
    monkeypatch.setattr(composition, "dispose_task_store_engine", failing_dispose)
    monkeypatch.setattr(composition.SqliteDatabase, "close", tracked_database_close)
    monkeypatch.setattr(composition, "GeminiEmbeddingAdapter", FakeEmbeddings)
    monkeypatch.setattr(composition.GeminiEvidenceGrader, "from_settings", lambda _settings: object())
    monkeypatch.setattr(composition.GeminiWorkflowModel, "from_settings", lambda _settings: object())
    settings = Settings(
        app_profile="full",
        rag_data_dir=tmp_path,
        gemini_api_key="test-gemini",
        admin_token="test-admin",
    )

    with pytest.raises(RuntimeError, match="A2A disposal failed") as exc_info:
        async with composition.compose_runtime(settings):
            pass

    assert exc_info.value is a2a_error
    assert events == ["checkpoints", "a2a_store", "sqlite"]
