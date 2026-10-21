"""Lazy local resource composition for the modular monolith."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from rag_app.a2a.store import create_sqlite_task_store, dispose_task_store_engine
from rag_app.api.app import AppServices
from rag_app.config import Settings
from rag_app.contracts import SourceType
from rag_app.errors import AppError, ErrorCode
from rag_app.ingestion.embeddings import GeminiEmbeddingAdapter
from rag_app.ingestion.models import IngestionResult, SourceRecord
from rag_app.ingestion.protocols import SourceRepository
from rag_app.ingestion.service import IngestionService
from rag_app.retrieval.chroma import ChromaRetriever
from rag_app.retrieval.citations import StrictCitationValidator
from rag_app.retrieval.grading import GeminiEvidenceGrader
from rag_app.retrieval.service import StrictRagService
from rag_app.storage.chroma import ChromaIndex, LocalSourceRepository
from rag_app.storage.sqlite import SqliteDatabase, SqliteSourceRepository, sqlite_checkpointer
from rag_app.threads import SqliteThreadStore
from rag_app.workflows.model import GeminiWorkflowModel
from rag_app.workflows.registry import WorkflowRegistry
from rag_app.workflows.runner import WorkflowRunner

_READY = {
    "sqlite": True,
    "chroma": True,
    "checkpoints": True,
    "sources": True,
    "runner": True,
    "a2a_store": True,
}


class FullSourceFacade:
    def __init__(
        self,
        repository: SourceRepository,
        ingestion: IngestionService,
        pdf_loader: Any,
        web_loader: Any,
        github_loader: Any,
    ) -> None:
        self._repository = repository
        self._ingestion = ingestion
        self._pdf_loader = pdf_loader
        self._web_loader = web_loader
        self._github_loader = github_loader

    async def list_sources(self) -> list[SourceRecord]:
        return await self._repository.list_sources()

    async def ingest_pdf(self, filename: str, data: bytes) -> IngestionResult:
        return await self._ingestion.ingest(await self._pdf_loader.load(filename, data))

    async def ingest_web(self, url: str) -> IngestionResult:
        return await self._ingestion.ingest(await self._web_loader.load(url))

    async def ingest_github(self, url: str) -> IngestionResult:
        return await self._ingestion.ingest(await self._github_loader.load(url))

    async def reindex(self, source_id: UUID) -> IngestionResult:
        source = next((item for item in await self.list_sources() if item.id == source_id), None)
        if source is None:
            raise AppError(ErrorCode.INVALID_SOURCE, "Unknown source")
        if source.source_type is SourceType.WEB:
            return await self.ingest_web(source.canonical_uri)
        if source.source_type is SourceType.GITHUB:
            return await self.ingest_github(source.canonical_uri)
        raise AppError(ErrorCode.INVALID_SOURCE, "PDF reindex requires a new upload")

    async def delete(self, source_id: UUID) -> None:
        await self._repository.delete(source_id)


@asynccontextmanager
async def compose_runtime(settings: Settings) -> AsyncIterator[AppServices]:
    """Open local resources in lifespan and close checkpoint, A2A, then SQLite."""

    if settings.gemini_api_key is None:
        raise RuntimeError("GEMINI_API_KEY is required at runtime")
    database = SqliteDatabase(settings.rag_data_dir)
    checkpoint_context: AbstractAsyncContextManager[Any] | None = None
    checkpoint_entered = False
    a2a_engine: AsyncEngine | None = None
    try:
        await database.initialize()
        checkpoint_context = sqlite_checkpointer(settings.rag_data_dir)
        checkpointer = await checkpoint_context.__aenter__()
        checkpoint_entered = True

        task_store, a2a_engine = create_sqlite_task_store(settings.rag_data_dir)
        await task_store.initialize()
        async with a2a_engine.connect() as connection:
            await connection.execute(text("SELECT 1 FROM tasks LIMIT 1"))

        api_key = settings.gemini_api_key.get_secret_value()
        embeddings = GeminiEmbeddingAdapter(api_key)
        index = ChromaIndex(settings.rag_data_dir / "chroma", embeddings)
        repository = LocalSourceRepository(SqliteSourceRepository(database), index)
        await repository.reconcile()
        citation_validator = StrictCitationValidator()
        rag = StrictRagService(
            ChromaRetriever(index.vector_store, repository),
            GeminiEvidenceGrader.from_settings(settings),
            citation_validator,
        )
        registry = WorkflowRegistry(
            rag=rag,
            model=GeminiWorkflowModel.from_settings(settings),
            citation_validator=citation_validator,
            settings=settings,
            checkpointer=checkpointer,
        )
        from rag_app.ingestion.loaders import GitHubLoader, PdfLoader, WebLoader

        sources = FullSourceFacade(
            repository,
            IngestionService(repository, embeddings),
            PdfLoader(max_bytes=settings.pdf_max_bytes, max_pages=settings.pdf_max_pages),
            WebLoader(max_bytes=settings.web_max_bytes, timeout_seconds=settings.web_timeout_seconds),
            GitHubLoader(max_files=settings.github_max_files, max_bytes=settings.github_max_bytes),
        )
        yield AppServices(
            runner=WorkflowRunner(registry, thread_store=SqliteThreadStore(database)),
            sources=sources,
            a2a_task_store=task_store,
            a2a_engine=a2a_engine,
            readiness=dict(_READY),
        )
    finally:
        try:
            if checkpoint_context is not None and checkpoint_entered:
                await checkpoint_context.__aexit__(None, None, None)
        finally:
            try:
                if a2a_engine is not None:
                    await dispose_task_store_engine(a2a_engine)
            finally:
                await database.close()
