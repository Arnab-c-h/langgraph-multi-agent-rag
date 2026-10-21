"""Dependency-injection protocols for source ingestion."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from rag_app.contracts import SourceType
from rag_app.ingestion.models import ChunkRecord, IngestionJob, SourceRecord


class IngestionPersistenceError(RuntimeError):
    """A secret-safe storage failure category suitable for job records."""

    def __init__(self, error_code: str) -> None:
        super().__init__("Source persistence failed")
        self.error_code = error_code


class SourceRepository(Protocol):
    async def find_by_identity(self, source_type: SourceType, canonical_uri: str) -> SourceRecord | None: ...

    async def replace(self, source: SourceRecord, chunks: list[ChunkRecord]) -> None: ...

    async def list_chunks(self, source_id: UUID) -> list[ChunkRecord]: ...

    async def list_sources(self) -> list[SourceRecord]: ...

    async def delete(self, source_id: UUID) -> None: ...

    async def start_job(self, job: IngestionJob) -> None: ...

    async def complete_job(self, job: IngestionJob) -> None: ...


class VectorRepository(Protocol):
    async def write_embeddings(self, chunks: list[ChunkRecord]) -> None: ...

    async def search(self, embedding: list[float], limit: int = 6) -> list[ChunkRecord]: ...


class EmbeddingAdapter(Protocol):
    dimensions: int

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def aembed_query(self, text: str) -> list[float]: ...
