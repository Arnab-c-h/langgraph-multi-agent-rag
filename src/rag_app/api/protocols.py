"""Structural dependencies accepted by the HTTP and A2A boundaries."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Protocol
from uuid import UUID

from rag_app.contracts import QueryRequest, StreamEvent
from rag_app.ingestion.models import IngestionResult, SourceRecord


class Runner(Protocol):
    def stream(
        self, request: QueryRequest, *, cancel_event: asyncio.Event | None = None
    ) -> AsyncIterator[StreamEvent]: ...


class SourceService(Protocol):
    async def list_sources(self) -> list[SourceRecord]: ...


class AdminSourceService(SourceService, Protocol):
    async def ingest_pdf(self, filename: str, data: bytes) -> IngestionResult: ...

    async def ingest_web(self, url: str) -> IngestionResult: ...

    async def ingest_github(self, url: str) -> IngestionResult: ...

    async def reindex(self, source_id: UUID) -> IngestionResult: ...

    async def delete(self, source_id: UUID) -> None: ...
