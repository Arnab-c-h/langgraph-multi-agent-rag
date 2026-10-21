"""Hash, chunk, embed, and atomically persist normalized source documents."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from rag_app.ingestion.models import ChunkRecord, IngestionJob, IngestionResult, NormalizedDocument, SourceRecord
from rag_app.ingestion.protocols import EmbeddingAdapter, IngestionPersistenceError, SourceRepository


class IngestionService:
    """Coordinates deterministic ingestion without coupling loaders to storage."""

    def __init__(self, sources: SourceRepository, embeddings: EmbeddingAdapter) -> None:
        self._sources = sources
        self._embeddings = embeddings

    @staticmethod
    def content_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def chunk_id(source: SourceRecord, chunk_hash: str, chunk_index: int) -> UUID:
        identity = f"{source.source_type}:{source.canonical_uri}:{source.content_hash}:{chunk_hash}:{chunk_index}"
        return uuid5(NAMESPACE_URL, identity)

    async def ingest(self, document: NormalizedDocument) -> IngestionResult:
        document_hash = self.content_hash(document.content)
        existing = await self._sources.find_by_identity(document.source_type, document.canonical_uri)
        if existing is not None and existing.content_hash == document_hash:
            started = await self._start_job(existing.id, document.provenance)
            job = await self._complete_job(started, "unchanged")
            chunks = await self._sources.list_chunks(existing.id)
            return IngestionResult(source=existing, job=job, chunks=chunks, status="unchanged")

        source = SourceRecord(
            id=existing.id if existing else uuid5(NAMESPACE_URL, f"{document.source_type}:{document.canonical_uri}"),
            source_type=document.source_type,
            canonical_uri=document.canonical_uri,
            title=document.title,
            content_hash=document_hash,
            metadata=document.metadata,
            provenance=document.provenance,
        )
        started = await self._start_job(source.id, document.provenance)
        try:
            texts_and_metadata = self._split(document)
            vectors = await self._embeddings.aembed_documents([text for text, _metadata in texts_and_metadata])
            if len(vectors) != len(texts_and_metadata) or any(len(vector) != 768 for vector in vectors):
                raise ValueError("Embedding adapter must return one 768-dimensional vector per chunk")
        except Exception:
            await self._fail_job(started, "EMBEDDING_FAILED")
            raise
        chunks = [
            ChunkRecord(
                id=self.chunk_id(source, self.content_hash(text), index),
                source_id=source.id,
                chunk_index=index,
                content=text,
                content_hash=self.content_hash(text),
                metadata={
                    **metadata,
                    "source_type": source.source_type.value,
                    "source_id_text": str(source.id),
                },
                provenance=document.provenance,
                embedding=vector,
            )
            for index, ((text, metadata), vector) in enumerate(zip(texts_and_metadata, vectors, strict=True))
        ]
        try:
            await self._sources.replace(source, chunks)
        except IngestionPersistenceError as exc:
            await self._fail_job(started, exc.error_code)
            raise
        except Exception:
            await self._fail_job(started, "PERSISTENCE_FAILED")
            raise
        job = await self._complete_job(started, "completed")
        return IngestionResult(source=source, job=job, chunks=chunks, status="ingested")

    async def _start_job(self, source_id: UUID, provenance: dict[str, object]) -> IngestionJob:
        started = IngestionJob(source_id=source_id, status="started", provenance=provenance)
        await self._sources.start_job(started)
        return started

    async def _complete_job(
        self, started: IngestionJob, status: Literal["completed", "unchanged"]
    ) -> IngestionJob:
        completed = started.model_copy(update={"status": status, "completed_at": datetime.now(UTC)})
        try:
            await self._sources.complete_job(completed)
        except Exception:
            await self._fail_job(started, "SQLITE_PERSISTENCE_FAILED")
            raise
        return completed

    async def _fail_job(self, started: IngestionJob, error_code: str) -> IngestionJob:
        failed = started.model_copy(
            update={"status": "failed", "error_code": error_code, "completed_at": datetime.now(UTC)}
        )
        try:
            await self._sources.complete_job(failed)
        except Exception:
            pass
        return failed

    @staticmethod
    def _split(document: NormalizedDocument) -> list[tuple[str, dict[str, object]]]:
        segments = document.segments or [
            type("Segment", (), {"content": document.content, "metadata": {}})()  # pragma: no cover - type shim
        ]
        chunks: list[tuple[str, dict[str, object]]] = []
        for segment in segments:
            texts = [segment.content] if document.source_type.value == "pdf" else _recursive_split(segment.content)
            for text in texts:
                chunks.append((text, {**document.provenance, **segment.metadata}))
        return chunks


def _recursive_split(text: str) -> list[str]:
    """Use LangChain where available; keep a deterministic offline fallback for minimal installs."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        return _fallback_split(text)
    return RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200).split_text(text)


def _fallback_split(text: str) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + 1200)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - 200
    return chunks
