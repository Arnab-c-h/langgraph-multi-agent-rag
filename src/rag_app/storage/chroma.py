"""Persistent local Chroma index coordinated with authoritative SQLite records."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import chromadb
from chromadb.api.collection_configuration import collection_configuration_to_json
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from rag_app.contracts import SourceType
from rag_app.ingestion.models import ChunkRecord, IngestionJob, SourceRecord
from rag_app.ingestion.protocols import IngestionPersistenceError
from rag_app.storage.sqlite import SqliteSourceRepository

COLLECTION_NAME = "rag_chunks"


class ChromaIndex:
    """One cosine collection whose records contain only filter/identity metadata."""

    def __init__(self, path: Path, embeddings: Embeddings) -> None:
        self.path = path.expanduser().resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.path)
        self.collection = self.client.get_or_create_collection(
            COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            configuration={"hnsw": {"space": "cosine"}},
        )
        configuration = getattr(self.collection, "configuration", None)
        if configuration is None:
            metric = (self.collection.metadata or {}).get("hnsw:space")
        else:
            effective_configuration = collection_configuration_to_json(configuration)
            index_configuration = effective_configuration.get("hnsw")
            if index_configuration is None:
                index_configuration = effective_configuration.get("spann")
            metric = None if index_configuration is None else index_configuration.get("space")
        if metric != "cosine":
            raise RuntimeError(
                "Existing Chroma rag_chunks collection does not use cosine distance; rebuild the local index"
            )
        self.vector_store = Chroma(
            client=self.client,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            create_collection_if_not_exists=False,
        )

    async def upsert(self, source: SourceRecord, chunks: list[ChunkRecord]) -> None:
        if not chunks:
            return
        embeddings = [chunk.embedding for chunk in chunks]
        if any(vector is None or len(vector) != 768 for vector in embeddings):
            raise ValueError("Chroma writes require one 768-dimensional embedding per chunk")
        vectors: list[Sequence[float]] = [cast(list[float], vector) for vector in embeddings]
        await asyncio.to_thread(
            self.collection.upsert,
            ids=[str(chunk.id) for chunk in chunks],
            embeddings=vectors,
            documents=[chunk.content for chunk in chunks],
            metadatas=[chroma_record_metadata(source, chunk) for chunk in chunks],
        )

    async def delete_ids(self, chunk_ids: list[UUID]) -> None:
        if chunk_ids:
            await asyncio.to_thread(self.collection.delete, ids=[str(chunk_id) for chunk_id in chunk_ids])

    async def all_ids(self) -> set[UUID]:
        return set(await self.records())

    async def records(self) -> dict[UUID, dict[str, Any]]:
        result = await asyncio.to_thread(self.collection.get, include=["metadatas"])
        metadatas = result.get("metadatas") or []
        if len(result["ids"]) != len(metadatas):
            raise RuntimeError("Chroma index contains records without identity metadata; rebuild the local index")
        return {
            UUID(value): dict(metadata or {})
            for value, metadata in zip(result["ids"], metadatas, strict=True)
        }


class LocalSourceRepository:
    """Preserve the source repository contract across SQLite and Chroma."""

    def __init__(self, sqlite: SqliteSourceRepository, chroma: ChromaIndex) -> None:
        self._sqlite = sqlite
        self._chroma = chroma

    async def find_by_identity(self, source_type: SourceType, canonical_uri: str) -> SourceRecord | None:
        return await self._sqlite.find_by_identity(source_type, canonical_uri)

    async def replace(self, source: SourceRecord, chunks: list[ChunkRecord]) -> None:
        existing = await self._sqlite.find_by_identity(source.source_type, source.canonical_uri)
        previous_chunks = await self._sqlite.list_chunks(existing.id) if existing is not None else []
        authoritative_source = (
            source
            if existing is None
            else source.model_copy(update={"id": existing.id, "created_at": existing.created_at})
        )
        normalized_chunks = [
            chunk.model_copy(update={"source_id": authoritative_source.id}) for chunk in chunks
        ]
        try:
            await self._chroma.upsert(authoritative_source, normalized_chunks)
        except Exception:
            raise IngestionPersistenceError("CHROMA_WRITE_FAILED") from None
        authoritative_chunks = [chunk.model_copy(update={"embedding": None}) for chunk in normalized_chunks]
        try:
            await self._sqlite.replace(authoritative_source, authoritative_chunks)
        except Exception:
            raise IngestionPersistenceError("SQLITE_PERSISTENCE_FAILED") from None
        current_ids = {chunk.id for chunk in normalized_chunks}
        try:
            await self._chroma.delete_ids([chunk.id for chunk in previous_chunks if chunk.id not in current_ids])
        except Exception:
            raise IngestionPersistenceError("CHROMA_CLEANUP_FAILED") from None

    async def list_chunks(self, source_id: UUID) -> list[ChunkRecord]:
        return await self._sqlite.list_chunks(source_id)

    async def get_chunks_by_ids(self, chunk_ids: list[UUID]) -> list[ChunkRecord]:
        return await self._sqlite.get_chunks_by_ids(chunk_ids)

    async def list_sources(self) -> list[SourceRecord]:
        return await self._sqlite.list_sources()

    async def delete(self, source_id: UUID) -> None:
        chunks = await self._sqlite.list_chunks(source_id)
        await self._sqlite.delete(source_id)
        await self._chroma.delete_ids([chunk.id for chunk in chunks])

    async def start_job(self, job: IngestionJob) -> None:
        await self._sqlite.start_job(job)

    async def complete_job(self, job: IngestionJob) -> None:
        await self._sqlite.complete_job(job)

    async def reconcile(self) -> None:
        expected_records = {
            chunk.id: chroma_record_metadata(source, chunk)
            for source in await self._sqlite.list_sources()
            for chunk in await self._sqlite.list_chunks(source.id)
        }
        active_ids = set(expected_records)
        records = await self._chroma.records()
        chroma_ids = set(records)
        missing = active_ids - chroma_ids
        invalid_versions = {
            chunk_id
            for chunk_id, expected in expected_records.items()
            if chunk_id in records and records[chunk_id].get("source_version_key") != expected["source_version_key"]
        }
        identity_keys = {"chunk_id", "source_id", "source_type", "chunk_index", "content_hash"}
        invalid_identities = {
            chunk_id
            for chunk_id, expected in expected_records.items()
            if chunk_id in records
            and (
                set(records[chunk_id]) != set(expected)
                or any(records[chunk_id].get(key) != expected[key] for key in identity_keys)
            )
        }
        cleanup_ids = (chroma_ids - active_ids) | invalid_versions | invalid_identities
        await self._chroma.delete_ids(sorted(cleanup_ids, key=str))
        if invalid_identities:
            raise RuntimeError("Chroma index has invalid active record identity metadata; reindex affected sources")
        if invalid_versions:
            raise RuntimeError("Chroma index has invalid active source version metadata; reindex the affected sources")
        if missing:
            raise RuntimeError("Chroma index is missing active SQLite chunks; reindex the affected sources")


def source_version_key(source: SourceRecord) -> str:
    return f"{source.id}:{source.content_hash}"


def chroma_record_metadata(source: SourceRecord, chunk: ChunkRecord) -> dict[str, str | int]:
    return {
        "chunk_id": str(chunk.id),
        "source_id": str(source.id),
        "source_type": source.source_type.value,
        "chunk_index": chunk.chunk_index,
        "content_hash": chunk.content_hash,
        "source_version_key": source_version_key(source),
    }


def chroma_metadata(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Typed compatibility helper for diagnostics without exposing document content."""
    return [dict(item) for item in result.get("metadatas") or [] if item is not None]
