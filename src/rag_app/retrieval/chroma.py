"""LangChain Chroma retrieval with authoritative SQLite hydration."""

from __future__ import annotations

from typing import Any, Protocol
from uuid import UUID

from rag_app.contracts import QueryRequest, SourceType
from rag_app.ingestion.models import ChunkRecord, SourceRecord
from rag_app.retrieval.models import RetrievedChunk
from rag_app.storage.chroma import chroma_record_metadata, source_version_key


class _VectorStore(Protocol):
    async def asimilarity_search_with_score(
        self, query: str, *, k: int, filter: dict[str, object] | None = None
    ) -> list[tuple[Any, float]]: ...


class _Sources(Protocol):
    async def list_sources(self) -> list[SourceRecord]: ...

    async def get_chunks_by_ids(self, chunk_ids: list[UUID]) -> list[ChunkRecord]: ...


class ChromaRetriever:
    """Filter to active versions, then ignore Chroma content during hydration."""

    def __init__(self, store: _VectorStore, sources: _Sources) -> None:
        self._store = store
        self._sources = sources

    async def retrieve(self, request: QueryRequest) -> list[RetrievedChunk]:
        active_sources = [
            source
            for source in await self._sources.list_sources()
            if (not request.source_types or source.source_type in request.source_types)
            and (not request.source_ids or source.id in request.source_ids)
        ]
        if not active_sources:
            return []
        version_keys = [source_version_key(source) for source in active_sources]
        documents = await self._store.asimilarity_search_with_score(
            request.question,
            k=request.top_k,
            filter={"source_version_key": {"$in": version_keys}},
        )
        candidates: list[tuple[UUID, Any, float]] = []
        seen_ids: set[UUID] = set()
        for document, distance in documents[: request.top_k]:
            try:
                chunk_id = UUID(str(getattr(document, "id", "")))
            except (TypeError, ValueError):
                continue
            if chunk_id not in seen_ids:
                candidates.append((chunk_id, document, float(distance)))
                seen_ids.add(chunk_id)
        chunks = await self._sources.get_chunks_by_ids([chunk_id for chunk_id, _document, _distance in candidates])
        chunks_by_id = {chunk.id: chunk for chunk in chunks}
        sources_by_id = {source.id: source for source in active_sources}
        retrieved: list[RetrievedChunk] = []
        for chunk_id, document, distance in candidates:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            source = sources_by_id.get(chunk.source_id)
            if source is None:
                continue
            metadata = dict(getattr(document, "metadata", {}) or {})
            if metadata != chroma_record_metadata(source, chunk):
                continue
            retrieved.append(_hydrate(source, chunk, distance))
        return retrieved


def _hydrate(source: SourceRecord, chunk: ChunkRecord, distance: float) -> RetrievedChunk:
    metadata = dict(chunk.metadata)
    provenance = {**source.provenance, **chunk.provenance}
    source_path = str(metadata["path"]) if metadata.get("path") else None
    if source.source_type is SourceType.PDF:
        source_path = source_path or source.canonical_uri
    source_url = source.canonical_uri if source.source_type in {SourceType.WEB, SourceType.GITHUB} else None
    return RetrievedChunk(
        chunk_id=chunk.id,
        source_id=source.id,
        source_type=source.source_type,
        source_title=source.title,
        content=chunk.content,
        supporting_excerpt=chunk.content[:1000],
        locator=_locator(source, chunk, metadata),
        source_url=source_url,
        source_path=source_path,
        metadata=metadata,
        provenance=provenance,
        similarity=max(-1.0, min(1.0, 1.0 - distance)),
    )


def _locator(source: SourceRecord, chunk: ChunkRecord, metadata: dict[str, Any]) -> str:
    heading = metadata.get("heading")
    if source.source_type is SourceType.PDF and metadata.get("page") is not None:
        base = f"page {metadata['page']}"
    elif source.source_type is SourceType.GITHUB and metadata.get("path"):
        base = str(metadata["path"])
    elif source.source_type is SourceType.WEB:
        base = source.canonical_uri
    else:
        base = f"chunk {chunk.chunk_index + 1}"
    return f"{base} — {heading}" if heading else base
