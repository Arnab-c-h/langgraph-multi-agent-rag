from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest

from rag_app.contracts import QueryRequest, SourceType
from rag_app.ingestion.models import ChunkRecord, NormalizedDocument, SourceRecord
from rag_app.ingestion.service import IngestionService


@dataclass
class DeterministicEmbeddings:
    dimensions: int = 768

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        vector[0 if "alpha" in text else 1] = 1.0
        return vector


def _source(
    *,
    source_type: SourceType = SourceType.WEB,
    content_hash: str = "a" * 64,
    canonical_uri: str | None = None,
    title: str = "Authoritative title",
) -> SourceRecord:
    return SourceRecord(
        id=uuid4(),
        source_type=source_type,
        canonical_uri=canonical_uri or ("https://example.com/guide" if source_type is SourceType.WEB else "report.pdf"),
        title=title,
        content_hash=content_hash,
        metadata={"owner": "sqlite"},
        provenance={"final_url": "https://example.com/guide", "private": "sqlite-only"},
    )


def _chunk(source: SourceRecord, index: int, content: str, vector: list[float]) -> ChunkRecord:
    return ChunkRecord(
        id=uuid4(),
        source_id=source.id,
        chunk_index=index,
        content=content,
        content_hash=f"{index + 1:064x}",
        metadata={"heading": f"Section {index + 1}"},
        provenance={"retrieved_at": "2026-08-18T00:00:00Z"},
        embedding=vector,
    )


@pytest.mark.asyncio
async def test_persistent_chroma_orders_top_k_and_hydrates_authoritative_sqlite(tmp_path: Path) -> None:
    from rag_app.retrieval.chroma import ChromaRetriever
    from rag_app.storage.chroma import ChromaIndex, LocalSourceRepository
    from rag_app.storage.sqlite import SqliteDatabase, SqliteSourceRepository

    embeddings = DeterministicEmbeddings()
    source = _source()
    alpha = [0.0] * 768
    alpha[0] = 1.0
    near_alpha = [0.0] * 768
    near_alpha[0], near_alpha[1] = 0.9, 0.1
    beta = [0.0] * 768
    beta[1] = 1.0
    chunks = [
        _chunk(source, 0, "authoritative alpha", alpha),
        _chunk(source, 1, "authoritative near alpha", near_alpha),
        _chunk(source, 2, "authoritative beta", beta),
    ]

    database = SqliteDatabase(tmp_path)
    await database.initialize()
    try:
        index = ChromaIndex(tmp_path / "chroma", embeddings)
        repository = LocalSourceRepository(SqliteSourceRepository(database), index)
        await repository.replace(source, chunks)

        assert await database.scalar("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL") == 0
        stored = index.collection.get(include=["metadatas"])
        assert stored["metadatas"]
        assert all(
            set(metadata)
            == {"chunk_id", "source_id", "source_type", "chunk_index", "content_hash", "source_version_key"}
            for metadata in stored["metadatas"]
        )

        reopened = ChromaIndex(tmp_path / "chroma", embeddings)
        result = await ChromaRetriever(reopened.vector_store, repository).retrieve(
            QueryRequest(question="alpha", top_k=2)
        )

        assert [item.chunk_id for item in result] == [chunks[0].id, chunks[1].id]
        assert result[0].content == "authoritative alpha"
        assert result[0].source_title == "Authoritative title"
        assert result[0].locator == "https://example.com/guide — Section 1"
        assert result[0].provenance == {
            "final_url": "https://example.com/guide",
            "private": "sqlite-only",
            "retrieved_at": "2026-08-18T00:00:00Z",
        }
        assert result[0].similarity == pytest.approx(1.0)
        assert result[1].similarity < result[0].similarity
    finally:
        await database.close()


def _document(content: str = "alpha evidence") -> NormalizedDocument:
    return NormalizedDocument(
        source_type=SourceType.WEB,
        canonical_uri="https://example.com/failed",
        title="Failed source",
        content=content,
        provenance={"final_url": "https://example.com/failed"},
    )


@pytest.mark.asyncio
async def test_first_time_chroma_write_failure_records_failed_job_without_activating_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from rag_app.storage.chroma import ChromaIndex, LocalSourceRepository
    from rag_app.storage.sqlite import SqliteDatabase, SqliteSourceRepository

    database = SqliteDatabase(tmp_path)
    await database.initialize()
    try:
        embeddings = DeterministicEmbeddings()
        index = ChromaIndex(tmp_path / "chroma", embeddings)
        repository = LocalSourceRepository(SqliteSourceRepository(database), index)

        async def fail_upsert(_source: SourceRecord, _chunks: list[ChunkRecord]) -> None:
            raise RuntimeError("api_key=must-not-be-stored")

        monkeypatch.setattr(index, "upsert", fail_upsert)
        with pytest.raises(RuntimeError):
            await IngestionService(repository, embeddings).ingest(_document())

        job = await database.fetchone("SELECT status, error_code FROM ingestion_jobs")
        assert tuple(job) == ("failed", "CHROMA_WRITE_FAILED")
        assert await database.scalar("SELECT count(*) FROM sources") == 0
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_sqlite_commit_failure_leaves_reconcilable_inactive_chroma_orphan_and_failed_job(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from rag_app.storage.chroma import ChromaIndex, LocalSourceRepository
    from rag_app.storage.sqlite import SqliteDatabase, SqliteSourceRepository

    database = SqliteDatabase(tmp_path)
    await database.initialize()
    try:
        connection = database._connection
        assert connection is not None
        embeddings = DeterministicEmbeddings()
        index = ChromaIndex(tmp_path / "chroma", embeddings)
        repository = LocalSourceRepository(SqliteSourceRepository(database), index)
        original_commit = connection.commit

        class FailNextSqliteCommit(DeterministicEmbeddings):
            async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
                async def fail_once() -> None:
                    monkeypatch.setattr(connection, "commit", original_commit)
                    raise RuntimeError("sqlite commit failed")

                monkeypatch.setattr(connection, "commit", fail_once)
                return self.embed_documents(texts)

        with pytest.raises(RuntimeError):
            await IngestionService(repository, FailNextSqliteCommit()).ingest(_document())

        job = await database.fetchone("SELECT status, error_code FROM ingestion_jobs")
        assert tuple(job) == ("failed", "SQLITE_PERSISTENCE_FAILED")
        assert await database.scalar("SELECT count(*) FROM sources") == 0
        assert len(await index.all_ids()) == 1
        await repository.reconcile()
        assert await index.all_ids() == set()
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_chroma_filters_to_active_source_type_and_id_and_ignores_index_document_text(tmp_path: Path) -> None:
    from rag_app.retrieval.chroma import ChromaRetriever
    from rag_app.storage.chroma import ChromaIndex, LocalSourceRepository
    from rag_app.storage.sqlite import SqliteDatabase, SqliteSourceRepository

    embeddings = DeterministicEmbeddings()
    web = _source(canonical_uri="https://example.com/web", title="Web source")
    pdf = _source(source_type=SourceType.PDF, canonical_uri="report.pdf", title="PDF source")
    web_chunk = _chunk(web, 0, "web alpha", embeddings.embed_query("alpha"))
    pdf_chunk = _chunk(pdf, 0, "pdf alpha", embeddings.embed_query("alpha"))
    database = SqliteDatabase(tmp_path)
    await database.initialize()
    try:
        index = ChromaIndex(tmp_path / "chroma", embeddings)
        repository = LocalSourceRepository(SqliteSourceRepository(database), index)
        await repository.replace(web, [web_chunk])
        await repository.replace(pdf, [pdf_chunk])
        index.collection.update(
            ids=[str(web_chunk.id)],
            documents=["poisoned Chroma text"],
            embeddings=[embeddings.embed_query("alpha")],
        )

        retriever = ChromaRetriever(index.vector_store, repository)
        by_type = await retriever.retrieve(
            QueryRequest(question="alpha", top_k=5, source_types=[SourceType.PDF])
        )
        by_id = await retriever.retrieve(QueryRequest(question="alpha", top_k=5, source_ids=[web.id]))
        no_match = await retriever.retrieve(QueryRequest(question="alpha", source_ids=[uuid4()]))

        assert [item.chunk_id for item in by_type] == [pdf_chunk.id]
        assert [item.chunk_id for item in by_id] == [web_chunk.id]
        assert by_id[0].content == "web alpha"
        assert "poisoned" not in by_id[0].supporting_excerpt
        assert no_match == []
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_poisoned_vector_identity_cannot_rebind_to_another_valid_sqlite_citation(tmp_path: Path) -> None:
    from rag_app.retrieval.chroma import ChromaRetriever
    from rag_app.storage.chroma import ChromaIndex, LocalSourceRepository
    from rag_app.storage.sqlite import SqliteDatabase, SqliteSourceRepository

    embeddings = DeterministicEmbeddings()
    source_a = _source(canonical_uri="https://example.com/a", title="Source A")
    source_b = _source(canonical_uri="https://example.com/b", title="Source B")
    chunk_a = _chunk(source_a, 0, "alpha evidence A", embeddings.embed_query("alpha"))
    chunk_b = _chunk(source_b, 0, "beta evidence B", embeddings.embed_query("beta"))
    database = SqliteDatabase(tmp_path)
    await database.initialize()
    try:
        index = ChromaIndex(tmp_path / "chroma", embeddings)
        repository = LocalSourceRepository(SqliteSourceRepository(database), index)
        await repository.replace(source_a, [chunk_a])
        await repository.replace(source_b, [chunk_b])
        record_a = index.collection.get(ids=[str(chunk_a.id)], include=["metadatas"])
        poisoned = dict(record_a["metadatas"][0])
        poisoned.update(
            {
                "chunk_id": str(chunk_b.id),
                "source_id": str(source_b.id),
                "source_type": source_b.source_type.value,
                "chunk_index": chunk_b.chunk_index,
                "content_hash": chunk_b.content_hash,
                "source_version_key": f"{source_a.id}:{source_a.content_hash}",
            }
        )
        index.collection.update(ids=[str(chunk_a.id)], metadatas=[poisoned])

        retrieved = await ChromaRetriever(index.vector_store, repository).retrieve(
            QueryRequest(question="alpha", top_k=1)
        )
        assert retrieved == []

        with pytest.raises(RuntimeError, match="identity metadata"):
            await repository.reconcile()
        assert await index.all_ids() == {chunk_b.id}
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_retriever_hydrates_only_actual_returned_candidate_ids() -> None:
    from langchain_core.documents import Document

    from rag_app.retrieval.chroma import ChromaRetriever

    source = _source(canonical_uri="https://example.com/bounded")
    returned = _chunk(source, 0, "alpha returned", DeterministicEmbeddings().embed_query("alpha"))
    not_returned = _chunk(source, 1, "beta not returned", DeterministicEmbeddings().embed_query("beta"))
    requested_ids: list[list[object]] = []

    class Store:
        async def asimilarity_search_with_score(
            self, query: str, *, k: int, filter: dict[str, object] | None = None
        ) -> list[tuple[Document, float]]:
            del query, k, filter
            return [
                (
                    Document(
                        id=str(returned.id),
                        page_content="ignored",
                        metadata={
                            "chunk_id": str(returned.id),
                            "source_id": str(source.id),
                            "source_type": source.source_type.value,
                            "chunk_index": returned.chunk_index,
                            "content_hash": returned.content_hash,
                            "source_version_key": f"{source.id}:{source.content_hash}",
                        },
                    ),
                    0.1,
                )
            ]

    class Sources:
        async def list_sources(self) -> list[SourceRecord]:
            return [source]

        async def get_chunks_by_ids(self, chunk_ids: list[object]) -> list[ChunkRecord]:
            requested_ids.append(chunk_ids)
            return [returned] if returned.id in chunk_ids else []

        async def list_chunks(self, _source_id: object) -> list[ChunkRecord]:
            raise AssertionError("retrieval must not hydrate the full source corpus")

    result = await ChromaRetriever(Store(), Sources()).retrieve(QueryRequest(question="alpha", top_k=1))

    assert requested_ids == [[returned.id]]
    assert [item.chunk_id for item in result] == [returned.id]
    assert not_returned.id not in requested_ids[0]


@pytest.mark.asyncio
async def test_sqlite_candidate_batch_lookup_is_parameterized_deduplicated_and_ordered(tmp_path: Path) -> None:
    from rag_app.storage.sqlite import SqliteDatabase, SqliteSourceRepository

    embeddings = DeterministicEmbeddings()
    source = _source(canonical_uri="https://example.com/batch")
    chunks = [
        _chunk(source, index, f"chunk {index}", embeddings.embed_query("alpha"))
        for index in range(3)
    ]
    database = SqliteDatabase(tmp_path)
    await database.initialize()
    try:
        repository = SqliteSourceRepository(database)
        await repository.replace(source, chunks)
        missing = uuid4()

        result = await repository.get_chunks_by_ids([chunks[2].id, missing, chunks[0].id, chunks[2].id])

        assert [chunk.id for chunk in result] == [chunks[2].id, chunks[0].id]
    finally:
        await database.close()


@pytest.mark.parametrize("metric", ["l2", "ip"])
def test_existing_non_cosine_collection_fails_readiness_closed(metric: str, tmp_path: Path) -> None:
    import chromadb

    from rag_app.storage.chroma import COLLECTION_NAME, ChromaIndex

    path = tmp_path / metric / "chroma"
    client = chromadb.PersistentClient(path=path)
    client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": metric})

    with pytest.raises(RuntimeError, match="cosine"):
        ChromaIndex(path, DeterministicEmbeddings())


def test_effective_configuration_metric_overrides_cosine_metadata(tmp_path: Path) -> None:
    import chromadb

    from rag_app.storage.chroma import COLLECTION_NAME, ChromaIndex

    path = tmp_path / "contradictory-metric" / "chroma"
    client = chromadb.PersistentClient(path=path)
    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        configuration={"hnsw": {"space": "ip"}},
    )
    assert collection.configuration["hnsw"]["space"] == "ip"
    assert collection.metadata["hnsw:space"] == "cosine"

    with pytest.raises(RuntimeError, match="cosine.*rebuild"):
        ChromaIndex(path, DeterministicEmbeddings())


@pytest.mark.asyncio
async def test_version_replacement_and_delete_remove_inactive_chroma_records(tmp_path: Path) -> None:
    from rag_app.retrieval.chroma import ChromaRetriever
    from rag_app.storage.chroma import ChromaIndex, LocalSourceRepository
    from rag_app.storage.sqlite import SqliteDatabase, SqliteSourceRepository

    embeddings = DeterministicEmbeddings()
    database = SqliteDatabase(tmp_path)
    await database.initialize()
    try:
        index = ChromaIndex(tmp_path / "chroma", embeddings)
        repository = LocalSourceRepository(SqliteSourceRepository(database), index)
        source = _source(content_hash="a" * 64)
        old_chunk = _chunk(source, 0, "old alpha", embeddings.embed_query("alpha"))
        await repository.replace(source, [old_chunk])

        replacement = source.model_copy(update={"content_hash": "b" * 64, "title": "New title"})
        new_chunk = _chunk(replacement, 0, "new alpha", embeddings.embed_query("alpha"))
        await repository.replace(replacement, [new_chunk])

        assert await index.all_ids() == {new_chunk.id}
        result = await ChromaRetriever(index.vector_store, repository).retrieve(
            QueryRequest(question="alpha", top_k=5)
        )
        assert [item.content for item in result] == ["new alpha"]

        await repository.delete(source.id)
        assert await index.all_ids() == set()
        assert await ChromaRetriever(index.vector_store, repository).retrieve(
            QueryRequest(question="alpha", top_k=5)
        ) == []
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_replacement_normalizes_caller_uuid_to_authoritative_identity_across_restart(tmp_path: Path) -> None:
    from rag_app.retrieval.chroma import ChromaRetriever
    from rag_app.storage.chroma import ChromaIndex, LocalSourceRepository
    from rag_app.storage.sqlite import SqliteDatabase, SqliteSourceRepository

    embeddings = DeterministicEmbeddings()
    original = _source(content_hash="a" * 64)
    original_chunk = _chunk(original, 0, "old alpha", embeddings.embed_query("alpha"))
    replacement = _source(
        content_hash="b" * 64,
        canonical_uri=original.canonical_uri,
        title="Replacement with caller UUID",
    )
    assert replacement.id != original.id
    replacement_chunk = _chunk(replacement, 0, "new alpha", embeddings.embed_query("alpha"))

    first_database = SqliteDatabase(tmp_path)
    await first_database.initialize()
    try:
        index = ChromaIndex(tmp_path / "chroma", embeddings)
        repository = LocalSourceRepository(SqliteSourceRepository(first_database), index)
        await repository.replace(original, [original_chunk])
        await repository.replace(replacement, [replacement_chunk])
    finally:
        await first_database.close()

    restarted_database = SqliteDatabase(tmp_path)
    await restarted_database.initialize()
    try:
        restarted_index = ChromaIndex(tmp_path / "chroma", embeddings)
        restarted_repository = LocalSourceRepository(SqliteSourceRepository(restarted_database), restarted_index)
        await restarted_repository.reconcile()
        result = await ChromaRetriever(restarted_index.vector_store, restarted_repository).retrieve(
            QueryRequest(question="alpha", top_k=5)
        )
        records = restarted_index.collection.get(ids=[str(replacement_chunk.id)], include=["metadatas"])

        assert [item.source_id for item in result] == [original.id]
        assert [item.content for item in result] == ["new alpha"]
        assert records["metadatas"] == [
            {
                "chunk_id": str(replacement_chunk.id),
                "source_id": str(original.id),
                "source_type": "web",
                "chunk_index": 0,
                "content_hash": replacement_chunk.content_hash,
                "source_version_key": f"{original.id}:{replacement.content_hash}",
            }
        ]
        stored_chunks = await restarted_repository.list_chunks(original.id)
        assert [chunk.source_id for chunk in stored_chunks] == [original.id]
    finally:
        await restarted_database.close()


@pytest.mark.asyncio
async def test_startup_reconciliation_deletes_orphans_and_fails_closed_for_missing_active_chunks(
    tmp_path: Path,
) -> None:
    from rag_app.storage.chroma import ChromaIndex, LocalSourceRepository
    from rag_app.storage.sqlite import SqliteDatabase, SqliteSourceRepository

    embeddings = DeterministicEmbeddings()
    database = SqliteDatabase(tmp_path)
    await database.initialize()
    try:
        index = ChromaIndex(tmp_path / "chroma", embeddings)
        repository = LocalSourceRepository(SqliteSourceRepository(database), index)
        active = _source(canonical_uri="https://example.com/active")
        active_chunk = _chunk(active, 0, "active alpha", embeddings.embed_query("alpha"))
        orphan = _source(canonical_uri="https://example.com/orphan")
        orphan_chunk = _chunk(orphan, 0, "orphan alpha", embeddings.embed_query("alpha"))
        await repository.replace(active, [active_chunk])
        await index.upsert(orphan, [orphan_chunk])

        await repository.reconcile()
        assert await index.all_ids() == {active_chunk.id}

        await index.delete_ids([active_chunk.id])
        with pytest.raises(RuntimeError, match="missing active SQLite chunks"):
            await repository.reconcile()
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_startup_reconciliation_fails_closed_for_poisoned_active_version_metadata(tmp_path: Path) -> None:
    from rag_app.storage.chroma import ChromaIndex, LocalSourceRepository
    from rag_app.storage.sqlite import SqliteDatabase, SqliteSourceRepository

    embeddings = DeterministicEmbeddings()
    database = SqliteDatabase(tmp_path)
    await database.initialize()
    try:
        index = ChromaIndex(tmp_path / "chroma", embeddings)
        repository = LocalSourceRepository(SqliteSourceRepository(database), index)
        source = _source(canonical_uri="https://example.com/version-poison")
        chunk = _chunk(source, 0, "active alpha", embeddings.embed_query("alpha"))
        await repository.replace(source, [chunk])
        record = index.collection.get(ids=[str(chunk.id)], include=["metadatas"])
        metadata = dict(record["metadatas"][0])
        metadata["source_version_key"] = f"{source.id}:{'f' * 64}"
        index.collection.update(ids=[str(chunk.id)], metadatas=[metadata])

        with pytest.raises(RuntimeError, match="version metadata"):
            await repository.reconcile()
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_unchanged_ingestion_is_idempotent_in_real_sqlite_and_chroma(tmp_path: Path) -> None:
    from rag_app.storage.chroma import ChromaIndex, LocalSourceRepository
    from rag_app.storage.sqlite import SqliteDatabase, SqliteSourceRepository

    embeddings = DeterministicEmbeddings()
    database = SqliteDatabase(tmp_path)
    await database.initialize()
    try:
        index = ChromaIndex(tmp_path / "chroma", embeddings)
        repository = LocalSourceRepository(SqliteSourceRepository(database), index)
        service = IngestionService(repository, embeddings)

        first = await service.ingest(_document())
        unchanged = await service.ingest(_document())

        assert first.status == "ingested"
        assert unchanged.status == "unchanged"
        assert unchanged.source.id == first.source.id
        assert await database.scalar("SELECT count(*) FROM sources") == 1
        assert await database.scalar("SELECT count(*) FROM chunks") == 1
        assert await index.all_ids() == {first.chunks[0].id}
        jobs = await database.fetchall("SELECT status FROM ingestion_jobs ORDER BY created_at, rowid")
        assert [row[0] for row in jobs] == ["completed", "unchanged"]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_failed_changed_sqlite_commit_keeps_prior_active_version_queryable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from rag_app.retrieval.chroma import ChromaRetriever
    from rag_app.storage.chroma import ChromaIndex, LocalSourceRepository
    from rag_app.storage.sqlite import SqliteDatabase, SqliteSourceRepository

    embeddings = DeterministicEmbeddings()
    database = SqliteDatabase(tmp_path)
    await database.initialize()
    try:
        connection = database._connection
        assert connection is not None
        index = ChromaIndex(tmp_path / "chroma", embeddings)
        repository = LocalSourceRepository(SqliteSourceRepository(database), index)
        service = IngestionService(repository, embeddings)
        original = await service.ingest(_document("old alpha evidence"))
        original_commit = connection.commit

        class FailNextSqliteCommit(DeterministicEmbeddings):
            async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
                async def fail_once() -> None:
                    monkeypatch.setattr(connection, "commit", original_commit)
                    raise RuntimeError("sqlite commit failed")

                monkeypatch.setattr(connection, "commit", fail_once)
                return self.embed_documents(texts)

        with pytest.raises(RuntimeError):
            await IngestionService(repository, FailNextSqliteCommit()).ingest(_document("changed alpha evidence"))

        retrieved = await ChromaRetriever(index.vector_store, repository).retrieve(
            QueryRequest(question="alpha", top_k=5)
        )
        assert [item.chunk_id for item in retrieved] == [original.chunks[0].id]
        assert [item.content for item in retrieved] == ["old alpha evidence"]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_delete_cleanup_failure_cannot_make_deleted_source_retrievable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from rag_app.retrieval.chroma import ChromaRetriever
    from rag_app.storage.chroma import ChromaIndex, LocalSourceRepository
    from rag_app.storage.sqlite import SqliteDatabase, SqliteSourceRepository

    embeddings = DeterministicEmbeddings()
    database = SqliteDatabase(tmp_path)
    await database.initialize()
    try:
        index = ChromaIndex(tmp_path / "chroma", embeddings)
        repository = LocalSourceRepository(SqliteSourceRepository(database), index)
        result = await IngestionService(repository, embeddings).ingest(_document())
        original_delete = index.delete_ids

        async def fail_delete(_chunk_ids: list[object]) -> None:
            raise RuntimeError("chroma cleanup failed")

        monkeypatch.setattr(index, "delete_ids", fail_delete)
        with pytest.raises(RuntimeError, match="cleanup failed"):
            await repository.delete(result.source.id)

        assert await repository.list_sources() == []
        assert await ChromaRetriever(index.vector_store, repository).retrieve(
            QueryRequest(question="alpha")
        ) == []
        monkeypatch.setattr(index, "delete_ids", original_delete)
        await repository.reconcile()
        assert await index.all_ids() == set()
    finally:
        await database.close()
