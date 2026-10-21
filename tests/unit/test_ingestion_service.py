from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

import pytest

from rag_app.contracts import SourceType
from rag_app.ingestion.models import ChunkRecord, DocumentSegment, IngestionJob, NormalizedDocument, SourceRecord
from rag_app.ingestion.service import IngestionService


@dataclass
class FakeSourceRepository:
    source: SourceRecord | None = None
    chunks: list[ChunkRecord] = field(default_factory=list)
    rolled_back: bool = False
    jobs: list[IngestionJob] = field(default_factory=list)

    async def find_by_identity(self, source_type: SourceType, canonical_uri: str) -> SourceRecord | None:
        if self.source and (self.source.source_type, self.source.canonical_uri) == (source_type, canonical_uri):
            return self.source
        return None

    async def replace(self, source: SourceRecord, chunks: list[ChunkRecord]) -> None:
        if any(chunk.content == "fail replacement" for chunk in chunks):
            self.rolled_back = True
            raise RuntimeError("write failed")
        self.source = source
        self.chunks = chunks

    async def list_chunks(self, source_id: UUID) -> list[ChunkRecord]:
        return list(self.chunks) if self.source and self.source.id == source_id else []

    async def delete(self, source_id: UUID) -> None:
        if self.source and self.source.id == source_id:
            self.source = None
            self.chunks = []

    async def start_job(self, job: IngestionJob) -> None:
        self.jobs.append(job)

    async def complete_job(self, job: IngestionJob) -> None:
        self.jobs.append(job)


@dataclass
class FakeEmbeddings:
    dimensions: int = 768

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.25] * self.dimensions for _ in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return [0.25] * self.dimensions


def web_document(content: str) -> NormalizedDocument:
    return NormalizedDocument(
        source_type=SourceType.WEB,
        canonical_uri="https://example.com/guide",
        title="Guide",
        content=content,
        provenance={"final_url": "https://example.com/guide", "title": "Guide"},
    )


@pytest.mark.asyncio
async def test_identical_canonical_content_returns_unchanged_without_replacement() -> None:
    repository = FakeSourceRepository()
    service = IngestionService(repository, FakeEmbeddings())

    first = await service.ingest(web_document("evidence " * 200))
    second = await service.ingest(web_document("evidence " * 200))

    assert first.status == "ingested"
    assert second.status == "unchanged"
    assert second.chunk_count == first.chunk_count
    assert repository.source is not None
    assert repository.source.id == first.source.id
    assert [job.status for job in repository.jobs] == ["started", "completed", "started", "unchanged"]


@pytest.mark.asyncio
async def test_changed_document_replaces_all_chunks_with_deterministic_ids() -> None:
    repository = FakeSourceRepository()
    service = IngestionService(repository, FakeEmbeddings())

    first = await service.ingest(web_document("first version " * 200))
    changed = await service.ingest(web_document("second version " * 200))

    assert changed.status == "ingested"
    assert changed.source.id == first.source.id
    assert changed.chunks[0].id != first.chunks[0].id
    assert [chunk.id for chunk in changed.chunks] == [
        service.chunk_id(changed.source, chunk.content_hash, chunk.chunk_index) for chunk in changed.chunks
    ]


@pytest.mark.asyncio
async def test_failed_changed_replacement_leaves_previous_source_and_chunks_intact() -> None:
    repository = FakeSourceRepository()
    service = IngestionService(repository, FakeEmbeddings())
    first = await service.ingest(web_document("first version " * 200))

    with pytest.raises(RuntimeError, match="write failed"):
        await service.ingest(web_document("fail replacement"))

    assert repository.rolled_back is True
    assert repository.source == first.source
    assert repository.chunks == first.chunks


@pytest.mark.asyncio
async def test_source_deletion_cascades_to_owned_chunks() -> None:
    repository = FakeSourceRepository()
    service = IngestionService(repository, FakeEmbeddings())
    result = await service.ingest(web_document("evidence " * 200))

    await repository.delete(result.source.id)

    assert repository.source is None
    assert repository.chunks == []


@pytest.mark.asyncio
async def test_pdf_ingestion_preserves_docling_native_chunks_without_recursive_rechunking() -> None:
    repository = FakeSourceRepository()
    service = IngestionService(repository, FakeEmbeddings())
    native_chunk = "p" * 1300
    document = NormalizedDocument(
        source_type=SourceType.PDF,
        canonical_uri="report.pdf",
        title="Report",
        content=native_chunk,
        segments=[DocumentSegment(content=native_chunk, metadata={"page": 1, "heading": "Intro"})],
    )

    result = await service.ingest(document)

    assert [chunk.content for chunk in result.chunks] == [native_chunk]
    assert result.chunks[0].metadata["page"] == 1


@pytest.mark.asyncio
async def test_ingestion_writes_stable_langchain_filter_metadata() -> None:
    repository = FakeSourceRepository()
    result = await IngestionService(repository, FakeEmbeddings()).ingest(web_document("grounded evidence"))

    assert result.chunks
    assert all(chunk.metadata["source_type"] == "web" for chunk in result.chunks)
    assert all(chunk.metadata["source_id_text"] == str(result.source.id) for chunk in result.chunks)


@pytest.mark.asyncio
async def test_embedding_failure_completes_the_preexisting_job_with_a_safe_code() -> None:
    repository = FakeSourceRepository()

    class BrokenEmbeddings(FakeEmbeddings):
        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            assert [job.status for job in repository.jobs] == ["started"]
            raise RuntimeError("api_key=must-not-be-persisted")

    service = IngestionService(repository, BrokenEmbeddings())
    with pytest.raises(RuntimeError, match="must-not-be-persisted"):
        await service.ingest(web_document("new evidence"))

    assert [job.status for job in repository.jobs] == ["started", "failed"]
    assert repository.jobs[-1].error_code == "EMBEDDING_FAILED"
    assert "must-not-be-persisted" not in str(repository.jobs[-1].model_dump())


@pytest.mark.asyncio
async def test_completed_job_finalize_failure_attempts_failed_status_without_masking_original() -> None:
    class FinalizeFailureRepository(FakeSourceRepository):
        attempts: list[IngestionJob] = field(default_factory=list)

        def __init__(self) -> None:
            super().__init__()
            self.attempts = []

        async def complete_job(self, job: IngestionJob) -> None:
            self.attempts.append(job)
            if job.status == "completed":
                raise RuntimeError("completed finalization failed")
            if job.status == "failed":
                raise RuntimeError("failed finalization also failed")

    repository = FinalizeFailureRepository()
    with pytest.raises(RuntimeError, match="completed finalization failed"):
        await IngestionService(repository, FakeEmbeddings()).ingest(web_document("new evidence"))

    assert [job.status for job in repository.attempts] == ["completed", "failed"]
    assert repository.attempts[-1].error_code == "SQLITE_PERSISTENCE_FAILED"


@pytest.mark.asyncio
async def test_unchanged_job_finalize_failure_is_rewritten_to_stable_failed_status() -> None:
    class FinalizeFailureRepository(FakeSourceRepository):
        fail_unchanged = False
        attempts: list[IngestionJob] = field(default_factory=list)

        def __init__(self) -> None:
            super().__init__()
            self.attempts = []

        async def complete_job(self, job: IngestionJob) -> None:
            self.attempts.append(job)
            if self.fail_unchanged and job.status == "unchanged":
                raise RuntimeError("unchanged finalization failed")
            self.jobs.append(job)

    repository = FinalizeFailureRepository()
    service = IngestionService(repository, FakeEmbeddings())
    await service.ingest(web_document("same evidence"))
    repository.fail_unchanged = True

    with pytest.raises(RuntimeError, match="unchanged finalization failed"):
        await service.ingest(web_document("same evidence"))

    assert [job.status for job in repository.attempts[-2:]] == ["unchanged", "failed"]
    assert repository.jobs[-1].status == "failed"
    assert repository.jobs[-1].error_code == "SQLITE_PERSISTENCE_FAILED"
