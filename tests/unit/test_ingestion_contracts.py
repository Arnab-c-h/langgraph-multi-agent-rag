from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import ValidationError

from rag_app.contracts import SourceType
from rag_app.ingestion.models import ChunkRecord, IngestionJob, IngestionResult, SourceRecord


def test_ingestion_models_preserve_stable_source_chunk_and_job_metadata() -> None:
    source = SourceRecord(
        source_type=SourceType.WEB,
        canonical_uri="https://example.com/guide",
        title="Guide",
        content_hash="a" * 64,
        metadata={"final_url": "https://example.com/guide"},
    )
    chunk = ChunkRecord(
        id=UUID("00000000-0000-0000-0000-000000000001"),
        source_id=source.id,
        chunk_index=0,
        content="Grounded answers cite evidence.",
        content_hash="b" * 64,
        metadata={"title": "Guide"},
    )
    job = IngestionJob(source_id=source.id, status="completed", provenance={"loader": "web"})
    result = IngestionResult(source=source, job=job, chunks=[chunk], status="ingested")

    assert source.id.version == 4
    assert result.chunk_count == 1
    assert result.status == "ingested"
    assert job.provenance == {"loader": "web"}


@pytest.mark.parametrize("bad_hash", ["short", "z" * 64])
def test_content_hashes_must_be_lowercase_sha256_hex(bad_hash: str) -> None:
    with pytest.raises(ValidationError):
        SourceRecord(
            source_type=SourceType.PDF,
            canonical_uri="file.pdf",
            title="File",
            content_hash=bad_hash,
        )
