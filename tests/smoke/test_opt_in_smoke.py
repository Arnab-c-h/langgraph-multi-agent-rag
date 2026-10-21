"""Live smoke tests that are skipped unless dedicated opt-ins are present."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from rag_app.config import Settings
from rag_app.contracts import SourceType
from rag_app.ingestion.embeddings import GeminiEmbeddingAdapter
from rag_app.retrieval.models import EvidenceAssessment, EvidenceBundle, RetrievedChunk
from rag_app.workflows import GeminiWorkflowModel, ModelOperation, ModelRequest

from .guards import require_loopback_a2a_url, require_smoke_secret


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_real_gemini_generation_and_embedding_are_explicitly_opted_in() -> None:
    api_key = require_smoke_secret("RAG_RUN_GEMINI_SMOKE", "RAG_SMOKE_GEMINI_API_KEY")
    embeddings = GeminiEmbeddingAdapter(api_key)
    vector = await embeddings.aembed_query("explicit opt-in smoke")
    assert len(vector) == 768

    chunk = RetrievedChunk(
        chunk_id=UUID("00000000-0000-0000-0000-000000000101"),
        source_id=UUID("00000000-0000-0000-0000-000000000201"),
        source_type=SourceType.WEB,
        source_title="Smoke evidence",
        content="The smoke answer is grounded in this sentence.",
        supporting_excerpt="The smoke answer is grounded in this sentence.",
        locator="smoke fixture",
        similarity=1.0,
    )
    evidence = EvidenceBundle(
        evidence=[EvidenceAssessment(chunk=chunk, relevant=True, confidence=1.0)],
        citations=[chunk.citation()],
        used_source_ids=[chunk.source_id],
    )
    settings = Settings(app_profile="test", gemini_api_key=api_key)
    result = await GeminiWorkflowModel.from_settings(settings).complete(
        ModelRequest(operation=ModelOperation.ANSWER, question="What is grounded?", evidence=evidence)
    )
    assert result.text


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_running_local_a2a_gateway_with_the_official_client() -> None:
    base_url = require_loopback_a2a_url()
    import httpx
    from a2a.client import ClientConfig, ClientFactory
    from a2a.types import GetTaskRequest, Message, Part, Role, SendMessageRequest

    http = httpx.AsyncClient(base_url=base_url, timeout=120)
    factory = ClientFactory(ClientConfig(httpx_client=http, supported_protocol_bindings=["JSONRPC"]))
    client = await factory.create_from_url(base_url)
    try:
        request = SendMessageRequest(
            message=Message(message_id=str(uuid4()), role=Role.ROLE_USER, parts=[Part(text="List sources")]),
            metadata={"skill_id": "list_sources"},
        )
        responses = [response async for response in client.send_message(request)]
        task_id = next(response.task.id for response in responses if response.HasField("task"))
        final = await client.get_task(GetTaskRequest(id=task_id))
    finally:
        await client.close()
        await http.aclose()
    from a2a.types import TaskState

    assert final.status.state == TaskState.TASK_STATE_COMPLETED
    assert any(artifact.name == "sources" for artifact in final.artifacts)
