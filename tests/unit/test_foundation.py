from __future__ import annotations

import json
import logging
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import TypeAdapter, ValidationError

from rag_app.config import Settings
from rag_app.contracts import (
    Citation,
    ErrorEvent,
    FinalEvent,
    QueryRequest,
    QueryResponse,
    ReasoningMode,
    SourceType,
    StreamEvent,
)
from rag_app.errors import AppError, ErrorCode
from rag_app.logging import redact_secrets


def test_settings_defaults_enforce_the_approved_limits() -> None:
    settings = Settings(app_profile="test")

    assert settings.model_name == "gemini-3.5-flash"
    assert settings.embedding_model == "gemini-embedding-2"
    assert settings.embedding_dimensions == 768
    assert settings.query_concurrency == 2
    assert settings.pdf_max_bytes == 25 * 1024 * 1024
    assert settings.pdf_max_pages == 200
    assert settings.web_max_bytes == 5 * 1024 * 1024
    assert settings.web_timeout_seconds == 30
    assert settings.github_max_files == 1000
    assert settings.github_max_bytes == 50 * 1024 * 1024
    assert settings.tot_branches == 3
    assert settings.react_max_actions == 6
    assert settings.reflection_revisions == 1
    assert settings.consistency_candidates == 3
    assert settings.decomposition_questions == 4
    assert settings.recursive_max_depth == 2
    assert settings.recursive_fanout == 3


def test_only_full_and_test_profiles_are_supported() -> None:
    assert Settings(app_profile="full").app_profile == "full"
    assert Settings(app_profile="test").app_profile == "test"

    with pytest.raises(ValidationError):
        Settings(app_profile="hosted")  # type: ignore[arg-type]


def test_full_profile_fails_closed_without_runtime_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)

    settings = Settings(app_profile="full", gemini_api_key=None, admin_token=None)

    with pytest.raises(ValueError, match="GEMINI_API_KEY, ADMIN_TOKEN"):
        settings.validate_runtime_credentials()


@pytest.mark.parametrize(
    ("credential", "value"),
    [("gemini_api_key", "\t"), ("admin_token", "   ")],
)
def test_full_profile_rejects_blank_runtime_secrets(credential: str, value: str) -> None:
    credentials = {
        "gemini_api_key": "live-key",
        "admin_token": "admin-secret",
    }
    credentials[credential] = value

    settings = Settings(app_profile="full", **credentials)

    with pytest.raises(ValueError, match=credential.upper()):
        settings.validate_runtime_credentials()


def test_full_profile_parses_without_secrets_but_runtime_validation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("GEMINI_API_KEY", "ADMIN_TOKEN"):
        monkeypatch.delenv(name, raising=False)

    settings = Settings(app_profile="full", gemini_api_key=None, admin_token=None)

    with pytest.raises(ValueError, match="GEMINI_API_KEY, ADMIN_TOKEN"):
        settings.validate_runtime_credentials()


@pytest.mark.parametrize("embedding_dimensions", [767, 769])
def test_embedding_dimension_is_fixed_to_the_migration_dimension(embedding_dimensions: int) -> None:
    with pytest.raises(ValidationError):
        Settings(app_profile="test", embedding_dimensions=embedding_dimensions)


@pytest.mark.parametrize(
    ("setting_name", "invalid_value"),
    [
        ("query_concurrency", 3),
        ("pdf_max_bytes", 25 * 1024 * 1024 + 1),
        ("pdf_max_pages", 201),
        ("web_max_bytes", 5 * 1024 * 1024 + 1),
        ("web_timeout_seconds", 31),
        ("web_max_urls", 2),
        ("github_max_files", 1001),
        ("github_max_bytes", 50 * 1024 * 1024 + 1),
        ("tot_branches", 4),
        ("react_max_actions", 7),
        ("reflection_revisions", 2),
        ("consistency_candidates", 4),
        ("decomposition_questions", 5),
        ("recursive_max_depth", 3),
        ("recursive_fanout", 4),
    ],
)
def test_approved_operational_limits_cannot_be_increased(setting_name: str, invalid_value: int) -> None:
    with pytest.raises(ValidationError):
        Settings(app_profile="test", **{setting_name: invalid_value})


def test_safe_summary_never_exposes_secret_values() -> None:
    settings = Settings(
        app_profile="test",
        gemini_api_key="real-api-key",
    )

    summary = settings.safe_summary()

    assert summary["gemini_configured"] is True
    assert "database_configured" not in summary
    assert "direct_database_configured" not in summary
    assert "database_url" not in Settings.model_fields
    assert "direct_database_url" not in Settings.model_fields
    assert "password" not in json.dumps(summary)
    assert "real-api-key" not in json.dumps(summary)


def test_reasoning_and_source_enums_expose_the_exact_public_values() -> None:
    assert [mode.value for mode in ReasoningMode] == [
        "standard",
        "cot",
        "tot",
        "react",
        "self_reflection",
        "self_consistency",
        "decomposed",
        "least_to_most",
        "recursive",
    ]
    assert [source.value for source in SourceType] == ["pdf", "web", "github"]


def test_query_request_rejects_blank_questions_and_out_of_range_retrieval() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(question="   ")
    with pytest.raises(ValidationError):
        QueryRequest(question="What is grounded retrieval?", top_k=0)
    with pytest.raises(ValidationError):
        QueryRequest(question="What is grounded retrieval?", top_k=21)


def test_query_request_accepts_additive_source_filters_and_citations_keep_evidence_excerpt() -> None:
    source_id = uuid4()
    request = QueryRequest(
        question="What is grounded retrieval?",
        source_types=[SourceType.WEB],
        source_ids=[source_id],
    )
    citation = Citation(
        source_id=source_id,
        chunk_id=uuid4(),
        source_title="RAG guide",
        locator="section Evidence",
        supporting_excerpt="Only retrieved evidence may be used.",
    )

    assert request.source_types == [SourceType.WEB]
    assert request.source_ids == [source_id]
    assert citation.supporting_excerpt == "Only retrieved evidence may be used."


def test_query_response_keeps_safe_retrieval_metadata_needed_by_later_workflows() -> None:
    source_id = uuid4()
    run_id = uuid4()
    response = QueryResponse(
        answer="Grounded answer",
        mode=ReasoningMode.REACT,
        run_id=run_id,
        used_source_ids=[source_id],
        trace_summary={"candidate_count": 2, "validated": True},
    )

    assert response.mode is ReasoningMode.REACT
    assert response.run_id == run_id
    assert response.used_source_ids == [source_id]
    assert response.trace_summary == {"candidate_count": 2, "validated": True}


def test_stream_events_are_discriminated_and_preserve_final_payload() -> None:
    citation = Citation(
        source_id=uuid4(),
        chunk_id=uuid4(),
        source_title="RAG guide",
        locator="page 3",
    )
    event = TypeAdapter(StreamEvent).validate_python(
        {
            "type": "final",
            "answer": "Ground the answer in retrieved chunks.",
            "citations": [citation.model_dump(mode="json")],
        }
    )

    assert isinstance(event, FinalEvent)
    assert event.citations == [citation]
    assert isinstance(ErrorEvent(code=ErrorCode.RUN_CANCELLED, message="Stopped"), ErrorEvent)
    with pytest.raises(ValidationError):
        TypeAdapter(StreamEvent).validate_python({"type": "unknown"})


def test_error_serialization_returns_stable_code_without_secret_detail() -> None:
    error = AppError(
        ErrorCode.PROVIDER_RATE_LIMITED,
        "Provider rejected token=live-secret",
        detail={
            "gemini_api_key": "live-secret",
            "access_token": "access-secret",
            "x-api-key": "header-secret",
            "retry_after_seconds": 10,
        },
    )

    payload = error.to_payload()

    assert payload == {
        "code": "PROVIDER_RATE_LIMITED",
        "message": "Provider request could not be completed.",
        "detail": {
            "gemini_api_key": "[REDACTED]",
            "access_token": "[REDACTED]",
            "x-api-key": "[REDACTED]",
            "retry_after_seconds": 10,
        },
    }
    assert str(error) == "Provider request could not be completed."
    assert error.args == ("Provider request could not be completed.",)


def test_log_redaction_replaces_known_secret_fields_recursively() -> None:
    rendered = redact_secrets(
        {
            "event": "configured",
            "database_url": "postgresql://user:password@host/db",
            "nested": {
                "authorization": "Bearer live-token",
                "gemini_api_key": "live-api-key",
                "access_token": "access-secret",
                "x-api-key": "header-secret",
                "safe": "visible",
            },
        }
    )

    assert rendered == {
        "event": "configured",
        "database_url": "[REDACTED]",
        "nested": {
            "authorization": "[REDACTED]",
            "gemini_api_key": "[REDACTED]",
            "access_token": "[REDACTED]",
            "x-api-key": "[REDACTED]",
            "safe": "visible",
        },
    }


def test_asgi_entrypoint_enables_json_secret_redacting_logging_before_app_construction() -> None:
    source = Path("src/rag_app/main.py").read_text(encoding="utf-8")
    assert source.index("configure_logging()") < source.index("create_app()")

    from rag_app.logging import JsonFormatter, configure_logging

    root = logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level
    try:
        configure_logging()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JsonFormatter)
        record = logging.LogRecord("rag_app", logging.INFO, "", 1, "ready", (), None)
        record.fields = {"gemini_api_key": "never-log-me"}  # type: ignore[attr-defined]
        rendered = root.handlers[0].formatter.format(record)
        assert "never-log-me" not in rendered
        assert json.loads(rendered)["fields"]["gemini_api_key"] == "[REDACTED]"
    finally:
        root.handlers[:] = old_handlers
        root.setLevel(old_level)
