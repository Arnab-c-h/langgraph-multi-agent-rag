"""Gemini-backed relevance grading with a stable public error boundary."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field, SecretStr

from rag_app.config import Settings
from rag_app.errors import AppError, ErrorCode
from rag_app.retrieval.models import EvidenceAssessment, RetrievedChunk


class _GradeDecision(BaseModel):
    relevant: bool
    confidence: float = Field(ge=0, le=1)
    summary: str = Field(default="", max_length=512)


class GeminiEvidenceGrader:
    """Grade each retrieved chunk through Gemini structured output."""

    def __init__(
        self,
        api_key: str | None = None,
        model: Any | None = None,
        model_name: str = "gemini-3.5-flash",
    ) -> None:
        self.model_name = model_name
        if model is None:
            if not api_key:
                raise ValueError("A Gemini API key or injected model is required")
            from langchain_google_genai import ChatGoogleGenerativeAI

            model = ChatGoogleGenerativeAI(model=model_name, google_api_key=SecretStr(api_key))
        self._structured_model: Any = model.with_structured_output(_GradeDecision)

    @classmethod
    def from_settings(cls, settings: Settings, model: Any | None = None) -> GeminiEvidenceGrader:
        """Compose a grader from application settings without exposing a secret."""
        api_key = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key is not None else None
        return cls(api_key=api_key, model=model, model_name=settings.model_name)

    async def grade(self, question: str, candidates: Sequence[RetrievedChunk]) -> list[EvidenceAssessment]:
        assessments: list[EvidenceAssessment] = []
        for chunk in candidates:
            try:
                raw = await self._structured_model.ainvoke(
                    {
                        "question": question,
                        "chunk": {
                            "content": chunk.content,
                            "source_title": chunk.source_title,
                            "locator": chunk.locator,
                        },
                        "instruction": (
                            "Mark only directly supporting evidence relevant. Return structured fields only."
                        ),
                    }
                )
                decision = _parse_decision(raw)
            except Exception as error:
                raise AppError(
                    ErrorCode.PROVIDER_RATE_LIMITED,
                    "Evidence grading provider failed",
                    {"provider_error_type": type(error).__name__},
                ) from error
            assessments.append(
                EvidenceAssessment(
                    chunk=chunk,
                    relevant=decision.relevant,
                    confidence=decision.confidence,
                    summary=decision.summary,
                )
            )
        return assessments


def _parse_decision(raw: Any) -> _GradeDecision:
    if isinstance(raw, BaseModel):
        return _GradeDecision.model_validate(raw.model_dump())
    return _GradeDecision.model_validate(raw)
