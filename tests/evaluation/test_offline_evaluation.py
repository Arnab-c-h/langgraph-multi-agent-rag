from __future__ import annotations

import pytest

from rag_app.contracts import ReasoningMode

from .offline_harness import load_fixture_corpus, run_offline_evaluation


def test_fixture_corpus_is_synthetic_complete_and_provenance_rich() -> None:
    corpus = load_fixture_corpus()

    assert {chunk.source_type.value for chunk in corpus} == {"pdf", "web", "github"}
    assert all(chunk.supporting_excerpt == chunk.content for chunk in corpus)
    assert all(chunk.locator and chunk.provenance.get("fixture") for chunk in corpus)


@pytest.mark.evaluation
@pytest.mark.asyncio
async def test_offline_acceptance_metrics_meet_all_thresholds_across_nine_modes() -> None:
    report = await run_offline_evaluation()

    assert report.passed, report.metrics
    assert report.metrics == {name: 1.0 for name in report.thresholds}
    assert "citation_precision" in report.metrics
    assert set(report.modes) == {mode.value for mode in ReasoningMode}
    assert set(report.cases) == {"pdf_retention", "web_safety", "multi_source"}
