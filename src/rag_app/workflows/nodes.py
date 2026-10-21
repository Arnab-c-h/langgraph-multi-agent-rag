"""Reusable, bounded workflow nodes."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from rag_app.config import Settings
from rag_app.contracts import QueryRequest
from rag_app.errors import AppError, ErrorCode
from rag_app.redaction import redact_inline_secrets
from rag_app.retrieval.models import EvidenceAssessment, EvidenceBundle
from rag_app.retrieval.protocols import CitationValidator
from rag_app.workflows.model import ModelOperation, ModelRequest, ModelResult, WorkflowModel
from rag_app.workflows.state import GraphState, RecursiveItem


class RagPreparer(Protocol):
    async def prepare(self, request: QueryRequest) -> EvidenceBundle: ...


_HIDDEN_REASONING_ENVELOPE = re.compile(
    r"(?is)(?:"
    r"<think\b[^>]*>.*?(?:</think>|$)|"
    r"\bBEGIN\s+(?:PRIVATE|HIDDEN|RAW)\s+(?:REASONING|SCRATCHPAD)\b.*?"
    r"(?:\bEND\s+(?:PRIVATE|HIDDEN|RAW)\s+(?:REASONING|SCRATCHPAD)\b|$)|"
    r"(?:^|[,{\n])\s*[\"']?(?:private_reasoning|hidden_reasoning|raw_reasoning|scratchpad)[\"']?\s*:"
    r")"
)
_MARKER_PATTERN = re.compile(r"\[chunk:([0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})\]")


def _safe_text(value: str, *, maximum: int = 4000) -> str:
    if _HIDDEN_REASONING_ENVELOPE.search(value):
        return "[redacted unsafe model output]"
    normalized = " ".join(value.split())
    return redact_inline_secrets(normalized)[:maximum]


def _safe_items(values: Sequence[str], maximum: int) -> list[str]:
    return [_safe_text(value, maximum=512) for value in values[:maximum] if value.strip()]


def _status(label: str, items: Sequence[str] = ()) -> str:
    detail = " | ".join(items)
    return _safe_text(f"{label}: {detail}" if detail else label, maximum=512)


def _merge_evidence(left: EvidenceBundle, right: EvidenceBundle) -> EvidenceBundle:
    by_id: dict[UUID, EvidenceAssessment] = {item.chunk.chunk_id: item for item in left.evidence}
    for item in right.evidence:
        by_id.setdefault(item.chunk.chunk_id, item)
    evidence = list(by_id.values())
    return EvidenceBundle(
        evidence=evidence,
        citations=[item.chunk.citation() for item in evidence],
        used_source_ids=list(dict.fromkeys(item.chunk.source_id for item in evidence)),
        retrieval_summary={"candidate_count": len(evidence), "accepted_count": len(evidence)},
        validation_summary={"citation_count": len(evidence), "rejected_count": 0},
    )


def _request(state: GraphState) -> QueryRequest:
    return QueryRequest.model_validate(state["request"])


def _evidence(state: GraphState) -> EvidenceBundle:
    return EvidenceBundle.model_validate(state["evidence"])


def _history(state: GraphState) -> list[str]:
    """Return only bounded, persisted public turns for model context."""
    return list(state.get("conversation_history", []))[-10:]


@dataclass(frozen=True)
class NodeFactory:
    rag: RagPreparer
    model: WorkflowModel
    citation_validator: CitationValidator
    settings: Settings

    async def _call(
        self,
        state: GraphState,
        operation: ModelOperation,
        *,
        question: str | None = None,
        context: Sequence[str] = (),
        maximum: int = 1,
    ) -> ModelResult:
        return await self.model.complete(
            ModelRequest(
                operation=operation,
                question=question or _request(state).question,
                evidence=_evidence(state),
                context=[_safe_text(item, maximum=1000) for item in context],
                max_items=min(maximum, 4),
            )
        )

    async def retrieve(self, state: GraphState) -> dict[str, object]:
        evidence = await self.rag.prepare(_request(state))
        return {
            "evidence": evidence.model_dump(mode="json"),
            "trace_summary": {
                "accepted_evidence": len(evidence.evidence),
                "candidate_evidence": evidence.retrieval_summary.get("candidate_count", len(evidence.evidence)),
            },
            "status": "Retrieved and graded supporting evidence.",
        }

    def _validated_grounded_result(
        self, result: ModelResult, evidence: EvidenceBundle, *, maximum: int = 16000
    ) -> tuple[str, list[str]]:
        answer = _safe_text(result.text, maximum=maximum)
        citation_ids = list(result.citation_chunk_ids)
        markers = {UUID(value) for value in _MARKER_PATTERN.findall(answer)}
        if not answer.strip() or not citation_ids or markers != set(citation_ids):
            raise AppError(ErrorCode.NO_SUPPORTING_EVIDENCE, "Grounded answer citations are absent or inconsistent")
        self.citation_validator.validate(evidence.evidence, citation_ids)
        return answer, [str(chunk_id) for chunk_id in citation_ids]

    async def plan(self, state: GraphState) -> dict[str, object]:
        result = await self._call(state, ModelOperation.PLAN, maximum=4)
        plan = _safe_items(result.items, 4)
        return {"public_plan": plan, "status": _status("Public plan", plan)}

    async def answer(self, state: GraphState) -> dict[str, object]:
        context = [
            *_history(state),
            *state.get("public_plan", []),
            *state.get("tool_actions", []),
            *state.get("subanswers", []),
        ]
        result = await self._call(state, ModelOperation.ANSWER, context=context)
        return {
            "answer": _safe_text(result.text, maximum=16000),
            "citation_chunk_ids": [str(chunk_id) for chunk_id in result.citation_chunk_ids],
            "status": "Generated a grounded answer draft.",
        }

    async def validate(self, state: GraphState) -> dict[str, object]:
        answer = state.get("answer", "")
        citation_ids = [UUID(chunk_id) for chunk_id in state.get("citation_chunk_ids", [])]
        markers = {UUID(value) for value in _MARKER_PATTERN.findall(answer)}
        if not markers or markers != set(citation_ids):
            raise AppError(ErrorCode.NO_SUPPORTING_EVIDENCE, "Answer citation markers are absent or inconsistent")
        result = await self._call(
            state,
            ModelOperation.VALIDATE,
            context=[answer, *[str(chunk_id) for chunk_id in citation_ids]],
        )
        if not result.approved:
            raise AppError(ErrorCode.NO_SUPPORTING_EVIDENCE, "Answer validation failed")
        citations = self.citation_validator.validate(_evidence(state).evidence, citation_ids)
        trace = dict(state.get("trace_summary", {}))
        trace["citation_count"] = len(citations)
        trace["validation_passed"] = True
        return {
            "citations": [citation.model_dump(mode="json") for citation in citations],
            "validation_passed": True,
            "trace_summary": trace,
            "status": "Validated answer citations and evidence support.",
        }

    async def approach(self, state: GraphState) -> dict[str, object]:
        approaches = list(state.get("approaches", []))
        result = await self._call(state, ModelOperation.APPROACH)
        approach = _safe_text(result.text, maximum=512)
        approaches.append(approach)
        return {"approaches": approaches, "status": _status(f"Public approach {len(approaches)}", [approach])}

    async def rank(self, state: GraphState) -> dict[str, object]:
        approaches = state.get("approaches", [])[: self.settings.tot_branches]
        result = await self._call(state, ModelOperation.RANK, context=approaches, maximum=len(approaches))
        selected = min(result.selected_index, max(0, len(approaches) - 1))
        rankings = _safe_items(result.items, self.settings.tot_branches) or [
            f"Selected approach {selected + 1} by evidence."
        ]
        return {
            "public_plan": [approaches[selected]] if approaches else [],
            "branch_rankings": rankings,
            "status": _status("Evidence ranking", rankings),
        }

    async def act(self, state: GraphState) -> dict[str, object]:
        actions = list(state.get("tool_actions", []))
        result = await self._call(state, ModelOperation.ACT, context=actions)
        inspection_query = _safe_text(result.text, maximum=400)
        if not inspection_query or inspection_query.startswith("[redacted"):
            inspection_query = _request(state).question
        action_request = _request(state).model_copy(update={"question": inspection_query})
        action_evidence = await self.rag.prepare(action_request)
        merged = _merge_evidence(_evidence(state), action_evidence)
        count = int(state.get("action_count", 0)) + 1
        action = f"source_inspection query={inspection_query} accepted={len(action_evidence.evidence)}"
        actions.append(action)
        return {
            "evidence": merged.model_dump(mode="json"),
            "tool_actions": actions,
            "action_count": count,
            "action_complete": result.complete,
            "status": _status(f"Allowlisted retrieval action {count}", [action]),
        }

    async def draft(self, state: GraphState) -> dict[str, object]:
        result = await self._call(state, ModelOperation.ANSWER, context=_history(state))
        return {
            "answer": _safe_text(result.text, maximum=16000),
            "citation_chunk_ids": [str(chunk_id) for chunk_id in result.citation_chunk_ids],
            "status": "Created a grounded draft.",
        }

    async def critique(self, state: GraphState) -> dict[str, object]:
        result = await self._call(state, ModelOperation.CRITIQUE, context=[state.get("answer", "")])
        return {
            "critique_approved": result.approved,
            "status": "Validator outcome: draft approved."
            if result.approved
            else "Validator outcome: one grounded revision requested.",
        }

    async def revise(self, state: GraphState) -> dict[str, object]:
        result = await self._call(
            state, ModelOperation.REVISE, context=[*_history(state), state.get("answer", "")]
        )
        return {
            "answer": _safe_text(result.text, maximum=16000),
            "citation_chunk_ids": [str(chunk_id) for chunk_id in result.citation_chunk_ids],
            "revision_count": int(state.get("revision_count", 0)) + 1,
            "status": "Applied the single permitted grounded revision.",
        }

    async def candidate(self, state: GraphState) -> dict[str, object]:
        candidates = list(state.get("candidates", []))
        candidate_citations = list(state.get("candidate_citations", []))
        result = await self._call(state, ModelOperation.CANDIDATE, context=_history(state))
        answer, citations = self._validated_grounded_result(result, _evidence(state))
        candidates.append(answer)
        candidate_citations.append(citations)
        return {
            "candidates": candidates,
            "candidate_citations": candidate_citations,
            "status": f"Generated grounded candidate {len(candidates)}.",
        }

    async def judge(self, state: GraphState) -> dict[str, object]:
        candidates = state.get("candidates", [])[: self.settings.consistency_candidates]
        result = await self._call(state, ModelOperation.JUDGE, context=candidates, maximum=len(candidates))
        normalized = [" ".join(answer.lower().split()) for answer in candidates]
        counts = Counter(normalized)
        selected = max(range(len(candidates)), key=lambda index: (counts[normalized[index]], -index))
        answer = candidates[selected]
        citations = state.get("candidate_citations", [])[selected]
        rankings = _safe_items(result.items, self.settings.consistency_candidates)
        return {
            "answer": answer,
            "citation_chunk_ids": citations,
            "branch_rankings": rankings,
            "status": _status("Deterministic evidence consensus", rankings),
        }

    async def decompose(self, state: GraphState) -> dict[str, object]:
        result = await self._call(state, ModelOperation.DECOMPOSE, maximum=self.settings.decomposition_questions)
        questions = _safe_items(result.items, self.settings.decomposition_questions)
        return {"subquestions": questions, "subquestion_index": 0, "status": _status("Subquestions", questions)}

    async def decompose_least_to_most(self, state: GraphState) -> dict[str, object]:
        result = await self._call(
            state,
            ModelOperation.DECOMPOSE,
            context=["Return questions ordered from foundational evidence to the final synthesis."],
            maximum=self.settings.decomposition_questions,
        )
        questions = _safe_items(result.items, self.settings.decomposition_questions)
        return {
            "subquestions": questions,
            "subquestion_index": 0,
            "status": _status("Foundational-to-final subquestions", questions),
        }

    async def solve_subquestion(self, state: GraphState) -> dict[str, object]:
        index = int(state.get("subquestion_index", 0))
        questions = state.get("subquestions", [])
        if index >= len(questions):
            return {"status": "Finished grounded subquestions."}
        question = questions[index]
        parent = _request(state)
        subrequest = parent.model_copy(update={"question": question})
        sub_evidence = await self.rag.prepare(subrequest)
        merged = _merge_evidence(_evidence(state), sub_evidence)
        result = await self.model.complete(
            ModelRequest(operation=ModelOperation.SUBANSWER, question=question, evidence=sub_evidence)
        )
        answers = list(state.get("subanswers", []))
        answer, _ = self._validated_grounded_result(result, sub_evidence, maximum=4000)
        answers.append(answer)
        return {
            "evidence": merged.model_dump(mode="json"),
            "subanswers": answers,
            "subquestion_index": index + 1,
            "status": f"Solved grounded subquestion {index + 1}.",
        }

    async def synthesize(self, state: GraphState) -> dict[str, object]:
        generated_context = state.get("subanswers", []) or state.get("candidates", [])
        context = [*_history(state), *generated_context]
        result = await self._call(state, ModelOperation.SYNTHESIZE, context=context)
        return {
            "answer": _safe_text(result.text, maximum=16000),
            "citation_chunk_ids": [str(chunk_id) for chunk_id in result.citation_chunk_ids],
            "status": "Synthesized a grounded answer.",
        }

    async def recursive_seed(self, state: GraphState) -> dict[str, object]:
        root: RecursiveItem = {"question": _request(state).question, "depth": 0}
        return {
            "recursive_queue": [root],
            "recursive_nodes": [],
            "recursive_node_count": 0,
            "status": "Started bounded recursive decomposition.",
        }

    async def recursive_step(self, state: GraphState) -> dict[str, object]:
        queue = list(state.get("recursive_queue", []))
        nodes = list(state.get("recursive_nodes", []))
        if not queue:
            return {"status": "Finished recursive decomposition."}
        item = queue.pop(0)
        nodes.append(item)
        subanswers = list(state.get("subanswers", []))
        evidence_update: dict[str, object] = {}
        maximum_nodes = sum(
            self.settings.recursive_fanout**depth for depth in range(self.settings.recursive_max_depth + 1)
        )
        if item["depth"] < self.settings.recursive_max_depth and len(nodes) < maximum_nodes:
            result = await self._call(
                state,
                ModelOperation.DECOMPOSE,
                question=item["question"],
                context=[entry["question"] for entry in nodes],
                maximum=self.settings.recursive_fanout,
            )
            remaining = maximum_nodes - len(nodes) - len(queue)
            children = _safe_items(result.items, min(self.settings.recursive_fanout, max(0, remaining)))
            queue.extend({"question": child, "depth": item["depth"] + 1} for child in children)
        else:
            parent = _request(state)
            leaf_request = parent.model_copy(update={"question": item["question"]})
            leaf_evidence = await self.rag.prepare(leaf_request)
            result = await self.model.complete(
                ModelRequest(
                    operation=ModelOperation.SUBANSWER,
                    question=item["question"],
                    evidence=leaf_evidence,
                    context=subanswers,
                )
            )
            answer, _ = self._validated_grounded_result(result, leaf_evidence, maximum=4000)
            subanswers.append(answer)
            merged = _merge_evidence(_evidence(state), leaf_evidence)
            evidence_update = {"evidence": merged.model_dump(mode="json")}
        return {
            **evidence_update,
            "recursive_queue": queue,
            "recursive_nodes": nodes,
            "recursive_node_count": len(nodes),
            "subanswers": subanswers,
            "status": _status(f"Recursive public node {len(nodes)}", [item["question"]]),
        }
