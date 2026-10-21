"""Workflow graph registry."""

from __future__ import annotations

from typing import Any

from langgraph.graph.state import CompiledStateGraph

from rag_app.config import Settings
from rag_app.contracts import ReasoningMode
from rag_app.retrieval.protocols import CitationValidator
from rag_app.workflows.builders import (
    build_cot,
    build_decomposed,
    build_least_to_most,
    build_react,
    build_recursive,
    build_self_consistency,
    build_self_reflection,
    build_standard,
    build_tot,
)
from rag_app.workflows.model import WorkflowModel
from rag_app.workflows.nodes import NodeFactory, RagPreparer
from rag_app.workflows.state import GraphState


class WorkflowRegistry:
    def __init__(
        self,
        *,
        rag: RagPreparer,
        model: WorkflowModel,
        citation_validator: CitationValidator,
        settings: Settings,
        checkpointer: Any = None,
    ) -> None:
        self.has_checkpointer = checkpointer is not None
        nodes = NodeFactory(rag=rag, model=model, citation_validator=citation_validator, settings=settings)
        self._graphs: dict[ReasoningMode, CompiledStateGraph[GraphState]] = {
            ReasoningMode.STANDARD: build_standard(nodes, checkpointer),
            ReasoningMode.COT: build_cot(nodes, checkpointer),
            ReasoningMode.TOT: build_tot(nodes, settings, checkpointer),
            ReasoningMode.REACT: build_react(nodes, settings, checkpointer),
            ReasoningMode.SELF_REFLECTION: build_self_reflection(nodes, settings, checkpointer),
            ReasoningMode.SELF_CONSISTENCY: build_self_consistency(nodes, settings, checkpointer),
            ReasoningMode.DECOMPOSED: build_decomposed(nodes, settings, checkpointer),
            ReasoningMode.LEAST_TO_MOST: build_least_to_most(nodes, settings, checkpointer),
            ReasoningMode.RECURSIVE: build_recursive(nodes, checkpointer),
        }

    def graph_for(self, mode: ReasoningMode) -> CompiledStateGraph[GraphState]:
        return self._graphs[mode]
