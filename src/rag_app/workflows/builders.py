"""Nine distinct bounded LangGraph builders."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from rag_app.config import Settings
from rag_app.contracts import ReasoningMode
from rag_app.workflows.nodes import NodeFactory
from rag_app.workflows.state import GraphState


def _compile(graph: StateGraph[GraphState], mode: ReasoningMode, checkpointer: Any) -> CompiledStateGraph[GraphState]:
    return graph.compile(checkpointer=checkpointer, name=f"workflow_{mode.value}")


def build_standard(nodes: NodeFactory, checkpointer: Any = None) -> CompiledStateGraph[GraphState]:
    graph = StateGraph(GraphState)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("answer", nodes.answer)
    graph.add_node("validate", nodes.validate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "validate")
    graph.add_edge("validate", END)
    return _compile(graph, ReasoningMode.STANDARD, checkpointer)


def build_cot(nodes: NodeFactory, checkpointer: Any = None) -> CompiledStateGraph[GraphState]:
    graph = StateGraph(GraphState)
    for name, node in (
        ("retrieve", nodes.retrieve),
        ("public_plan", nodes.plan),
        ("answer", nodes.answer),
        ("validate", nodes.validate),
    ):
        graph.add_node(name, node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "public_plan")
    graph.add_edge("public_plan", "answer")
    graph.add_edge("answer", "validate")
    graph.add_edge("validate", END)
    return _compile(graph, ReasoningMode.COT, checkpointer)


def build_tot(nodes: NodeFactory, settings: Settings, checkpointer: Any = None) -> CompiledStateGraph[GraphState]:
    graph = StateGraph(GraphState)
    for name, node in (
        ("retrieve", nodes.retrieve),
        ("approach", nodes.approach),
        ("rank", nodes.rank),
        ("answer", nodes.answer),
        ("validate", nodes.validate),
    ):
        graph.add_node(name, node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "approach")
    graph.add_conditional_edges(
        "approach",
        lambda state: "rank" if len(state.get("approaches", [])) >= settings.tot_branches else "approach",
        {"approach": "approach", "rank": "rank"},
    )
    graph.add_edge("rank", "answer")
    graph.add_edge("answer", "validate")
    graph.add_edge("validate", END)
    return _compile(graph, ReasoningMode.TOT, checkpointer)


def build_react(nodes: NodeFactory, settings: Settings, checkpointer: Any = None) -> CompiledStateGraph[GraphState]:
    graph = StateGraph(GraphState)
    for name, node in (
        ("retrieve", nodes.retrieve),
        ("act", nodes.act),
        ("answer", nodes.answer),
        ("validate", nodes.validate),
    ):
        graph.add_node(name, node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "act")
    graph.add_conditional_edges(
        "act",
        lambda state: (
            "answer"
            if state.get("action_complete", False) or int(state.get("action_count", 0)) >= settings.react_max_actions
            else "act"
        ),
        {"act": "act", "answer": "answer"},
    )
    graph.add_edge("answer", "validate")
    graph.add_edge("validate", END)
    return _compile(graph, ReasoningMode.REACT, checkpointer)


def build_self_reflection(
    nodes: NodeFactory, settings: Settings, checkpointer: Any = None
) -> CompiledStateGraph[GraphState]:
    graph = StateGraph(GraphState)
    for name, node in (
        ("retrieve", nodes.retrieve),
        ("draft", nodes.draft),
        ("critique", nodes.critique),
        ("revise", nodes.revise),
        ("validate", nodes.validate),
    ):
        graph.add_node(name, node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "draft")
    graph.add_edge("draft", "critique")
    graph.add_conditional_edges(
        "critique",
        lambda state: (
            "validate"
            if state.get("critique_approved", False)
            or int(state.get("revision_count", 0)) >= settings.reflection_revisions
            else "revise"
        ),
        {"revise": "revise", "validate": "validate"},
    )
    graph.add_edge("revise", "validate")
    graph.add_edge("validate", END)
    return _compile(graph, ReasoningMode.SELF_REFLECTION, checkpointer)


def build_self_consistency(
    nodes: NodeFactory, settings: Settings, checkpointer: Any = None
) -> CompiledStateGraph[GraphState]:
    graph = StateGraph(GraphState)
    for name, node in (
        ("retrieve", nodes.retrieve),
        ("candidate", nodes.candidate),
        ("judge", nodes.judge),
        ("validate", nodes.validate),
    ):
        graph.add_node(name, node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "candidate")
    graph.add_conditional_edges(
        "candidate",
        lambda state: "judge" if len(state.get("candidates", [])) >= settings.consistency_candidates else "candidate",
        {"candidate": "candidate", "judge": "judge"},
    )
    graph.add_edge("judge", "validate")
    graph.add_edge("validate", END)
    return _compile(graph, ReasoningMode.SELF_CONSISTENCY, checkpointer)


def _build_subquestions(
    nodes: NodeFactory,
    settings: Settings,
    mode: ReasoningMode,
    checkpointer: Any,
) -> CompiledStateGraph[GraphState]:
    graph = StateGraph(GraphState)
    decompose_node = nodes.decompose_least_to_most if mode is ReasoningMode.LEAST_TO_MOST else nodes.decompose
    for name, node in (
        ("retrieve", nodes.retrieve),
        ("decompose", decompose_node),
        ("solve", nodes.solve_subquestion),
        ("synthesize", nodes.synthesize),
        ("validate", nodes.validate),
    ):
        graph.add_node(name, node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "decompose")
    graph.add_conditional_edges(
        "decompose",
        lambda state: "solve" if state.get("subquestions", []) else "synthesize",
        {"solve": "solve", "synthesize": "synthesize"},
    )
    graph.add_conditional_edges(
        "solve",
        lambda state: (
            "synthesize"
            if int(state.get("subquestion_index", 0))
            >= min(len(state.get("subquestions", [])), settings.decomposition_questions)
            else "solve"
        ),
        {"solve": "solve", "synthesize": "synthesize"},
    )
    graph.add_edge("synthesize", "validate")
    graph.add_edge("validate", END)
    return _compile(graph, mode, checkpointer)


def build_decomposed(
    nodes: NodeFactory, settings: Settings, checkpointer: Any = None
) -> CompiledStateGraph[GraphState]:
    return _build_subquestions(nodes, settings, ReasoningMode.DECOMPOSED, checkpointer)


def build_least_to_most(
    nodes: NodeFactory, settings: Settings, checkpointer: Any = None
) -> CompiledStateGraph[GraphState]:
    return _build_subquestions(nodes, settings, ReasoningMode.LEAST_TO_MOST, checkpointer)


def build_recursive(nodes: NodeFactory, checkpointer: Any = None) -> CompiledStateGraph[GraphState]:
    graph = StateGraph(GraphState)
    for name, node in (
        ("retrieve", nodes.retrieve),
        ("seed", nodes.recursive_seed),
        ("recurse", nodes.recursive_step),
        ("synthesize", nodes.synthesize),
        ("validate", nodes.validate),
    ):
        graph.add_node(name, node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "seed")
    graph.add_edge("seed", "recurse")
    graph.add_conditional_edges(
        "recurse",
        lambda state: "recurse" if state.get("recursive_queue", []) else "synthesize",
        {"recurse": "recurse", "synthesize": "synthesize"},
    )
    graph.add_edge("synthesize", "validate")
    graph.add_edge("validate", END)
    return _compile(graph, ReasoningMode.RECURSIVE, checkpointer)
