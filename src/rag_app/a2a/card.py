"""Profile-aware official protobuf Agent Card."""

from __future__ import annotations

from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill

from rag_app.config import Settings


def build_agent_card(settings: Settings) -> AgentCard:
    skills = [
        AgentSkill(
            id="query",
            name="Grounded query",
            description="Answer a question from the indexed corpus with citations.",
            tags=["rag", "langgraph", "citations"],
            examples=["What does the indexed documentation say about retries?"],
            input_modes=["text/plain"],
            output_modes=["text/plain", "application/json"],
        ),
        AgentSkill(
            id="list_sources",
            name="List sources",
            description="List the public metadata for indexed knowledge sources.",
            tags=["rag", "sources"],
            examples=["List available sources"],
            input_modes=["text/plain"],
            output_modes=["application/json"],
        ),
    ]
    skills.append(
        AgentSkill(
            id="ingestion",
            name="Ingest source",
            description=(
                "Ingest a bounded PDF, public web URL, or public GitHub repository; "
                "requires admin Bearer authentication."
            ),
            tags=["ingestion", "admin"],
            input_modes=["text/plain", "application/pdf"],
            output_modes=["application/json"],
        )
    )
    base = str(settings.public_base_url).rstrip("/")
    return AgentCard(
        name="LangGraph Multi-Agent RAG",
        description="Strict evidence-grounded retrieval with nine bounded LangGraph workflows.",
        supported_interfaces=[
            AgentInterface(
                url=f"{base}/a2a",
                protocol_binding="JSONRPC",
                protocol_version="1.0",
            )
        ],
        version="0.1.0",
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain", "application/json"],
        skills=skills,
    )
