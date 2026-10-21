"""Construct the official A2A handler and FastAPI-compatible routes."""

from __future__ import annotations

from dataclasses import dataclass

from a2a.server.request_handlers import DefaultRequestHandlerV2
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import TaskStore
from starlette.routing import BaseRoute

from rag_app.a2a.card import build_agent_card
from rag_app.a2a.executor import RagAgentExecutor
from rag_app.api.limits import QueryCapacity
from rag_app.api.protocols import Runner, SourceService
from rag_app.api.security import AdminAuthorizer
from rag_app.config import Settings


@dataclass(frozen=True)
class A2AGateway:
    handler: DefaultRequestHandlerV2
    routes: list[BaseRoute]

    async def aclose(self) -> None:
        await self.handler.aclose()


def build_gateway(
    *,
    runner: Runner,
    sources: SourceService,
    settings: Settings,
    task_store: TaskStore,
    authorizer: AdminAuthorizer,
    capacity: QueryCapacity,
) -> A2AGateway:
    card = build_agent_card(settings)
    executor = RagAgentExecutor(
        runner=runner,
        sources=sources,
        settings=settings,
        authorizer=authorizer,
        capacity=capacity,
    )
    handler = DefaultRequestHandlerV2(agent_executor=executor, task_store=task_store, agent_card=card)
    routes: list[BaseRoute] = [
        *create_agent_card_routes(card),
        *create_jsonrpc_routes(handler, rpc_url="/a2a"),
    ]
    return A2AGateway(handler=handler, routes=routes)
