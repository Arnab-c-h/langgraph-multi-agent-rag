"""Importable ASGI entrypoint; external resources open only in lifespan."""

from rag_app.logging import configure_logging

configure_logging()

from rag_app.api.app import create_app  # noqa: E402  (logging must precede app imports)

app = create_app()
