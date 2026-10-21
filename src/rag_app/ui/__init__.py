"""Gradio presentation layer for the co-located FastAPI service."""

from rag_app.ui.app import UiController, build_ui, mount_ui

__all__ = ["UiController", "build_ui", "mount_ui"]
