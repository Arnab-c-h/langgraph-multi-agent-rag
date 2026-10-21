"""Gemini embedding adapter with an asynchronous, fake-friendly boundary."""

from __future__ import annotations

import asyncio
from typing import Any, cast

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import SecretStr


class GeminiEmbeddingAdapter(Embeddings):
    dimensions = 768

    def __init__(self, api_key: str, client: Any | None = None) -> None:
        self._client = client or GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2", google_api_key=SecretStr(api_key)
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._client.embed_documents(texts, output_dimensionality=self.dimensions)
        if any(len(vector) != self.dimensions for vector in vectors):
            raise ValueError(f"Gemini embeddings must contain {self.dimensions} dimensions")
        return cast(list[list[float]], vectors)

    def embed_query(self, text: str) -> list[float]:
        vector = self._client.embed_query(text, output_dimensionality=self.dimensions)
        if len(vector) != self.dimensions:
            raise ValueError(f"Gemini embeddings must contain {self.dimensions} dimensions")
        return cast(list[float], vector)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)
