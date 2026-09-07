from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from fastapi import UploadFile

from app.models import RetrievedChunk, SearchResult, SourceChunk, StoredSource


class SourceRepository(Protocol):
    async def save_many(self, sources: Sequence[StoredSource]) -> None: ...

    async def list_for_workspace(self, workspace_id: str) -> Sequence[StoredSource]: ...


class SourceIngestor(Protocol):
    async def ingest(
        self, workspace_id: str, files: Sequence[UploadFile]
    ) -> Sequence[StoredSource]: ...


class EmbeddingProvider(Protocol):
    async def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...

    async def embed_query(self, text: str) -> Sequence[float]: ...


class VectorStore(Protocol):
    async def add(
        self,
        chunks: Sequence[SourceChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None: ...

    async def search(
        self, workspace_id: str, vector: Sequence[float], limit: int
    ) -> Sequence[RetrievedChunk]: ...


class WebSearchProvider(Protocol):
    async def search(self, query: str, limit: int) -> Sequence[SearchResult]: ...


class LanguageModel(Protocol):
    async def create_search_queries(self, request: str) -> Sequence[str]: ...

    def stream_answer(
        self,
        request: str,
        uploaded_context: Sequence[RetrievedChunk],
        web_context: Sequence[SearchResult],
    ) -> AsyncIterator[str]: ...


class ResearchAgent(Protocol):
    async def prepare_answer(
        self, workspace_id: str, request: str
    ) -> AsyncIterator[str]: ...
