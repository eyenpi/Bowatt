from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from fastapi import UploadFile

from app.models import (
    IndexedSource,
    RetrievedChunk,
    SearchPlan,
    SearchResult,
    StoredSource,
)


class SourceRepository(Protocol):
    async def initialize(self) -> None: ...

    async def close(self) -> None: ...

    async def existing_hashes(
        self,
        workspace_id: str,
        content_hashes: Sequence[str],
        embedding_model: str,
        index_signature: str,
    ) -> frozenset[str]: ...

    async def save_indexed_sources(self, sources: Sequence[IndexedSource]) -> None: ...

    async def list_for_workspace(self, workspace_id: str) -> Sequence[StoredSource]: ...

    async def search(
        self,
        workspace_id: str,
        vector: Sequence[float],
        embedding_model: str,
        limit: int,
    ) -> Sequence[RetrievedChunk]: ...


class SourceIngestor(Protocol):
    async def ingest(
        self, workspace_id: str, files: Sequence[UploadFile]
    ) -> Sequence[StoredSource]: ...


class EmbeddingProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    async def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...

    async def embed_query(self, text: str) -> Sequence[float]: ...

    async def close(self) -> None: ...


class SourceRetriever(Protocol):
    async def retrieve(
        self, workspace_id: str, query: str, limit: int | None = None
    ) -> Sequence[RetrievedChunk]: ...


class ResearchProvider(Protocol):
    async def create_search_plan(
        self,
        request: str,
        prior_web_context: Sequence[SearchResult],
        round_number: int,
    ) -> SearchPlan: ...

    async def search(self, query: str, limit: int) -> Sequence[SearchResult]: ...

    async def start_answer(
        self,
        request: str,
        uploaded_context: Sequence[RetrievedChunk],
        web_context: Sequence[SearchResult],
    ) -> AsyncIterator[str]: ...

    async def close(self) -> None: ...


class ResearchAgent(Protocol):
    async def prepare_answer(
        self, workspace_id: str, request: str
    ) -> AsyncIterator[str]: ...
