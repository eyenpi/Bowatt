from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    environment: str
    mode: str


class ResearchRequest(BaseModel):
    request: str = Field(min_length=1)


class UploadedFileResponse(BaseModel):
    name: str
    size: int
    type: str


class UploadResponse(BaseModel):
    uploaded: list[UploadedFileResponse]


@dataclass(frozen=True, slots=True)
class StoredSource:
    workspace_id: str
    name: str
    size: int
    media_type: str
    content: str
    content_hash: str


@dataclass(frozen=True, slots=True)
class SourceChunk:
    source_hash: str
    source_name: str
    workspace_id: str
    index: int
    text: str
    heading_path: tuple[str, ...] = ()
    start_offset: int | None = None
    end_offset: int | None = None
    context_prefix: str = ""

    @property
    def embedding_text(self) -> str:
        return self.context_prefix + self.text

    @property
    def location(self) -> str:
        location = f"chunk {self.index + 1}"
        if self.start_offset is not None and self.end_offset is not None:
            location += f", characters {self.start_offset + 1}–{self.end_offset}"
        return location


@dataclass(frozen=True, slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str


@dataclass(frozen=True, slots=True)
class SearchPlan:
    queries: tuple[str, ...]
    is_complete: bool


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk: SourceChunk
    score: float


@dataclass(frozen=True, slots=True)
class IndexedSource:
    source: StoredSource
    chunks: tuple[SourceChunk, ...]
    embeddings: tuple[tuple[float, ...], ...]
    embedding_model: str
    index_signature: str = ""
