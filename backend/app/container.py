from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.ports import (
    EmbeddingProvider,
    ResearchAgent,
    SourceIngestor,
    SourceRepository,
    SourceRetriever,
)
from app.providers.openai_embeddings import (
    OpenAIEmbeddingProvider,
    UnavailableEmbeddingProvider,
)
from app.services.chunking import TextChunker
from app.services.research import ScaffoldResearchAgent
from app.services.retrieval import SourceRetrievalService
from app.services.source_ingestion import SourceIngestionService
from app.storage import SqliteSourceRepository


@dataclass(frozen=True, slots=True)
class AppContainer:
    source_ingestor: SourceIngestor
    source_retriever: SourceRetriever
    research_agent: ResearchAgent
    source_repository: SourceRepository
    embedding_provider: EmbeddingProvider

    async def initialize(self) -> None:
        await self.source_repository.initialize()

    async def close(self) -> None:
        await self.embedding_provider.close()
        await self.source_repository.close()


def build_container(
    settings: Settings,
    *,
    embedding_provider: EmbeddingProvider | None = None,
) -> AppContainer:
    repository = SqliteSourceRepository(settings.database_path)
    resolved_embedding_provider = embedding_provider or _build_embedding_provider(settings)
    chunker = TextChunker(settings.chunk_size, settings.chunk_overlap)
    return AppContainer(
        source_ingestor=SourceIngestionService(
            settings,
            repository,
            resolved_embedding_provider,
            chunker,
        ),
        source_retriever=SourceRetrievalService(
            settings,
            repository,
            resolved_embedding_provider,
        ),
        research_agent=ScaffoldResearchAgent(settings, repository),
        source_repository=repository,
        embedding_provider=resolved_embedding_provider,
    )


def _build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if not settings.openai_api_key:
        return UnavailableEmbeddingProvider(settings.embedding_model)

    return OpenAIEmbeddingProvider(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        timeout_seconds=settings.embedding_timeout_seconds,
    )
