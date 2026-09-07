from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.ports import (
    EmbeddingProvider,
    ResearchAgent,
    ResearchProvider,
    SourceIngestor,
    SourceRepository,
    SourceRetriever,
)
from app.providers.openai_embeddings import (
    OpenAIEmbeddingProvider,
    UnavailableEmbeddingProvider,
)
from app.providers.openai_research import (
    OpenAIResearchProvider,
    UnavailableResearchProvider,
)
from app.services.chunking import TextChunker
from app.services.research import ResearchAgentService
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
    research_provider: ResearchProvider

    async def initialize(self) -> None:
        await self.source_repository.initialize()

    async def close(self) -> None:
        try:
            await self.research_provider.close()
        finally:
            try:
                await self.embedding_provider.close()
            finally:
                await self.source_repository.close()


def build_container(
    settings: Settings,
    *,
    embedding_provider: EmbeddingProvider | None = None,
    research_provider: ResearchProvider | None = None,
) -> AppContainer:
    repository = SqliteSourceRepository(settings.database_path)
    resolved_embedding_provider = embedding_provider or _build_embedding_provider(settings)
    resolved_research_provider = research_provider or _build_research_provider(settings)
    chunker = TextChunker(
        settings.chunk_size, settings.chunk_overlap, settings.chunk_max_tokens
    )
    source_retriever = SourceRetrievalService(
        settings,
        repository,
        resolved_embedding_provider,
    )
    return AppContainer(
        source_ingestor=SourceIngestionService(
            settings,
            repository,
            resolved_embedding_provider,
            chunker,
        ),
        source_retriever=source_retriever,
        research_agent=ResearchAgentService(
            settings,
            source_retriever,
            resolved_research_provider,
        ),
        source_repository=repository,
        embedding_provider=resolved_embedding_provider,
        research_provider=resolved_research_provider,
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


def _build_research_provider(settings: Settings) -> ResearchProvider:
    if not settings.openai_api_key:
        return UnavailableResearchProvider()

    return OpenAIResearchProvider(
        api_key=settings.openai_api_key,
        model=settings.research_model,
        timeout_seconds=settings.research_provider_timeout_seconds,
        max_output_tokens=settings.research_max_output_tokens,
        close_timeout_seconds=settings.research_stream_close_timeout_seconds,
    )
