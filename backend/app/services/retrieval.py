from __future__ import annotations

from collections.abc import Sequence

from app.config import Settings
from app.errors import ApiError, EmbeddingProviderError, StorageError
from app.models import RetrievedChunk
from app.ports import EmbeddingProvider, SourceRepository
from app.vectors import validated_vector


class SourceRetrievalService:
    def __init__(
        self,
        settings: Settings,
        repository: SourceRepository,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._embedding_provider = embedding_provider

    async def retrieve(
        self, workspace_id: str, query: str, limit: int | None = None
    ) -> Sequence[RetrievedChunk]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ApiError(400, "Retrieval query must not be blank.")

        resolved_limit = limit or self._settings.retrieval_limit
        if resolved_limit <= 0:
            raise ApiError(400, "Retrieval limit must be greater than zero.")

        try:
            query_vector = validated_vector(
                await self._embedding_provider.embed_query(normalized_query)
            )
            return await self._repository.search(
                workspace_id,
                query_vector,
                self._embedding_provider.model_name,
                resolved_limit,
            )
        except EmbeddingProviderError as error:
            raise ApiError(error.status_code, error.public_message) from error
        except StorageError as error:
            raise ApiError(500, "Source retrieval failed.") from error

