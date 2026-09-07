from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from app.config import Settings
from app.errors import ApiError
from app.ports import SourceRepository


class ScaffoldResearchAgent:
    """Contract-valid placeholder to be replaced by the research orchestration service."""

    def __init__(self, settings: Settings, repository: SourceRepository) -> None:
        self._settings = settings
        self._repository = repository

    async def prepare_answer(
        self, workspace_id: str, request: str
    ) -> AsyncIterator[str]:
        normalized_request = request.strip()
        if not normalized_request:
            raise ApiError(400, "Research request must not be blank.")

        if len(normalized_request) > self._settings.max_request_characters:
            raise ApiError(
                413,
                "Research request exceeds the configured character limit.",
            )

        sources = await self._repository.list_for_workspace(workspace_id)
        return self._stream_placeholder(len(sources))

    async def _stream_placeholder(self, source_count: int) -> AsyncIterator[str]:
        chunks = (
            "# Backend scaffold ready\n\n",
            f"The current workspace contains **{source_count}** uploaded source(s).\n\n",
            "> Research planning, retrieval, web search, and LLM synthesis "
            "are not implemented yet.\n",
        )

        for chunk in chunks:
            await asyncio.sleep(2)  # Simulate async processing delay
            yield chunk
