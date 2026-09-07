from __future__ import annotations

import asyncio
from collections.abc import Sequence

from app.models import StoredSource


class InMemorySourceRepository:
    """Temporary process-local storage; replace with persisted source/vector storage."""

    def __init__(self) -> None:
        self._sources: dict[str, dict[str, StoredSource]] = {}
        self._lock = asyncio.Lock()

    async def save_many(self, sources: Sequence[StoredSource]) -> None:
        async with self._lock:
            for source in sources:
                workspace_sources = self._sources.setdefault(source.workspace_id, {})
                workspace_sources[source.content_hash] = source

    async def list_for_workspace(self, workspace_id: str) -> Sequence[StoredSource]:
        async with self._lock:
            return tuple(self._sources.get(workspace_id, {}).values())

