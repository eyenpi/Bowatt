from __future__ import annotations

import asyncio
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

from fastapi import UploadFile

from app.config import Settings
from app.errors import ApiError
from app.models import StoredSource
from app.ports import SourceRepository


class ValidatingSourceIngestor:
    """Validates and stores source text; chunking and embedding are the next implementation step."""

    def __init__(self, settings: Settings, repository: SourceRepository) -> None:
        self._settings = settings
        self._repository = repository

    async def ingest(
        self, workspace_id: str, files: Sequence[UploadFile]
    ) -> Sequence[StoredSource]:
        if not files:
            raise ApiError(400, "Select at least one source file.")

        if len(files) > self._settings.max_upload_files:
            raise ApiError(
                413,
                f"A maximum of {self._settings.max_upload_files} files can be uploaded at once.",
            )

        sources = await asyncio.gather(
            *(self._read_source(workspace_id, upload) for upload in files)
        )
        await self._repository.save_many(sources)
        return sources

    async def _read_source(self, workspace_id: str, upload: UploadFile) -> StoredSource:
        name = Path(upload.filename or "").name
        if not name:
            raise ApiError(400, "Every source must have a filename.")

        media_type = upload.content_type or "application/octet-stream"
        if not media_type.startswith("text/") and media_type != "application/json":
            raise ApiError(415, f"{name} is not a supported text file.")

        content_bytes = await upload.read(self._settings.max_upload_bytes + 1)
        if len(content_bytes) > self._settings.max_upload_bytes:
            raise ApiError(
                413,
                f"{name} exceeds the {self._settings.max_upload_bytes}-byte upload limit.",
            )

        if not content_bytes:
            raise ApiError(400, f"{name} is empty.")

        try:
            content = content_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ApiError(415, f"{name} must contain UTF-8 text.") from error

        return StoredSource(
            workspace_id=workspace_id,
            name=name,
            size=len(content_bytes),
            media_type=media_type,
            content=content,
            content_hash=sha256(content_bytes).hexdigest(),
        )

