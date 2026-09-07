from __future__ import annotations

import asyncio
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

from fastapi import UploadFile

from app.config import Settings
from app.errors import ApiError, EmbeddingProviderError
from app.models import IndexedSource, SourceChunk, StoredSource
from app.ports import EmbeddingProvider, SourceRepository
from app.services.chunking import TextChunker
from app.vectors import validated_embeddings


class SourceIngestionService:
    def __init__(
        self,
        settings: Settings,
        repository: SourceRepository,
        embedding_provider: EmbeddingProvider,
        chunker: TextChunker,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._embedding_provider = embedding_provider
        self._chunker = chunker
        self._ingestion_lock = asyncio.Lock()

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

        async with self._ingestion_lock:
            await self._index_new_sources(workspace_id, sources)

        return sources

    async def _index_new_sources(
        self, workspace_id: str, sources: Sequence[StoredSource]
    ) -> None:
        unique_sources = tuple({source.content_hash: source for source in sources}.values())
        existing_hashes = await self._repository.existing_hashes(
            workspace_id,
            tuple(source.content_hash for source in unique_sources),
            self._embedding_provider.model_name,
        )
        new_sources = tuple(
            source for source in unique_sources if source.content_hash not in existing_hashes
        )
        if not new_sources:
            return

        grouped_chunks: list[tuple[StoredSource, tuple[SourceChunk, ...]]] = []
        all_chunks: list[SourceChunk] = []
        for source in new_sources:
            chunks = self._chunker.chunk(source)
            if not chunks:
                raise ApiError(400, f"{source.name} does not contain indexable text.")
            grouped_chunks.append((source, chunks))
            all_chunks.extend(chunks)

        embeddings = await self._embed_chunks(all_chunks)
        indexed_sources: list[IndexedSource] = []
        offset = 0
        for source, chunks in grouped_chunks:
            next_offset = offset + len(chunks)
            indexed_sources.append(
                IndexedSource(
                    source=source,
                    chunks=chunks,
                    embeddings=embeddings[offset:next_offset],
                    embedding_model=self._embedding_provider.model_name,
                )
            )
            offset = next_offset

        await self._repository.save_indexed_sources(indexed_sources)

    async def _embed_chunks(
        self, chunks: Sequence[SourceChunk]
    ) -> tuple[tuple[float, ...], ...]:
        batches = tuple(
            chunks[index : index + self._settings.embedding_batch_size]
            for index in range(0, len(chunks), self._settings.embedding_batch_size)
        )
        semaphore = asyncio.Semaphore(self._settings.embedding_concurrency)

        async def embed_batch(
            batch: Sequence[SourceChunk],
        ) -> Sequence[Sequence[float]]:
            async with semaphore:
                return await self._embedding_provider.embed_documents(
                    tuple(chunk.text for chunk in batch)
                )

        try:
            batch_results = await asyncio.gather(*(embed_batch(batch) for batch in batches))
            validated_batches = tuple(
                validated_embeddings(result, len(batch))
                for batch, result in zip(batches, batch_results, strict=True)
            )
            flattened = tuple(vector for result in validated_batches for vector in result)
            return validated_embeddings(flattened, len(chunks))
        except EmbeddingProviderError as error:
            raise ApiError(error.status_code, error.public_message) from error

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
