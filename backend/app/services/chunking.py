from __future__ import annotations

from app.models import SourceChunk, StoredSource


class TextChunker:
    """Creates deterministic, whitespace-normalized character chunks."""

    def __init__(self, chunk_size: int, overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be zero or greater and smaller than chunk_size")

        self._chunk_size = chunk_size
        self._overlap = overlap

    def chunk(self, source: StoredSource) -> tuple[SourceChunk, ...]:
        text = " ".join(source.content.split())
        if not text:
            return ()

        chunks: list[SourceChunk] = []
        start = 0

        while start < len(text):
            hard_end = min(start + self._chunk_size, len(text))
            end = hard_end

            if hard_end < len(text):
                soft_boundary = text.rfind(" ", start + self._chunk_size // 2, hard_end + 1)
                if soft_boundary > start:
                    end = soft_boundary

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    SourceChunk(
                        source_hash=source.content_hash,
                        source_name=source.name,
                        workspace_id=source.workspace_id,
                        index=len(chunks),
                        text=chunk_text,
                    )
                )

            if end >= len(text):
                break

            next_start = max(end - self._overlap, start + 1)
            if next_start > start:
                previous_space = text.rfind(" ", start + 1, next_start + 1)
                if previous_space > start:
                    next_start = previous_space + 1

            start = next_start

        return tuple(chunks)

