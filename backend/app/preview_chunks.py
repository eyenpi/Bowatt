"""Inspect source chunking locally, without writing a database or calling a model."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

from app.config import Settings
from app.models import StoredSource
from app.services.chunking import TextChunker


def main() -> None:
    settings = Settings.from_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=settings.chunk_max_tokens)
    args = parser.parse_args()
    try:
        with args.source.open("rb") as source_file:
            data = source_file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise ValueError(f"Source exceeds the {settings.max_upload_bytes}-byte limit")
        overlap = args.overlap
        if overlap is None:
            overlap = min(settings.chunk_overlap, max(0, args.chunk_size - 1))
        chunker = TextChunker(args.chunk_size, overlap, args.max_tokens)
        source = StoredSource(
            workspace_id="preview", name=args.source.name, size=len(data),
            media_type="text/plain", content=data.decode("utf-8"),
            content_hash=sha256(data).hexdigest(),
        )
        chunks = chunker.chunk(source)
        if not chunks:
            raise ValueError("Source does not contain indexable text")
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))

    print(json.dumps({
        "source": source.name,
        "chunking": chunker.signature,
        "chunks": [
            {
                "index": chunk.index,
                "headings": chunk.heading_path,
                "start_offset": chunk.start_offset,
                "end_offset": chunk.end_offset,
                "tokens": chunker.count_tokens(chunk.embedding_text),
                "text": chunk.text,
                "embedding_text": chunk.embedding_text,
            }
            for chunk in chunks
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
