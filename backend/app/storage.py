from __future__ import annotations

import asyncio
import math
import struct
from collections.abc import Sequence
from pathlib import Path

import aiosqlite

from app.errors import StorageError
from app.models import IndexedSource, RetrievedChunk, SourceChunk, StoredSource

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id TEXT NOT NULL,
    name TEXT NOT NULL,
    size INTEGER NOT NULL,
    media_type TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (workspace_id, content_hash)
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    embedding_dimensions INTEGER NOT NULL,
    embedding_model TEXT NOT NULL,
    UNIQUE (source_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS sources_workspace_idx ON sources(workspace_id);
CREATE INDEX IF NOT EXISTS chunks_source_idx ON chunks(source_id);
CREATE INDEX IF NOT EXISTS chunks_model_idx ON chunks(embedding_model);
"""


def _encode_vector(vector: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vector)}d", *vector)


def _decode_vector(data: bytes, dimensions: int) -> tuple[float, ...]:
    expected_bytes = dimensions * 8
    if dimensions <= 0 or len(data) != expected_bytes:
        raise StorageError("Stored embedding data has an invalid size.")
    return tuple(struct.unpack(f"<{dimensions}d", data))


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise StorageError("Stored and query embedding dimensions do not match.")

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise StorageError("Stored or query embedding has zero length.")

    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def _rank_candidates(
    candidates: Sequence[tuple[SourceChunk, tuple[float, ...]]],
    query_vector: Sequence[float],
    limit: int,
) -> tuple[RetrievedChunk, ...]:
    ranked = [
        RetrievedChunk(chunk=chunk, score=_cosine_similarity(vector, query_vector))
        for chunk, vector in candidates
    ]
    ranked.sort(key=lambda result: (-result.score, result.chunk.source_name, result.chunk.index))
    return tuple(ranked[:limit])


class SqliteSourceRepository:
    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path

    async def initialize(self) -> None:
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._connect() as database:
            await database.execute("PRAGMA journal_mode=WAL")
            await database.executescript(SCHEMA)
            await database.commit()

    async def close(self) -> None:
        return None

    async def existing_hashes(
        self,
        workspace_id: str,
        content_hashes: Sequence[str],
        embedding_model: str,
    ) -> frozenset[str]:
        if not content_hashes:
            return frozenset()

        placeholders = ",".join("?" for _ in content_hashes)
        query = f"""
            SELECT DISTINCT sources.content_hash
            FROM sources
            JOIN chunks ON chunks.source_id = sources.id
            WHERE sources.workspace_id = ?
              AND chunks.embedding_model = ?
              AND sources.content_hash IN ({placeholders})
        """
        parameters = (workspace_id, embedding_model, *content_hashes)

        async with self._connect() as database:
            cursor = await database.execute(query, parameters)
            rows = await cursor.fetchall()

        return frozenset(str(row[0]) for row in rows)

    async def save_indexed_sources(self, sources: Sequence[IndexedSource]) -> None:
        if not sources:
            return

        async with self._connect() as database:
            await database.execute("PRAGMA foreign_keys=ON")
            await database.execute("BEGIN IMMEDIATE")
            try:
                for indexed_source in sources:
                    source = indexed_source.source
                    await database.execute(
                        """
                        INSERT INTO sources (
                            workspace_id, name, size, media_type, content, content_hash
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(workspace_id, content_hash) DO UPDATE SET
                            name = excluded.name,
                            size = excluded.size,
                            media_type = excluded.media_type,
                            content = excluded.content
                        """,
                        (
                            source.workspace_id,
                            source.name,
                            source.size,
                            source.media_type,
                            source.content,
                            source.content_hash,
                        ),
                    )
                    cursor = await database.execute(
                        "SELECT id FROM sources WHERE workspace_id = ? AND content_hash = ?",
                        (source.workspace_id, source.content_hash),
                    )
                    row = await cursor.fetchone()
                    if row is None:
                        raise StorageError("Failed to load the stored source identifier.")

                    source_id = int(row[0])
                    await database.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))
                    await database.executemany(
                        """
                        INSERT INTO chunks (
                            source_id,
                            chunk_index,
                            text,
                            embedding,
                            embedding_dimensions,
                            embedding_model
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                source_id,
                                chunk.index,
                                chunk.text,
                                _encode_vector(vector),
                                len(vector),
                                indexed_source.embedding_model,
                            )
                            for chunk, vector in zip(
                                indexed_source.chunks,
                                indexed_source.embeddings,
                                strict=True,
                            )
                        ],
                    )
                await database.commit()
            except Exception:
                await database.rollback()
                raise

    async def list_for_workspace(self, workspace_id: str) -> Sequence[StoredSource]:
        async with self._connect() as database:
            cursor = await database.execute(
                """
                SELECT workspace_id, name, size, media_type, content, content_hash
                FROM sources
                WHERE workspace_id = ?
                ORDER BY id
                """,
                (workspace_id,),
            )
            rows = await cursor.fetchall()

        return tuple(
            StoredSource(
                workspace_id=str(row[0]),
                name=str(row[1]),
                size=int(row[2]),
                media_type=str(row[3]),
                content=str(row[4]),
                content_hash=str(row[5]),
            )
            for row in rows
        )

    async def search(
        self,
        workspace_id: str,
        vector: Sequence[float],
        embedding_model: str,
        limit: int,
    ) -> Sequence[RetrievedChunk]:
        async with self._connect() as database:
            cursor = await database.execute(
                """
                SELECT
                    sources.content_hash,
                    sources.name,
                    sources.workspace_id,
                    chunks.chunk_index,
                    chunks.text,
                    chunks.embedding,
                    chunks.embedding_dimensions
                FROM chunks
                JOIN sources ON sources.id = chunks.source_id
                WHERE sources.workspace_id = ? AND chunks.embedding_model = ?
                """,
                (workspace_id, embedding_model),
            )
            rows = await cursor.fetchall()

        candidates = tuple(
            (
                SourceChunk(
                    source_hash=str(row[0]),
                    source_name=str(row[1]),
                    workspace_id=str(row[2]),
                    index=int(row[3]),
                    text=str(row[4]),
                ),
                _decode_vector(bytes(row[5]), int(row[6])),
            )
            for row in rows
        )
        return await asyncio.to_thread(_rank_candidates, candidates, vector, limit)

    def _connect(self) -> aiosqlite.Connection:
        return aiosqlite.connect(self._database_path)

