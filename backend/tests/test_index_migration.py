from __future__ import annotations

import asyncio
import sqlite3
import struct
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.container import build_container
from app.main import create_app
from tests.fakes import DeterministicEmbeddingProvider


def _snapshot(path: Path) -> tuple[list[tuple], list[tuple]]:
    with sqlite3.connect(path) as database:
        return (
            database.execute("SELECT * FROM sources ORDER BY id").fetchall(),
            database.execute("SELECT * FROM chunks ORDER BY id").fetchall(),
        )


@pytest.mark.parametrize("changes", [
    {"chunk_size": 48}, {"chunk_overlap": 0}, {"chunk_max_tokens": 16},
    {"embedding_dimensions": 64},
])
def test_changed_index_settings_reembed_once_and_replace_old_chunks(
    settings: Settings, changes: dict[str, int]
) -> None:
    upload = {
        "files": ("notes.txt", b"First source sentence. Second source sentence.", "text/plain")
    }
    provider = DeterministicEmbeddingProvider()
    first = build_container(settings, embedding_provider=provider)
    with TestClient(create_app(settings=settings, container=first)) as client:
        assert client.post("/api/sources", files=upload).status_code == 201
    before_sources, _ = _snapshot(settings.database_path)

    updated_settings = replace(settings, **changes)
    updated_provider = DeterministicEmbeddingProvider()
    updated = build_container(updated_settings, embedding_provider=updated_provider)
    with TestClient(create_app(settings=updated_settings, container=updated)) as client:
        assert client.post("/api/sources", files=upload).status_code == 201
        call_count = len(updated_provider.document_calls)
        assert call_count > 0
        after_sources, after_chunks = _snapshot(settings.database_path)
        assert len(after_sources) == 1
        assert after_sources[0][0] == before_sources[0][0]
        assert after_sources != before_sources  # The persisted signature changed.
        assert len(after_chunks) == sum(map(len, updated_provider.document_calls))
        assert client.post("/api/sources", files=upload).status_code == 201
        assert len(updated_provider.document_calls) == call_count
        assert _snapshot(settings.database_path) == (after_sources, after_chunks)


@pytest.mark.parametrize("failure", ["embedding", "database"])
def test_failed_reindex_preserves_the_entire_previous_index(
    settings: Settings, failure: str
) -> None:
    upload = {"files": ("notes.txt", b"Previously indexed source material.", "text/plain")}
    first = build_container(settings, embedding_provider=DeterministicEmbeddingProvider())
    with TestClient(create_app(settings=settings, container=first)) as client:
        assert client.post("/api/sources", files=upload).status_code == 201
    before = _snapshot(settings.database_path)

    if failure == "database":
        with sqlite3.connect(settings.database_path) as database:
            database.executescript("""
                CREATE TRIGGER fail_chunk_insert BEFORE INSERT ON chunks
                BEGIN SELECT RAISE(ABORT, 'synthetic database failure'); END;
            """)
    updated_settings = replace(settings, chunk_size=48)
    provider = DeterministicEmbeddingProvider(
        fail_on_document_call=1 if failure == "embedding" else None
    )
    updated = build_container(updated_settings, embedding_provider=provider)
    with TestClient(
        create_app(settings=updated_settings, container=updated),
        raise_server_exceptions=False,
    ) as client:
        response = client.post("/api/sources", files=upload)
        assert response.status_code == (502 if failure == "embedding" else 500)
        assert _snapshot(settings.database_path) == before
        matches = asyncio.run(updated.source_retriever.retrieve("local-default", "source material"))
        assert matches


def test_legacy_database_upgrades_without_losing_sources_and_reindexes_on_upload(
    settings: Settings,
) -> None:
    content = "# Legacy\nExisting evidence."
    content_hash = sha256(content.encode()).hexdigest()
    # This is the schema from before chunk metadata/signatures were introduced.
    with sqlite3.connect(settings.database_path) as database:
        database.executescript("""
            CREATE TABLE sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workspace_id TEXT NOT NULL, name TEXT NOT NULL, size INTEGER NOT NULL,
                media_type TEXT NOT NULL, content TEXT NOT NULL, content_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (workspace_id, content_hash)
            );
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL, text TEXT NOT NULL, embedding BLOB NOT NULL,
                embedding_dimensions INTEGER NOT NULL, embedding_model TEXT NOT NULL,
                UNIQUE (source_id, chunk_index)
            );
        """)
        database.execute(
            """INSERT INTO sources
               (workspace_id, name, size, media_type, content, content_hash)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                "local-default", "legacy.md", len(content.encode()),
                "text/markdown", content, content_hash,
            ),
        )
        database.execute(
            """INSERT INTO chunks
               (source_id, chunk_index, text, embedding, embedding_dimensions, embedding_model)
               VALUES (1, 0, ?, ?, 64, 'test-embedding-v1')""",
            (content, struct.pack("<64d", 1.0, *([0.0] * 63))),
        )

    provider = DeterministicEmbeddingProvider()
    container = build_container(settings, embedding_provider=provider)
    with TestClient(create_app(settings=settings, container=container)) as client:
        asyncio.run(container.source_repository.initialize())  # Migration is idempotent.
        sources = asyncio.run(container.source_repository.list_for_workspace("local-default"))
        assert len(sources) == 1 and sources[0].content == content
        legacy = asyncio.run(container.source_retriever.retrieve("local-default", "evidence"))
        assert legacy[0].chunk.text == content
        assert legacy[0].chunk.heading_path == ()
        assert legacy[0].chunk.location == "chunk 1"
        assert provider.document_calls == []

        response = client.post(
            "/api/sources", files={"files": ("legacy.md", content.encode(), "text/markdown")}
        )
        assert response.status_code == 201
        assert provider.document_calls
        refreshed = asyncio.run(container.source_retriever.retrieve("local-default", "evidence"))
        assert refreshed[0].chunk.heading_path == ("Legacy",)
        assert refreshed[0].chunk.start_offset == 0
        assert refreshed[0].chunk.end_offset == len(content)


def test_heading_context_and_source_offsets_survive_restart(settings: Settings) -> None:
    settings = replace(settings, max_upload_bytes=2_000, chunk_size=150, chunk_overlap=0)
    content = "# Support\n\n## Enterprise\n" + "Priority response within four hours. " * 12
    first_provider = DeterministicEmbeddingProvider()
    first = build_container(settings, embedding_provider=first_provider)
    with TestClient(create_app(settings=settings, container=first)) as client:
        response = client.post(
            "/api/sources", files={"files": ("support.md", content.encode(), "text/markdown")}
        )
        assert response.status_code == 201
        inputs = [text for batch in first_provider.document_calls for text in batch]
        assert len(inputs) > 1
        assert all(text.startswith("Section: Support > Enterprise\n\n") for text in inputs)

    restarted = build_container(settings, embedding_provider=DeterministicEmbeddingProvider())
    with TestClient(create_app(settings=settings, container=restarted)):
        results = asyncio.run(
            restarted.source_retriever.retrieve("local-default", "Enterprise", 20)
        )
        assert len(results) == len(inputs)
        for result in results:
            chunk = result.chunk
            assert chunk.heading_path == ("Support", "Enterprise")
            assert chunk.embedding_text in inputs
            assert content[chunk.start_offset:chunk.end_offset] == chunk.text
