from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.container import build_container
from app.main import create_app
from app.services.chunking import TextChunker
from tests.fakes import DeterministicEmbeddingProvider


def _database_counts(database_path: Path) -> tuple[int, int]:
    with sqlite3.connect(database_path) as database:
        source_count = int(database.execute("SELECT COUNT(*) FROM sources").fetchone()[0])
        chunk_count = int(database.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
    return source_count, chunk_count


def test_upload_chunks_embeds_in_batches_and_persists(
    client: TestClient,
    settings: Settings,
    embedding_provider: DeterministicEmbeddingProvider,
) -> None:
    content = b"alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"

    response = client.post(
        "/api/sources",
        files={"files": ("long.txt", content, "text/plain")},
    )

    source_count, chunk_count = _database_counts(settings.database_path)
    assert response.status_code == 201
    assert source_count == 1
    assert chunk_count > 1
    assert sum(len(call) for call in embedding_provider.document_calls) == chunk_count
    assert all(
        len(call) <= settings.embedding_batch_size
        for call in embedding_provider.document_calls
    )
    assert embedding_provider.query_calls == []


def test_duplicate_upload_does_not_embed_or_store_again(
    client: TestClient,
    settings: Settings,
    embedding_provider: DeterministicEmbeddingProvider,
) -> None:
    upload = {"files": ("duplicate.txt", b"same source content", "text/plain")}

    first = client.post("/api/sources", files=upload)
    calls_after_first_upload = len(embedding_provider.document_calls)
    second = client.post("/api/sources", files=upload)

    assert first.status_code == 201
    assert second.status_code == 201
    assert len(embedding_provider.document_calls) == calls_after_first_upload
    assert _database_counts(settings.database_path)[0] == 1


def test_embedding_failure_leaves_no_partial_database_records(
    settings: Settings,
) -> None:
    isolated_settings = replace(
        settings,
        database_path=settings.database_path.with_name("atomic.db"),
        chunk_size=24,
        chunk_overlap=4,
        embedding_batch_size=1,
    )
    embedding_provider = DeterministicEmbeddingProvider(fail_on_document_call=2)
    container = build_container(isolated_settings, embedding_provider=embedding_provider)

    with TestClient(create_app(settings=isolated_settings, container=container)) as client:
        response = client.post(
            "/api/sources",
            files=[
                ("files", ("first.txt", b"alpha source material", "text/plain")),
                ("files", ("second.txt", b"beta source material", "text/plain")),
            ],
        )

    assert response.status_code == 502
    assert response.text == "Synthetic embedding failure."
    assert _database_counts(isolated_settings.database_path) == (0, 0)


def test_embedding_batches_use_configured_concurrency(settings: Settings) -> None:
    isolated_settings = replace(
        settings,
        database_path=settings.database_path.with_name("concurrency.db"),
        chunk_size=20,
        chunk_overlap=4,
        embedding_batch_size=1,
        embedding_concurrency=2,
    )
    embedding_provider = DeterministicEmbeddingProvider(delay_seconds=0.02)
    container = build_container(isolated_settings, embedding_provider=embedding_provider)
    content = b"one two three four five six seven eight nine ten eleven twelve"

    with TestClient(create_app(settings=isolated_settings, container=container)) as client:
        response = client.post(
            "/api/sources",
            files={"files": ("parallel.txt", content, "text/plain")},
        )

    assert response.status_code == 201
    assert len(embedding_provider.document_calls) > 2
    assert embedding_provider.peak_document_calls == 2


def test_missing_embedding_configuration_returns_safe_error(settings: Settings) -> None:
    isolated_settings = replace(
        settings,
        database_path=settings.database_path.with_name("unconfigured.db"),
        openai_api_key=None,
    )

    with TestClient(create_app(settings=isolated_settings)) as client:
        response = client.post(
            "/api/sources",
            files={"files": ("source.txt", b"source text", "text/plain")},
        )

    assert response.status_code == 503
    assert response.text == "Embedding provider is not configured. Set OPENAI_API_KEY."
    assert _database_counts(isolated_settings.database_path) == (0, 0)


def test_invalid_embedding_vectors_are_rejected_before_storage(settings: Settings) -> None:
    class ZeroVectorProvider(DeterministicEmbeddingProvider):
        async def embed_documents(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
            return tuple((0.0, 0.0) for _ in texts)

    isolated_settings = replace(
        settings,
        database_path=settings.database_path.with_name("invalid-vectors.db"),
    )
    container = build_container(isolated_settings, embedding_provider=ZeroVectorProvider())

    with TestClient(create_app(settings=isolated_settings, container=container)) as client:
        response = client.post(
            "/api/sources",
            files={"files": ("source.txt", b"source text", "text/plain")},
        )

    assert response.status_code == 502
    assert response.text == "Embedding provider returned a zero-length vector."
    assert _database_counts(isolated_settings.database_path) == (0, 0)


def test_wrong_batch_vector_count_is_rejected_before_storage(settings: Settings) -> None:
    class MissingVectorProvider(DeterministicEmbeddingProvider):
        async def embed_documents(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
            del texts
            return ()

    isolated_settings = replace(
        settings,
        database_path=settings.database_path.with_name("wrong-count.db"),
    )
    container = build_container(isolated_settings, embedding_provider=MissingVectorProvider())

    with TestClient(create_app(settings=isolated_settings, container=container)) as client:
        response = client.post(
            "/api/sources",
            files={"files": ("source.txt", b"source text", "text/plain")},
        )

    assert response.status_code == 502
    assert response.text == "Embedding provider returned an unexpected number of vectors."
    assert _database_counts(isolated_settings.database_path) == (0, 0)


def test_whitespace_source_is_not_embedded_or_stored(
    client: TestClient,
    settings: Settings,
    embedding_provider: DeterministicEmbeddingProvider,
) -> None:
    response = client.post(
        "/api/sources",
        files={"files": ("empty.txt", b"  \n\t  ", "text/plain")},
    )

    assert response.status_code == 400
    assert response.text == "empty.txt does not contain indexable text."
    assert embedding_provider.document_calls == []
    assert _database_counts(settings.database_path) == (0, 0)


def test_embedding_batches_respect_aggregate_token_budget(settings: Settings) -> None:
    settings = replace(
        settings, chunk_size=1, chunk_overlap=0, chunk_max_tokens=8_191,
        embedding_batch_size=100,
    )
    provider = DeterministicEmbeddingProvider()
    container = build_container(settings, embedding_provider=provider)
    with TestClient(create_app(settings=settings, container=container)) as client:
        response = client.post(
            "/api/sources", files={"files": ("many.txt", b"a" * 100, "text/plain")}
        )
    assert response.status_code == 201
    assert sum(map(len, provider.document_calls)) == 100
    assert all(
        len(batch) * settings.chunk_max_tokens <= 300_000
        for batch in provider.document_calls
    )
    tokenizer = TextChunker(1, 0, settings.chunk_max_tokens)
    assert all(
        sum(tokenizer.count_tokens(text) for text in batch) <= 300_000
        for batch in provider.document_calls
    )
