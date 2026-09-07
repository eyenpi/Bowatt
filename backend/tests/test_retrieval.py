from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.container import AppContainer, build_container
from app.errors import ApiError
from app.main import create_app
from tests.fakes import DeterministicEmbeddingProvider


def test_retrieval_embeds_query_once_and_ranks_related_source_first(
    client: TestClient,
    container: AppContainer,
    embedding_provider: DeterministicEmbeddingProvider,
) -> None:
    client.post(
        "/api/sources",
        headers={"X-Workspace-ID": "animals"},
        files=[
            ("files", ("cats.txt", b"cats purr whiskers feline naps", "text/plain")),
            ("files", ("mars.txt", b"mars planet orbit space rover", "text/plain")),
        ],
    )

    results = asyncio.run(
        container.source_retriever.retrieve("animals", "cats purr feline", limit=2)
    )

    assert len(embedding_provider.query_calls) == 1
    assert embedding_provider.query_calls == ["cats purr feline"]
    assert [result.chunk.source_name for result in results] == ["cats.txt", "mars.txt"]
    assert results[0].score > results[1].score


def test_retrieval_is_isolated_by_workspace(
    client: TestClient,
    container: AppContainer,
) -> None:
    client.post(
        "/api/sources",
        headers={"X-Workspace-ID": "workspace-a"},
        files={"files": ("private.txt", b"private research evidence", "text/plain")},
    )

    results = asyncio.run(
        container.source_retriever.retrieve("workspace-b", "private research evidence")
    )

    assert results == ()


def test_sources_and_vectors_survive_application_restart(
    settings: Settings,
) -> None:
    first_provider = DeterministicEmbeddingProvider()
    first_container = build_container(settings, embedding_provider=first_provider)
    with TestClient(create_app(settings=settings, container=first_container)) as first_client:
        upload = first_client.post(
            "/api/sources",
            headers={"X-Workspace-ID": "persistent"},
            files={"files": ("durable.txt", b"durable persisted evidence", "text/plain")},
        )
    assert upload.status_code == 201
    assert first_provider.closed is True

    restarted_provider = DeterministicEmbeddingProvider()
    restarted_container = build_container(settings, embedding_provider=restarted_provider)
    restarted_app = create_app(settings=settings, container=restarted_container)
    with TestClient(restarted_app) as restarted_client:
        response = restarted_client.post(
            "/api/research",
            headers={"X-Workspace-ID": "persistent"},
            json={"request": "What was persisted?"},
        )
        results = asyncio.run(
            restarted_container.source_retriever.retrieve(
                "persistent", "durable persisted evidence"
            )
        )

    assert response.status_code == 200
    assert "**1** uploaded source(s)" in response.text
    assert results[0].chunk.source_name == "durable.txt"


def test_dimension_mismatch_returns_safe_retrieval_error(
    client: TestClient,
    container: AppContainer,
    embedding_provider: DeterministicEmbeddingProvider,
) -> None:
    client.post(
        "/api/sources",
        files={"files": ("source.txt", b"dimension test source", "text/plain")},
    )

    async def mismatched_query(_text: str) -> tuple[float, ...]:
        return (1.0, 2.0)

    embedding_provider.embed_query = mismatched_query  # type: ignore[method-assign]

    with pytest.raises(ApiError) as captured:
        asyncio.run(container.source_retriever.retrieve("local-default", "dimension test"))

    assert captured.value.status_code == 500
    assert captured.value.message == "Source retrieval failed."


def test_same_source_is_reembedded_after_embedding_model_change(settings: Settings) -> None:
    content = b"model migration source"
    first_provider = DeterministicEmbeddingProvider(model_name="embedding-v1")
    first_container = build_container(settings, embedding_provider=first_provider)
    with TestClient(create_app(settings=settings, container=first_container)) as first_client:
        assert (
            first_client.post(
                "/api/sources",
                files={"files": ("source.txt", content, "text/plain")},
            ).status_code
            == 201
        )

    second_settings = replace(settings, embedding_model="embedding-v2")
    second_provider = DeterministicEmbeddingProvider(model_name="embedding-v2")
    second_container = build_container(second_settings, embedding_provider=second_provider)
    with TestClient(
        create_app(settings=second_settings, container=second_container)
    ) as second_client:
        response = second_client.post(
            "/api/sources",
            files={"files": ("source.txt", content, "text/plain")},
        )

    assert response.status_code == 201
    assert len(first_provider.document_calls) == 1
    assert len(second_provider.document_calls) == 1
