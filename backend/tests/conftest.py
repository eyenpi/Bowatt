from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.container import AppContainer, build_container
from app.main import create_app
from tests.fakes import DeterministicEmbeddingProvider


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        environment="test",
        max_upload_files=2,
        max_upload_bytes=128,
        max_request_characters=200,
        database_path=tmp_path / "research.db",
        chunk_size=32,
        chunk_overlap=8,
        embedding_batch_size=2,
        embedding_concurrency=2,
        scaffold_stream_delay_seconds=0,
    )


@pytest.fixture
def embedding_provider() -> DeterministicEmbeddingProvider:
    return DeterministicEmbeddingProvider()


@pytest.fixture
def container(
    settings: Settings,
    embedding_provider: DeterministicEmbeddingProvider,
) -> AppContainer:
    return build_container(settings, embedding_provider=embedding_provider)


@pytest.fixture
def client(settings: Settings, container: AppContainer) -> Iterator[TestClient]:
    with TestClient(create_app(settings=settings, container=container)) as test_client:
        yield test_client
