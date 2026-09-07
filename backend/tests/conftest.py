from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture
def client() -> Iterator[TestClient]:
    settings = Settings(
        environment="test",
        max_upload_files=2,
        max_upload_bytes=128,
        max_request_characters=200,
    )
    with TestClient(create_app(settings=settings)) as test_client:
        yield test_client

