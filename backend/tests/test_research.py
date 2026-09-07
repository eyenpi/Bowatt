from fastapi.testclient import TestClient

from app.config import Settings
from app.container import build_container
from app.main import create_app
from tests.fakes import DeterministicEmbeddingProvider, FakeResearchProvider


def test_research_returns_grounded_raw_markdown_stream(
    client: TestClient,
    research_provider: FakeResearchProvider,
) -> None:
    client.post(
        "/api/sources",
        headers={"X-Workspace-ID": "browser-one"},
        files={"files": ("notes.txt", b"Source contents", "text/plain")},
    )

    with client.stream(
        "POST",
        "/api/research",
        headers={"X-Workspace-ID": "browser-one"},
        json={"request": "What is in my source?"},
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert response.headers["x-research-agent-mode"] == "agent"
    assert body.startswith("# Research answer")
    assert "## Sources" in body
    assert "[U1] Uploaded: `notes.txt` (chunk 1, characters 1–15)" in body
    assert "[W1] [Example evidence](https://example.test/evidence)" in body
    assert research_provider.answer_calls[0][1][0].chunk.source_name == "notes.txt"


def test_sources_are_isolated_by_workspace(client: TestClient) -> None:
    client.post(
        "/api/sources",
        headers={"X-Workspace-ID": "browser-one"},
        files={"files": ("private.txt", b"Private source", "text/plain")},
    )

    response = client.post(
        "/api/research",
        headers={"X-Workspace-ID": "browser-two"},
        json={"request": "What sources are available?"},
    )

    assert response.status_code == 200
    assert "private.txt" not in response.text


def test_research_rejects_blank_requests(client: TestClient) -> None:
    response = client.post("/api/research", json={"request": "   "})

    assert response.status_code == 400
    assert response.text == "Research request must not be blank."


def test_research_reports_missing_provider_configuration(settings: Settings) -> None:
    container = build_container(
        settings,
        embedding_provider=DeterministicEmbeddingProvider(),
    )

    with TestClient(create_app(settings=settings, container=container)) as test_client:
        response = test_client.post(
            "/api/research",
            json={"request": "Find current evidence"},
        )

    assert response.status_code == 503
    assert response.text == "Research provider is not configured. Set OPENAI_API_KEY."
