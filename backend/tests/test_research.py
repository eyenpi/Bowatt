from fastapi.testclient import TestClient


def test_research_returns_raw_markdown_stream(client: TestClient) -> None:
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
    assert response.headers["x-research-agent-mode"] == "scaffold"
    assert body.startswith("# Backend scaffold ready")
    assert "**1** uploaded source(s)" in body


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
    assert "**0** uploaded source(s)" in response.text


def test_research_rejects_blank_requests(client: TestClient) -> None:
    response = client.post("/api/research", json={"request": "   "})

    assert response.status_code == 400
    assert response.text == "Research request must not be blank."

