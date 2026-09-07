from fastapi.testclient import TestClient


def test_upload_matches_frontend_contract(client: TestClient) -> None:
    response = client.post(
        "/api/sources",
        files=[
            ("files", ("notes.txt", b"A useful source", "text/plain")),
            ("files", ("facts.md", b"# Facts", "text/markdown")),
        ],
    )

    assert response.status_code == 201
    assert response.json() == {
        "uploaded": [
            {"name": "notes.txt", "size": 15, "type": "text/plain"},
            {"name": "facts.md", "size": 7, "type": "text/markdown"},
        ]
    }


def test_upload_rejects_non_text_files(client: TestClient) -> None:
    response = client.post(
        "/api/sources",
        files={"files": ("image.png", b"not really an image", "image/png")},
    )

    assert response.status_code == 415
    assert response.text == "image.png is not a supported text file."


def test_upload_rejects_files_over_the_limit(client: TestClient) -> None:
    response = client.post(
        "/api/sources",
        files={"files": ("large.txt", b"a" * 129, "text/plain")},
    )

    assert response.status_code == 413
    assert "128-byte upload limit" in response.text

