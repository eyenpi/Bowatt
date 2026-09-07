from __future__ import annotations

import json
from collections.abc import Sequence
from email.message import Message
from typing import NamedTuple
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

pytestmark = pytest.mark.integration
FRONTEND_ORIGIN = "http://localhost:5173"


class LiveResponse(NamedTuple):
    status: int
    headers: Message
    body: bytes


def _send(request: Request) -> LiveResponse:
    with urlopen(request, timeout=3) as response:
        return LiveResponse(response.status, response.headers, response.read())


def _multipart_body(
    files: Sequence[tuple[str, bytes, str]],
) -> tuple[bytes, str]:
    boundary = "bowatt-integration-boundary"
    body = bytearray()

    for filename, content, media_type in files:
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            f'Content-Disposition: form-data; name="files"; filename="{filename}"\r\n'.encode()
        )
        body.extend(f"Content-Type: {media_type}\r\n\r\n".encode())
        body.extend(content)
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode())
    return bytes(body), boundary


def _upload_request(
    base_url: str,
    files: Sequence[tuple[str, bytes, str]],
    workspace_id: str | None = None,
) -> Request:
    body, boundary = _multipart_body(files)
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Origin": FRONTEND_ORIGIN,
    }
    if workspace_id:
        headers["X-Workspace-ID"] = workspace_id

    return Request(
        f"{base_url}/api/sources",
        data=body,
        headers=headers,
        method="POST",
    )


def _research_request(
    base_url: str,
    query: str,
    workspace_id: str | None = None,
) -> Request:
    headers = {
        "Content-Type": "application/json",
        "Origin": FRONTEND_ORIGIN,
    }
    if workspace_id:
        headers["X-Workspace-ID"] = workspace_id

    return Request(
        f"{base_url}/api/research",
        data=json.dumps({"request": query}).encode(),
        headers=headers,
        method="POST",
    )


def test_frontend_cors_preflight_is_allowed(live_api_url: str) -> None:
    request = Request(
        f"{live_api_url}/api/research",
        headers={
            "Origin": FRONTEND_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-workspace-id",
        },
        method="OPTIONS",
    )

    response = _send(request)
    allowed_headers = response.headers["Access-Control-Allow-Headers"].lower()

    assert response.status == 200
    assert response.headers["Access-Control-Allow-Origin"] == FRONTEND_ORIGIN
    assert "POST" in response.headers["Access-Control-Allow-Methods"]
    assert "content-type" in allowed_headers
    assert "x-workspace-id" in allowed_headers


def test_upload_then_research_works_over_live_http(live_api_url: str) -> None:
    workspace_id = "integration-round-trip"
    upload = _send(
        _upload_request(
            live_api_url,
            [
                ("notes.txt", b"Alpha source", "text/plain"),
                ("facts.md", b"# Beta source", "text/markdown"),
            ],
            workspace_id,
        )
    )

    assert upload.status == 201
    assert upload.headers["Access-Control-Allow-Origin"] == FRONTEND_ORIGIN
    assert json.loads(upload.body) == {
        "uploaded": [
            {"name": "notes.txt", "size": 12, "type": "text/plain"},
            {"name": "facts.md", "size": 13, "type": "text/markdown"},
        ]
    }

    research = _send(
        _research_request(live_api_url, "Compare my two sources", workspace_id)
    )
    markdown = research.body.decode()

    assert research.status == 200
    assert research.headers["Content-Type"].startswith("text/markdown")
    assert research.headers["Transfer-Encoding"].lower() == "chunked"
    assert research.headers["Access-Control-Allow-Origin"] == FRONTEND_ORIGIN
    assert markdown.startswith("# Backend scaffold ready")
    assert "**2** uploaded source(s)" in markdown
    assert not markdown.startswith("data:")


def test_unmodified_frontend_uses_default_workspace(live_api_url: str) -> None:
    upload = _send(
        _upload_request(
            live_api_url,
            [("default.txt", b"Default workspace source", "text/plain")],
        )
    )
    research = _send(_research_request(live_api_url, "Use my uploaded source"))

    assert upload.status == 201
    assert research.status == 200
    assert "**1** uploaded source(s)" in research.body.decode()


def test_errors_are_plain_text_for_frontend_display(live_api_url: str) -> None:
    request = _upload_request(
        live_api_url,
        [("image.png", b"not an image", "image/png")],
        "integration-errors",
    )

    with pytest.raises(HTTPError) as captured:
        urlopen(request, timeout=3)

    error = captured.value
    try:
        body = error.read().decode()
    finally:
        error.close()

    assert error.code == 415
    assert error.headers["Content-Type"].startswith("text/plain")
    assert body == "image.png is not a supported text file."

