"""Check the real Nginx/Compose path using the existing deterministic providers."""

from __future__ import annotations

import json
import re
import sys
import time
from http.client import HTTPConnection
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tests.integration.test_frontend_contract import (
    _research_request,
    _upload_request,
    test_errors_are_plain_text_for_frontend_display,
    test_frontend_cors_preflight_is_allowed,
    test_unmodified_frontend_uses_default_workspace,
    test_upload_then_research_works_over_live_http,
)

BASE_URL = "http://frontend:8080"
STATE_URL = "http://backend:8787/__test__/provider-state"


def read(request: str | Request) -> bytes:
    with urlopen(request, timeout=30) as response:
        return response.read()


def check_persistence() -> None:
    # Nginx may briefly cache the previous backend container's address.
    deadline = time.monotonic() + 15
    while True:
        try:
            answer = read(_research_request(BASE_URL, "Use my uploaded source")).decode()
            assert "Uploaded: `default.txt`" in answer
            break
        except (HTTPError, URLError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.25)
    print("PASS: stored sources survive backend container replacement", flush=True)


def check_streaming_and_cancellation() -> None:
    before = json.loads(read(STATE_URL))
    connection = HTTPConnection("frontend", 8080, timeout=10)
    try:
        connection.request(
            "POST", "/api/research",
            body=json.dumps({"request": "Stream an answer"}),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        assert response.status == 200
        assert response.read(1)
        # A byte must arrive while generation is still active, not after buffering.
        assert json.loads(read(STATE_URL))["active_answer_streams"] == 1
        response.close()
    finally:
        connection.close()

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = json.loads(read(STATE_URL))
        if state["active_answer_streams"] == 0:
            assert state["interrupted_answer_streams"] > before["interrupted_answer_streams"]
            print("PASS: unbuffered streaming and cancellation through Nginx", flush=True)
            return
        time.sleep(0.05)
    raise AssertionError("Generation did not stop after the proxied client disconnected")


def main() -> None:
    if "--verify-persistence" in sys.argv:
        check_persistence()
        return

    html = read(BASE_URL).decode()
    assert '<div id="root"></div>' in html
    scripts = re.findall(r'src="(/assets/[^\"]+\.js)"', html)
    assert scripts
    for script in scripts:
        bundle = read(BASE_URL + script).decode()
        assert "http://localhost:8787" not in bundle
        assert "/api/research" in bundle and "/api/sources" in bundle
    assert json.loads(read(BASE_URL + "/health"))["status"] == "ok"
    print("PASS: UI assets, same-origin API configuration, and backend health", flush=True)

    for check in (
        test_frontend_cors_preflight_is_allowed,
        test_upload_then_research_works_over_live_http,
        test_unmodified_frontend_uses_default_workspace,
        test_errors_are_plain_text_for_frontend_display,
    ):
        check(BASE_URL)
        print(f"PASS: {check.__name__}", flush=True)

    # Nginx defaults to a 1 MiB body limit, below the backend's supported file size.
    upload = _upload_request(
        BASE_URL, [("large.txt", b"evidence " * 130_000, "text/plain")], "large-upload"
    )
    assert json.loads(read(upload))["uploaded"][0]["size"] > 1_048_576
    print("PASS: uploads larger than Nginx's default body limit", flush=True)
    check_streaming_and_cancellation()


if __name__ == "__main__":
    main()
