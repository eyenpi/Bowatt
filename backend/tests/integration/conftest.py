from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(("127.0.0.1", 0))
        return int(server_socket.getsockname()[1])


@pytest.fixture(scope="module")
def live_api_url() -> Iterator[str]:
    port = _available_port()
    base_url = f"http://127.0.0.1:{port}"
    environment = os.environ.copy()
    environment.update(
        {
            "BOWATT_ENVIRONMENT": "integration-test",
            "BOWATT_HOST": "127.0.0.1",
            "BOWATT_PORT": str(port),
            "BOWATT_CORS_ORIGINS": "http://localhost:5173",
        }
    )

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=BACKEND_ROOT,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Integration server exited with code {process.returncode}.")

        try:
            with urlopen(f"{base_url}/health", timeout=0.25) as response:
                if response.status == 200:
                    break
        except (TimeoutError, URLError):
            time.sleep(0.05)
    else:
        process.terminate()
        process.wait(timeout=2)
        raise RuntimeError("Integration server did not become ready within five seconds.")

    try:
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)

