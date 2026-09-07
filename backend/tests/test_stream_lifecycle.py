from __future__ import annotations

import asyncio

import pytest
from starlette.requests import ClientDisconnect, Request

from app.api.streaming import (
    DisconnectAwareStreamingResponse,
    prepare_stream_until_disconnect,
)


def _http_scope() -> dict[str, object]:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/research",
        "raw_path": b"/api/research",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8787),
    }


def test_preparation_is_cancelled_when_the_request_disconnects() -> None:
    async def scenario() -> tuple[bool, int]:
        disconnect = asyncio.Event()
        operation_started = asyncio.Event()
        operation_stopped = asyncio.Event()
        active_operations = 0

        async def receive() -> dict[str, str]:
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def operation():
            nonlocal active_operations
            active_operations += 1
            operation_started.set()
            try:
                await asyncio.Future()
            finally:
                active_operations -= 1
                operation_stopped.set()

        request = Request(_http_scope(), receive)
        task = asyncio.create_task(
            prepare_stream_until_disconnect(request, operation())
        )
        await operation_started.wait()
        disconnect.set()

        with pytest.raises(ClientDisconnect):
            await asyncio.wait_for(task, timeout=0.2)

        return operation_stopped.is_set(), active_operations

    stopped, active = asyncio.run(scenario())

    assert stopped is True
    assert active == 0


def test_disconnect_race_closes_an_already_prepared_stream() -> None:
    class PreparedStream:
        def __init__(self) -> None:
            self.closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def aclose(self) -> None:
            self.closed = True

    async def scenario() -> bool:
        prepared_stream = PreparedStream()

        async def receive() -> dict[str, str]:
            return {"type": "http.disconnect"}

        async def operation():
            return prepared_stream

        request = Request(_http_scope(), receive)
        with pytest.raises(ClientDisconnect):
            await prepare_stream_until_disconnect(request, operation())
        return prepared_stream.closed

    assert asyncio.run(scenario()) is True


def test_stream_disconnect_closes_the_body_iterator() -> None:
    async def scenario() -> tuple[bool, list[dict[str, object]]]:
        disconnect = asyncio.Event()
        first_body_sent = asyncio.Event()
        iterator_closed = asyncio.Event()
        messages: list[dict[str, object]] = []

        async def body():
            try:
                yield "first"
                await asyncio.Future()
            finally:
                iterator_closed.set()

        async def receive() -> dict[str, str]:
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, object]) -> None:
            messages.append(message)
            if message.get("type") == "http.response.body" and message.get("body"):
                first_body_sent.set()

        response = DisconnectAwareStreamingResponse(
            body(),
            media_type="text/plain",
            close_timeout_seconds=0.2,
        )
        task = asyncio.create_task(response(_http_scope(), receive, send))
        await first_body_sent.wait()
        disconnect.set()
        await asyncio.wait_for(task, timeout=0.2)
        return iterator_closed.is_set(), messages

    closed, messages = asyncio.run(scenario())

    assert closed is True
    assert any(message.get("body") == b"first" for message in messages)


def test_stream_respects_asgi_send_backpressure() -> None:
    async def scenario() -> tuple[int, int]:
        release_send = asyncio.Event()
        first_send_started = asyncio.Event()
        produced = 0
        bodies_sent = 0

        async def body():
            nonlocal produced
            produced += 1
            yield "first"
            produced += 1
            yield "second"

        async def receive() -> dict[str, str]:
            await asyncio.Future()
            raise AssertionError("unreachable")

        async def send(message: dict[str, object]) -> None:
            nonlocal bodies_sent
            if message.get("type") != "http.response.body" or not message.get("body"):
                return
            bodies_sent += 1
            if bodies_sent == 1:
                first_send_started.set()
                await release_send.wait()

        response = DisconnectAwareStreamingResponse(
            body(),
            media_type="text/plain",
            close_timeout_seconds=0.2,
        )
        task = asyncio.create_task(response(_http_scope(), receive, send))
        await first_send_started.wait()
        produced_while_blocked = produced
        release_send.set()
        await asyncio.wait_for(task, timeout=0.2)
        return produced_while_blocked, produced

    produced_while_blocked, produced_at_end = asyncio.run(scenario())

    assert produced_while_blocked == 1
    assert produced_at_end == 2
