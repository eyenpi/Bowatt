from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import AsyncIterator, Awaitable
from contextlib import suppress

from starlette.requests import ClientDisconnect, Request
from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

logger = logging.getLogger(__name__)


async def prepare_stream_until_disconnect(
    request: Request,
    operation: Awaitable[AsyncIterator[str]],
    *,
    close_timeout_seconds: float = 2.0,
) -> AsyncIterator[str]:
    """Run response preparation while separately watching for client disconnect."""

    operation_task = asyncio.create_task(operation)
    disconnect_task = asyncio.create_task(_wait_for_disconnect(request.receive))
    try:
        completed, _pending = await asyncio.wait(
            (operation_task, disconnect_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if disconnect_task in completed:
            if operation_task.done() and not operation_task.cancelled():
                try:
                    prepared_stream = operation_task.result()
                except BaseException:
                    pass
                else:
                    await _close_async_resource(
                        prepared_stream,
                        close_timeout_seconds,
                    )
            else:
                operation_task.cancel()
                await asyncio.gather(operation_task, return_exceptions=True)
            raise ClientDisconnect()

        disconnect_task.cancel()
        await asyncio.gather(disconnect_task, return_exceptions=True)
        return await operation_task
    finally:
        for task in (operation_task, disconnect_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(
            operation_task,
            disconnect_task,
            return_exceptions=True,
        )


class DisconnectAwareStreamingResponse(StreamingResponse):
    """Streaming response that closes its iterator on disconnect for every ASGI version."""

    def __init__(
        self,
        content: AsyncIterator[str],
        *,
        close_timeout_seconds: float,
        **kwargs,
    ) -> None:
        super().__init__(content, **kwargs)
        self._close_timeout_seconds = close_timeout_seconds
        self._body_iterator_closed = False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        stream_task = asyncio.create_task(self.stream_response(send))
        disconnect_task = asyncio.create_task(_wait_for_disconnect(receive))
        try:
            completed, _pending = await asyncio.wait(
                (stream_task, disconnect_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stream_task in completed:
                disconnect_task.cancel()
                await asyncio.gather(disconnect_task, return_exceptions=True)
                with suppress(OSError):
                    await stream_task
            else:
                stream_task.cancel()
                await asyncio.gather(stream_task, return_exceptions=True)
        finally:
            for task in (stream_task, disconnect_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(
                stream_task,
                disconnect_task,
                return_exceptions=True,
            )
            await self._close_body_iterator()

        if self.background is not None:
            await self.background()

    async def _close_body_iterator(self) -> None:
        if self._body_iterator_closed:
            return
        self._body_iterator_closed = True
        await _close_async_resource(
            self.body_iterator,
            self._close_timeout_seconds,
        )


async def _wait_for_disconnect(receive: Receive) -> None:
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return


async def _close_async_resource(resource: object, timeout_seconds: float) -> None:
    close = getattr(resource, "aclose", None)
    if not callable(close):
        return

    try:
        async with asyncio.timeout(timeout_seconds):
            result = close()
            if inspect.isawaitable(result):
                await result
    except TimeoutError:
        logger.warning("Timed out while closing a disconnected response stream.")
    except Exception:
        logger.exception("Failed to close a disconnected response stream.")
