from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from app.api.routes import router
from app.config import Settings
from app.container import AppContainer, build_container
from app.errors import ApiError

logger = logging.getLogger(__name__)


def create_app(
    *,
    settings: Settings | None = None,
    container: AppContainer | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_environment()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = resolved_settings
        app.state.container = container or build_container(resolved_settings)
        yield

    application = FastAPI(
        title=resolved_settings.service_name,
        version="0.1.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Workspace-ID"],
    )

    @application.exception_handler(ApiError)
    async def handle_api_error(_request: Request, error: ApiError) -> PlainTextResponse:
        return PlainTextResponse(error.message, status_code=error.status_code)

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request, error: RequestValidationError
    ) -> PlainTextResponse:
        details = error.errors()
        message = details[0].get("msg", "Invalid request.") if details else "Invalid request."
        return PlainTextResponse(str(message), status_code=422)

    @application.exception_handler(Exception)
    async def handle_unexpected_error(_request: Request, error: Exception) -> PlainTextResponse:
        logger.exception("Unhandled API error", exc_info=error)
        return PlainTextResponse("Internal server error.", status_code=500)

    application.include_router(router)
    return application


app = create_app()

