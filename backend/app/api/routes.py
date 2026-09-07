from __future__ import annotations

import re
from typing import Annotated

from fastapi import APIRouter, Depends, File, Header, Request, UploadFile, status
from fastapi.responses import StreamingResponse

from app.config import Settings
from app.container import AppContainer
from app.errors import ApiError
from app.models import (
    HealthResponse,
    ResearchRequest,
    UploadedFileResponse,
    UploadResponse,
)

router = APIRouter()
WORKSPACE_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def get_container(request: Request) -> AppContainer:
    return request.app.state.container


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_workspace_id(
    settings: Annotated[Settings, Depends(get_settings)],
    workspace_id: Annotated[str | None, Header(alias="X-Workspace-ID")] = None,
) -> str:
    resolved_workspace_id = workspace_id or settings.default_workspace_id
    if not WORKSPACE_ID_PATTERN.fullmatch(resolved_workspace_id):
        raise ApiError(
            400,
            "X-Workspace-ID must be 1-128 letters, numbers, dots, underscores, or hyphens.",
        )
    return resolved_workspace_id


@router.get("/health", response_model=HealthResponse)
async def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.service_name,
        environment=settings.environment,
        mode="scaffold",
    )


@router.post(
    "/api/sources",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_sources(
    files: Annotated[list[UploadFile], File(description="UTF-8 text source files")],
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    container: Annotated[AppContainer, Depends(get_container)],
) -> UploadResponse:
    stored_sources = await container.source_ingestor.ingest(workspace_id, files)
    return UploadResponse(
        uploaded=[
            UploadedFileResponse(
                name=source.name,
                size=source.size,
                type=source.media_type,
            )
            for source in stored_sources
        ]
    )


@router.post("/api/research")
async def research(
    body: ResearchRequest,
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    container: Annotated[AppContainer, Depends(get_container)],
) -> StreamingResponse:
    stream = await container.research_agent.prepare_answer(workspace_id, body.request)
    return StreamingResponse(
        stream,
        media_type="text/markdown",
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "X-Research-Agent-Mode": "scaffold",
        },
    )
