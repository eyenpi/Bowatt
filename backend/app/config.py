from __future__ import annotations

import os
from dataclasses import dataclass


def _positive_integer(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer") from error

    if value <= 0:
        raise RuntimeError(f"{name} must be greater than zero")

    return value


@dataclass(frozen=True, slots=True)
class Settings:
    service_name: str = "Bowatt Research Agent API"
    environment: str = "development"
    host: str = "0.0.0.0"
    port: int = 8787
    cors_origins: tuple[str, ...] = ("http://localhost:5173",)
    default_workspace_id: str = "local-default"
    max_upload_files: int = 10
    max_upload_bytes: int = 5_000_000
    max_request_characters: int = 10_000

    @classmethod
    def from_environment(cls) -> Settings:
        origins = tuple(
            origin.strip()
            for origin in os.getenv("BOWATT_CORS_ORIGINS", "http://localhost:5173").split(",")
            if origin.strip()
        )

        if not origins:
            raise RuntimeError("BOWATT_CORS_ORIGINS must contain at least one origin")

        return cls(
            environment=os.getenv("BOWATT_ENVIRONMENT", "development"),
            host=os.getenv("BOWATT_HOST", "0.0.0.0"),
            port=_positive_integer("BOWATT_PORT", 8787),
            cors_origins=origins,
            default_workspace_id=os.getenv(
                "BOWATT_DEFAULT_WORKSPACE_ID", "local-default"
            ),
            max_upload_files=_positive_integer("BOWATT_MAX_UPLOAD_FILES", 10),
            max_upload_bytes=_positive_integer("BOWATT_MAX_UPLOAD_BYTES", 5_000_000),
            max_request_characters=_positive_integer(
                "BOWATT_MAX_REQUEST_CHARACTERS", 10_000
            ),
        )

