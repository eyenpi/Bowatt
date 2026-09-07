from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


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


def _optional_positive_integer(name: str) -> int | None:
    raw_value = os.getenv(name)
    if not raw_value:
        return None

    return _positive_integer(name, 1)


def _non_negative_integer(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer") from error

    if value < 0:
        raise RuntimeError(f"{name} must be zero or greater")

    return value


def _non_negative_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = float(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a number") from error

    if value < 0:
        raise RuntimeError(f"{name} must be zero or greater")

    return value


def _positive_float(name: str, default: float) -> float:
    value = _non_negative_float(name, default)
    if value == 0:
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
    database_path: Path = Path("data/research.db")
    chunk_size: int = 2_000
    chunk_overlap: int = 200
    embedding_batch_size: int = 64
    embedding_concurrency: int = 4
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int | None = None
    embedding_timeout_seconds: float = 30.0
    retrieval_limit: int = 5
    scaffold_stream_delay_seconds: float = 2.0
    openai_api_key: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        positive_values = {
            "max_upload_files": self.max_upload_files,
            "max_upload_bytes": self.max_upload_bytes,
            "max_request_characters": self.max_request_characters,
            "chunk_size": self.chunk_size,
            "embedding_batch_size": self.embedding_batch_size,
            "embedding_concurrency": self.embedding_concurrency,
            "retrieval_limit": self.retrieval_limit,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be zero or greater")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.embedding_batch_size > 2_048:
            raise ValueError("embedding_batch_size cannot exceed 2048")
        if self.embedding_dimensions is not None and self.embedding_dimensions <= 0:
            raise ValueError("embedding_dimensions must be greater than zero")
        if self.embedding_timeout_seconds <= 0:
            raise ValueError("embedding_timeout_seconds must be greater than zero")
        if self.scaffold_stream_delay_seconds < 0:
            raise ValueError("scaffold_stream_delay_seconds must be zero or greater")
        if not self.embedding_model.strip():
            raise ValueError("embedding_model must not be blank")

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
            database_path=Path(
                os.getenv("BOWATT_DATABASE_PATH", "data/research.db")
            ).expanduser(),
            chunk_size=_positive_integer("BOWATT_CHUNK_SIZE", 2_000),
            chunk_overlap=_non_negative_integer("BOWATT_CHUNK_OVERLAP", 200),
            embedding_batch_size=_positive_integer("BOWATT_EMBEDDING_BATCH_SIZE", 64),
            embedding_concurrency=_positive_integer("BOWATT_EMBEDDING_CONCURRENCY", 4),
            embedding_model=os.getenv(
                "BOWATT_EMBEDDING_MODEL", "text-embedding-3-small"
            ),
            embedding_dimensions=_optional_positive_integer(
                "BOWATT_EMBEDDING_DIMENSIONS"
            ),
            embedding_timeout_seconds=_positive_float(
                "BOWATT_EMBEDDING_TIMEOUT_SECONDS", 30.0
            ),
            retrieval_limit=_positive_integer("BOWATT_RETRIEVAL_LIMIT", 5),
            scaffold_stream_delay_seconds=_non_negative_float(
                "BOWATT_SCAFFOLD_STREAM_DELAY_SECONDS", 2.0
            ),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        )
