from __future__ import annotations

from dataclasses import replace

import pytest

from app.config import Settings


def test_environment_configuration_accepts_zero_chunk_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BOWATT_CHUNK_OVERLAP", "0")

    settings = Settings.from_environment()

    assert settings.chunk_overlap == 0


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"chunk_size": 10, "chunk_overlap": 10}, "smaller than chunk_size"),
        ({"embedding_batch_size": 2_049}, "cannot exceed 2048"),
        ({"embedding_timeout_seconds": 0}, "must be greater than zero"),
        ({"embedding_model": "   "}, "must not be blank"),
    ],
)
def test_settings_reject_invalid_embedding_configuration(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(Settings(), **changes)


def test_settings_repr_does_not_expose_api_key() -> None:
    settings = Settings(openai_api_key="secret-value")

    assert "secret-value" not in repr(settings)
