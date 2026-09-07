from __future__ import annotations

import math
from collections.abc import Sequence

from app.errors import EmbeddingProviderError


def validated_vector(values: Sequence[float]) -> tuple[float, ...]:
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise EmbeddingProviderError("Embedding provider returned invalid vector data.") from error

    if not vector:
        raise EmbeddingProviderError("Embedding provider returned an empty vector.")
    if not all(math.isfinite(value) for value in vector):
        raise EmbeddingProviderError("Embedding provider returned non-finite vector data.")
    if math.sqrt(sum(value * value for value in vector)) == 0:
        raise EmbeddingProviderError("Embedding provider returned a zero-length vector.")

    return vector


def validated_embeddings(
    vectors: Sequence[Sequence[float]], expected_count: int
) -> tuple[tuple[float, ...], ...]:
    if len(vectors) != expected_count:
        raise EmbeddingProviderError(
            "Embedding provider returned an unexpected number of vectors."
        )

    validated = tuple(validated_vector(vector) for vector in vectors)
    dimensions = {len(vector) for vector in validated}
    if len(dimensions) > 1:
        raise EmbeddingProviderError(
            "Embedding provider returned vectors with inconsistent dimensions."
        )

    return validated

