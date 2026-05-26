from __future__ import annotations

import hashlib
import math

from evolab.backends.embeddings.base import EmbeddingBackend
from evolab.contracts.embeddings import EmbeddingResponse


class FakeEmbeddingBackend(EmbeddingBackend):
    def __init__(self, backend_id: str = "fake-embedding", dimensions: int = 8):
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.backend_id = backend_id
        self.dimensions = dimensions
        self.instantiated_state_refs: list[str | None] = []

    def instantiate(self, state_ref: str | None) -> "FakeEmbeddingRuntime":
        self.instantiated_state_refs.append(state_ref)
        return FakeEmbeddingRuntime(self.backend_id, self.dimensions)


class FakeEmbeddingRuntime:
    def __init__(self, backend_id: str, dimensions: int):
        self.backend_id = backend_id
        self.dimensions = dimensions
        self.calls: list[dict[str, object]] = []

    def embed(self, texts: list[str], *, purpose: str) -> EmbeddingResponse:
        self.calls.append({"texts": list(texts), "purpose": purpose})
        return EmbeddingResponse(
            backend_id=self.backend_id,
            model="fake-deterministic",
            vectors=[_vector(text, self.dimensions) for text in texts],
            metadata={"purpose": purpose},
        )


def _vector(text: str, dimensions: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = [((digest[index % len(digest)] / 255.0) * 2.0) - 1.0 for index in range(dimensions)]
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]
