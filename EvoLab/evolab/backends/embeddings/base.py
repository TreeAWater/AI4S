from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from evolab.contracts.embeddings import EmbeddingResponse


class EmbeddingRuntime(Protocol):
    def embed(self, texts: list[str], *, purpose: str) -> EmbeddingResponse:
        raise NotImplementedError


class EmbeddingBackend(ABC):
    backend_id: str

    @abstractmethod
    def instantiate(self, state_ref: str | None) -> EmbeddingRuntime:
        raise NotImplementedError
