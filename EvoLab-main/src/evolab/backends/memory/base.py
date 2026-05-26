from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from evolab.contracts.common import Message
from evolab.contracts.retrieval import MemoryBundle, RetrievalRequest


class MemoryBackend(ABC):
    backend_id: str

    @abstractmethod
    def search(self, request: RetrievalRequest) -> MemoryBundle:
        ...

    @abstractmethod
    def add(self, task_id: str, role: str, messages: list[Message]) -> Any:
        ...
