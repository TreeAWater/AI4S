from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from evolab.contracts.retrieval import RetrievalRequest, SkillBundle, SkillObservationRequest, SkillUpdateResult


class SkillBackend(ABC):
    backend_id: str

    @abstractmethod
    def get(self, request: RetrievalRequest) -> SkillBundle:
        raise NotImplementedError

    @abstractmethod
    def look_at(self, event: dict[str, Any] | SkillObservationRequest) -> SkillUpdateResult | dict[str, Any]:
        raise NotImplementedError
