from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol

from evolab.contracts.common import Message
from evolab.contracts.llm import LLMGenerationConfig, LLMRuntimeResponse


class LLMRuntime(Protocol):
    def generate(
        self,
        messages: list[Message],
        tool_specs: list[dict[str, Any]],
        generation_config: LLMGenerationConfig,
    ) -> LLMRuntimeResponse:
        ...


class LLMBackend(ABC):
    backend_id: str

    @abstractmethod
    def instantiate(self, state_ref: str | None) -> LLMRuntime:
        raise NotImplementedError
