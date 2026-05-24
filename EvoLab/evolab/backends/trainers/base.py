from __future__ import annotations

from abc import ABC, abstractmethod

from evolab.contracts.evolution import LLMEvolutionRequest, LLMEvolutionResult


class LLMTrainer(ABC):
    trainer_id: str

    @abstractmethod
    def train(self, request: LLMEvolutionRequest) -> LLMEvolutionResult:
        raise NotImplementedError
