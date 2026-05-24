from __future__ import annotations

from evolab.backends.trainers.base import LLMTrainer
from evolab.contracts.evolution import LLMEvolutionRequest, LLMEvolutionResult


class BlankTrainer(LLMTrainer):
    trainer_id = "blank"

    def train(self, request: LLMEvolutionRequest) -> LLMEvolutionResult:
        raise NotImplementedError("training algorithm is not implemented")
