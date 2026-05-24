from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from evolab.backends.rewards.base import (
    RewardSnapshotResolver,
    RewardCalculationRequest,
    RewardCalculationResult,
    RewardCalculator,
    RewardScore,
)

RewardCombinationMode = Literal["sum", "mean", "weighted_sum", "max", "min"]


@dataclass(frozen=True)
class RewardComponent:
    calculator: RewardCalculator
    weight: float = 1.0


class CompositeRewardCalculator(RewardCalculator):
    def __init__(
        self,
        components: list[RewardComponent],
        *,
        mode: RewardCombinationMode = "weighted_sum",
        calculator_id: str | None = None,
    ) -> None:
        if not components:
            raise ValueError("CompositeRewardCalculator requires at least one component")
        if mode not in {"sum", "mean", "weighted_sum", "max", "min"}:
            raise ValueError(f"unsupported reward combination mode: {mode!r}")
        self.components = components
        self.mode = mode
        self.calculator_id = calculator_id or f"composite:{mode}"

    def calculate(
        self,
        request: RewardCalculationRequest,
        context: RewardSnapshotResolver | None = None,
    ) -> RewardCalculationResult:
        child_request = request.model_copy(update={"compute_advantages": False})
        component_results = [
            (component, component.calculator.calculate(child_request, context))
            for component in self.components
        ]
        component_scores = [
            (component, _scores_by_sample_id(result))
            for component, result in component_results
        ]

        scores = []
        for example in request.examples:
            values = []
            metadata = {"components": []}
            for component, scores_by_sample_id in component_scores:
                try:
                    score = scores_by_sample_id[example.sample_id]
                except KeyError as exc:
                    raise ValueError(
                        f"component {component.calculator.calculator_id!r} did not score "
                        f"sample {example.sample_id!r}"
                    ) from exc
                values.append((score.reward, component.weight))
                metadata["components"].append(
                    {
                        "calculator_id": component.calculator.calculator_id,
                        "reward": score.reward,
                        "weight": component.weight,
                    }
                )
            scores.append(
                RewardScore(
                    sample_id=example.sample_id,
                    reward=_combine(values, self.mode),
                    metadata=metadata,
                )
            )

        return self._finalize(
            request,
            scores,
            metadata={
                "mode": self.mode,
                "component_ids": [
                    component.calculator.calculator_id for component in self.components
                ],
            },
        )


def _scores_by_sample_id(result: RewardCalculationResult) -> dict[str, RewardScore]:
    return {score.sample_id: score for score in result.scores}


def _combine(values: list[tuple[float, float]], mode: RewardCombinationMode) -> float:
    rewards = [reward for reward, _weight in values]
    if mode == "sum":
        return sum(rewards)
    if mode == "mean":
        return sum(rewards) / len(rewards)
    if mode == "weighted_sum":
        return sum(reward * weight for reward, weight in values)
    if mode == "max":
        return max(rewards)
    if mode == "min":
        return min(rewards)
    raise ValueError(f"unsupported reward combination mode: {mode!r}")
