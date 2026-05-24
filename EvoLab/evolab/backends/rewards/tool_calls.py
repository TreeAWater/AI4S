from __future__ import annotations

from typing import Literal

from evolab.backends.rewards.base import (
    RewardCalculationRequest,
    RewardCalculationResult,
    RewardCalculator,
    RewardExample,
    RewardScore,
    RewardSnapshotResolver,
)


class NumToolCallRewardCalculator(RewardCalculator):
    calculator_id = "num_toolcall"

    def __init__(
        self,
        *,
        tool_name: str | None = None,
        status: Literal["ok", "error"] | None = None,
        scale: float = 1.0,
        offset: float = 0.0,
        calculator_id: str | None = None,
    ) -> None:
        if calculator_id is not None:
            self.calculator_id = calculator_id
        self.tool_name = tool_name
        self.status = status
        self.scale = scale
        self.offset = offset

    def calculate(
        self,
        request: RewardCalculationRequest,
        context: RewardSnapshotResolver | None = None,
    ) -> RewardCalculationResult:
        scores = []
        for example in request.examples:
            count = self._count_tool_calls(example)
            scores.append(
                RewardScore(
                    sample_id=example.sample_id,
                    reward=self.offset + self.scale * count,
                    raw_score=float(count),
                    metadata={
                        "tool_name": self.tool_name,
                        "status": self.status,
                    },
                )
            )
        return self._finalize(request, scores)

    def _count_tool_calls(self, example: RewardExample) -> float:
        count = 0
        for record in example.tool_calls:
            if self.tool_name is not None and record.tool_call.name != self.tool_name:
                continue
            if self.status is not None and record.result.status != self.status:
                continue
            count += 1
        return float(count)
