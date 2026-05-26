from __future__ import annotations

from abc import ABC, abstractmethod
from math import sqrt
from typing import Any, Literal, Protocol

from pydantic import Field

from evolab.contracts.common import StrictBaseModel
from evolab.contracts.records import SubagentRunRecord
from evolab.contracts.snapshots import SnapshotRecord, SnapshotRef
from evolab.contracts.tools import ToolCallRecord


class RewardExample(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    sample_id: str
    trajectory_ref: str | None = None
    task_id: str | None = None
    role: str | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_subagent_run(cls, record: SubagentRunRecord) -> "RewardExample":
        return cls(
            sample_id=record.run_ref,
            trajectory_ref=record.run_ref,
            task_id=record.task_id,
            role=record.role,
            tool_calls=record.tool_calls,
            metadata={
                "task_origin": record.task_origin.value,
                "task_purpose": record.task_purpose.value,
                "stage_index": record.stage_index,
            },
        )


class RewardCalculationRequest(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    examples: list[RewardExample] = Field(default_factory=list)
    reward_policy_snapshot_ref: str | None = None
    before_snapshot_refs: list[SnapshotRef] = Field(default_factory=list)
    after_snapshot_refs: list[SnapshotRef] = Field(default_factory=list)
    curriculum_state_ref: str | None = None
    compute_advantages: bool = True
    advantage_baseline: float | None = None
    normalize_advantages: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class RewardSnapshotResolver(Protocol):
    def get_snapshot(self, snapshot_ref: str) -> SnapshotRecord | None:
        ...


class RewardCalculationContext(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    metadata: dict[str, Any] = Field(default_factory=dict)

    def get_snapshot(self, snapshot_ref: str) -> SnapshotRecord | None:
        return None


class RewardScore(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    sample_id: str
    reward: float
    raw_score: float | None = None
    advantage: float | None = None
    passed: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RewardCalculationResult(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    calculator_id: str
    scores: list[RewardScore] = Field(default_factory=list)
    aggregate_reward: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RewardVerification(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    sample_id: str
    passed: bool
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RewardCalculator(ABC):
    calculator_id: str

    @abstractmethod
    def calculate(
        self,
        request: RewardCalculationRequest,
        context: RewardSnapshotResolver | None = None,
    ) -> RewardCalculationResult:
        raise NotImplementedError

    def _finalize(
        self,
        request: RewardCalculationRequest,
        scores: list[RewardScore],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> RewardCalculationResult:
        aggregate = _mean([score.reward for score in scores])
        return RewardCalculationResult(
            calculator_id=self.calculator_id,
            scores=self._with_advantages(request, scores, aggregate),
            aggregate_reward=aggregate,
            metadata=metadata or {},
        )

    def _with_advantages(
        self,
        request: RewardCalculationRequest,
        scores: list[RewardScore],
        aggregate: float | None,
    ) -> list[RewardScore]:
        if not request.compute_advantages:
            return scores
        baseline = request.advantage_baseline
        if baseline is None:
            baseline = aggregate
        if baseline is None:
            return scores

        scale = 1.0
        if request.normalize_advantages and scores:
            variance = sum((score.reward - baseline) ** 2 for score in scores) / len(scores)
            scale = sqrt(variance) or 1.0

        return [
            score.model_copy(update={"advantage": (score.reward - baseline) / scale})
            for score in scores
        ]


class VerifierRewardCalculator(RewardCalculator):
    def __init__(
        self,
        *,
        calculator_id: str,
        pass_reward: float = 1.0,
        fail_reward: float = 0.0,
    ) -> None:
        self.calculator_id = calculator_id
        self.pass_reward = pass_reward
        self.fail_reward = fail_reward

    @abstractmethod
    def verify(self, example: RewardExample) -> RewardVerification:
        raise NotImplementedError

    def calculate(
        self,
        request: RewardCalculationRequest,
        context: RewardSnapshotResolver | None = None,
    ) -> RewardCalculationResult:
        scores = []
        for example in request.examples:
            verification = self.verify(example)
            reward = self.pass_reward if verification.passed else self.fail_reward
            scores.append(
                RewardScore(
                    sample_id=example.sample_id,
                    reward=reward,
                    raw_score=reward,
                    passed=verification.passed,
                    metadata={
                        "reason": verification.reason,
                        "verification": verification.model_dump(mode="json"),
                    },
                )
            )
        return self._finalize(request, scores)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
