from __future__ import annotations

from datetime import datetime

try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        pass

from typing import Any, Literal

from pydantic import Field, model_validator

from evolab.contracts.common import StrictBaseModel


class TaskOrigin(StrEnum):
    HUMAN = "human"
    PROPOSER = "proposer"
    BENCHMARK = "benchmark"
    SCHEDULER = "scheduler"


class TaskPurpose(StrEnum):
    SCIENCE = "science"
    TRAINING_ROLLOUT = "training_rollout"
    EVALUATION = "evaluation"
    REGRESSION = "regression"


class ProposedTaskRelationType(StrEnum):
    SUBPROBLEM = "subproblem"
    ANALOGY = "analogy"
    DIFFICULTY_VARIANT = "difficulty_variant"
    SKILL_PROBE = "skill_probe"
    FAILURE_REPAIR = "failure_repair"
    COUNTEREXAMPLE = "counterexample"
    DATA_VARIANT = "data_variant"
    ABLATION = "ablation"
    REGRESSION = "regression"
    CURRICULUM_STEP = "curriculum_step"


class ProposerInputRef(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    ref_type: str
    ref_id: str
    role: str
    summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProposedTaskRelation(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    human_anchor_task_refs: list[str] = Field(default_factory=list)
    human_anchor_trajectory_refs: list[str] = Field(default_factory=list)
    proposer_input_refs: list[ProposerInputRef] = Field(default_factory=list)
    relation_type: ProposedTaskRelationType
    relation_rationale: str
    target_capabilities: list[str] = Field(default_factory=list)
    expected_transfer: str
    eval_target_task_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_human_anchor(self) -> "ProposedTaskRelation":
        if not self.human_anchor_task_refs and not self.human_anchor_trajectory_refs:
            raise ValueError("proposed task relation requires a human anchor task or trajectory")
        return self


class TaskRequest(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    task_id: str
    origin: TaskOrigin
    purpose: TaskPurpose
    goal: str
    task_config_ref: str | None = None
    task_payload_uri: str | None = None
    producer_ref: str | None = None
    parent_task_id: str | None = None
    round_id: str | None = None
    proposed_task_relation: ProposedTaskRelation | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_proposer_relation(self) -> "TaskRequest":
        if self.origin == TaskOrigin.PROPOSER and self.proposed_task_relation is None:
            raise ValueError("origin=proposer requires proposed_task_relation")
        return self


class TaskJob(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    job_id: str
    request_payload_uri: str
    enqueued_at: datetime
