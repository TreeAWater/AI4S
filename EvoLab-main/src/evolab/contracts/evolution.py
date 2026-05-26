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

from evolab.contracts.common import ArtifactRef, EvolutionBudget, StrictBaseModel
from evolab.contracts.task import ProposerInputRef


class LLMEvolutionMode(StrEnum):
    BASICS = "basics"
    CONSOLIDATION = "consolidation"


class LabSignals(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    solve_rate: float | None = Field(default=None, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class InstanceSnapshot(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    snapshot_ref: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMEvolutionRequest(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    mode: LLMEvolutionMode
    backend_id: str
    previous_state_ref: str | None = None
    artifact_root_uri: str
    budget: EvolutionBudget = Field(default_factory=EvolutionBudget)
    trigger_trajectory_ref: str | None = None
    proposer_input_refs: list[ProposerInputRef] = Field(default_factory=list)
    lab_signals: LabSignals | None = None
    instance_snapshots: list[InstanceSnapshot] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StandardEvolutionMetrics(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    n_train_samples: int | None = Field(default=None, ge=0)
    eval_score_before: float | None = None
    eval_score_after: float | None = None
    eval_metric_name: str | None = None
    promotion_threshold: float | None = None
    promotion_margin: float | None = None


class LLMEvolutionResult(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    status: Literal["promoted_candidate", "not_recommended", "skipped", "failed"]
    new_state_ref: str | None = None
    recommend_for_promotion: bool = False
    lora_role: Literal["solver", "skill_distilled", "composed"] | None = None
    standard_metrics: StandardEvolutionMetrics = Field(default_factory=StandardEvolutionMetrics)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_promotion_fields_match_status(self) -> "LLMEvolutionResult":
        if self.status == "promoted_candidate":
            if not self.recommend_for_promotion:
                raise ValueError("status=promoted_candidate requires recommend_for_promotion=True")
            if not self.new_state_ref:
                raise ValueError("status=promoted_candidate requires new_state_ref")
            return self

        if self.recommend_for_promotion:
            raise ValueError("recommend_for_promotion=True requires status=promoted_candidate")
        if self.new_state_ref is not None:
            raise ValueError("new_state_ref requires status=promoted_candidate")
        return self


class EvolutionRunEventType(StrEnum):
    RUN_STARTED = "run_started"
    SNAPSHOT_CAPTURED = "snapshot_captured"
    TASK_PROPOSED = "task_proposed"
    TASK_ENQUEUED = "task_enqueued"
    ROLLOUT_COMPLETED = "rollout_completed"
    REWARD_POLICY_UPDATED = "reward_policy_updated"
    REWARD_CALCULATED = "reward_calculated"
    SAMPLE_ACCEPTED = "sample_accepted"
    SAMPLE_REJECTED = "sample_rejected"
    CURRICULUM_UPDATED = "curriculum_updated"
    TRAINER_INVOKED = "trainer_invoked"
    TRAINER_COMPLETED = "trainer_completed"
    CANDIDATE_CREATED = "candidate_created"
    EVOLUTION_RECORD_SAVED = "evolution_record_saved"
    PROMOTION_DECIDED = "promotion_decided"
    RUN_FINISHED = "run_finished"
    RUN_FAILED = "run_failed"
    RUN_SKIPPED = "run_skipped"


class EvolutionRunEvent(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    run_ref: str
    event_type: EvolutionRunEventType
    created_at: datetime = Field(default_factory=datetime.utcnow)
    backend_id: str | None = None
    task_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
