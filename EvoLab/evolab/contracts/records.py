from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field, model_validator

from evolab.contracts.common import ArtifactRef, Message, StrictBaseModel
from evolab.contracts.dispatch import DispatchDecision
from evolab.contracts.evolution import LLMEvolutionMode, LLMEvolutionResult
from evolab.contracts.retrieval import MemoryBundle, RetrievalRequest, SkillBundle
from evolab.contracts.task import ProposedTaskRelationType, TaskOrigin, TaskPurpose
from evolab.contracts.tools import ToolCallRecord


class LLMCallRecord(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    call_ref: str
    run_ref: str
    backend_id: str
    model: str
    input_messages: list[Message]
    output_messages: list[Message] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrajectoryEventRecord(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    event_ref: str
    event_type: str
    subject_type: str
    subject_ref: str | None = None
    task_id: str | None = None
    run_ref: str | None = None
    parent_ref: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCallTrajectoryRecord(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    record_ref: str
    run_ref: str
    task_id: str
    tool_call_id: str
    tool_name: str
    role: str | None = None
    runtime_stage: str | None = None
    step_index: int | None = None
    workflow_node_id: str | None = None
    record: ToolCallRecord
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetaAgentRunRecord(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    run_ref: str
    task_id: str
    decision: DispatchDecision
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SubagentRunRecord(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    run_ref: str
    task_id: str
    task_origin: TaskOrigin
    task_purpose: TaskPurpose
    producer_ref: str | None = None
    round_id: str | None = None
    human_anchor_task_refs: list[str] = Field(default_factory=list)
    human_anchor_trajectory_refs: list[str] = Field(default_factory=list)
    proposed_relation_type: ProposedTaskRelationType | None = None
    expected_transfer: str | None = None
    stage_index: int
    role: str
    instruction: str
    retrieval_request: RetrievalRequest
    memory_bundle: MemoryBundle
    skill_bundle: SkillBundle
    prompt_messages: list[Message]
    llm_call_refs: list[str] = Field(default_factory=list)
    llm_backend_id: str
    llm_backend_config_ref: str | None = None
    llm_backend_state_ref: str | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    output_messages: list[Message] = Field(default_factory=list)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvolutionRunRecord(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    run_ref: str
    mode: LLMEvolutionMode
    backend_id: str
    result_status: Literal["promoted_candidate", "not_recommended", "skipped", "failed"]
    result: LLMEvolutionResult
    training_trajectory_refs: list[str] = Field(default_factory=list)
    input_snapshot_refs: list[str] = Field(default_factory=list)
    consumed_instance_snapshot_refs: list[str] = Field(default_factory=list)
    output_snapshot_refs: list[str] = Field(default_factory=list)
    parent_evolution_run_refs: list[str] = Field(default_factory=list)
    lora_role: Literal["solver", "skill_distilled", "composed"] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_result_consistency(self) -> "EvolutionRunRecord":
        if self.result_status != self.result.status:
            raise ValueError("result_status must match result.status")
        if self.lora_role is not None and self.lora_role != self.result.lora_role:
            raise ValueError("lora_role must match result.lora_role")
        return self
