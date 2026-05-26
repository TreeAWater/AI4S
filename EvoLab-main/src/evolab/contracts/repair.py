from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from evolab.contracts.common import StrictBaseModel
from evolab.contracts.generated_tools import GeneratedToolPackage
from evolab.contracts.tools import ToolCall, ToolResult


class SkillOverlayPatch(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    patch_id: str
    target_skill_id: str | None = None
    target_skill_name: str | None = None
    principles: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    recovery_strategies: list[str] = Field(default_factory=list)
    tool_use_policies: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolOverlayPatch(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    patch_id: str
    name: str
    base_tool_name: str
    strategy: str
    description: str | None = None
    replace_existing: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class FailureSignal(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    failure_id: str
    task_id: str
    subagent_id: str
    step_id: str
    failure_type: str
    severity: Literal["low", "medium", "high", "critical"]
    failed_tool_call: ToolCall | None = None
    failed_tool_result: ToolResult | None = None
    active_skill_ids: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    suspected_cause: str
    suggested_repair_actions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RepairPlan(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    repair_id: str
    failure_id: str
    diagnosis: str
    repair_action: str
    rationale: str
    skill_overlay_patch: SkillOverlayPatch | None = None
    tool_overlay_patch: ToolOverlayPatch | None = None
    new_runtime_skill: dict[str, Any] | None = None
    new_runtime_tool: GeneratedToolPackage | None = None
    retry_plan: dict[str, Any] = Field(default_factory=dict)
    validation_plan: dict[str, Any] = Field(default_factory=dict)
    promotion_candidate_policy: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RepairValidationResult(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    valid: bool
    status: Literal["passed", "failed", "skipped"]
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    before_summary: dict[str, Any] = Field(default_factory=dict)
    after_summary: dict[str, Any] = Field(default_factory=dict)
    normal_behavior_ok: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromotionCandidate(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    candidate_id: str
    candidate_type: Literal["skill_patch", "new_skill", "tool_patch", "new_tool"]
    target_id: str | None = None
    supporting_evidence: list[str] = Field(default_factory=list)
    validation_result: RepairValidationResult | None = None
    affected_ids: list[str] = Field(default_factory=list)
    recommended_decision: Literal["review", "promote", "reject"] = "review"
    metadata: dict[str, Any] = Field(default_factory=dict)
