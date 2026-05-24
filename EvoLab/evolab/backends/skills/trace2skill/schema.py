from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from evolab.contracts.common import StrictBaseModel


TraceOutcome = Literal[
    "runtime_success",
    "runtime_failure",
    "evaluation_success",
    "evaluation_failure",
    "partial_success",
    "unknown",
]

LessonType = Literal[
    "error_lesson",
    "success_lesson",
    "coverage_lesson",
    "tool_lesson",
    "validation_lesson",
]

SkillPatchType = Literal[
    "skill_deepen_patch",
    "skill_create_patch",
    "required_tools_patch",
    "example_memory_patch",
    "procedure_step_patch",
    "precondition_patch",
    "failure_case_patch",
    "validation_rule_patch",
    "relationship_patch",
    "metadata_patch",
]

Trace2SkillMode = Literal[
    "error_only",
    "success_only",
    "combined",
    "skill_deepening",
    "skill_creation_from_scratch",
    "mixed",
]

RiskLevel = Literal["low", "medium", "high"]
PatchDecision = Literal["keep", "merge", "reject", "defer"]


class TraceRecord(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    trace_id: str
    task_id: str | None = None
    task_summary: str | None = None
    task_type: str | None = None
    target_skill_ids: list[str] = Field(default_factory=list)
    retrieved_skill_ids: list[str] = Field(default_factory=list)
    selected_skill_ids: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    missing_tools: list[str] = Field(default_factory=list)
    artifacts: list[Any] = Field(default_factory=list)
    final_status: TraceOutcome = "unknown"
    evaluation_metrics: dict[str, Any] = Field(default_factory=dict)
    error_summary: str | None = None
    compact_execution_summary: str | None = None
    observation_id: str | None = None
    run_record_ref: str | None = None
    created_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class TracePool(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    pool_id: str
    traces: list[TraceRecord] = Field(default_factory=list)
    success_traces: list[TraceRecord] = Field(default_factory=list)
    failure_traces: list[TraceRecord] = Field(default_factory=list)
    mixed_traces: list[TraceRecord] = Field(default_factory=list)
    task_type: str | None = None
    target_skill_ids: list[str] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class TrajectoryLesson(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    lesson_id: str
    source_trace_ids: list[str] = Field(default_factory=list)
    lesson_type: LessonType
    target_skill_id: str | None = None
    evidence_summary: str
    reusable_principle: str
    proposed_delta: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.5, ge=0, le=1)
    support_count: int = Field(default=1, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillPatchProposal(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    patch_id: str
    patch_type: SkillPatchType
    target_skill_id: str | None = None
    candidate_skill_id: str | None = None
    source_lesson_ids: list[str] = Field(default_factory=list)
    source_trace_ids: list[str] = Field(default_factory=list)
    patch_content: dict[str, Any] = Field(default_factory=dict)
    evidence_summary: str
    confidence: float = Field(default=0.5, ge=0, le=1)
    support_count: int = Field(default=1, ge=0)
    risk_level: RiskLevel = "medium"
    created_at: str


class SkillPatchBundle(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    bundle_id: str
    local_patches: list[SkillPatchProposal] = Field(default_factory=list)
    target_skill_ids: list[str] = Field(default_factory=list)
    source_trace_ids: list[str] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)


class PatchConflict(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    conflict_id: str
    conflict_type: str
    patch_ids: list[str] = Field(default_factory=list)
    target_skill_ids: list[str] = Field(default_factory=list)
    description: str
    resolution: PatchDecision = "defer"
    severity: RiskLevel = "medium"


class ConsolidatedSkillPatch(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    consolidated_patch_id: str
    patch_type: SkillPatchType
    target_skill_id: str | None = None
    candidate_skill_id: str | None = None
    merged_content: dict[str, Any] = Field(default_factory=dict)
    source_patch_ids: list[str] = Field(default_factory=list)
    source_lesson_ids: list[str] = Field(default_factory=list)
    source_trace_ids: list[str] = Field(default_factory=list)
    support_count: int = Field(default=1, ge=0)
    confidence: float = Field(default=0.5, ge=0, le=1)
    conflicts_resolved: list[PatchConflict] = Field(default_factory=list)
    validation_hints: list[str] = Field(default_factory=list)
    risk_level: RiskLevel = "medium"
    created_at: str


class PatchConsolidationResult(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    result_id: str
    consolidated_patches: list[ConsolidatedSkillPatch] = Field(default_factory=list)
    rejected_patches: list[SkillPatchProposal] = Field(default_factory=list)
    deferred_patches: list[SkillPatchProposal] = Field(default_factory=list)
    conflicts: list[PatchConflict] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)


class Trace2SkillRunConfig(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    mode: Trace2SkillMode = "combined"
    policy_mode: str = "conservative"
    target_skill_ids: list[str] = Field(default_factory=list)
    task_type: str | None = None
    max_traces: int | None = Field(default=None, ge=1)
    min_support_count: int = Field(default=1, ge=1)
    min_confidence: float = Field(default=0.25, ge=0, le=1)
    enable_llm_analysts: bool = False
    llm_config_ref: str | None = None
    max_llm_retries: int = Field(default=1, ge=0)
    llm_temperature: float = Field(default=0, ge=0)
    max_trace_summary_chars: int = Field(default=1200, ge=100)
    enable_deterministic_fallback: bool = True
    analyst_execution_mode: Literal["sequential", "thread"] = "sequential"
    analyst_max_workers: int = Field(default=4, ge=1)
    analyst_timeout_seconds: float | None = Field(default=None, gt=0)
    enable_validation_gate: bool = True
    enable_regression_gate: bool = False
    regression_metrics: list[str] = Field(default_factory=lambda: ["accuracy"])
    regression_no_regression_threshold: float = Field(default=0.0, ge=0)
    max_examples_per_skill: int = Field(default=5, ge=1)
    max_procedure_notes_per_skill: int = Field(default=10, ge=1)
    max_preconditions_per_skill: int = Field(default=10, ge=1)
    max_failure_cases_per_skill: int = Field(default=10, ge=1)
    max_validation_rules_per_skill: int = Field(default=10, ge=1)
    max_evolution_text_chars: int = Field(default=600, ge=1)
    dry_run: bool = True
    output_dir: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PatchValidationResult(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    valid_patches: list[ConsolidatedSkillPatch] = Field(default_factory=list)
    invalid_patches: list[ConsolidatedSkillPatch] = Field(default_factory=list)
    conflicts: list[PatchConflict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)


class SkillLibraryUpdateTransaction(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    transaction_id: str
    proposal_ids: list[str] = Field(default_factory=list)
    decision_ids: list[str] = Field(default_factory=list)
    applied_updates: list[dict[str, Any]] = Field(default_factory=list)
    staged_updates: list[dict[str, Any]] = Field(default_factory=list)
    rejected_updates: list[dict[str, Any]] = Field(default_factory=list)
    before_graph_hash: str | None = None
    after_graph_hash: str | None = None
    changed_library: bool = False
    status: Literal["applied", "staged", "rejected", "dry_run", "no_op"] = "no_op"
    created_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Trace2SkillRunResult(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    run_id: str
    config: Trace2SkillRunConfig
    trace_pool_summary: dict[str, Any] = Field(default_factory=dict)
    lessons: list[TrajectoryLesson] = Field(default_factory=list)
    local_patches: list[SkillPatchProposal] = Field(default_factory=list)
    consolidation_result: PatchConsolidationResult
    validation_result: PatchValidationResult | None = None
    converted_skill_update_proposals: list[Any] = Field(default_factory=list)
    policy_decisions: list[Any] = Field(default_factory=list)
    applied_transactions: list[SkillLibraryUpdateTransaction] = Field(default_factory=list)
    staged_updates: list[Any] = Field(default_factory=list)
    rejected_updates: list[Any] = Field(default_factory=list)
    regression_gate_result: dict[str, Any] | None = None
    before_metrics: dict[str, Any] = Field(default_factory=dict)
    after_metrics: dict[str, Any] = Field(default_factory=dict)
    blocked_by_regression: bool = False
    before_graph_hash: str | None = None
    after_graph_hash: str | None = None
    retrieval_change_summary: dict[str, Any] = Field(default_factory=dict)
    changed_library: bool = False
    report_paths: dict[str, str] = Field(default_factory=dict)
    status: Literal["completed", "failed", "partial", "dry_run"] = "completed"
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnalystRunResult(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    lessons: list[TrajectoryLesson] = Field(default_factory=list)
    patches: list[SkillPatchProposal] = Field(default_factory=list)
    failures: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
