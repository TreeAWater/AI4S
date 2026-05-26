from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from evolab.contracts.common import ArtifactRef, StrictBaseModel


class RunLedgerRecord(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    task_run_id: str
    task_id: str
    task_goal: str
    status: Literal["queued", "running", "completed", "failed", "skipped", "interrupted"]
    config_ref: str | None = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    final_answer: str | None = None
    failure_reason: str | None = None
    meta_run_refs: list[str] = Field(default_factory=list)
    subagent_run_refs: list[str] = Field(default_factory=list)
    final_artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SubagentReportRecord(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    report_ref: str
    task_id: str
    run_ref: str
    role: str
    status: Literal["completed", "failed", "guard_failed", "interrupted", "partial"]
    assigned_task: str
    summary: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    coverage: dict[str, Any] = Field(default_factory=dict)
    failures: list[dict[str, Any]] = Field(default_factory=list)
    skipped_items: list[dict[str, Any]] = Field(default_factory=list)
    next_recommendations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactIndexRecord(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    artifact_ref: str
    task_id: str
    producer_run_ref: str | None = None
    uri: str
    artifact_type: str
    role: str | None = None
    status: Literal["intermediate", "final", "validation", "audit", "candidate", "rejected"] = "intermediate"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrainingIndexRecord(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    sample_ref: str
    task_id: str
    source_run_ref: str
    source_llm_call_refs: list[str] = Field(default_factory=list)
    sample_kind: Literal["meta_route", "subagent_trace", "tool_use", "failure_repair"]
    quality_label: Literal["accepted", "candidate", "rejected", "unknown"] = "candidate"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvolutionProductRecord(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    product_ref: str
    product_type: Literal["skill", "tool", "checkpoint", "policy", "other"]
    status: Literal["candidate", "validated", "promoted", "rejected"] = "candidate"
    source_run_refs: list[str] = Field(default_factory=list)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    validation: dict[str, Any] = Field(default_factory=dict)
    promotion: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LabStateIndexTaskSummary(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    task_id: str
    task_goal: str
    task_request_ref: str | None = None
    status: str | None = None


class LabStateTrajectorySummary(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    meta_agent_run_count: int = 0
    subagent_run_count: int = 0
    llm_call_count: int = 0
    tool_call_count: int = 0
    event_count: int = 0
    evolution_run_count: int = 0


class LabStateIndex(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    index_ref: str
    task: LabStateIndexTaskSummary
    trajectory: LabStateTrajectorySummary = Field(default_factory=LabStateTrajectorySummary)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    queue_summary: dict[str, Any] = Field(default_factory=dict)
    subagent_reports: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    backend_states: list[dict[str, Any]] = Field(default_factory=list)
    evolution_products: list[dict[str, Any]] = Field(default_factory=list)
    training_samples: list[dict[str, Any]] = Field(default_factory=list)
    detail_refs: dict[str, list[str]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LabStateDigest(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    digest_ref: str
    index_ref: str
    task_id: str
    summary: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    sections: dict[str, Any] = Field(default_factory=dict)
    detail_refs: dict[str, list[str]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
