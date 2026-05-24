from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from evolab.contracts.common import StrictBaseModel


WorkflowNodeStatus = Literal["planned", "running", "completed", "failed", "skipped"]
PlanExecutionStatus = Literal["completed", "failed", "partial"]


class WorkflowNode(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    node_id: str
    skill_id: str
    name: str
    purpose: str
    required_inputs: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    resource_refs: list[Any] = Field(default_factory=list)
    status: WorkflowNodeStatus = "planned"
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowEdge(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    source_node_id: str
    target_node_id: str
    relation: str
    reason: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowPlan(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    plan_id: str
    task_id: str | None = None
    task_goal: str
    role: str | None = None
    nodes: list[WorkflowNode] = Field(default_factory=list)
    edges: list[WorkflowEdge] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    expected_artifacts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class NodeExecutionRecord(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    node_id: str
    skill_id: str
    status: WorkflowNodeStatus = "planned"
    started_at: str | None = None
    ended_at: str | None = None
    tool_calls: list[Any] = Field(default_factory=list)
    artifact_refs: list[Any] = Field(default_factory=list)
    output_summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlanExecutionTrace(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    plan_id: str
    run_ref: str
    status: PlanExecutionStatus
    node_records: list[NodeExecutionRecord] = Field(default_factory=list)
    tool_trace: Any | None = None
    artifact_refs: list[Any] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

