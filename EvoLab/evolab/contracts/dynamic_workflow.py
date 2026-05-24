from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from evolab.contracts.common import StrictBaseModel


class DynamicBackendBinding(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    backend_id: str
    config_ref: str | None = None
    state_ref: str | None = None


class DynamicSubagentsConfig(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    enabled: bool = False
    mode: Literal["dynamic", "static", "hybrid"] = "dynamic"
    scope: Literal["per_task", "per_work_item"] = "per_task"
    planner_backend: DynamicBackendBinding | None = None
    default_worker_backend: DynamicBackendBinding | None = None
    max_subagents: int = Field(default=8, ge=1)
    max_subagents_per_work_item: int = Field(default=8, ge=1)
    max_planner_retries: int = Field(default=2, ge=0)
    fallback_to_static: bool = True
    allow_skill_required_tools: bool = True
    require_output_schema: bool = True
    allowed_tool_names: list[str] = Field(default_factory=list)
    allowed_worker_backend_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_backends_when_enabled(self) -> "DynamicSubagentsConfig":
        if self.enabled and self.mode != "static":
            if self.planner_backend is None:
                raise ValueError("dynamic_subagents.enabled requires planner_backend")
            if self.default_worker_backend is None:
                raise ValueError("dynamic_subagents.enabled requires default_worker_backend")
        return self


class DynamicSubAgentSpec(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    subagent_id: str
    role_name: str
    goal: str
    system_prompt: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    allowed_tools: list[str] = Field(default_factory=list)
    required_skills: list[str] = Field(default_factory=list)
    skill_retrieval_request: dict[str, Any] | None = None
    artifact_inputs: list[str] = Field(default_factory=list)
    artifact_outputs: list[str] = Field(default_factory=list)
    max_turns: int = Field(default=8, ge=1)
    max_tool_calls: int = Field(default=8, ge=0)
    llm_backend_id: str | None = None
    constraints: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_private_reasoning_fields(self) -> "DynamicSubAgentSpec":
        for key in ("chain_of_thought", "reasoning", "hidden_reasoning"):
            if key in self.metadata:
                raise ValueError(f"dynamic subagent metadata must not include {key}")
        return self


class DynamicWorkflowNodeSpec(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    node_id: str
    subagent_id: str
    input_artifacts: list[str] = Field(default_factory=list)
    output_artifacts: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    execution_constraints: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DynamicWorkflowEdgeSpec(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    source_node_id: str
    target_node_id: str
    relation: str = "depends_on"
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DynamicWorkflowSpec(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    workflow_id: str
    work_item_id: str | None = None
    task_summary: str
    article_context_summary: str | None = None
    dynamic_subagents: list[DynamicSubAgentSpec] = Field(default_factory=list)
    workflow_nodes: list[DynamicWorkflowNodeSpec] = Field(default_factory=list)
    workflow_edges: list[DynamicWorkflowEdgeSpec] = Field(default_factory=list)
    artifact_contracts: dict[str, Any] = Field(default_factory=dict)
    validation_rules: list[str] = Field(default_factory=list)
    planner_rationale_summary: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_basic_references(self) -> "DynamicWorkflowSpec":
        if not self.dynamic_subagents:
            raise ValueError("dynamic workflow requires at least one dynamic_subagent")
        if not self.workflow_nodes:
            raise ValueError("dynamic workflow requires at least one workflow_node")
        subagent_ids = [agent.subagent_id for agent in self.dynamic_subagents]
        node_ids = [node.node_id for node in self.workflow_nodes]
        if len(subagent_ids) != len(set(subagent_ids)):
            raise ValueError("dynamic_subagents contains duplicate subagent_id")
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("workflow_nodes contains duplicate node_id")
        subagent_set = set(subagent_ids)
        node_set = set(node_ids)
        for node in self.workflow_nodes:
            if node.subagent_id not in subagent_set:
                raise ValueError(f"workflow node {node.node_id!r} references unknown subagent_id {node.subagent_id!r}")
            for dependency in node.dependencies:
                if dependency not in node_set:
                    raise ValueError(f"workflow node {node.node_id!r} references unknown dependency {dependency!r}")
        for edge in self.workflow_edges:
            if edge.source_node_id not in node_set or edge.target_node_id not in node_set:
                raise ValueError("workflow edge references an unknown node")
            if edge.source_node_id == edge.target_node_id:
                raise ValueError("workflow edge cannot be self-referential")
        for key in ("chain_of_thought", "reasoning", "hidden_reasoning"):
            if key in self.metadata:
                raise ValueError(f"dynamic workflow metadata must not include {key}")
        return self


class DynamicWorkflowValidationReport(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    planner_backend_id: str | None = None
    default_worker_backend_id: str | None = None
    worker_backend_ids: list[str] = Field(default_factory=list)
    allowed_tool_names: list[str] = Field(default_factory=list)
    resolved_skill_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DynamicWorkflowTrace(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    workflow_id: str
    work_item_id: str | None = None
    execution_mode: Literal["dynamic", "static_fallback"]
    status: Literal["completed", "failed", "partial", "fallback"]
    planner_backend_id: str | None = None
    default_worker_backend_id: str | None = None
    run_refs: list[str] = Field(default_factory=list)
    node_results: list[dict[str, Any]] = Field(default_factory=list)
    fallback_reason: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
