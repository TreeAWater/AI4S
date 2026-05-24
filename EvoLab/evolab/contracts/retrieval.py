from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from evolab.contracts.common import ArtifactRef, Message, StrictBaseModel
from evolab.contracts.task import TaskOrigin, TaskPurpose
from evolab.contracts.tools import ToolTrace


class RetrievalRequest(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    task_id: str
    role: str
    query: str
    task_origin: TaskOrigin | None = None
    task_purpose: TaskPurpose | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


BackendScope = Literal["agent", "task"]


class MemoryAddRequest(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    task_id: str
    role: str
    messages: list[Message]
    filters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryUpdateResult(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    status: Literal["updated", "skipped", "failed", "degraded"]
    state_ref: str | None = None
    previous_state_ref: str | None = None
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryItem(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    memory_id: str
    content: str
    score: float | None = Field(default=None, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryBundle(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    items: list[MemoryItem] = Field(default_factory=list)
    backend_id: str
    state_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillItem(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    skill_id: str
    name: str
    content: str
    required_tools: list[str] = Field(default_factory=list)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    script_refs: list[ArtifactRef] = Field(default_factory=list)
    resource_refs: list[ArtifactRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


SkillRef = SkillItem


class SkillBundle(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    skills: list[SkillItem] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    backend_id: str
    graph_version_ref: str | None = None
    skill_state_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillObservationRequest(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    task_id: str
    run_ref: str
    role: str
    retrieval_request: RetrievalRequest
    skill_bundle: SkillBundle
    graph_version_ref: str | None = None
    skill_state_ref: str | None = None
    tool_trace: ToolTrace | None = None
    final_answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillUpdateResult(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    status: str
    update_summary: dict[str, Any] = Field(default_factory=dict)
    graph_version_ref: str | None = None
    skill_state_ref: str | None = None
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
