from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, strict=True)


class Message(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactRef(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    uri: str
    type: Literal["text", "code", "dataset", "model_adapter", "image", "paper", "log", "other"]
    metadata: dict[str, Any] = Field(default_factory=dict)


class GitRef(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    uri: str
    commit: str | None = None
    branch: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class OutputSpec(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    name: str
    description: str
    required: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimePolicy(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    max_tool_steps: int = Field(default=20, ge=0)
    allow_human_tools: bool = True
    allowed_human_tools: list[str] = Field(
        default_factory=lambda: ["ask_human", "request_human_review", "notify_human"]
    )
    max_human_requests_per_run: int = Field(default=3, ge=0)
    human_tool_mock_mode: bool = True
    enable_workflow_planning: bool = False
    max_workflow_nodes: int = Field(default=20, ge=1)
    max_tool_steps_per_node: int | None = Field(default=None, ge=0)
    enable_runtime_capability_repair: bool = False
    max_repair_attempts_per_step: int = Field(default=1, ge=0)
    max_repair_attempts_per_task: int = Field(default=5, ge=0)
    allow_runtime_tool_patch: bool = True
    allow_runtime_tool_creation: bool = True
    allow_runtime_skill_overlay: bool = True
    allow_runtime_skill_creation: bool = True
    allow_global_skill_mutation: bool = False
    allow_global_tool_mutation: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvolutionBudget(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    max_wall_clock_s: int | None = Field(default=None, ge=0)
    max_train_samples: int | None = Field(default=None, ge=0)
    max_cost_usd: float | None = Field(default=None, ge=0)
    deadline_at: datetime | None = None
