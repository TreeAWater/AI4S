from __future__ import annotations

from typing import Literal

from pydantic import Field

from evolab.contracts.common import RuntimePolicy, StrictBaseModel
from evolab.contracts.dynamic_workflow import DynamicSubagentsConfig


class BackendBinding(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    backend_id: str
    config_ref: str | None = None
    state_ref: str | None = None


class RoleSpec(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    name: str
    system_prompt: str
    llm_backend: BackendBinding
    agent_memory_backend: BackendBinding | None = None
    allowed_tools: list[str] = Field(default_factory=list)


class MetaAgentSpec(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    name: str = "meta"
    system_prompt: str
    llm_backend: BackendBinding
    memory_backend: BackendBinding | None = None
    instruction_ref: str | None = None


class TaskConfig(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    task_id: str
    goal: str
    task_memory_backend: BackendBinding | None = None
    meta_agent: MetaAgentSpec | None = None
    roles: dict[str, RoleSpec] = Field(default_factory=dict)
    dynamic_subagents: DynamicSubagentsConfig | None = None
    max_dispatch_steps: int = Field(default=20, ge=0)
    runtime_policy: RuntimePolicy = Field(default_factory=RuntimePolicy)
