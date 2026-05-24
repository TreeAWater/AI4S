from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import Field

from evolab.contracts.common import ArtifactRef, StrictBaseModel
from evolab.contracts.retrieval import SkillItem
from evolab.contracts.tools import ToolSpec


class SnapshotRef(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    snapshot_ref: str
    kind: Literal["toolset", "skill", "environment", "reward_policy", "other"]
    uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolsetSnapshot(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    snapshot_ref: str
    kind: Literal["toolset"] = "toolset"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    tool_specs: list[ToolSpec] = Field(default_factory=list)
    implementation_refs: list[ArtifactRef] = Field(default_factory=list)
    parent_snapshot_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillSnapshot(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    snapshot_ref: str
    kind: Literal["skill"] = "skill"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    skill_backend_id: str | None = None
    skill_state_ref: str | None = None
    graph_version_ref: str | None = None
    skills: list[SkillItem] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    parent_snapshot_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RewardPolicyComponent(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    calculator_id: str
    weight: float = 1.0
    config_ref: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RewardPolicySnapshot(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    snapshot_ref: str
    kind: Literal["reward_policy"] = "reward_policy"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    components: list[RewardPolicyComponent] = Field(default_factory=list)
    combination_mode: Literal["sum", "mean", "weighted_sum", "max", "min"] = "weighted_sum"
    parent_snapshot_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EnvironmentSnapshot(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    snapshot_ref: str
    kind: Literal["environment"] = "environment"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    task_config_ref: str | None = None
    toolset_snapshot_ref: str | None = None
    skill_snapshot_ref: str | None = None
    reward_policy_snapshot_ref: str | None = None
    memory_state_refs: dict[str, str] = Field(default_factory=dict)
    llm_state_refs: dict[str, str] = Field(default_factory=dict)
    backend_state_refs: dict[str, str] = Field(default_factory=dict)
    parent_snapshot_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OtherSnapshot(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    snapshot_ref: str
    kind: Literal["other"] = "other"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    payload: dict[str, Any] = Field(default_factory=dict)
    parent_snapshot_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


SnapshotRecord = Annotated[
    ToolsetSnapshot | SkillSnapshot | EnvironmentSnapshot | RewardPolicySnapshot | OtherSnapshot,
    Field(discriminator="kind"),
]
