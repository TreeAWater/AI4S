from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import Field, model_validator

from evolab.backends.skills.candidates import CandidateSkill
from evolab.backends.skills.package_schema import SkillGraphSkillNode
from evolab.contracts.common import StrictBaseModel


SCIENTIFIC_PROCESS_CAPABILITIES = (
    "Research",
    "Literature",
    "Data Preparation",
    "Analysis",
    "Validation",
    "Execution",
    "Writing",
)


class SkillCategoryNode(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    category_id: str
    name: str
    layer: Literal["scientific_process_capability", "scientific_task"] = "scientific_task"
    description: str | None = None
    parent_category_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_scientific_process_capability_name(self):
        if self.layer == "scientific_process_capability" and self.name not in SCIENTIFIC_PROCESS_CAPABILITIES:
            allowed = ", ".join(SCIENTIFIC_PROCESS_CAPABILITIES)
            raise ValueError(f"scientific_process_capability name must be one of: {allowed}")
        return self


class SkillGraphEdge(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    source_id: str
    target_id: str
    relation: str
    weight: float | None = Field(default=None, ge=0, le=1)
    deprecated: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class MissingSkillReport(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    missing_capability: str
    reason: str
    can_be_solved_by_existing_tools: bool
    risk_level: Literal["low", "medium", "high"]
    on_demand_synthesis_allowed: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillUpdateSummary(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    source_run_id: str | None = None
    candidate_skill_id: str | None = None
    update_type: str
    affected_skill_ids: list[str] = Field(default_factory=list)
    affected_edges: list[dict[str, Any]] = Field(default_factory=list)
    decision_rationale: str | None = None
    validation_signals: list[str] = Field(default_factory=list)
    graph_version_before: str | None = None
    graph_version_after: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance: dict[str, Any] = Field(default_factory=dict)


class SkillGraph(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    version: str = "v1"
    skills: list[CandidateSkill | SkillGraphSkillNode] = Field(default_factory=list)
    categories: list[SkillCategoryNode] = Field(default_factory=list)
    edges: list[SkillGraphEdge] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
