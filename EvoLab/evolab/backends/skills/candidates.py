from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from evolab.contracts.common import StrictBaseModel


SkillSourceType = Literal[
    "github",
    "paper",
    "pubmed",
    "package",
    "notebook",
    "database",
    "web",
    "human",
]


class CandidateSkill(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    skill_id: str
    name: str
    description: str

    source_type: SkillSourceType
    source_uri: str
    provenance: dict[str, Any] = Field(default_factory=dict)

    domain_tags: list[str] = Field(default_factory=list)
    task_types: list[str] = Field(default_factory=list)
    target_category: str | None = None

    scope: str
    applicability: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    required_inputs: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    environment_assumptions: list[str] = Field(default_factory=list)

    procedure: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    scripts: list[str] = Field(default_factory=list)
    resources: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)

    smoke_tests: list[str] = Field(default_factory=list)
    synthetic_tests: list[str] = Field(default_factory=list)
    system_tests: list[str] = Field(default_factory=list)
    benchmark_tests: list[str] = Field(default_factory=list)
    validation_signals: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
