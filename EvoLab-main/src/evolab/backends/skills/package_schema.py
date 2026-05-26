from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from evolab.backends.skills.candidates import CandidateSkill, SkillSourceType
from evolab.contracts.common import StrictBaseModel


class SkillGraphSkillNode(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    id: str
    name: str
    summary: str
    package_ref: str | None = None
    group: str | None = None
    status: Literal["active", "dormant", "pinned"] = "active"
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def skill_id(self) -> str:
        return self.id

    @property
    def description(self) -> str:
        return self.summary


class SkillPackageTests(StrictBaseModel):
    smoke: list[str] = Field(default_factory=list)
    synthetic: list[str] = Field(default_factory=list)
    system: list[str] = Field(default_factory=list)
    benchmark: list[str] = Field(default_factory=list)


class SkillPackage(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    skill_id: str
    name: str
    version: str
    summary: str

    source_type: SkillSourceType = "human"
    source_uri: str = "package://local"
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

    tests: SkillPackageTests = Field(default_factory=SkillPackageTests)
    validation_signals: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    package_path: str
    skill_markdown: str | None = None

    def to_candidate_skill(self) -> CandidateSkill:
        metadata = {
            **self.metadata,
            "package_path": self.package_path,
            "skill_version": self.version,
        }
        if self.skill_markdown:
            metadata["skill_markdown"] = self.skill_markdown
        return CandidateSkill(
            skill_id=self.skill_id,
            name=self.name,
            description=self.summary,
            source_type=self.source_type,
            source_uri=self.source_uri,
            provenance=self.provenance,
            domain_tags=self.domain_tags,
            task_types=self.task_types,
            target_category=self.target_category,
            scope=self.scope,
            applicability=self.applicability,
            limitations=self.limitations,
            required_inputs=self.required_inputs,
            expected_outputs=self.expected_outputs,
            dependencies=self.dependencies,
            environment_assumptions=self.environment_assumptions,
            procedure=self.procedure,
            required_tools=self.required_tools,
            scripts=self.scripts,
            resources=self.resources,
            examples=self.examples,
            smoke_tests=self.tests.smoke,
            synthetic_tests=self.tests.synthetic,
            system_tests=self.tests.system,
            benchmark_tests=self.tests.benchmark,
            validation_signals=self.validation_signals,
            confidence=self.confidence,
            metadata=metadata,
        )

    @property
    def path(self) -> Path:
        return Path(self.package_path)


class SkillGroupConfig(StrictBaseModel):
    group_name: str
    description: str | None = None
    graph: str | None = None
    skill_roots: list[str] = Field(default_factory=list)
    domain_packages: list[str] = Field(default_factory=list)
    default_active_status: Literal["active", "dormant", "pinned"] = "active"
    include_patterns: list[str] = Field(default_factory=lambda: ["*/metadata.yaml", "*/metadata.json"])
    exclude_patterns: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
