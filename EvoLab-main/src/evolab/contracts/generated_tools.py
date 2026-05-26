from __future__ import annotations

from hashlib import sha256
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import Field, model_validator

from evolab.contracts.common import StrictBaseModel
from evolab.contracts.tools import ToolSpec


class GeneratedToolFile(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    path: str
    content: str

    @model_validator(mode="after")
    def validate_relative_package_path(self) -> "GeneratedToolFile":
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts or not self.path.strip():
            raise ValueError("generated tool file path must be a relative path inside the package")
        return self


class GeneratedToolSmokeTest(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    expect_status: Literal["ok", "error"] | None = None


class GeneratedToolCapabilityGrant(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    allowed_read_roots: list[str] = Field(default_factory=list)
    allowed_write_root: str | None = None
    allowed_env_keys: list[str] = Field(default_factory=list)
    allow_network: bool = False
    allow_subprocess: bool = False
    allowed_imports: list[str] = Field(default_factory=list)


class GeneratedToolPackage(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    tool_name: str
    reason: str
    manifest: dict[str, Any] = Field(default_factory=dict)
    files: list[GeneratedToolFile] = Field(default_factory=list)
    primary_module: str = "tool.py"
    smoke_tests: list[GeneratedToolSmokeTest] = Field(default_factory=list)
    capability_grant: GeneratedToolCapabilityGrant | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_package(self) -> "GeneratedToolPackage":
        if not self.tool_name.strip():
            raise ValueError("generated tool package requires tool_name")
        if not self.reason.strip():
            raise ValueError("generated tool package requires reason")
        if not self.files:
            raise ValueError("generated tool package requires at least one file")
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("generated tool package contains duplicate file paths")
        if self.primary_module not in set(paths):
            raise ValueError("generated tool primary_module must match one package file")
        return self

    @property
    def source_bytes(self) -> int:
        return sum(len(item.content.encode("utf-8")) for item in self.files)

    @property
    def package_hash(self) -> str:
        digest = sha256()
        for item in sorted(self.files, key=lambda file: file.path):
            digest.update(item.path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(item.content.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()


class GeneratedToolValidationResult(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    valid: bool
    status: Literal["passed", "failed", "skipped"]
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    smoke_test_results: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GeneratedToolRegistration(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    requested_tool_name: str
    registered_tool_name: str
    task_id: str
    run_ref: str
    tool_spec: ToolSpec
    module_path: str
    code_hash: str
    validation: GeneratedToolValidationResult
    capability_grant: GeneratedToolCapabilityGrant = Field(default_factory=GeneratedToolCapabilityGrant)
    artifact_refs: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskEffectiveToolCatalog(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    task_id: str
    builtin_allowed_tool_names: list[str] = Field(default_factory=list)
    generated_tool_names: list[str] = Field(default_factory=list)
    tool_specs_by_name: dict[str, ToolSpec] = Field(default_factory=dict)
    provenance_by_name: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @property
    def effective_allowed_tool_names(self) -> list[str]:
        seen: set[str] = set()
        names: list[str] = []
        for name in [*self.builtin_allowed_tool_names, *self.generated_tool_names]:
            if name not in seen:
                names.append(name)
                seen.add(name)
        return names
