from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from evolab.contracts.common import ArtifactRef, StrictBaseModel


class ToolSpec(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    name: str
    description: str
    parameters_schema: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolBundle(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    tool_specs: list[ToolSpec] = Field(default_factory=list)


class ToolCall(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    call_id: str
    status: Literal["ok", "error"]
    content: str
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCallRecord(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    tool_call: ToolCall
    result: ToolResult

    @model_validator(mode="after")
    def validate_call_ids_match(self) -> "ToolCallRecord":
        if self.tool_call.call_id != self.result.call_id:
            raise ValueError("tool call and result call_id must match")
        return self


class ToolTrace(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    run_ref: str
    calls: list[ToolCallRecord] = Field(default_factory=list)
