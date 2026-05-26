from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from evolab.contracts.common import Message, StrictBaseModel
from evolab.contracts.tools import ToolCall


class LLMGenerationConfig(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    model: str
    temperature: float | None = Field(default=None, ge=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    response_json_schema: dict[str, Any] | None = None
    previous_response_id: str | None = None
    response_input_items: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMRuntimeRequest(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    messages: list[Message]
    tool_specs: list[dict[str, Any]] = Field(default_factory=list)
    generation_config: LLMGenerationConfig


class SubAgentAction(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    action: Literal["tool_call", "final_answer", "ask_human", "abort"]
    content: str | None = None
    tool_call: ToolCall | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_payload_for_action(self) -> "SubAgentAction":
        if self.action == "tool_call":
            if self.tool_call is None and not self.tool_calls:
                raise ValueError("action=tool_call requires tool_call or tool_calls")
            if self.content is not None:
                raise ValueError("action=tool_call must not include content")
            if self.tool_call is not None and not self.tool_calls:
                self.tool_calls = [self.tool_call]
            if self.tool_call is None and self.tool_calls:
                self.tool_call = self.tool_calls[0]
            if self.tool_call is not None and self.tool_calls and self.tool_call != self.tool_calls[0]:
                raise ValueError("action=tool_call tool_call must match the first tool_calls item")
            return self

        if self.tool_call is not None:
            raise ValueError(f"action={self.action} must not include tool_call")
        if self.tool_calls:
            raise ValueError(f"action={self.action} must not include tool_calls")
        if not self.content:
            raise ValueError(f"action={self.action} requires non-empty content")
        return self


class LLMRuntimeResponse(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    action: SubAgentAction
    raw_response: dict[str, Any] = Field(default_factory=dict)
