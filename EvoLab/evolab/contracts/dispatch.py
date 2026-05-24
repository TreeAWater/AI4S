from __future__ import annotations

try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        pass

from typing import Any, Literal

from pydantic import Field, model_validator

from evolab.contracts.common import OutputSpec, StrictBaseModel


class DispatchAction(StrEnum):
    RUN_SUBAGENT = "run_subagent"
    FINISH_TASK = "finish_task"
    ASK_HUMAN = "ask_human"
    ABORT = "abort"


class DispatchDecision(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    action: DispatchAction
    target_role: str | None = None
    instruction: str | None = None
    retrieval_query: str | None = None
    expected_outputs: list[OutputSpec] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_required_fields(self) -> "DispatchDecision":
        if self.action == DispatchAction.RUN_SUBAGENT:
            if not self.target_role:
                raise ValueError("run_subagent requires target_role")
            if not self.instruction:
                raise ValueError("run_subagent requires instruction")
        return self
