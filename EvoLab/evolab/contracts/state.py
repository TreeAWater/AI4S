from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from evolab.contracts.common import ArtifactRef, StrictBaseModel


class BackendStateRecord(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    state_ref: str
    backend_id: str
    backend_type: Literal["llm", "memory", "skill"]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_from_task_id: str | None = None
    created_from_run_ref: str | None = None
    parent_state_refs: list[str] = Field(default_factory=list)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    active: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
