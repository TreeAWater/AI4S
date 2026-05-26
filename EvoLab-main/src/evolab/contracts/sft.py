from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from evolab.contracts.common import Message, StrictBaseModel


class SFTDatasetSample(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    sample_id: str
    messages: list[Message]
    source_llm_call_ref: str
    source_run_ref: str
    task_id: str | None = None
    role: str | None = None
    teacher_backend_id: str
    teacher_model: str
    action: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SFTDatasetManifest(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    dataset_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    train_path: str
    val_path: str
    sample_count: int
    train_count: int
    val_count: int
    source_llm_call_refs: list[str] = Field(default_factory=list)
    source_run_refs: list[str] = Field(default_factory=list)
    teacher_backend_ids: list[str] = Field(default_factory=list)
    teacher_models: list[str] = Field(default_factory=list)
    selection: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
