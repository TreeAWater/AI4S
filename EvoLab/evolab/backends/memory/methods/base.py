from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import Field

from evolab.contracts.common import Message, StrictBaseModel
from evolab.contracts.retrieval import MemoryItem


class MemorySearchRequest(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    task_id: str
    role: str
    query: str
    scope: Literal["agent", "task"]
    scope_id: str
    filters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    top_k: int | None = Field(default=None, ge=0)
    threshold: float | None = Field(default=None, ge=0, le=1)


class MemoryIngestRequest(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    task_id: str
    role: str
    messages: list[Message]
    scope: Literal["agent", "task"]
    scope_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemorySearchResult(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    items: list[MemoryItem] = Field(default_factory=list)
    state_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryIngestResult(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    status: Literal["updated", "skipped", "failed", "degraded"]
    state_ref: str | None = None
    previous_state_ref: str | None = None
    added_memory_ids: list[str] = Field(default_factory=list)
    skipped_memory_ids: list[str] = Field(default_factory=list)
    linked_memory_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryMethod(Protocol):
    method_name: str

    def instantiate(self, state_ref: str | None) -> "MemoryMethod":
        raise NotImplementedError

    def search(self, request: MemorySearchRequest) -> MemorySearchResult:
        raise NotImplementedError

    def add(self, request: MemoryIngestRequest) -> MemoryIngestResult:
        raise NotImplementedError
