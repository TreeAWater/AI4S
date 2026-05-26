from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from evolab.contracts.common import StrictBaseModel


class EmbeddingResponse(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    vectors: list[list[float]]
    backend_id: str
    model: str
    metadata: dict[str, Any] = Field(default_factory=dict)
