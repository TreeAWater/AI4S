from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import Field

from evolab.contracts.common import StrictBaseModel


LOCAL_TRAINABLE_STATE_SCHEME = "local-trainable"


class LocalTrainableStateManifest(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    backend_id: str
    state_ref: str
    parent_state_ref: str | None = None
    created_by_trainer: str
    adapter_uri: str | None = None
    dataset_manifest_uri: str | None = None
    default_content: str = "local trainable state response"
    serving: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def new_local_trainable_state_ref(backend_id: str) -> str:
    return f"{LOCAL_TRAINABLE_STATE_SCHEME}://{backend_id}/state/{uuid4()}"
