from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import Field

from evolab.backends.embeddings.base import EmbeddingBackend
from evolab.contracts.common import StrictBaseModel
from evolab.contracts.embeddings import EmbeddingResponse


class ApiEmbeddingBackendConfig(StrictBaseModel):
    provider: Literal["openai"]
    api: Literal["openai-embeddings"] = "openai-embeddings"
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    timeout_seconds: float | None = Field(default=None, gt=0)


class ApiEmbeddingBackend(EmbeddingBackend):
    def __init__(
        self,
        config: ApiEmbeddingBackendConfig,
        *,
        backend_id: str = "api-embedding",
        api_key: str | None = None,
        client: Any | None = None,
    ):
        self.config = config
        self.backend_id = backend_id
        self._client = client
        self._api_key = api_key

    def instantiate(self, state_ref: str | None) -> "ApiEmbeddingRuntime":
        return ApiEmbeddingRuntime(
            backend_id=self.backend_id,
            model=self.config.model,
            client=self._client or _openai_client(self.config, self._api_key),
            timeout_seconds=self.config.timeout_seconds,
        )


class ApiEmbeddingRuntime:
    def __init__(self, *, backend_id: str, model: str, client: Any, timeout_seconds: float | None = None):
        self.backend_id = backend_id
        self.model = model
        self.client = client
        self.timeout_seconds = timeout_seconds

    def embed(self, texts: list[str], *, purpose: str) -> EmbeddingResponse:
        kwargs: dict[str, Any] = {"model": self.model, "input": texts}
        if self.timeout_seconds is not None:
            kwargs["timeout"] = self.timeout_seconds
        response = self.client.embeddings.create(**kwargs)
        vectors = [_embedding_item_vector(item) for item in response.data]
        return EmbeddingResponse(
            backend_id=self.backend_id,
            model=getattr(response, "model", None) or self.model,
            vectors=vectors,
            metadata={"purpose": purpose},
        )


def _openai_client(config: ApiEmbeddingBackendConfig, api_key: str | None) -> Any:
    resolved_api_key = api_key or os.environ.get(config.api_key_env)
    if not resolved_api_key:
        raise ValueError(f"missing API key in environment variable {config.api_key_env}")
    from openai import OpenAI

    kwargs: dict[str, Any] = {"api_key": resolved_api_key}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return OpenAI(**kwargs)


def _embedding_item_vector(item: Any) -> list[float]:
    embedding = item.get("embedding") if isinstance(item, dict) else getattr(item, "embedding", None)
    if not isinstance(embedding, list):
        raise ValueError("embedding response item missing embedding list")
    return [float(value) for value in embedding]
