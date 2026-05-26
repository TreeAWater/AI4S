import math

import pytest

from evolab.backends.embeddings import (
    ApiEmbeddingBackend,
    ApiEmbeddingBackendConfig,
    EmbeddingBackend,
    FakeEmbeddingBackend,
)
from evolab.contracts.embeddings import EmbeddingResponse


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def test_fake_embedding_backend_is_exported_backend_subclass():
    assert issubclass(FakeEmbeddingBackend, EmbeddingBackend)


def test_fake_embedding_runtime_returns_deterministic_vectors():
    backend = FakeEmbeddingBackend(backend_id="embed-fake", dimensions=6)
    runtime = backend.instantiate("state-1")

    first = runtime.embed(["alpha", "beta"], purpose="search")
    second = runtime.embed(["alpha", "beta"], purpose="search")

    assert backend.instantiated_state_refs == ["state-1"]
    assert isinstance(first, EmbeddingResponse)
    assert first.backend_id == "embed-fake"
    assert first.model == "fake-deterministic"
    assert first.vectors == second.vectors
    assert len(first.vectors) == 2
    assert all(len(vector) == 6 for vector in first.vectors)
    assert all(_norm(vector) == pytest.approx(1.0) for vector in first.vectors)
    assert first.metadata["purpose"] == "search"
    assert runtime.calls == [
        {"texts": ["alpha", "beta"], "purpose": "search"},
        {"texts": ["alpha", "beta"], "purpose": "search"},
    ]


@pytest.mark.parametrize("dimensions", [0, -1])
def test_fake_embedding_backend_rejects_invalid_dimensions(dimensions):
    with pytest.raises(ValueError, match="dimensions"):
        FakeEmbeddingBackend(dimensions=dimensions)


class _FakeEmbeddingCreate:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item0 = type("EmbeddingItem", (), {"embedding": [1.0, 0.0, 0.0]})()
        item1 = type("EmbeddingItem", (), {"embedding": [0.0, 1.0, 0.0]})()
        return type(
            "EmbeddingResult",
            (),
            {"data": [item0, item1], "model": kwargs["model"], "model_dump": lambda self: {"data": []}},
        )()


class _FakeOpenAIClient:
    def __init__(self):
        self.embeddings = _FakeEmbeddingCreate()


def test_api_embedding_runtime_calls_openai_compatible_embeddings_api():
    client = _FakeOpenAIClient()
    backend = ApiEmbeddingBackend(
        ApiEmbeddingBackendConfig(provider="openai", model="text-embedding-test"),
        backend_id="embed-api",
        client=client,
    )

    response = backend.instantiate(None).embed(["alpha", "beta"], purpose="add")

    assert client.embeddings.calls == [{"model": "text-embedding-test", "input": ["alpha", "beta"]}]
    assert response.backend_id == "embed-api"
    assert response.model == "text-embedding-test"
    assert response.vectors == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert response.metadata["purpose"] == "add"
