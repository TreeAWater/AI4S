# Memory Methods And Native Mem0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a shared memory-method architecture, add first-class embedding backends, and replace the old Mem0-compatible adapter with a production-grade EvoLab-native mem0 memory method.

**Architecture:** Keep `MemoryBackend` as the runtime-facing interface, add `MethodMemoryBackend` as the adapter from EvoLab contracts to algorithm-specific `MemoryMethod`s, and implement native mem0 as the first method. Native mem0 persists scoped memories in SQLite, calls EvoLab LLM runtimes for ADD-only extraction, calls EvoLab embedding runtimes for semantic retrieval, and uses hybrid search with keyword and entity signals.

**Tech Stack:** Python 3.10, Pydantic, SQLite stdlib, OpenAI-compatible HTTP client through existing `openai` package, pytest, existing EvoLab registry and runtime contracts.

---

## File Structure

- Create `evolab/contracts/embeddings.py`: embedding request/response contracts.
- Create `evolab/backends/embeddings/base.py`: abstract embedding backend and runtime protocol.
- Create `evolab/backends/embeddings/fake.py`: deterministic test embedding backend.
- Create `evolab/backends/embeddings/api.py`: OpenAI-compatible embedding backend.
- Create `evolab/backends/embeddings/__init__.py`: exports.
- Modify `evolab/cli.py`: build `backends.embedding`; pass embedding backends to `TaskWorker`; build method-backed memory configs; remove default in-memory Mem0 path.
- Modify `evolab/runtime/task_worker.py`: initialize embedding runtimes before memory runtimes and bind method memory dependencies.
- Create `evolab/backends/memory/method_backend.py`: runtime-facing adapter that wraps `MemoryMethod`.
- Create `evolab/backends/memory/methods/base.py`: method contracts and scope helpers.
- Create `evolab/backends/memory/methods/store.py`: SQLite store and audit log writer.
- Create `evolab/backends/memory/methods/retrieval.py`: cosine similarity, BM25 scoring, entity boost, score fusion.
- Create `evolab/backends/memory/methods/mem0.py`: native mem0 method, prompts, extraction parsing, add/search flow.
- Modify `evolab/backends/memory/__init__.py`: export native method backend and remove `InMemoryMem0Client` from production exports.
- Delete `evolab/backends/memory/mem0.py`: remove the old external-package adapter after native mem0 config is wired.
- Modify `docs/memory.md`: document memory methods and native mem0.
- Modify configs that currently use `client: in_memory`.
- Create tests listed per task below.

## Task 1: Embedding Contracts And Fake Backend

**Files:**
- Create: `evolab/contracts/embeddings.py`
- Create: `evolab/backends/embeddings/base.py`
- Create: `evolab/backends/embeddings/fake.py`
- Create: `evolab/backends/embeddings/__init__.py`
- Test: `tests/test_embedding_backends.py`

- [ ] **Step 1: Write failing tests for embedding contracts and fake backend**

Add this to `tests/test_embedding_backends.py`:

```python
import pytest

from evolab.backends.embeddings import EmbeddingBackend, FakeEmbeddingBackend
from evolab.contracts.embeddings import EmbeddingResponse


def test_fake_embedding_backend_is_exported_backend_subclass():
    assert issubclass(FakeEmbeddingBackend, EmbeddingBackend)


def test_fake_embedding_runtime_returns_deterministic_vectors():
    backend = FakeEmbeddingBackend(backend_id="embed-fake", dimensions=6)
    runtime = backend.instantiate(None)

    first = runtime.embed(["alpha", "beta"], purpose="search")
    second = runtime.embed(["alpha", "beta"], purpose="search")

    assert isinstance(first, EmbeddingResponse)
    assert first.backend_id == "embed-fake"
    assert first.model == "fake-deterministic"
    assert first.vectors == second.vectors
    assert len(first.vectors) == 2
    assert all(len(vector) == 6 for vector in first.vectors)
    assert first.metadata["purpose"] == "search"


def test_fake_embedding_backend_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match="dimensions"):
        FakeEmbeddingBackend(dimensions=0)
```

- [ ] **Step 2: Run the tests and verify import failure**

Run:

```bash
pytest -q tests/test_embedding_backends.py
```

Expected: FAIL because `evolab.backends.embeddings` does not exist.

- [ ] **Step 3: Add embedding contracts**

Create `evolab/contracts/embeddings.py`:

```python
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
```

- [ ] **Step 4: Add embedding backend base protocol**

Create `evolab/backends/embeddings/base.py`:

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from evolab.contracts.embeddings import EmbeddingResponse


class EmbeddingRuntime(Protocol):
    def embed(self, texts: list[str], *, purpose: str) -> EmbeddingResponse:
        raise NotImplementedError


class EmbeddingBackend(ABC):
    backend_id: str

    @abstractmethod
    def instantiate(self, state_ref: str | None) -> EmbeddingRuntime:
        raise NotImplementedError
```

- [ ] **Step 5: Add deterministic fake embedding backend**

Create `evolab/backends/embeddings/fake.py`:

```python
from __future__ import annotations

import hashlib
import math

from evolab.backends.embeddings.base import EmbeddingBackend
from evolab.contracts.embeddings import EmbeddingResponse


class FakeEmbeddingBackend(EmbeddingBackend):
    def __init__(self, backend_id: str = "fake-embedding", dimensions: int = 8):
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.backend_id = backend_id
        self.dimensions = dimensions
        self.instantiated_state_refs: list[str | None] = []

    def instantiate(self, state_ref: str | None) -> "FakeEmbeddingRuntime":
        self.instantiated_state_refs.append(state_ref)
        return FakeEmbeddingRuntime(self.backend_id, self.dimensions)


class FakeEmbeddingRuntime:
    def __init__(self, backend_id: str, dimensions: int):
        self.backend_id = backend_id
        self.dimensions = dimensions
        self.calls: list[dict[str, object]] = []

    def embed(self, texts: list[str], *, purpose: str) -> EmbeddingResponse:
        self.calls.append({"texts": list(texts), "purpose": purpose})
        return EmbeddingResponse(
            backend_id=self.backend_id,
            model="fake-deterministic",
            vectors=[_vector(text, self.dimensions) for text in texts],
            metadata={"purpose": purpose},
        )


def _vector(text: str, dimensions: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = [((digest[index % len(digest)] / 255.0) * 2.0) - 1.0 for index in range(dimensions)]
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]
```

- [ ] **Step 6: Export embedding package**

Create `evolab/backends/embeddings/__init__.py`:

```python
from evolab.backends.embeddings.base import EmbeddingBackend, EmbeddingRuntime
from evolab.backends.embeddings.fake import FakeEmbeddingBackend, FakeEmbeddingRuntime

__all__ = [
    "EmbeddingBackend",
    "EmbeddingRuntime",
    "FakeEmbeddingBackend",
    "FakeEmbeddingRuntime",
]
```

- [ ] **Step 7: Run tests**

Run:

```bash
pytest -q tests/test_embedding_backends.py
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add evolab/contracts/embeddings.py evolab/backends/embeddings tests/test_embedding_backends.py
git commit -m "Add embedding backend contracts"
```

## Task 2: API Embedding Backend And CLI Builder

**Files:**
- Create: `evolab/backends/embeddings/api.py`
- Modify: `evolab/backends/embeddings/__init__.py`
- Modify: `evolab/cli.py`
- Modify: `evolab/runtime/task_worker.py`
- Test: `tests/test_embedding_backends.py`
- Test: `tests/test_cli_clean_run.py`

- [ ] **Step 1: Add tests for API embedding backend**

Append to `tests/test_embedding_backends.py`:

```python
from evolab.backends.embeddings import ApiEmbeddingBackend, ApiEmbeddingBackendConfig


class _FakeEmbeddingCreate:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item0 = type("EmbeddingItem", (), {"embedding": [1.0, 0.0, 0.0]})()
        item1 = type("EmbeddingItem", (), {"embedding": [0.0, 1.0, 0.0]})()
        return type("EmbeddingResult", (), {"data": [item0, item1], "model": kwargs["model"], "model_dump": lambda self: {"data": []}})()


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
```

- [ ] **Step 2: Add tests for CLI embedding builder and worker runtime initialization**

Add to `tests/test_cli_clean_run.py` near the existing LLM builder tests:

```python
from evolab.backends.embeddings import FakeEmbeddingBackend


def test_clean_run_config_builds_fake_embedding_backend():
    config = {
        "backends": {
            "embedding": {
                "memory-embedding": {
                    "type": "fake",
                    "dimensions": 5,
                }
            }
        }
    }

    backends = _build_embedding_backends(config)

    backend = backends["memory-embedding"]
    assert isinstance(backend, FakeEmbeddingBackend)
    assert backend.backend_id == "memory-embedding"
    assert backend.dimensions == 5
```

- [ ] **Step 3: Run tests and verify failures**

Run:

```bash
pytest -q tests/test_embedding_backends.py tests/test_cli_clean_run.py::test_clean_run_config_builds_fake_embedding_backend
```

Expected: FAIL because `ApiEmbeddingBackend` and `_build_embedding_backends` do not exist.

- [ ] **Step 4: Implement API embedding backend**

Create `evolab/backends/embeddings/api.py`:

```python
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
```

- [ ] **Step 5: Export API embedding backend**

Update `evolab/backends/embeddings/__init__.py`:

```python
from evolab.backends.embeddings.api import ApiEmbeddingBackend, ApiEmbeddingBackendConfig, ApiEmbeddingRuntime
from evolab.backends.embeddings.base import EmbeddingBackend, EmbeddingRuntime
from evolab.backends.embeddings.fake import FakeEmbeddingBackend, FakeEmbeddingRuntime

__all__ = [
    "ApiEmbeddingBackend",
    "ApiEmbeddingBackendConfig",
    "ApiEmbeddingRuntime",
    "EmbeddingBackend",
    "EmbeddingRuntime",
    "FakeEmbeddingBackend",
    "FakeEmbeddingRuntime",
]
```

- [ ] **Step 6: Add CLI embedding builder**

Modify `evolab/cli.py` imports:

```python
from evolab.backends.embeddings import ApiEmbeddingBackend, ApiEmbeddingBackendConfig, FakeEmbeddingBackend
```

Add a builder near `_build_llm_backends`:

```python
def _build_embedding_backends(
    config: dict[str, Any],
    config_dir: Path | None = None,
) -> dict[str, Any]:
    backends = {}
    env_values = _load_dotenv_values(config, config_dir=config_dir)
    for backend_id, payload in _backend_section(config, "embedding").items():
        backend_type = payload.get("type")
        if backend_type == "fake":
            backends[backend_id] = FakeEmbeddingBackend(
                backend_id=backend_id,
                dimensions=_int_backend_option(payload, {}, "dimensions", 8),
            )
            continue
        if backend_type == "api":
            backends[backend_id] = _build_api_embedding_backend(
                backend_id=backend_id,
                payload=payload,
                env_values=env_values,
            )
            continue
        raise ValueError(f"backend {backend_id!r} has unsupported embedding type {backend_type!r}")
    return backends


def _build_api_embedding_backend(
    *,
    backend_id: str,
    payload: dict[str, Any],
    env_values: dict[str, str],
) -> ApiEmbeddingBackend:
    env_ref = payload.get("env_ref")
    env_entry = _api_env_entry(env_values, env_ref, backend_id)
    api_kind = payload.get("api") or env_entry.get("api") or "openai-embeddings"
    if api_kind != "openai-embeddings":
        raise ValueError(f"backend {backend_id!r} has unsupported embedding api {api_kind!r}")
    model = payload.get("model") or env_entry.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError(f"backend {backend_id!r} requires a non-empty model")
    if "api_key" in payload or "apiKey" in payload:
        raise ValueError(f"backend {backend_id!r} must not include inline api keys; use env_ref or api_key_env")
    api_key_env = payload.get("api_key_env", "OPENAI_API_KEY")
    api_key = env_entry.get("api_key") or env_entry.get("apiKey") or _first_env_value(env_values, api_key_env)
    base_url = payload.get("base_url") or payload.get("baseUrl") or env_entry.get("base_url") or env_entry.get("baseUrl")
    return ApiEmbeddingBackend(
        ApiEmbeddingBackendConfig(
            provider="openai",
            api=api_kind,
            model=model,
            api_key_env=api_key_env,
            base_url=base_url,
            timeout_seconds=_optional_float_backend_option(payload, env_entry, "timeout_seconds"),
        ),
        backend_id=backend_id,
        api_key=api_key,
    )
```

- [ ] **Step 7: Add embedding runtime maps to TaskWorker**

Modify `TaskWorker.__init__` parameters and attributes:

```python
embedding_backends: dict[str, Any] | None = None,
embedding_runtimes: dict[str, Any] | None = None,
```

Set:

```python
self.embedding_backends = embedding_backends or {}
self.embedding_runtimes = embedding_runtimes or {}
```

In `startup`, initialize embeddings after LLMs and before memories:

```python
self._initialize_backend_map(
    self.embedding_backends,
    None,
    self.embedding_runtimes,
)
for backend in self.memory_backends.values():
    bind = getattr(backend, "bind_runtimes", None)
    if bind is not None:
        bind(llm_runtimes=self.llm_runtimes, embedding_runtimes=self.embedding_runtimes)
```

- [ ] **Step 8: Pass embedding backends from clean-run**

In `run_clean_demo`, build locals before constructing `TaskWorker`:

```python
llm_backends = _build_llm_backends(
    config,
    config_dir=config_path.resolve().parent,
    backend_state_registry=resolver.backend_state_registry(),
)
embedding_backends = _build_embedding_backends(config, config_dir=config_path.resolve().parent)
memory_backends = _build_memory_backends(config, config_dir=layout.root)
```

Pass:

```python
llm_backends=llm_backends,
embedding_backends=embedding_backends,
memory_backends=memory_backends,
```

- [ ] **Step 9: Run tests**

Run:

```bash
pytest -q tests/test_embedding_backends.py tests/test_cli_clean_run.py::test_clean_run_config_builds_fake_embedding_backend
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add evolab/backends/embeddings evolab/cli.py evolab/runtime/task_worker.py tests/test_embedding_backends.py tests/test_cli_clean_run.py
git commit -m "Add embedding backend builder"
```

## Task 3: Memory Method Contracts And Backend Adapter

**Files:**
- Create: `evolab/backends/memory/methods/base.py`
- Create: `evolab/backends/memory/method_backend.py`
- Modify: `evolab/backends/memory/__init__.py`
- Test: `tests/test_memory_method_backend.py`

- [ ] **Step 1: Write failing tests for method backend mapping and dependency binding**

Create `tests/test_memory_method_backend.py`:

```python
from evolab.backends.memory import MethodMemoryBackend
from evolab.backends.memory.methods.base import MemoryIngestResult, MemorySearchResult
from evolab.contracts.common import Message
from evolab.contracts.retrieval import RetrievalRequest


class _Method:
    method_name = "test-method"

    def __init__(self):
        self.search_requests = []
        self.add_requests = []
        self.instantiated_state_refs = []
        self.bound = None

    def bind_runtimes(self, *, llm_runtimes, embedding_runtimes):
        self.bound = {"llm": llm_runtimes, "embedding": embedding_runtimes}

    def instantiate(self, state_ref):
        self.instantiated_state_refs.append(state_ref)
        return self

    def search(self, request):
        self.search_requests.append(request)
        return MemorySearchResult(
            items=[],
            state_ref="method://memory-local/12:agent:solver/v0",
            metadata={"seen_scope_id": request.scope_id},
        )

    def add(self, request):
        self.add_requests.append(request)
        return MemoryIngestResult(
            status="updated",
            state_ref="method://memory-local/12:agent:solver/v1",
            previous_state_ref="method://memory-local/12:agent:solver/v0",
            metadata={"added": True},
        )


def test_method_memory_backend_maps_search_scope_and_result():
    method = _Method()
    backend = MethodMemoryBackend(backend_id="memory-local", method=method)

    bundle = backend.search(
        RetrievalRequest(
            task_id="task-1",
            role="solver",
            query="prior work",
            filters={"memory_scope": "agent", "memory_scope_id": "agent:solver"},
        )
    )

    assert method.search_requests[0].scope == "agent"
    assert method.search_requests[0].scope_id == "agent:solver"
    assert bundle.backend_id == "memory-local"
    assert bundle.state_ref == "method://memory-local/12:agent:solver/v0"
    assert bundle.metadata["seen_scope_id"] == "agent:solver"


def test_method_memory_backend_maps_add_result():
    method = _Method()
    backend = MethodMemoryBackend(backend_id="memory-local", method=method)

    result = backend.add("task-1", "solver", [Message(role="assistant", content="remember this")])

    assert method.add_requests[0].scope == "agent"
    assert method.add_requests[0].scope_id == "agent:solver"
    assert result.status == "updated"
    assert result.state_ref == "method://memory-local/12:agent:solver/v1"
    assert result.metadata["added"] is True


def test_method_memory_backend_binds_dependency_runtimes():
    method = _Method()
    backend = MethodMemoryBackend(backend_id="memory-local", method=method)

    backend.bind_runtimes(llm_runtimes={"llm": object()}, embedding_runtimes={"embed": object()})

    assert method.bound["llm"].keys() == {"llm"}
    assert method.bound["embedding"].keys() == {"embed"}
```

- [ ] **Step 2: Run tests and verify failures**

Run:

```bash
pytest -q tests/test_memory_method_backend.py
```

Expected: FAIL because method contracts do not exist.

- [ ] **Step 3: Add method contracts**

Create `evolab/backends/memory/methods/base.py`:

```python
from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import Field

from evolab.contracts.common import Message, StrictBaseModel
from evolab.contracts.retrieval import MemoryItem


class MemorySearchRequest(StrictBaseModel):
    task_id: str
    role: str
    query: str
    scope: Literal["agent", "task"]
    scope_id: str
    filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int | None = None
    threshold: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryIngestRequest(StrictBaseModel):
    task_id: str
    role: str
    scope: Literal["agent", "task"]
    scope_id: str
    messages: list[Message]
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemorySearchResult(StrictBaseModel):
    items: list[MemoryItem] = Field(default_factory=list)
    state_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryIngestResult(StrictBaseModel):
    status: Literal["updated", "skipped", "failed", "degraded"]
    state_ref: str | None = None
    previous_state_ref: str | None = None
    added_memory_ids: list[str] = Field(default_factory=list)
    skipped_memory_ids: list[str] = Field(default_factory=list)
    linked_memory_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryMethod(Protocol):
    method_name: str

    def search(self, request: MemorySearchRequest) -> MemorySearchResult:
        raise NotImplementedError

    def add(self, request: MemoryIngestRequest) -> MemoryIngestResult:
        raise NotImplementedError

    def instantiate(self, state_ref: str | None) -> "MemoryMethod":
        raise NotImplementedError
```

- [ ] **Step 4: Add method backend adapter**

Create `evolab/backends/memory/method_backend.py`:

```python
from __future__ import annotations

from typing import Any, Literal

from evolab.backends.memory.base import MemoryBackend
from evolab.backends.memory.methods.base import MemoryIngestRequest, MemoryMethod, MemorySearchRequest
from evolab.contracts.common import Message
from evolab.contracts.retrieval import MemoryBundle, MemoryUpdateResult, RetrievalRequest


class MethodMemoryBackend(MemoryBackend):
    def __init__(
        self,
        *,
        backend_id: str,
        method: MemoryMethod,
        default_search_top_k: int | None = None,
        default_search_threshold: float | None = None,
    ):
        self.backend_id = backend_id
        self.method = method
        self.default_search_top_k = default_search_top_k
        self.default_search_threshold = default_search_threshold

    def bind_runtimes(self, *, llm_runtimes: dict[str, Any], embedding_runtimes: dict[str, Any]) -> None:
        bind = getattr(self.method, "bind_runtimes", None)
        if bind is not None:
            bind(llm_runtimes=llm_runtimes, embedding_runtimes=embedding_runtimes)

    def search(self, request: RetrievalRequest) -> MemoryBundle:
        scope, scope_id = _scope_context(request.task_id, request.role, request.filters)
        result = self.method.search(
            MemorySearchRequest(
                task_id=request.task_id,
                role=request.role,
                query=request.query,
                scope=scope,
                scope_id=scope_id,
                filters=dict(request.filters),
                top_k=self.default_search_top_k,
                threshold=self.default_search_threshold,
                metadata=dict(request.metadata),
            )
        )
        return MemoryBundle(
            backend_id=self.backend_id,
            items=result.items,
            state_ref=result.state_ref,
            metadata=result.metadata,
        )

    def add(self, task_id: str, role: str, messages: list[Message]) -> MemoryUpdateResult:
        scope, scope_id = _scope_context(task_id, role, None)
        result = self.method.add(
            MemoryIngestRequest(
                task_id=task_id,
                role=role,
                scope=scope,
                scope_id=scope_id,
                messages=messages,
            )
        )
        return MemoryUpdateResult(
            status=result.status,
            state_ref=result.state_ref,
            previous_state_ref=result.previous_state_ref,
            metadata={
                **result.metadata,
                "added_memory_ids": result.added_memory_ids,
                "skipped_memory_ids": result.skipped_memory_ids,
                "linked_memory_ids": result.linked_memory_ids,
                "memory_method": self.method.method_name,
            },
        )

    def instantiate(self, state_ref: str | None) -> "MethodMemoryBackend":
        self.method = self.method.instantiate(state_ref)
        return self


def _scope_context(task_id: str, role: str, filters: dict[str, Any] | None) -> tuple[Literal["agent", "task"], str]:
    filters = filters or {}
    default_scope = "task" if role == "task" else "agent"
    default_scope_id = f"task:{task_id}" if role == "task" else f"agent:{role}"
    scope = filters.get("memory_scope", default_scope)
    scope_id = filters.get("memory_scope_id", default_scope_id)
    if scope not in {"agent", "task"}:
        raise ValueError("memory_scope must be 'agent' or 'task'")
    if not isinstance(scope_id, str) or not scope_id:
        raise ValueError("memory_scope_id must be a non-empty string")
    return scope, scope_id
```

- [ ] **Step 5: Export method backend**

Update `evolab/backends/memory/__init__.py` to include:

```python
from evolab.backends.memory.method_backend import MethodMemoryBackend

__all__ = [
    "FakeMemoryBackend",
    "MemoryBackend",
    "MethodMemoryBackend",
    "NullMemoryBackend",
]
```

Keep any existing exports required by unchanged tests until the cleanup task removes them.

- [ ] **Step 6: Run tests**

Run:

```bash
pytest -q tests/test_memory_method_backend.py
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add evolab/backends/memory/method_backend.py evolab/backends/memory/methods/base.py evolab/backends/memory/__init__.py tests/test_memory_method_backend.py
git commit -m "Add memory method backend adapter"
```

## Task 4: SQLite Memory Store

**Files:**
- Create: `evolab/backends/memory/methods/store.py`
- Test: `tests/test_memory_method_store.py`

- [ ] **Step 1: Write failing SQLite store tests**

Create `tests/test_memory_method_store.py`:

```python
import json
from pathlib import Path

from evolab.backends.memory.methods.store import SQLiteMemoryStore
from evolab.contracts.common import Message


def test_sqlite_store_persists_messages_and_memories_across_instances(tmp_path: Path):
    db_path = tmp_path / "memory.sqlite"
    store = SQLiteMemoryStore(db_path)
    store.save_messages(
        scope="agent",
        scope_id="agent:solver",
        messages=[Message(role="assistant", content="first run")],
        metadata={"run_ref": "run-1"},
    )
    memory_id = store.insert_memory(
        scope="agent",
        scope_id="agent:solver",
        content="User wants exact extraction records preserved",
        embedding=[1.0, 0.0],
        metadata={"attributed_to": "user"},
        linked_memory_ids=[],
        entities=[],
    )

    reopened = SQLiteMemoryStore(db_path)

    assert [item.content for item in reopened.list_memories("agent:solver")] == [
        "User wants exact extraction records preserved"
    ]
    assert reopened.recent_messages("agent:solver", limit=1)[0]["content"] == "first run"
    assert reopened.memory_history(memory_id)[0]["event"] == "ADD"


def test_sqlite_store_enforces_scope_isolation(tmp_path: Path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    store.insert_memory("agent", "agent:solver", "solver memory", [1.0], {}, [], [])
    store.insert_memory("task", "task:task-1", "task memory", [1.0], {}, [], [])

    assert [item.content for item in store.list_memories("agent:solver")] == ["solver memory"]
    assert [item.content for item in store.list_memories("task:task-1")] == ["task memory"]


def test_sqlite_store_audit_log_records_events(tmp_path: Path):
    audit_path = tmp_path / "memory.audit.jsonl"
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite", audit_log_path=audit_path)

    store.insert_memory("agent", "agent:solver", "audit memory", [0.5], {}, [], [])

    lines = audit_path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["event"] == "memory.add"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest -q tests/test_memory_method_store.py
```

Expected: FAIL because `SQLiteMemoryStore` does not exist.

- [ ] **Step 3: Implement SQLite store**

Create `evolab/backends/memory/methods/store.py` with a `SQLiteMemoryStore`
class that implements these public methods exactly:

```python
class SQLiteMemoryStore:
    def __init__(self, path: Path | str, audit_log_path: Path | str | None = None):
        self.path = Path(path)
        self.audit_log_path = Path(audit_log_path) if audit_log_path is not None else None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def save_messages(self, *, scope: str, scope_id: str, messages: list[Message], metadata: dict[str, Any]) -> None:
        created_at = _now()
        rows = [
            (str(uuid.uuid4()), scope, scope_id, message.role, message.content, created_at, _json(metadata))
            for message in messages
        ]
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO message_history(message_id, scope, scope_id, role, content, created_at, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        self._audit("messages.add", {"scope": scope, "scope_id": scope_id, "count": len(rows)})

    def recent_messages(self, scope_id: str, *, limit: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content, created_at, metadata_json FROM message_history WHERE scope_id = ? ORDER BY created_at DESC LIMIT ?",
                (scope_id, limit),
            ).fetchall()
        return [
            {"role": row["role"], "content": row["content"], "created_at": row["created_at"], "metadata": _loads(row["metadata_json"])}
            for row in reversed(rows)
        ]

    def insert_memory(self, scope: str, scope_id: str, content: str, embedding: list[float], metadata: dict[str, Any], linked_memory_ids: list[str], entities: list[dict[str, Any]]) -> str:
        memory_id = str(uuid.uuid4())
        created_at = _now()
        payload = {**metadata, "content_hash": content_hash(content)}
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO memory_records(memory_id, scope, scope_id, content, content_hash, embedding_json, text_lemmatized, attributed_to, created_at, updated_at, metadata_json, deleted_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                (memory_id, scope, scope_id, content, payload["content_hash"], _json(embedding), lemmatize_text(content), metadata.get("attributed_to"), created_at, created_at, _json(payload)),
            )
            conn.execute(
                "INSERT INTO memory_history(history_id, memory_id, event, old_content, new_content, created_at, metadata_json) VALUES (?, ?, ?, NULL, ?, ?, ?)",
                (str(uuid.uuid4()), memory_id, "ADD", content, created_at, _json(payload)),
            )
        self._audit("memory.add", {"memory_id": memory_id, "scope": scope, "scope_id": scope_id})
        return memory_id

    def list_memories(self, scope_id: str, *, include_deleted: bool = False) -> list[MemoryItem]:
        deleted_clause = "" if include_deleted else "AND deleted_at IS NULL"
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT memory_id, content, metadata_json FROM memory_records WHERE scope_id = ? {deleted_clause} ORDER BY created_at ASC",
                (scope_id,),
            ).fetchall()
        return [MemoryItem(memory_id=row["memory_id"], content=row["content"], metadata=_loads(row["metadata_json"])) for row in rows]

    def memory_history(self, memory_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT event, old_content, new_content, created_at, metadata_json FROM memory_history WHERE memory_id = ? ORDER BY created_at ASC",
                (memory_id,),
            ).fetchall()
        return [dict(row) | {"metadata": _loads(row["metadata_json"])} for row in rows]

    def semantic_candidates(self, scope_id: str) -> list[dict[str, Any]]:
        return self._candidate_rows(scope_id)

    def keyword_candidates(self, scope_id: str) -> list[dict[str, Any]]:
        return self._candidate_rows(scope_id)

    def entity_links(self, scope_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT e.entity_text, e.entity_type, l.memory_id FROM memory_entities e JOIN entity_memory_links l ON e.entity_id = l.entity_id WHERE e.scope_id = ?",
                (scope_id,),
            ).fetchall()
        return [dict(row) for row in rows]
```

Use these implementation requirements:

```python
CREATE TABLE IF NOT EXISTS memory_records (
    memory_id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    embedding_json TEXT NOT NULL,
    text_lemmatized TEXT NOT NULL,
    attributed_to TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    deleted_at TEXT
)
```

Also create `message_history`, `memory_history`, `memory_links`, `memory_entities`, and `entity_memory_links` as defined in the spec. Use one transaction for `insert_memory`, write JSON with `ensure_ascii=False`, and compute IDs with `uuid.uuid4()`.

- [ ] **Step 4: Add helper functions in the store file**

Add:

```python
def content_hash(content: str) -> str:
    return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()


def lemmatize_text(text: str) -> str:
    return " ".join(token.lower() for token in re.findall(r"[A-Za-z0-9_]+", text))
```

This is a deterministic local lemmatization baseline inside EvoLab-owned code. It is not a retrieval downgrade because semantic embeddings remain required.

- [ ] **Step 5: Run tests**

Run:

```bash
pytest -q tests/test_memory_method_store.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add evolab/backends/memory/methods/store.py tests/test_memory_method_store.py
git commit -m "Add SQLite memory method store"
```

## Task 5: Native Mem0 Extraction And Add Flow

**Files:**
- Create: `evolab/backends/memory/methods/mem0.py`
- Modify: `evolab/backends/memory/methods/__init__.py`
- Test: `tests/test_native_mem0_method.py`

- [ ] **Step 1: Write failing tests for LLM extraction, duplicate handling, and empty extraction**

Create `tests/test_native_mem0_method.py`:

```python
import json
from pathlib import Path

from evolab.backends.embeddings import FakeEmbeddingBackend
from evolab.backends.memory.methods.mem0 import Mem0MemoryMethod
from evolab.contracts.common import Message
from evolab.contracts.llm import LLMRuntimeResponse, SubAgentAction
from evolab.backends.memory.methods.base import MemoryIngestRequest


class _LLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def generate(self, messages, tool_specs, generation_config):
        self.calls.append({"messages": messages, "generation_config": generation_config})
        return LLMRuntimeResponse(
            action=SubAgentAction(action="final_answer", content=json.dumps(self.payload)),
            raw_response={"ok": True},
        )


def _method(tmp_path: Path, payload):
    embedding = FakeEmbeddingBackend(dimensions=6).instantiate(None)
    method = Mem0MemoryMethod(
        store_path=tmp_path / "mem0.sqlite",
        llm_backend_id="llm",
        embedding_backend_id="embed",
    )
    method.bind_runtimes(llm_runtimes={"llm": _LLM(payload)}, embedding_runtimes={"embed": embedding})
    return method


def test_mem0_add_calls_llm_and_persists_extracted_memory(tmp_path: Path):
    method = _method(tmp_path, {"memory": [{"id": "0", "text": "User needs biology component records preserved", "attributed_to": "user"}]})

    result = method.add(
        MemoryIngestRequest(
            task_id="task-1",
            role="solver",
            scope="agent",
            scope_id="agent:solver",
            messages=[Message(role="user", content="Please preserve biology component records.")],
        )
    )

    assert result.status == "updated"
    assert len(result.added_memory_ids) == 1
    assert method.store.list_memories("agent:solver")[0].content == "User needs biology component records preserved"
    assert method.llm_runtime.calls[0]["generation_config"].response_json_schema["type"] == "object"


def test_mem0_add_skips_duplicate_hash_in_same_scope(tmp_path: Path):
    payload = {"memory": [{"id": "0", "text": "Duplicate fact", "attributed_to": "user"}]}
    method = _method(tmp_path, payload)
    request = MemoryIngestRequest(
        task_id="task-1",
        role="solver",
        scope="agent",
        scope_id="agent:solver",
        messages=[Message(role="user", content="duplicate")],
    )

    first = method.add(request)
    second = method.add(request)

    assert first.status == "updated"
    assert second.status == "skipped"
    assert len(method.store.list_memories("agent:solver")) == 1


def test_mem0_add_empty_extraction_saves_message_history(tmp_path: Path):
    method = _method(tmp_path, {"memory": []})

    result = method.add(
        MemoryIngestRequest(
            task_id="task-1",
            role="solver",
            scope="agent",
            scope_id="agent:solver",
            messages=[Message(role="assistant", content="No durable fact here.")],
        )
    )

    assert result.status == "skipped"
    assert method.store.recent_messages("agent:solver", limit=1)[0]["content"] == "No durable fact here."
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest -q tests/test_native_mem0_method.py
```

Expected: FAIL because `Mem0MemoryMethod` does not exist.

- [ ] **Step 3: Implement native mem0 method constructor and dependency binding**

Create `evolab/backends/memory/methods/mem0.py` with:

```python
class Mem0MemoryMethod:
    method_name = "mem0"

    def __init__(
        self,
        *,
        store_path: Path | str,
        llm_backend_id: str,
        embedding_backend_id: str,
        audit_log_path: Path | str | None = None,
        top_k_existing: int = 10,
    ):
        self.store = SQLiteMemoryStore(store_path, audit_log_path=audit_log_path)
        self.llm_backend_id = llm_backend_id
        self.embedding_backend_id = embedding_backend_id
        self.top_k_existing = top_k_existing
        self.llm_runtime = None
        self.embedding_runtime = None
        self.instantiated_state_refs: list[str | None] = []

    def bind_runtimes(self, *, llm_runtimes: dict[str, Any], embedding_runtimes: dict[str, Any]) -> None:
        self.llm_runtime = llm_runtimes[self.llm_backend_id]
        self.embedding_runtime = embedding_runtimes[self.embedding_backend_id]

    def instantiate(self, state_ref: str | None) -> "Mem0MemoryMethod":
        self.instantiated_state_refs.append(state_ref)
        return self
```

- [ ] **Step 4: Implement LLM extraction**

Add:

```python
MEM0_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "memory": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "attributed_to": {"type": "string"},
                    "linked_memory_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "text", "attributed_to", "linked_memory_ids"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["memory"],
    "additionalProperties": False,
}


def _extraction_messages(request: MemoryIngestRequest, existing: list[MemoryItem], recent_messages: list[dict[str, Any]]) -> list[Message]:
    prompt = {
        "scope": request.scope,
        "scope_id": request.scope_id,
        "existing_memories": [{"id": item.memory_id, "text": item.content} for item in existing],
        "recent_messages": recent_messages,
        "new_messages": [{"role": item.role, "content": item.content} for item in request.messages],
    }
    return [
        Message(role="system", content=MEM0_ADDITIVE_EXTRACTION_PROMPT),
        Message(role="user", content=json.dumps(prompt, ensure_ascii=False)),
    ]
```

`MEM0_ADDITIVE_EXTRACTION_PROMPT` must be a concise EvoLab-owned prompt that captures mem0 v3 requirements: extract every durable fact from user and assistant messages, use existing memories only for deduplication and linking, return JSON only, preserve proper nouns/numbers/dates, and emit `linked_memory_ids`.

- [ ] **Step 5: Implement add flow**

Implement:

```python
def add(self, request: MemoryIngestRequest) -> MemoryIngestResult:
    self._require_dependencies()
    recent = self.store.recent_messages(request.scope_id, limit=10)
    existing = self.store.list_memories(request.scope_id)
    llm_response = self.llm_runtime.generate(
        _extraction_messages(request, existing, recent),
        [],
        LLMGenerationConfig(model="", temperature=0, response_json_schema=MEM0_EXTRACTION_SCHEMA),
    )
    extracted = _parse_extraction(llm_response.action.content or "")
    self.store.save_messages(scope=request.scope, scope_id=request.scope_id, messages=request.messages, metadata=request.metadata)
    if not extracted:
        return MemoryIngestResult(status="skipped", previous_state_ref=self._state_ref(request.scope_id), state_ref=self._state_ref(request.scope_id), metadata={"extracted_count": 0})
    added_ids = []
    skipped_ids = []
    existing_hashes = {item.metadata.get("content_hash") for item in existing}
    batch_hashes: set[str] = set()
    texts_to_embed = []
    for item in extracted:
        digest = content_hash(item["text"])
        if digest in existing_hashes or digest in batch_hashes:
            skipped_ids.append(item["id"])
            continue
        batch_hashes.add(digest)
        texts_to_embed.append(item)
    if not texts_to_embed:
        return MemoryIngestResult(status="skipped", previous_state_ref=self._state_ref(request.scope_id), state_ref=self._state_ref(request.scope_id), skipped_memory_ids=skipped_ids, metadata={"duplicate_count": len(skipped_ids)})
    vectors = self.embedding_runtime.embed([item["text"] for item in texts_to_embed], purpose="add").vectors
    for item, vector in zip(texts_to_embed, vectors):
        memory_id = self.store.insert_memory(
            request.scope,
            request.scope_id,
            item["text"],
            vector,
            {"attributed_to": item.get("attributed_to"), "content_hash": content_hash(item["text"])},
            item.get("linked_memory_ids") or [],
            extract_entities(item["text"]),
        )
        added_ids.append(memory_id)
    return MemoryIngestResult(status="updated", previous_state_ref=None, state_ref=self._state_ref(request.scope_id), added_memory_ids=added_ids, skipped_memory_ids=skipped_ids, metadata={"extracted_count": len(extracted)})
```

Adjust `_state_ref` to include backend/method state through `MethodMemoryBackend` if the backend owns state refs. The tests only require non-empty state refs.

- [ ] **Step 6: Export method package**

Create `evolab/backends/memory/methods/__init__.py`:

```python
from evolab.backends.memory.methods.base import MemoryIngestRequest, MemoryIngestResult, MemoryMethod, MemorySearchRequest, MemorySearchResult
from evolab.backends.memory.methods.mem0 import Mem0MemoryMethod
from evolab.backends.memory.methods.store import SQLiteMemoryStore

__all__ = [
    "Mem0MemoryMethod",
    "MemoryIngestRequest",
    "MemoryIngestResult",
    "MemoryMethod",
    "MemorySearchRequest",
    "MemorySearchResult",
    "MemoryStore",
    "SQLiteMemoryStore",
]
```

- [ ] **Step 7: Run tests**

Run:

```bash
pytest -q tests/test_native_mem0_method.py tests/test_memory_method_store.py tests/test_memory_method_backend.py
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add evolab/backends/memory/methods tests/test_native_mem0_method.py
git commit -m "Add native mem0 memory ingestion"
```

## Task 6: Native Mem0 Hybrid Search

**Files:**
- Create: `evolab/backends/memory/methods/retrieval.py`
- Modify: `evolab/backends/memory/methods/mem0.py`
- Test: `tests/test_native_mem0_retrieval.py`

- [ ] **Step 1: Write failing retrieval tests**

Create `tests/test_native_mem0_retrieval.py`:

```python
from pathlib import Path

from evolab.backends.embeddings import FakeEmbeddingBackend
from evolab.backends.memory.methods.base import MemorySearchRequest
from evolab.backends.memory.methods.mem0 import Mem0MemoryMethod


class _NoLLM:
    def generate(self, messages, tool_specs, generation_config):
        raise AssertionError("search must not call LLM")


def _seeded_method(tmp_path: Path):
    embedding = FakeEmbeddingBackend(dimensions=6).instantiate(None)
    method = Mem0MemoryMethod(
        store_path=tmp_path / "mem0.sqlite",
        llm_backend_id="llm",
        embedding_backend_id="embed",
    )
    method.bind_runtimes(llm_runtimes={"llm": _NoLLM()}, embedding_runtimes={"embed": embedding})
    for text in [
        "Sigma70 promoter records must preserve exact sequence coordinates",
        "Synthetic promoter diffusion records contain generated promoter sequences",
        "Unrelated scheduling note",
    ]:
        vector = embedding.embed([text], purpose="add").vectors[0]
        method.store.insert_memory("task", "task:bio", text, vector, {"attributed_to": "assistant"}, [], [])
    return method


def test_mem0_search_returns_ranked_hybrid_results(tmp_path: Path):
    method = _seeded_method(tmp_path)

    result = method.search(
        MemorySearchRequest(
            task_id="bio",
            role="task",
            scope="task",
            scope_id="task:bio",
            query="promoter sequence coordinates",
            top_k=2,
            threshold=0.0,
        )
    )

    assert len(result.items) == 2
    assert "promoter" in result.items[0].content.lower()
    assert result.items[0].score is not None


def test_mem0_search_enforces_scope_isolation(tmp_path: Path):
    method = _seeded_method(tmp_path)
    vector = method.embedding_runtime.embed(["promoter only other scope"], purpose="add").vectors[0]
    method.store.insert_memory("task", "task:other", "promoter only other scope", vector, {}, [], [])

    result = method.search(
        MemorySearchRequest(
            task_id="bio",
            role="task",
            scope="task",
            scope_id="task:bio",
            query="other scope",
            top_k=10,
            threshold=0.0,
        )
    )

    assert all(item.content != "promoter only other scope" for item in result.items)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest -q tests/test_native_mem0_retrieval.py
```

Expected: FAIL because search is not implemented.

- [ ] **Step 3: Implement retrieval helpers**

Create `evolab/backends/memory/methods/retrieval.py`:

```python
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(0.0, min(1.0, (numerator / (left_norm * right_norm) + 1.0) / 2.0))


def keyword_score(query: str, content: str) -> float:
    query_terms = _terms(query)
    content_terms = Counter(_terms(content))
    if not query_terms or not content_terms:
        return 0.0
    hits = sum(1 for term in query_terms if content_terms[term] > 0)
    return hits / len(set(query_terms))


def fuse_scores(semantic: float, keyword: float, entity_boost: float) -> float:
    max_possible = 1.0
    if keyword > 0:
        max_possible += 1.0
    if entity_boost > 0:
        max_possible += 0.5
    return min((semantic + keyword + entity_boost) / max_possible, 1.0)


def _terms(text: str) -> list[str]:
    return [term.lower() for term in re.findall(r"[A-Za-z0-9_]+", text)]
```

- [ ] **Step 4: Implement entity extraction helper**

In `mem0.py`, add:

```python
def extract_entities(text: str) -> list[dict[str, str]]:
    entities = []
    seen = set()
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*)*\b", text):
        value = match.group(0).strip()
        key = value.lower()
        if key and key not in seen and len(value) > 2:
            seen.add(key)
            entities.append({"entity_text": value, "entity_type": "PROPER"})
    return entities
```

- [ ] **Step 5: Implement `Mem0MemoryMethod.search`**

Add:

```python
def search(self, request: MemorySearchRequest) -> MemorySearchResult:
    self._require_dependencies()
    top_k = request.top_k or 20
    threshold = 0.1 if request.threshold is None else request.threshold
    query_vector = self.embedding_runtime.embed([request.query], purpose="search").vectors[0]
    candidates = self.store.semantic_candidates(request.scope_id)
    ranked = []
    for candidate in candidates:
        semantic = cosine_similarity(query_vector, candidate["embedding"])
        if semantic < threshold:
            continue
        keyword = keyword_score(request.query, candidate["content"])
        entity_boost = self._entity_boost(request.query, candidate["memory_id"], request.scope_id)
        score = fuse_scores(semantic, keyword, entity_boost)
        ranked.append((score, candidate))
    ranked.sort(key=lambda item: item[0], reverse=True)
    items = [
        MemoryItem(
            memory_id=candidate["memory_id"],
            content=candidate["content"],
            score=score,
            metadata=candidate["metadata"],
        )
        for score, candidate in ranked[:top_k]
    ]
    return MemorySearchResult(items=items, state_ref=self._state_ref(request.scope_id), metadata={"memory_method": "mem0"})
```

Implement `_entity_boost` by matching extracted query entity text against stored entity links for the same scope. Cap the returned boost at `0.5`.

- [ ] **Step 6: Run tests**

Run:

```bash
pytest -q tests/test_native_mem0_retrieval.py tests/test_native_mem0_method.py
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add evolab/backends/memory/methods/retrieval.py evolab/backends/memory/methods/mem0.py tests/test_native_mem0_retrieval.py
git commit -m "Add native mem0 hybrid retrieval"
```

## Task 7: CLI Method Memory Config And Cleanup Old Mem0 Adapter

**Files:**
- Modify: `evolab/cli.py`
- Modify: `evolab/backends/memory/__init__.py`
- Delete or reduce: `evolab/backends/memory/mem0.py`
- Modify: `tests/test_cli_clean_run.py`
- Modify: `tests/test_mem0_memory_backend.py`
- Modify: configs using `client: in_memory`

- [ ] **Step 1: Write config builder tests for native mem0**

Replace the existing `test_clean_run_config_builds_mem0_memory_backend_with_in_memory_client` with:

```python
def test_clean_run_config_builds_native_mem0_method_backend(tmp_path: Path):
    config = {
        "backends": {
            "memory": {
                "mem0-agent-memory": {
                    "type": "method",
                    "method": "mem0",
                    "store_path": "registries/memory/mem0-agent.sqlite",
                    "llm_backend": "llm-main",
                    "embedding_backend": "embed-main",
                    "default_search_top_k": 5,
                }
            }
        }
    }

    backends = _build_memory_backends(config, config_dir=tmp_path)

    backend = backends["mem0-agent-memory"]
    assert isinstance(backend, MethodMemoryBackend)
    assert backend.backend_id == "mem0-agent-memory"
    assert backend.method.method_name == "mem0"
    assert backend.default_search_top_k == 5
```

Add a rejection test:

```python
def test_clean_run_config_rejects_in_memory_mem0_client():
    config = {
        "backends": {
            "memory": {
                "mem0-agent-memory": {"type": "mem0", "client": "in_memory"}
            }
        }
    }

    with pytest.raises(ValueError, match="in_memory"):
        _build_memory_backends(config)
```

- [ ] **Step 2: Run tests and verify failures**

Run:

```bash
pytest -q tests/test_cli_clean_run.py::test_clean_run_config_builds_native_mem0_method_backend tests/test_cli_clean_run.py::test_clean_run_config_rejects_in_memory_mem0_client
```

Expected: FAIL until CLI builder supports method config and rejects old in-memory Mem0.

- [ ] **Step 3: Modify memory builder**

In `_build_memory_backends`, support:

```python
if backend_type == "method":
    method_name = payload.get("method")
    if method_name == "mem0":
        backends[backend_id] = _build_native_mem0_memory_backend(
            backend_id=backend_id,
            payload=payload,
            config_dir=config_dir,
        )
        continue
    raise ValueError(f"backend {backend_id!r} has unsupported memory method {method_name!r}")
if backend_type == "mem0":
    implementation = payload.get("implementation", "native")
    if payload.get("client") == "in_memory":
        raise ValueError("Mem0 in_memory client has been removed; use type='method', method='mem0' with fake LLM and fake embedding in tests")
    if implementation == "native":
        backends[backend_id] = _build_native_mem0_memory_backend(
            backend_id=backend_id,
            payload={**payload, "method": "mem0"},
            config_dir=config_dir,
        )
        continue
    raise ValueError(f"backend {backend_id!r} has unsupported Mem0 implementation {implementation!r}")
```

Add:

```python
def _build_native_mem0_memory_backend(*, backend_id: str, payload: dict[str, Any], config_dir: Path | None) -> MethodMemoryBackend:
    llm_backend_id = payload.get("llm_backend") or payload.get("llmBackend")
    embedding_backend_id = payload.get("embedding_backend") or payload.get("embeddingBackend")
    if not isinstance(llm_backend_id, str) or not llm_backend_id:
        raise ValueError(f"backend {backend_id!r} requires llm_backend")
    if not isinstance(embedding_backend_id, str) or not embedding_backend_id:
        raise ValueError(f"backend {backend_id!r} requires embedding_backend")
    store_path = _resolve_lab_relative_path(payload.get("store_path"), config_dir, "store_path", backend_id)
    audit_log_path = _resolve_optional_lab_relative_path(payload.get("audit_log_path"), config_dir, "audit_log_path", backend_id)
    return MethodMemoryBackend(
        backend_id=backend_id,
        method=Mem0MemoryMethod(
            store_path=store_path,
            audit_log_path=audit_log_path,
            llm_backend_id=llm_backend_id,
            embedding_backend_id=embedding_backend_id,
        ),
        default_search_top_k=payload.get("default_search_top_k"),
        default_search_threshold=payload.get("default_search_threshold"),
    )
```

Add path helpers in `evolab/cli.py`:

```python
def _resolve_lab_relative_path(value: Any, config_dir: Path | None, key: str, backend_id: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"backend {backend_id!r} requires non-empty {key}")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    root = Path(config_dir or Path.cwd())
    return (root / path).resolve()


def _resolve_optional_lab_relative_path(value: Any, config_dir: Path | None, key: str, backend_id: str) -> Path | None:
    if value is None:
        return None
    return _resolve_lab_relative_path(value, config_dir, key, backend_id)
```

- [ ] **Step 4: Clean old exports**

Remove `InMemoryMem0Client`, `Mem0MemoryBackend`, and `Mem0MemoryConfig` from production `__all__`. Delete `evolab/backends/memory/mem0.py` and replace old adapter tests with native method tests from this plan.

- [ ] **Step 5: Migrate configs**

For CI demos that should avoid real API calls, use:

```yaml
backends:
  llm:
    fake-llm:
      type: fake
      responses:
        - action:
            action: final_answer
            content: '{"memory": []}'
  embedding:
    fake-embedding:
      type: fake
      dimensions: 8
  memory:
    mem0-agent-memory:
      type: method
      method: mem0
      store_path: registries/memory/mem0-agent.sqlite
      llm_backend: fake-llm
      embedding_backend: fake-embedding
```

For biology configs using OpenRouter chat, add a real embedding backend before enabling native mem0. If a real embedding provider is not configured for the ablation, use `type: null` for the no-memory condition and leave native mem0 config disabled until credentials are provided.

- [ ] **Step 6: Run targeted tests**

Run:

```bash
pytest -q tests/test_cli_clean_run.py tests/test_memory_method_backend.py tests/test_native_mem0_method.py tests/test_native_mem0_retrieval.py
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add evolab/cli.py evolab/backends/memory configs tests
git commit -m "Wire native mem0 memory config"
```

## Task 8: Runtime Integration, Replay, Docs, And Validation

**Files:**
- Modify: `docs/memory.md`
- Modify: `docs/implementation_plan.md` if it still describes old Mem0 adapter behavior.
- Modify: `tests/test_memory_replay.py`
- Modify: `tests/test_task_worker.py`
- Modify: `tests/test_cli_clean_run.py`

- [ ] **Step 1: Add clean-run integration test using native mem0 method**

Update `test_clean_run_demo_v1_records_mem0_memory_lineage` or add a new test:

```python
def test_clean_run_demo_v1_records_native_mem0_memory_lineage(tmp_path: Path):
    lab_root = tmp_path / "demo-v1-native-mem0"

    result = run_clean_demo(Path("configs/demo_v1_ci.yaml"), lab_root)

    assert result["task_id"] == "demo-v1"
    trajectory_registry = FileTrajectoryRegistry(lab_root / "registries" / "trajectory")
    saved = trajectory_registry.list_subagent_runs()[0]
    assert saved.metadata["agent_memory_update_result"]["metadata"]["memory_method"] == "mem0"
    assert saved.metadata["task_memory_update_result"]["metadata"]["memory_method"] == "mem0"
    assert (lab_root / "registries" / "memory" / "mem0-agent.sqlite").exists()
```

- [ ] **Step 2: Run integration test and verify failure**

Run:

```bash
pytest -q tests/test_cli_clean_run.py::test_clean_run_demo_v1_records_native_mem0_memory_lineage
```

Expected: FAIL until `configs/demo_v1_ci.yaml` and memory update metadata are migrated.

- [ ] **Step 3: Update docs**

Rewrite the Mem0-compatible sections in `docs/memory.md`:

```markdown
## Memory Methods

Runtime calls `MemoryBackend`, but algorithmic behavior lives in
`MemoryMethod`. Native `mem0` is the first production method. It uses local
SQLite persistence, EvoLab LLM runtimes for ADD-only extraction, and EvoLab
embedding runtimes for semantic retrieval.

The old `client: in_memory` Mem0 path has been removed from production configs.
Use `FakeMemoryBackend` only for tests that are not validating mem0 behavior,
and use `NullMemoryBackend` for ablations where memory is intentionally off.
```

- [ ] **Step 4: Update replay tests if state refs changed**

Native mem0 state refs use `method://<backend_id>/<len>:<encoded_scope_id>/v<N>`. Update replay expectations to accept this scheme while keeping parent state refs and scope metadata checks unchanged.

- [ ] **Step 5: Run full memory-focused suite**

Run:

```bash
pytest -q tests/test_embedding_backends.py tests/test_memory_method_backend.py tests/test_memory_method_store.py tests/test_native_mem0_method.py tests/test_native_mem0_retrieval.py tests/test_memory_replay.py tests/test_task_worker.py tests/test_cli_clean_run.py
```

Expected: PASS.

- [ ] **Step 6: Run broader regression tests likely affected by CLI/runtime changes**

Run:

```bash
pytest -q tests/test_api_llm_backend.py tests/test_runtime_contracts.py tests/test_registries.py tests/test_imports.py
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add docs configs tests evolab
git commit -m "Document native mem0 memory methods"
```

## Task 9: Real API Smoke And Ablation Readiness

**Files:**
- Create: `scripts/smoke_native_mem0.py` if the repo has a scripts convention; otherwise use a documented command in `docs/memory.md`.
- Modify: `docs/memory.md`

- [ ] **Step 1: Add a real API smoke command to docs**

Add a section to `docs/memory.md`:

```markdown
## Native Mem0 Real API Smoke

Use this only when `.env` has both a chat LLM endpoint and an embedding endpoint:

```bash
python3 -m evolab.cli clean-run configs/demo_v1_native_mem0_api.yaml --lab-root /tmp/evolab-native-mem0-smoke
```

Expected checks:

- `registries/memory/*.sqlite` exists.
- subagent metadata includes `memory_method="mem0"`.
- memory update result is `updated` or `skipped` with extractor diagnostics.
- no `client: in_memory` config appears in the copied lab config.
```

- [ ] **Step 2: Run static config scan**

Run:

```bash
rg -n "client:\\s*in_memory|InMemoryMem0Client|Mem0MemoryBackend|Mem0MemoryConfig" evolab configs docs tests
```

Expected: no production references. Test references are allowed only if they are explicitly testing a retained external adapter.

- [ ] **Step 3: Run real API smoke when credentials are present**

Check:

```bash
python3 - <<'PY'
from pathlib import Path
from evolab.config.env import parse_dotenv
values = {k.upper(): v for k, v in parse_dotenv(Path(".env")).items()}
print("has_chat_key", bool(values.get("OPENROUTER_API_KEY") or values.get("OPENAI_API_KEY")))
print("has_embedding_key", bool(values.get("OPENAI_API_KEY")))
PY
```

If keys are present, run the documented clean-run smoke. If keys are absent, record that the real API smoke was skipped because required credentials were not configured.

- [ ] **Step 4: Commit docs or smoke script**

```bash
git add docs scripts configs
git commit -m "Add native mem0 smoke guidance"
```

## Final Validation

- [ ] **Step 1: Run targeted memory validation**

```bash
pytest -q tests/test_embedding_backends.py tests/test_memory_method_backend.py tests/test_memory_method_store.py tests/test_native_mem0_method.py tests/test_native_mem0_retrieval.py tests/test_memory_replay.py tests/test_task_worker.py tests/test_cli_clean_run.py
```

Expected: PASS.

- [ ] **Step 2: Run broader regression validation**

```bash
pytest -q tests/test_api_llm_backend.py tests/test_runtime_contracts.py tests/test_registries.py tests/test_imports.py
```

Expected: PASS.

- [ ] **Step 3: Verify old Mem0 shim is gone from production paths**

```bash
rg -n "client:\\s*in_memory|InMemoryMem0Client|Mem0MemoryBackend|Mem0MemoryConfig" evolab configs docs
```

Expected: no matches.

- [ ] **Step 4: Verify worktree cleanliness**

```bash
git status --short
```

Expected: no output.
