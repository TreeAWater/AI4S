from evolab.backends.memory import MethodMemoryBackend
from evolab.backends.memory.methods.base import MemoryIngestResult, MemoryMethod, MemorySearchResult
from evolab.backends.memory.methods.mem0 import Mem0MemoryMethod
from evolab.contracts.common import Message
from evolab.contracts.embeddings import EmbeddingResponse
from evolab.contracts.retrieval import RetrievalRequest


class _Method:
    method_name = "test-method"

    def __init__(self):
        self.search_requests = []
        self.add_requests = []
        self.instantiated_state_refs = []
        self.bound_backend_ids = []
        self.bound = None

    def bind_backend_id(self, backend_id):
        self.bound_backend_ids.append(backend_id)

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


class _MethodWithoutRuntimeBinding:
    method_name = "test-method-without-runtime-binding"

    def instantiate(self, state_ref):
        return self

    def search(self, request):
        return MemorySearchResult()

    def add(self, request):
        return MemoryIngestResult(status="skipped")


class _MethodWithoutBackendBinding:
    method_name = "test-method-without-backend-binding"

    def instantiate(self, state_ref):
        return self

    def search(self, request):
        return MemorySearchResult()

    def add(self, request):
        return MemoryIngestResult(status="skipped")


class _ReplacingMethod:
    method_name = "replacing-method"

    def __init__(self):
        self.replacement = _Method()

    def instantiate(self, state_ref):
        self.replacement.instantiated_state_refs.append(state_ref)
        return self.replacement

    def search(self, request):
        return MemorySearchResult()

    def add(self, request):
        return MemoryIngestResult(status="skipped")


class _OverridingMetadataMethod:
    method_name = "override-attempt"

    def instantiate(self, state_ref):
        return self

    def search(self, request):
        return MemorySearchResult(
            metadata={
                "memory_scope": "task",
                "memory_scope_id": "task:wrong",
                "memory_method": "wrong-method",
                "method_value": True,
            }
        )

    def add(self, request):
        return MemoryIngestResult(
            status="updated",
            metadata={
                "memory_scope": "task",
                "memory_scope_id": "task:wrong",
                "memory_method": "wrong-method",
                "method_value": True,
            },
        )


class _NoLLM:
    def generate(self, messages, tool_specs, generation_config):
        raise AssertionError("search must not call LLM")


class _StaticEmbeddingRuntime:
    def __init__(self, vectors_by_text):
        self.vectors_by_text = vectors_by_text

    def embed(self, texts, *, purpose):
        return EmbeddingResponse(
            backend_id="static",
            model="static",
            vectors=[self.vectors_by_text[text] for text in texts],
            metadata={"purpose": purpose},
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


def test_method_memory_backend_binds_backend_id_on_construction():
    method = _Method()

    MethodMemoryBackend(backend_id="memory-local", method=method)

    assert method.bound_backend_ids == ["memory-local"]


def test_method_memory_backend_backend_id_binding_is_optional():
    method = _MethodWithoutBackendBinding()

    MethodMemoryBackend(backend_id="memory-local", method=method)

    assert not hasattr(MemoryMethod, "bind_backend_id")


def test_method_memory_backend_runtime_binding_is_optional():
    method = _MethodWithoutRuntimeBinding()
    backend = MethodMemoryBackend(backend_id="memory-local", method=method)

    backend.bind_runtimes(llm_runtimes={"llm": object()}, embedding_runtimes={"embed": object()})

    assert not hasattr(MemoryMethod, "bind_runtimes")


def test_method_memory_backend_rebinds_replacement_method_after_instantiate():
    method = _ReplacingMethod()
    backend = MethodMemoryBackend(backend_id="memory-local", method=method)

    backend.bind_runtimes(llm_runtimes={"llm": object()}, embedding_runtimes={"embed": object()})
    backend.instantiate("method://memory-local/12:agent:solver/v0")

    assert backend.method is method.replacement
    assert method.replacement.bound_backend_ids == ["memory-local"]
    assert method.replacement.bound["llm"].keys() == {"llm"}
    assert method.replacement.bound["embedding"].keys() == {"embed"}


def test_method_memory_backend_search_metadata_uses_canonical_scope():
    backend = MethodMemoryBackend(backend_id="memory-local", method=_OverridingMetadataMethod())

    bundle = backend.search(RetrievalRequest(task_id="task-1", role="solver", query="prior work"))

    assert bundle.metadata["memory_scope"] == "agent"
    assert bundle.metadata["memory_scope_id"] == "agent:solver"
    assert bundle.metadata["memory_method"] == "override-attempt"
    assert bundle.metadata["method_value"] is True


def test_method_memory_backend_add_metadata_uses_canonical_scope():
    backend = MethodMemoryBackend(backend_id="memory-local", method=_OverridingMetadataMethod())

    result = backend.add("task-1", "solver", [Message(role="assistant", content="remember this")])

    assert result.metadata["memory_scope"] == "agent"
    assert result.metadata["memory_scope_id"] == "agent:solver"
    assert result.metadata["memory_method"] == "override-attempt"
    assert result.metadata["method_value"] is True


def test_method_memory_backend_search_accepts_native_mem0_scope_filters(tmp_path):
    embedding = _StaticEmbeddingRuntime(
        {
            "agent scoped memory": [1.0, 0.0],
            "agent scoped": [1.0, 0.0],
        }
    )
    method = Mem0MemoryMethod(
        store_path=tmp_path / "mem0.sqlite",
        llm_backend_id="llm",
        embedding_backend_id="embed",
    )
    backend = MethodMemoryBackend(backend_id="memory-local", method=method)
    backend.bind_runtimes(
        llm_runtimes={"llm": _NoLLM()},
        embedding_runtimes={"embed": embedding},
    )
    method.store.insert_memory(
        "agent",
        "agent:solver",
        "agent scoped memory",
        [1.0, 0.0],
        {},
        [],
        [],
    )

    bundle = backend.search(
        RetrievalRequest(
            task_id="task-1",
            role="solver",
            query="agent scoped",
            filters={"memory_scope": "agent", "memory_scope_id": "agent:solver"},
        )
    )

    assert len(bundle.items) == 1
    assert bundle.metadata["memory_scope"] == "agent"
    assert bundle.metadata["memory_scope_id"] == "agent:solver"


def test_method_memory_backend_binds_native_mem0_backend_id_into_state_refs(tmp_path):
    embedding = _StaticEmbeddingRuntime({"agent scoped memory": [1.0, 0.0], "agent scoped": [1.0, 0.0]})
    method = Mem0MemoryMethod(
        store_path=tmp_path / "mem0.sqlite",
        llm_backend_id="llm",
        embedding_backend_id="embed",
    )
    backend = MethodMemoryBackend(backend_id="memory-local", method=method)
    backend.bind_runtimes(
        llm_runtimes={"llm": _NoLLM()},
        embedding_runtimes={"embed": embedding},
    )
    method.store.insert_memory("agent", "agent:solver", "agent scoped memory", [1.0, 0.0], {}, [], [])

    bundle = backend.search(RetrievalRequest(task_id="task-1", role="solver", query="agent scoped"))

    assert bundle.state_ref == "method://mem0/12:memory-local/5:agent/12:agent:solver/v1"


def test_native_mem0_backends_with_same_scope_emit_distinct_state_refs(tmp_path):
    embedding = _StaticEmbeddingRuntime({"agent scoped memory": [1.0, 0.0], "agent scoped": [1.0, 0.0]})
    first_method = Mem0MemoryMethod(
        store_path=tmp_path / "first.sqlite",
        llm_backend_id="llm",
        embedding_backend_id="embed",
    )
    second_method = Mem0MemoryMethod(
        store_path=tmp_path / "second.sqlite",
        llm_backend_id="llm",
        embedding_backend_id="embed",
    )
    first = MethodMemoryBackend(backend_id="memory-a", method=first_method)
    second = MethodMemoryBackend(backend_id="memory-b", method=second_method)
    for backend, method in [(first, first_method), (second, second_method)]:
        backend.bind_runtimes(
            llm_runtimes={"llm": _NoLLM()},
            embedding_runtimes={"embed": embedding},
        )
        method.store.insert_memory("agent", "agent:solver", "agent scoped memory", [1.0, 0.0], {}, [], [])

    first_ref = first.search(RetrievalRequest(task_id="task-1", role="solver", query="agent scoped")).state_ref
    second_ref = second.search(RetrievalRequest(task_id="task-1", role="solver", query="agent scoped")).state_ref

    assert first_ref == "method://mem0/8:memory-a/5:agent/12:agent:solver/v1"
    assert second_ref == "method://mem0/8:memory-b/5:agent/12:agent:solver/v1"
    assert first_ref != second_ref
