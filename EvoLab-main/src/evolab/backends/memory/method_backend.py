from __future__ import annotations

from typing import Any, Literal

from evolab.backends.memory.base import MemoryBackend
from evolab.backends.memory.methods.base import MemoryIngestRequest, MemoryMethod, MemorySearchRequest
from evolab.contracts.common import Message
from evolab.contracts.retrieval import MemoryBundle, MemoryUpdateResult, RetrievalRequest


class MethodMemoryBackend(MemoryBackend):
    def __init__(
        self,
        backend_id: str,
        method: MemoryMethod,
        default_search_top_k: int | None = None,
        default_search_threshold: float | None = None,
    ):
        self.backend_id = backend_id
        self.method = method
        self.default_search_top_k = default_search_top_k
        self.default_search_threshold = default_search_threshold
        self._llm_runtimes: Any | None = None
        self._embedding_runtimes: Any | None = None
        self._has_bound_runtimes = False
        self._bind_current_method_backend_id()

    def bind_runtimes(self, *, llm_runtimes: Any, embedding_runtimes: Any) -> None:
        self._llm_runtimes = llm_runtimes
        self._embedding_runtimes = embedding_runtimes
        self._has_bound_runtimes = True
        self._bind_current_method()

    def _bind_current_method_backend_id(self) -> None:
        bind_backend_id = getattr(self.method, "bind_backend_id", None)
        if bind_backend_id is not None:
            bind_backend_id(self.backend_id)

    def _bind_current_method(self) -> None:
        if not self._has_bound_runtimes:
            return
        bind_runtimes = getattr(self.method, "bind_runtimes", None)
        if bind_runtimes is not None:
            bind_runtimes(llm_runtimes=self._llm_runtimes, embedding_runtimes=self._embedding_runtimes)

    def search(self, request: RetrievalRequest) -> MemoryBundle:
        scope, scope_id = _scope_context(request.task_id, request.role, request.filters)
        method_result = self.method.search(
            MemorySearchRequest(
                task_id=request.task_id,
                role=request.role,
                query=request.query,
                scope=scope,
                scope_id=scope_id,
                filters=request.filters,
                metadata=request.metadata,
                top_k=self.default_search_top_k,
                threshold=self.default_search_threshold,
            )
        )
        return MemoryBundle(
            backend_id=self.backend_id,
            items=method_result.items,
            state_ref=method_result.state_ref,
            metadata={
                **method_result.metadata,
                "memory_scope": scope,
                "memory_scope_id": scope_id,
                "memory_method": getattr(self.method, "method_name", type(self.method).__name__),
            },
        )

    def add(self, task_id: str, role: str, messages: list[Message]) -> MemoryUpdateResult:
        scope, scope_id = _scope_context(task_id, role)
        method_result = self.method.add(
            MemoryIngestRequest(
                task_id=task_id,
                role=role,
                messages=messages,
                scope=scope,
                scope_id=scope_id,
                metadata={"memory_scope": scope, "memory_scope_id": scope_id},
            )
        )
        return MemoryUpdateResult(
            status=method_result.status,
            state_ref=method_result.state_ref,
            previous_state_ref=method_result.previous_state_ref,
            metadata={
                **method_result.metadata,
                "added_memory_ids": method_result.added_memory_ids,
                "skipped_memory_ids": method_result.skipped_memory_ids,
                "linked_memory_ids": method_result.linked_memory_ids,
                "memory_scope": scope,
                "memory_scope_id": scope_id,
                "memory_method": getattr(self.method, "method_name", type(self.method).__name__),
            },
        )

    def instantiate(self, state_ref: str | None) -> "MethodMemoryBackend":
        instantiate = getattr(self.method, "instantiate", None)
        if instantiate is not None:
            method = instantiate(state_ref)
            if method is not None:
                self.method = method
                self._bind_current_method_backend_id()
                self._bind_current_method()
        return self


def _scope_context(
    task_id: str,
    role: str,
    filters: dict[str, Any] | None = None,
) -> tuple[Literal["agent", "task"], str]:
    filters = filters or {}
    default_scope = "task" if role == "task" else "agent"
    default_scope_id = f"task:{task_id}" if role == "task" else f"agent:{role}"
    memory_scope = filters.get("memory_scope", default_scope)
    memory_scope_id = filters.get("memory_scope_id", default_scope_id)
    if not isinstance(memory_scope, str) or memory_scope not in {"agent", "task"}:
        raise ValueError("memory_scope must be 'agent' or 'task'")
    if not isinstance(memory_scope_id, str) or not memory_scope_id:
        raise ValueError("memory_scope_id must be a non-empty string")
    return memory_scope, memory_scope_id
