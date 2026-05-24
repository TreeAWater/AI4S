from __future__ import annotations

from evolab.backends.memory.base import MemoryBackend
from evolab.contracts.common import Message
from evolab.contracts.retrieval import MemoryBundle, MemoryUpdateResult, RetrievalRequest


class NullMemoryBackend(MemoryBackend):
    """Memory backend for ablations where retrieval and persistence are disabled."""

    def __init__(self, backend_id: str = "null-memory"):
        self.backend_id = backend_id
        self.instantiated_state_refs: list[str | None] = []

    def search(self, request: RetrievalRequest) -> MemoryBundle:
        scope_context = _memory_scope_context(request.task_id, request.role, request.filters)
        return MemoryBundle(
            backend_id=self.backend_id,
            items=[],
            state_ref=None,
            metadata={
                **scope_context,
                "memory_disabled": True,
            },
        )

    def add(self, task_id: str, role: str, messages: list[Message]) -> MemoryUpdateResult:
        scope_context = _memory_scope_context(task_id, role)
        return MemoryUpdateResult(
            status="skipped",
            state_ref=None,
            previous_state_ref=None,
            metadata={
                **scope_context,
                "memory_disabled": True,
                "message_count": len(messages),
            },
        )

    def instantiate(self, state_ref: str | None) -> NullMemoryBackend:
        self.instantiated_state_refs.append(state_ref)
        return self


def _memory_scope_context(
    task_id: str,
    role: str,
    filters: dict[str, object] | None = None,
) -> dict[str, str]:
    filters = filters or {}
    default_scope = "task" if role == "task" else "agent"
    default_scope_id = f"task:{task_id}" if role == "task" else f"agent:{role}"
    memory_scope = filters.get("memory_scope", default_scope)
    memory_scope_id = filters.get("memory_scope_id", default_scope_id)
    return {
        "memory_scope": str(memory_scope),
        "memory_scope_id": str(memory_scope_id),
    }
