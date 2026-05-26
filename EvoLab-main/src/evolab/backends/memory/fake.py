from __future__ import annotations

from urllib.parse import quote

from evolab.backends.memory.base import MemoryBackend
from evolab.contracts.common import Message
from evolab.contracts.retrieval import MemoryBundle, MemoryItem, MemoryUpdateResult, RetrievalRequest


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


def _scope_ref_part(memory_scope_id: str) -> str:
    encoded = quote(memory_scope_id, safe=":")
    return f"{len(memory_scope_id)}:{encoded}"


class FakeMemoryBackend(MemoryBackend):
    def __init__(self, backend_id: str = "fake-memory"):
        self.backend_id = backend_id
        self.records_by_scope_id: dict[str, list[MemoryItem]] = {}
        self.instantiated_state_refs: list[str | None] = []

    def search(self, request: RetrievalRequest) -> MemoryBundle:
        scope_context = _memory_scope_context(request.task_id, request.role, request.filters)
        memory_scope_id = scope_context["memory_scope_id"]
        items = [
            item.model_copy(deep=True)
            for item in self.records_by_scope_id.get(memory_scope_id, [])
        ]
        return MemoryBundle(
            items=items,
            backend_id=self.backend_id,
            state_ref=self._state_ref(memory_scope_id),
            metadata=scope_context,
        )

    def add(self, task_id: str, role: str, messages: list[Message]) -> MemoryUpdateResult:
        scope_context = _memory_scope_context(task_id, role)
        memory_scope_id = scope_context["memory_scope_id"]
        records = self.records_by_scope_id.setdefault(memory_scope_id, [])
        previous_state_ref = self._state_ref(memory_scope_id)
        version = len(records) + 1
        memory_id = f"{memory_scope_id}:memory-{version}"
        item = MemoryItem(
            memory_id=memory_id,
            content=_last_non_empty_content(messages),
            metadata={
                "task_id": task_id,
                "role": role,
                "memory_scope": scope_context["memory_scope"],
                "memory_scope_id": memory_scope_id,
            },
        )
        records.append(item.model_copy(deep=True))
        return MemoryUpdateResult(
            status="updated",
            previous_state_ref=previous_state_ref,
            state_ref=self._state_ref(memory_scope_id),
            metadata={
                **scope_context,
                "added_memory_ids": [memory_id],
            },
        )

    def instantiate(self, state_ref: str | None) -> FakeMemoryBackend:
        self.instantiated_state_refs.append(state_ref)
        return self

    def _state_ref(self, memory_scope_id: str) -> str:
        version = len(self.records_by_scope_id.get(memory_scope_id, []))
        return f"fake-memory://{self.backend_id}/{_scope_ref_part(memory_scope_id)}/v{version}"


def _last_non_empty_content(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message.content.strip():
            return message.content
    return ""
