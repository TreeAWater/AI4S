import json
import sqlite3
from pathlib import Path

import pytest

from evolab.backends.embeddings import FakeEmbeddingBackend
from evolab.backends.memory.methods.base import MemoryIngestRequest, MemorySearchRequest
from evolab.backends.memory.methods.mem0 import Mem0MemoryMethod
from evolab.contracts.common import Message
from evolab.contracts.embeddings import EmbeddingResponse
from evolab.contracts.llm import LLMRuntimeResponse, SubAgentAction


class _LLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def generate(self, messages, tool_specs, generation_config):
        self.calls.append({"messages": messages, "generation_config": generation_config})
        content = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        return LLMRuntimeResponse(
            action=SubAgentAction(action="final_answer", content=content),
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


class _ShortEmbeddingRuntime:
    def embed(self, texts, *, purpose):
        return EmbeddingResponse(
            backend_id="short",
            model="short",
            vectors=[[1.0] * 6 for _ in texts[:-1]],
            metadata={"purpose": purpose},
        )


def test_mem0_add_calls_llm_and_persists_extracted_memory(tmp_path: Path):
    method = _method(
        tmp_path,
        {
            "memory": [
                {
                    "id": "0",
                    "text": "User needs biology component records preserved",
                    "attributed_to": "user",
                }
            ]
        },
    )

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
    stored = method.store.list_memories("agent", "agent:solver")[0]
    assert stored.content == "User needs biology component records preserved"
    assert stored.metadata["task_id"] == "task-1"
    assert stored.metadata["role"] == "solver"
    assert stored.metadata["memory_scope"] == "agent"
    assert stored.metadata["memory_scope_id"] == "agent:solver"
    assert stored.metadata["memory_method"] == "mem0"
    assert stored.metadata["source_extraction_id"] == "0"
    assert stored.metadata["attributed_to"] == "user"
    assert "content_hash" in stored.metadata
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
    assert second.status == "updated"
    assert second.previous_state_ref != second.state_ref
    assert second.metadata["stored_count"] == 0
    assert second.metadata["duplicate_count"] > 0
    assert second.skipped_memory_ids == first.added_memory_ids
    assert len(method.store.list_memories("agent", "agent:solver")) == 1


def test_mem0_add_isolates_duplicate_context_and_versions_by_scope(tmp_path: Path):
    payload = {"memory": [{"id": "0", "text": "Shared durable fact", "attributed_to": "user"}]}
    method = _method(tmp_path, payload)

    agent = method.add(
        MemoryIngestRequest(
            task_id="task-1",
            role="solver",
            scope="agent",
            scope_id="shared",
            messages=[Message(role="user", content="agent-only message")],
        )
    )
    method.llm_runtime.payload = {
        "memory": [{"id": "1", "text": "Shared durable fact", "attributed_to": "user"}]
    }
    task = method.add(
        MemoryIngestRequest(
            task_id="task-1",
            role="task",
            scope="task",
            scope_id="shared",
            messages=[Message(role="user", content="task-only message")],
        )
    )

    task_context = json.loads(method.llm_runtime.calls[-1]["messages"][-1].content)
    assert task_context["existing_memories"] == []
    assert task_context["recent_messages"] == []
    assert task.metadata["stored_count"] == 1
    assert task.metadata["duplicate_count"] == 0
    assert method.store.list_memories("agent", "shared")[0].memory_id == agent.added_memory_ids[0]
    assert method.store.list_memories("task", "shared")[0].memory_id == task.added_memory_ids[0]
    assert agent.previous_state_ref != task.previous_state_ref
    assert agent.state_ref != task.state_ref
    assert agent.previous_state_ref.endswith("/v0")
    assert task.previous_state_ref.endswith("/v0")


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

    assert result.status == "updated"
    assert result.previous_state_ref == "method://mem0/4:mem0/5:agent/12:agent:solver/v0"
    assert result.state_ref == "method://mem0/4:mem0/5:agent/12:agent:solver/v1"
    assert result.metadata["extracted_count"] == 0
    assert result.metadata["stored_count"] == 0
    assert result.metadata["duplicate_count"] == 0
    assert method.store.recent_messages("agent", "agent:solver", limit=1)[0]["content"] == "No durable fact here."


def test_mem0_instantiate_restores_scope_version_floor(tmp_path: Path):
    method = _method(tmp_path, {"memory": []})
    method.instantiate("method://mem0/12:agent:solver/v3")

    result = method.add(
        MemoryIngestRequest(
            task_id="task-1",
            role="solver",
            scope="agent",
            scope_id="agent:solver",
            messages=[Message(role="assistant", content="No durable fact here.")],
        )
    )

    assert result.status == "updated"
    assert result.previous_state_ref == "method://mem0/4:mem0/5:agent/12:agent:solver/v3"
    assert result.state_ref == "method://mem0/4:mem0/5:agent/12:agent:solver/v4"


def test_mem0_instantiate_restores_previous_scope_aware_ref_version_floor(tmp_path: Path):
    method = _method(tmp_path, {"memory": []})
    method.instantiate("method://mem0/5:agent/12:agent:solver/v3")

    result = method.add(
        MemoryIngestRequest(
            task_id="task-1",
            role="solver",
            scope="agent",
            scope_id="agent:solver",
            messages=[Message(role="assistant", content="No durable fact here.")],
        )
    )

    assert result.status == "updated"
    assert result.previous_state_ref == "method://mem0/4:mem0/5:agent/12:agent:solver/v3"
    assert result.state_ref == "method://mem0/4:mem0/5:agent/12:agent:solver/v4"


def test_mem0_instantiate_restores_backend_bound_scope_version_floor(tmp_path: Path):
    method = _method(tmp_path, {"memory": []})
    method.instantiate("method://mem0/12:memory-local/5:agent/12:agent:solver/v3")

    result = method.add(
        MemoryIngestRequest(
            task_id="task-1",
            role="solver",
            scope="agent",
            scope_id="agent:solver",
            messages=[Message(role="assistant", content="No durable fact here.")],
        )
    )

    assert result.status == "updated"
    assert result.previous_state_ref == "method://mem0/4:mem0/5:agent/12:agent:solver/v3"
    assert result.state_ref == "method://mem0/4:mem0/5:agent/12:agent:solver/v4"


def test_mem0_reopened_store_derives_state_ref_from_durable_state(tmp_path: Path):
    store_path = tmp_path / "mem0.sqlite"
    embedding = FakeEmbeddingBackend(dimensions=6).instantiate(None)
    method = Mem0MemoryMethod(
        store_path=store_path,
        llm_backend_id="llm",
        embedding_backend_id="embed",
    )
    method.bind_runtimes(
        llm_runtimes={
            "llm": _LLM(
                {
                    "memory": [
                        {"id": "0", "text": "Durable reopened fact", "attributed_to": "user"},
                    ]
                }
            )
        },
        embedding_runtimes={"embed": embedding},
    )
    method.add(
        MemoryIngestRequest(
            task_id="task-1",
            role="solver",
            scope="agent",
            scope_id="agent:solver",
            messages=[Message(role="user", content="remember across reopen")],
        )
    )
    reopened = Mem0MemoryMethod(
        store_path=store_path,
        llm_backend_id="llm",
        embedding_backend_id="embed",
    )
    reopened.bind_runtimes(
        llm_runtimes={"llm": _LLM({"memory": []})},
        embedding_runtimes={"embed": embedding},
    )

    result = reopened.search(
        MemorySearchRequest(
            task_id="task-1",
            role="solver",
            scope="agent",
            scope_id="agent:solver",
            query="durable",
        )
    )

    assert result.state_ref != "method://mem0/4:mem0/5:agent/12:agent:solver/v0"


def test_mem0_add_invalid_llm_json_fails_without_persisting(tmp_path: Path):
    method = _method(tmp_path, "not json")

    result = method.add(
        MemoryIngestRequest(
            task_id="task-1",
            role="solver",
            scope="agent",
            scope_id="agent:solver",
            messages=[Message(role="user", content="remember this")],
        )
    )

    assert result.status == "failed"
    assert "error" in result.metadata
    assert method.store.list_memories("agent", "agent:solver") == []
    assert method.store.recent_messages("agent", "agent:solver", limit=1) == []


def test_mem0_add_embedding_cardinality_failure_does_not_persist(tmp_path: Path):
    method = Mem0MemoryMethod(
        store_path=tmp_path / "mem0.sqlite",
        llm_backend_id="llm",
        embedding_backend_id="embed",
    )
    method.bind_runtimes(
        llm_runtimes={
            "llm": _LLM(
                {
                    "memory": [
                        {"id": "a", "text": "First fact", "attributed_to": "user"},
                        {"id": "b", "text": "Second fact", "attributed_to": "user"},
                    ]
                }
            )
        },
        embedding_runtimes={"embed": _ShortEmbeddingRuntime()},
    )

    result = method.add(
        MemoryIngestRequest(
            task_id="task-1",
            role="solver",
            scope="agent",
            scope_id="agent:solver",
            messages=[Message(role="user", content="remember two facts")],
        )
    )

    assert result.status == "failed"
    assert result.metadata["error_type"] == "embedding_cardinality"
    assert method.store.list_memories("agent", "agent:solver") == []
    assert method.store.recent_messages("agent", "agent:solver", limit=1) == []


def test_mem0_add_requires_bound_runtimes(tmp_path: Path):
    method = Mem0MemoryMethod(
        store_path=tmp_path / "mem0.sqlite",
        llm_backend_id="llm",
        embedding_backend_id="embed",
    )

    with pytest.raises(RuntimeError, match="llm.*embed|embed.*llm"):
        method.add(
            MemoryIngestRequest(
                task_id="task-1",
                role="solver",
                scope="agent",
                scope_id="agent:solver",
                messages=[Message(role="user", content="remember this")],
            )
        )


def test_mem0_add_linked_memory_ids_persist_same_scope_and_skip_invalid_links(tmp_path: Path):
    method = _method(
        tmp_path,
        {"memory": [{"id": "seed", "text": "Seed memory", "attributed_to": "user"}]},
    )
    seed = method.add(
        MemoryIngestRequest(
            task_id="task-1",
            role="solver",
            scope="agent",
            scope_id="agent:solver",
            messages=[Message(role="user", content="seed")],
        )
    )
    same_scope_memory_id = seed.added_memory_ids[0]
    cross_scope_memory_id = method.store.insert_memory(
        "task",
        "task:other",
        "Other task memory",
        [1.0] * 6,
        {},
        [],
        [],
    )
    method.llm_runtime.payload = {
        "memory": [
            {
                "id": "linked",
                "text": "Linked memory",
                "attributed_to": "assistant",
                "linked_memory_ids": [same_scope_memory_id, "missing-memory", cross_scope_memory_id],
            }
        ]
    }

    result = method.add(
        MemoryIngestRequest(
            task_id="task-1",
            role="solver",
            scope="agent",
            scope_id="agent:solver",
            messages=[Message(role="assistant", content="link it")],
        )
    )

    assert result.status == "updated"
    assert result.linked_memory_ids == [same_scope_memory_id, "missing-memory", cross_scope_memory_id]

    stored = method.store.list_memories("agent", "agent:solver")
    linked_item = stored[1]
    assert set(linked_item.metadata["skipped_linked_memory_ids"]) == {
        "missing-memory",
        cross_scope_memory_id,
    }
    with sqlite3.connect(tmp_path / "mem0.sqlite") as conn:
        rows = conn.execute(
            "SELECT target_memory_id FROM memory_links WHERE source_memory_id = ?",
            (linked_item.memory_id,),
        ).fetchall()
    assert rows == [(same_scope_memory_id,)]
