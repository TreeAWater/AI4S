from evolab.backends.memory import FakeMemoryBackend, MemoryBackend, NullMemoryBackend
from evolab.contracts.common import Message
from evolab.contracts.retrieval import MemoryAddRequest, MemoryBundle, MemoryUpdateResult, RetrievalRequest


def test_fake_backend_is_exported_package_backend_subclass():
    from evolab.backends.memory.fake import FakeMemoryBackend as PackageFakeMemoryBackend

    assert FakeMemoryBackend is PackageFakeMemoryBackend
    assert issubclass(FakeMemoryBackend, MemoryBackend)


def test_null_backend_is_exported_package_backend_subclass():
    from evolab.backends.memory.null import NullMemoryBackend as PackageNullMemoryBackend

    assert NullMemoryBackend is PackageNullMemoryBackend
    assert issubclass(NullMemoryBackend, MemoryBackend)


def test_null_backend_disables_memory_reads_and_writes():
    backend = NullMemoryBackend(backend_id="memory-off")

    bundle = backend.search(
        RetrievalRequest(
            task_id="task-1",
            role="solver",
            query="prior context",
            filters={"memory_scope": "agent", "memory_scope_id": "agent:solver"},
        )
    )
    update = backend.add(
        "task-1",
        "solver",
        [Message(role="assistant", content="do not persist this")],
    )

    assert bundle == MemoryBundle(
        backend_id="memory-off",
        items=[],
        state_ref=None,
        metadata={
            "memory_scope": "agent",
            "memory_scope_id": "agent:solver",
            "memory_disabled": True,
        },
    )
    assert update == MemoryUpdateResult(
        status="skipped",
        state_ref=None,
        previous_state_ref=None,
        metadata={
            "memory_scope": "agent",
            "memory_scope_id": "agent:solver",
            "memory_disabled": True,
            "message_count": 1,
        },
    )


def test_memory_add_contracts_are_lightweight_strict_models():
    request = MemoryAddRequest(
        task_id="task-1",
        role="solver",
        messages=[Message(role="assistant", content="remember this")],
        filters={"memory_scope": "agent", "memory_scope_id": "agent:solver"},
    )
    result = MemoryUpdateResult(
        status="updated",
        previous_state_ref=None,
        state_ref="fake-memory://fake-memory/agent:solver/v1",
        artifact_refs=[],
        metadata={"memory_scope": "agent", "memory_scope_id": "agent:solver"},
    )

    assert request.filters["memory_scope"] == "agent"
    assert request.filters["memory_scope_id"] == "agent:solver"
    assert result.status == "updated"
    assert result.metadata["memory_scope_id"] == "agent:solver"


def test_search_returns_only_items_for_requested_scope():
    backend = FakeMemoryBackend()
    backend.add("task-1", "solver", [Message(role="assistant", content="agent memory")])
    backend.add("task-1", "task", [Message(role="assistant", content="task memory")])

    agent_bundle = backend.search(
        RetrievalRequest(
            task_id="task-1",
            role="solver",
            query="context",
            filters={"memory_scope": "agent", "memory_scope_id": "agent:solver"},
        )
    )
    task_bundle = backend.search(
        RetrievalRequest(
            task_id="task-1",
            role="task",
            query="context",
            filters={"memory_scope": "task", "memory_scope_id": "task:task-1"},
        )
    )

    assert isinstance(agent_bundle, MemoryBundle)
    assert agent_bundle.backend_id == "fake-memory"
    assert agent_bundle.state_ref == "fake-memory://fake-memory/12:agent:solver/v1"
    assert agent_bundle.metadata == {"memory_scope": "agent", "memory_scope_id": "agent:solver"}
    assert [item.content for item in agent_bundle.items] == ["agent memory"]
    assert [item.content for item in task_bundle.items] == ["task memory"]
    assert task_bundle.state_ref == "fake-memory://fake-memory/11:task:task-1/v1"


def test_search_returns_deep_copies_not_mutable_backend_state():
    backend = FakeMemoryBackend()
    backend.add("task-1", "solver", [Message(role="assistant", content="agent memory")])

    first_bundle = backend.search(RetrievalRequest(task_id="task-1", role="solver", query="context"))
    first_bundle.items[0].content = "corrupted"
    first_bundle.items[0].metadata["corrupted"] = True

    second_bundle = backend.search(RetrievalRequest(task_id="task-1", role="solver", query="context"))

    assert second_bundle.items[0].content == "agent memory"
    assert "corrupted" not in second_bundle.items[0].metadata


def test_state_ref_encoding_does_not_collapse_distinct_scope_ids():
    backend = FakeMemoryBackend()
    backend.add("task-1", "a/b", [Message(role="assistant", content="slash")])
    backend.add("task-1", "a b", [Message(role="assistant", content="space")])

    slash_bundle = backend.search(RetrievalRequest(task_id="task-1", role="a/b", query="context"))
    space_bundle = backend.search(RetrievalRequest(task_id="task-1", role="a b", query="context"))
    empty_bundle = backend.search(
        RetrievalRequest(
            task_id="task-1",
            role="solver",
            query="context",
            filters={"memory_scope": "agent", "memory_scope_id": ""},
        )
    )
    underscore_bundle = backend.search(
        RetrievalRequest(
            task_id="task-1",
            role="solver",
            query="context",
            filters={"memory_scope": "agent", "memory_scope_id": "_"},
        )
    )

    assert slash_bundle.state_ref != space_bundle.state_ref
    assert empty_bundle.state_ref != underscore_bundle.state_ref


def test_add_uses_last_non_empty_message_accepts_tool_messages_and_advances_state():
    backend = FakeMemoryBackend(backend_id="test-backend")

    first = backend.add(
        "task-2",
        "solver",
        [
            Message(role="assistant", content="first memory"),
            Message(role="tool", content="tool output", tool_call_id="call-1"),
            Message(role="assistant", content="   "),
        ],
    )
    second = backend.add("task-2", "solver", [Message(role="user", content="second memory")])
    bundle = backend.search(RetrievalRequest(task_id="task-2", role="solver", query="context"))

    assert first == MemoryUpdateResult(
        status="updated",
        previous_state_ref="fake-memory://test-backend/12:agent:solver/v0",
        state_ref="fake-memory://test-backend/12:agent:solver/v1",
        metadata={
            "memory_scope": "agent",
            "memory_scope_id": "agent:solver",
            "added_memory_ids": ["agent:solver:memory-1"],
        },
    )
    assert second.previous_state_ref == "fake-memory://test-backend/12:agent:solver/v1"
    assert second.state_ref == "fake-memory://test-backend/12:agent:solver/v2"
    assert second.metadata["added_memory_ids"] == ["agent:solver:memory-2"]
    assert [item.memory_id for item in bundle.items] == ["agent:solver:memory-1", "agent:solver:memory-2"]
    assert [item.content for item in bundle.items] == ["tool output", "second memory"]


def test_instantiate_records_state_ref_and_returns_backend():
    backend = FakeMemoryBackend()
    state_ref = "fake-memory://fake-memory/12:agent:solver/v1"

    instantiated = backend.instantiate(state_ref)

    assert instantiated is backend
    assert backend.instantiated_state_refs == [state_ref]
