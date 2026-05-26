import pytest

from evolab.contracts.common import ArtifactRef, RuntimePolicy
from evolab.contracts.tools import ToolCall, ToolResult, ToolSpec
from evolab.tools.runtime import ToolRegistry, ToolRuntime


def _spec(name: str, **metadata) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"{name} tool",
        parameters_schema={"type": "object"},
        metadata=metadata,
    )


def test_prepare_filters_allowed_and_registered_tools():
    registry = ToolRegistry()
    registry.register(_spec("lookup"), lambda arguments: "ok")
    registry.register(_spec("write_file"), lambda arguments: "ok")

    bundle = ToolRuntime(registry).prepare(
        required_tools=["lookup", "write_file", "missing"],
        allowed_tools=["lookup", "missing"],
        policy=RuntimePolicy(),
    )

    assert [spec.name for spec in bundle.tool_specs] == ["lookup"]


def test_prepare_deduplicates_required_tools():
    registry = ToolRegistry()
    registry.register(_spec("lookup"), lambda arguments: "ok")

    bundle = ToolRuntime(registry).prepare(
        required_tools=["lookup", "lookup"],
        allowed_tools=["lookup"],
        policy=RuntimePolicy(),
    )

    assert [spec.name for spec in bundle.tool_specs] == ["lookup"]


def test_prepare_validates_runtime_policy():
    registry = ToolRegistry()
    registry.register(_spec("lookup"), lambda arguments: "ok")
    registry.register(_spec("human_review", requires_human=True), lambda arguments: "ok")
    runtime = ToolRuntime(registry)

    no_steps = runtime.prepare(
        required_tools=["lookup"],
        allowed_tools=["lookup"],
        policy=RuntimePolicy(max_tool_steps=0),
    )
    no_human = runtime.prepare(
        required_tools=["lookup", "human_review"],
        allowed_tools=["lookup", "human_review"],
        policy=RuntimePolicy(allow_human_tools=False),
    )

    assert no_steps.tool_specs == []
    assert [spec.name for spec in no_human.tool_specs] == ["lookup"]


def test_runtime_rejects_missing_registry_and_unprepared_tool_calls():
    with pytest.raises(ValueError, match="requires a ToolRegistry"):
        ToolRuntime(None)  # type: ignore[arg-type]

    registry = ToolRegistry()
    registry.register(_spec("lookup"), lambda arguments: "ok")
    registry.register(_spec("write_file"), lambda arguments: "written")
    runtime = ToolRuntime(registry)
    runtime.prepare(
        required_tools=["lookup"],
        allowed_tools=["lookup", "write_file"],
        policy=RuntimePolicy(),
    )

    result = runtime.execute(ToolCall(call_id="call-1", name="write_file"))

    assert result.status == "error"
    assert "not prepared" in result.content
    assert result.metadata["error_type"] == "unprepared_tool"


def test_execute_returns_ok_result():
    registry = ToolRegistry()
    registry.register(_spec("lookup"), lambda arguments: f"found {arguments['query']}")

    result = ToolRuntime(registry).execute(
        ToolCall(call_id="call-1", name="lookup", arguments={"query": "alpha"})
    )

    assert result.call_id == "call-1"
    assert result.status == "ok"
    assert result.content == "found alpha"


def test_execute_preserves_tool_result_handler_artifacts_and_metadata():
    registry = ToolRegistry()
    artifact = ArtifactRef(
        uri="file:///tmp/tool-output.json",
        type="dataset",
        metadata={"role": "tool_output"},
    )
    registry.register(
        _spec("lookup"),
        lambda arguments: ToolResult(
            call_id="handler-local-call",
            status="ok",
            content=f"wrote {arguments['query']}",
            artifact_refs=[artifact],
            metadata={"rows": 3},
        ),
    )

    result = ToolRuntime(registry).execute(
        ToolCall(call_id="call-1", name="lookup", arguments={"query": "alpha"})
    )

    assert result.call_id == "call-1"
    assert result.status == "ok"
    assert result.content == "wrote alpha"
    assert result.artifact_refs == [artifact]
    assert result.metadata == {"rows": 3}


def test_execute_unknown_tool_returns_error_result():
    result = ToolRuntime(ToolRegistry()).execute_tool_name(
        call_id="call-1",
        name="missing",
        arguments={},
    )

    assert result.call_id == "call-1"
    assert result.status == "error"
    assert "not registered" in result.content


def test_register_rejects_duplicate_tool_names():
    registry = ToolRegistry()
    registry.register(_spec("lookup"), lambda arguments: "first")

    with pytest.raises(ValueError, match="already registered"):
        registry.register(_spec("lookup"), lambda arguments: "second")


def test_execute_non_string_handler_output_returns_error_result():
    registry = ToolRegistry()
    registry.register(_spec("lookup"), lambda arguments: 123)  # type: ignore[arg-type]

    result = ToolRuntime(registry).execute_tool_name(
        call_id="call-1",
        name="lookup",
        arguments={},
    )

    assert result.call_id == "call-1"
    assert result.status == "error"
    assert "returned non-string" in result.content


def test_generated_tools_survive_prepare_until_task_reset():
    registry = ToolRegistry()
    registry.register(_spec("read_text"), lambda arguments: "read")
    runtime = ToolRuntime(registry)
    runtime.activate_generated_tool_scope("task-1")
    runtime.register_task_generated_tool(
        _spec("gt_extract", generated_tool=True, task_id="task-1"),
        lambda arguments: ToolResult(
            call_id="local",
            status="ok",
            content="generated",
            metadata={"source": "generated"},
        ),
        provenance={"code_hash": "abc"},
    )

    first = runtime.prepare(
        required_tools=["gt_extract"],
        allowed_tools=["gt_extract"],
        policy=RuntimePolicy(),
    )
    second = runtime.prepare(
        required_tools=["read_text", "gt_extract"],
        allowed_tools=["read_text", "gt_extract"],
        policy=RuntimePolicy(),
    )

    assert [spec.name for spec in first.tool_specs] == ["gt_extract"]
    assert [spec.name for spec in second.tool_specs] == ["read_text", "gt_extract"]

    result = runtime.execute_tool_name("call-1", "gt_extract", {})
    assert result.status == "ok"
    assert result.metadata["generated_tool"]["code_hash"] == "abc"

    runtime.reset_task_generated_tools("task-1")
    after_reset = runtime.prepare(
        required_tools=["gt_extract"],
        allowed_tools=["gt_extract"],
        policy=RuntimePolicy(),
    )
    assert after_reset.tool_specs == []


def test_generated_tool_scope_rejects_interleaved_task_switch():
    runtime = ToolRuntime(ToolRegistry())
    runtime.activate_generated_tool_scope("task-1")
    runtime.register_task_generated_tool(
        _spec("gt_extract", generated_tool=True),
        lambda arguments: "ok",
        provenance={},
    )

    with pytest.raises(ValueError, match="cannot switch generated tool scope"):
        runtime.activate_generated_tool_scope("task-2")


def test_unprepared_generated_tool_call_is_rejected():
    runtime = ToolRuntime(ToolRegistry())
    runtime.activate_generated_tool_scope("task-1")
    runtime.register_task_generated_tool(_spec("gt_extract", generated_tool=True), lambda arguments: "ok")
    runtime.prepare(required_tools=[], allowed_tools=["gt_extract"], policy=RuntimePolicy())

    result = runtime.execute_tool_name("call-1", "gt_extract", {})

    assert result.status == "error"
    assert result.metadata["error_type"] == "unprepared_tool"


def test_mark_tool_prepared_allows_registered_generated_tool_in_active_prepare_scope():
    runtime = ToolRuntime(ToolRegistry())
    runtime.activate_generated_tool_scope("task-1")
    runtime.prepare(required_tools=[], allowed_tools=["gt_extract"], policy=RuntimePolicy())
    runtime.register_task_generated_tool(_spec("gt_extract", generated_tool=True), lambda arguments: "ok")

    blocked = runtime.execute_tool_name("call-1", "gt_extract", {})
    runtime.mark_tool_prepared("gt_extract")
    result = runtime.execute_tool_name("call-2", "gt_extract", {})

    assert blocked.status == "error"
    assert blocked.metadata["error_type"] == "unprepared_tool"
    assert result.status == "ok"
    assert result.content == "ok"


def test_generated_tool_failure_preserves_provenance():
    runtime = ToolRuntime(ToolRegistry())
    runtime.activate_generated_tool_scope("task-1")
    runtime.register_task_generated_tool(
        _spec("gt_extract", generated_tool=True),
        lambda arguments: (_ for _ in ()).throw(RuntimeError("boom")),
        provenance={"code_hash": "abc"},
    )
    runtime.prepare(required_tools=["gt_extract"], allowed_tools=["gt_extract"], policy=RuntimePolicy())

    result = runtime.execute_tool_name("call-1", "gt_extract", {})

    assert result.status == "error"
    assert result.content == "boom"
    assert result.metadata["generated_tool"]["task_id"] == "task-1"
    assert result.metadata["generated_tool"]["code_hash"] == "abc"


def test_registered_tool_execution_does_not_add_generated_provenance_for_builtin_replacement():
    registry = ToolRegistry()
    registry.register(_spec("lookup"), lambda arguments: "builtin")
    runtime = ToolRuntime(registry)
    runtime.activate_generated_tool_scope("task-1")
    runtime.register_task_generated_tool(
        _spec("lookup", generated_tool=True),
        lambda arguments: "generated",
        provenance={"code_hash": "abc"},
        replace_existing=True,
    )
    runtime.prepare(required_tools=["lookup"], allowed_tools=["lookup"], policy=RuntimePolicy())

    generated = runtime.execute_tool_name("call-generated", "lookup", {})
    builtin = runtime.execute_registered_tool_name("call-builtin", "lookup", {})

    assert generated.content == "generated"
    assert generated.metadata["generated_tool"]["code_hash"] == "abc"
    assert builtin.content == "builtin"
    assert "generated_tool" not in builtin.metadata


def test_generated_scope_can_advance_to_next_task_after_reset():
    runtime = ToolRuntime(ToolRegistry())
    runtime.activate_generated_tool_scope("task-1")
    runtime.register_task_generated_tool(_spec("gt_extract", generated_tool=True), lambda arguments: "ok")
    runtime.prepare(required_tools=["gt_extract"], allowed_tools=["gt_extract"], policy=RuntimePolicy())

    runtime.reset_task_generated_tools("task-1")
    runtime.activate_generated_tool_scope("task-2")
    runtime.register_task_generated_tool(_spec("gt_next", generated_tool=True), lambda arguments: "next")
    bundle = runtime.prepare(required_tools=["gt_next"], allowed_tools=["gt_next"], policy=RuntimePolicy())

    assert [spec.name for spec in bundle.tool_specs] == ["gt_next"]


def test_reset_generated_replacement_keeps_prepared_builtin_available():
    registry = ToolRegistry()
    registry.register(_spec("lookup"), lambda arguments: "builtin")
    runtime = ToolRuntime(registry)
    runtime.activate_generated_tool_scope("task-1")
    runtime.register_task_generated_tool(
        _spec("lookup", generated_tool=True),
        lambda arguments: "generated",
        replace_existing=True,
    )
    runtime.prepare(required_tools=["lookup"], allowed_tools=["lookup"], policy=RuntimePolicy())

    assert runtime.execute_tool_name("call-generated", "lookup", {}).content == "generated"

    runtime.reset_task_generated_tools("task-1")
    result = runtime.execute_tool_name("call-builtin", "lookup", {})

    assert result.status == "ok"
    assert result.content == "builtin"
    assert "generated_tool" not in result.metadata
