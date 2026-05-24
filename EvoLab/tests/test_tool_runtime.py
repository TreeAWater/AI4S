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
