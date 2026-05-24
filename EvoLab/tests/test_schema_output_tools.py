from pathlib import Path

from evolab.contracts.common import RuntimePolicy
from evolab.contracts.tools import ToolCall
from evolab.tools.output import register_output_tools
from evolab.tools.runtime import ToolRegistry, ToolRuntime
from evolab.tools.schema import register_schema_tools


def test_schema_and_output_tools_smoke(tmp_path):
    registry = ToolRegistry()
    register_schema_tools(registry)
    register_output_tools(registry, artifact_root=tmp_path)
    runtime = ToolRuntime(registry)
    runtime.prepare(
        required_tools=["json_schema_validate", "write_jsonl", "write_report"],
        allowed_tools=["json_schema_validate", "write_jsonl", "write_report"],
        policy=RuntimePolicy(),
    )
    schema = {
        "type": "object",
        "required": ["name", "count"],
        "properties": {"name": {"type": "string"}, "count": {"type": "integer"}},
    }

    validation = runtime.execute(
        ToolCall(
            call_id="schema",
            name="json_schema_validate",
            arguments={"schema": schema, "instance": {"name": "alpha", "count": 2}},
        )
    )
    invalid = runtime.execute(
        ToolCall(
            call_id="schema-bad",
            name="json_schema_validate",
            arguments={"schema": schema, "instance": {"name": "alpha"}},
        )
    )
    jsonl = runtime.execute(
        ToolCall(call_id="jsonl", name="write_jsonl", arguments={"records": [{"name": "alpha", "count": 2}]})
    )
    report = runtime.execute(
        ToolCall(call_id="report", name="write_report", arguments={"content": {"ok": True}, "format": "json"})
    )

    assert validation.status == "ok"
    assert validation.metadata["valid"] is True
    assert invalid.status == "ok"
    assert invalid.metadata["valid"] is False
    assert jsonl.status == "ok"
    assert jsonl.artifact_refs[0].type == "dataset"
    assert Path(jsonl.artifact_refs[0].uri).read_text(encoding="utf-8") == '{"count": 2, "name": "alpha"}\n'
    assert report.status == "ok"
    assert report.artifact_refs[0].type == "log"


def test_output_tools_write_relative_paths_under_artifact_root(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    artifact_root = tmp_path / "artifacts"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    registry = ToolRegistry()
    register_output_tools(registry, artifact_root=artifact_root)
    runtime = ToolRuntime(registry)
    runtime.prepare(required_tools=["write_report"], allowed_tools=["write_report"], policy=RuntimePolicy())

    result = runtime.execute(
        ToolCall(
            call_id="report",
            name="write_report",
            arguments={"path": "biology_component_report.md", "content": "ok"},
        )
    )

    assert result.status == "ok"
    output_path = artifact_root / "biology_component_report.md"
    assert Path(result.artifact_refs[0].uri) == output_path
    assert output_path.read_text(encoding="utf-8") == "ok"
    assert not (cwd / "biology_component_report.md").exists()


def test_write_jsonl_defaults_missing_records_to_explicit_empty_dataset(tmp_path):
    registry = ToolRegistry()
    register_output_tools(registry, artifact_root=tmp_path)
    runtime = ToolRuntime(registry)
    runtime.prepare(required_tools=["write_jsonl"], allowed_tools=["write_jsonl"], policy=RuntimePolicy())

    result = runtime.execute(
        ToolCall(call_id="jsonl", name="write_jsonl", arguments={"artifact_name": "records.jsonl"})
    )

    assert result.status == "ok"
    assert result.metadata["record_count"] == 0
    assert result.metadata["warnings"] == ["records argument missing; wrote explicit empty JSONL dataset"]
    assert Path(result.artifact_refs[0].uri).read_text(encoding="utf-8") == ""


def test_write_jsonl_preserves_existing_non_empty_artifact_when_empty_records_are_requested(tmp_path):
    registry = ToolRegistry()
    register_output_tools(registry, artifact_root=tmp_path)
    runtime = ToolRuntime(registry)
    runtime.prepare(required_tools=["write_jsonl"], allowed_tools=["write_jsonl"], policy=RuntimePolicy())

    first = runtime.execute(
        ToolCall(
            call_id="jsonl-first",
            name="write_jsonl",
            arguments={"artifact_name": "records.jsonl", "records": [{"name": "alpha"}]},
        )
    )
    second = runtime.execute(
        ToolCall(
            call_id="jsonl-second",
            name="write_jsonl",
            arguments={"artifact_name": "records.jsonl", "records": []},
        )
    )

    output_path = Path(first.artifact_refs[0].uri)
    assert second.status == "ok"
    assert second.metadata["record_count"] == 1
    assert second.metadata["warnings"] == [
        "preserved existing non-empty JSONL artifact; empty records would have overwritten prior output"
    ]
    assert output_path.read_text(encoding="utf-8") == '{"name": "alpha"}\n'
