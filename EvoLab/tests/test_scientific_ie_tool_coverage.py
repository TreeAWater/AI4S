import json
from pathlib import Path
from zipfile import ZipFile

import yaml
import pytest

from evolab.contracts.common import RuntimePolicy
from evolab.contracts.retrieval import RetrievalRequest, SkillBundle, SkillRef
from evolab.contracts.tools import ToolCall
from evolab.runtime.skill_retrieval import MissingRequiredToolError, prepare_skill_runtime_context
from evolab.tools.runtime import ToolRegistry, ToolRuntime
from evolab.tools.scientific_ie import register_scientific_ie_tools


SKILL_ROOT = Path("skills/scientific_ie")
HUMAN_TOOLS = {"ask_human", "request_human_review", "notify_human"}


def _required_tools_by_skill() -> dict[str, list[str]]:
    by_skill: dict[str, list[str]] = {}
    for path in sorted(SKILL_ROOT.glob("*/metadata.*")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        by_skill[payload["skill_id"]] = list(payload.get("required_tools", []))
    return by_skill


def _all_required_tools() -> list[str]:
    return sorted({tool for tools in _required_tools_by_skill().values() for tool in tools})


def test_scientific_ie_required_tools_have_specs_and_handlers():
    required_tools = _all_required_tools()
    registry = ToolRegistry()
    register_scientific_ie_tools(registry)

    missing_specs = [tool for tool in required_tools if registry.get_spec(tool) is None]
    missing_handlers = []
    for tool in required_tools:
        try:
            registry.get_handler(tool)
        except ValueError:
            missing_handlers.append(tool)

    assert required_tools == [
        "detect_table_header",
        "extract_sections",
        "inspect_excel_workbook",
        "inspect_file_metadata",
        "inspect_table",
        "json_schema_validate",
        "list_files",
        "normalize_table",
        "profile_table",
        "read_excel_sheet",
        "read_table_slice",
        "read_text",
        "search_text",
        "write_jsonl",
        "write_report",
    ]
    assert missing_specs == []
    assert missing_handlers == []


def test_each_scientific_ie_required_tool_smoke_executes_ok(tmp_path):
    registry = ToolRegistry()
    register_scientific_ie_tools(registry, artifact_root=tmp_path / "artifacts")
    required_tools = _all_required_tools()
    runtime = ToolRuntime(registry)
    runtime.prepare(required_tools=required_tools, allowed_tools=required_tools, policy=RuntimePolicy())
    fixtures = _tool_fixtures(tmp_path)

    results = {
        tool: runtime.execute(ToolCall(call_id=f"call-{tool}", name=tool, arguments=fixtures[tool]))
        for tool in required_tools
    }

    assert {tool: result.status for tool, result in results.items()} == {tool: "ok" for tool in required_tools}
    assert results["write_jsonl"].artifact_refs
    assert results["write_report"].artifact_refs


def test_scientific_ie_tools_resolve_relative_paths_against_base_dir(tmp_path):
    base_dir = tmp_path / "lab"
    input_dir = base_dir / "inputs"
    input_dir.mkdir(parents=True)
    (input_dir / "paper.md").write_text("Alpha beta\n", encoding="utf-8")

    registry = ToolRegistry()
    register_scientific_ie_tools(registry, base_dir=base_dir, artifact_root=base_dir / "artifacts")
    runtime = ToolRuntime(registry)

    result = runtime.execute(
        ToolCall(
            call_id="read",
            name="read_text",
            arguments={"path": "inputs/paper.md"},
        )
    )

    assert result.status == "ok"
    assert result.metadata["path"] == str(input_dir / "paper.md")
    assert result.metadata["text"] == "Alpha beta\n"


def test_tool_runtime_prepare_contains_all_scientific_ie_required_tools():
    registry = ToolRegistry()
    register_scientific_ie_tools(registry)
    required_tools = _all_required_tools()

    bundle = ToolRuntime(registry).prepare(
        required_tools=required_tools,
        allowed_tools=required_tools,
        policy=RuntimePolicy(),
    )

    assert [spec.name for spec in bundle.tool_specs] == required_tools


def test_scientific_ie_tool_specs_describe_required_arguments():
    registry = ToolRegistry()
    register_scientific_ie_tools(registry)

    expected_required = {
        "detect_table_header": set(),
        "extract_sections": set(),
        "inspect_excel_workbook": {"path"},
        "inspect_file_metadata": {"path"},
        "inspect_table": {"path"},
        "json_schema_validate": set(),
        "list_files": {"root"},
        "normalize_table": set(),
        "profile_table": set(),
        "read_excel_sheet": {"path"},
        "read_table_slice": {"path"},
        "read_text": {"path"},
        "search_text": {"query"},
        "write_jsonl": {"records"},
        "write_report": {"content"},
    }

    for tool in _all_required_tools():
        spec = registry.get_spec(tool)
        assert spec is not None
        schema = spec.parameters_schema
        assert schema["type"] == "object", tool
        assert isinstance(schema.get("properties"), dict), tool
        assert schema["properties"], tool
        assert set(schema.get("required", [])) == expected_required[tool], tool


def test_scientific_ie_tool_specs_define_nested_array_items():
    registry = ToolRegistry()
    register_scientific_ie_tools(registry)

    missing_items = []
    for tool in _all_required_tools():
        spec = registry.get_spec(tool)
        assert spec is not None
        missing_items.extend(_array_paths_missing_items(spec.parameters_schema, tool))

    assert missing_items == []


def test_missing_required_tool_still_raises_through_skill_runtime_context():
    class MissingToolSkillBackend:
        def get(self, request: RetrievalRequest) -> SkillBundle:
            return SkillBundle(
                backend_id="skill-local",
                required_tools=["missing_required"],
                skills=[
                    SkillRef(
                        skill_id="skill.test.v1",
                        name="Test Skill",
                        content="Requires missing tool.",
                        required_tools=["missing_required"],
                    )
                ],
            )

    with pytest.raises(MissingRequiredToolError, match="missing_required"):
        prepare_skill_runtime_context(
            retrieval_request=RetrievalRequest(task_id="task-1", role="solver", query="test"),
            skill_backend=MissingToolSkillBackend(),
            tool_runtime=ToolRuntime(ToolRegistry()),
            allowed_tools=["missing_required"],
            policy=RuntimePolicy(),
        )


def test_human_tools_are_optional_and_not_required_by_stable_scientific_ie_skills():
    required_tools = set(_all_required_tools())
    registry = ToolRegistry()
    register_scientific_ie_tools(registry)

    assert required_tools.isdisjoint(HUMAN_TOOLS)
    hidden = ToolRuntime(registry).prepare(
        required_tools=[],
        allowed_tools=sorted(HUMAN_TOOLS),
        optional_tools=sorted(HUMAN_TOOLS),
        policy=RuntimePolicy(allow_human_tools=False),
    )
    exposed = ToolRuntime(registry).prepare(
        required_tools=[],
        allowed_tools=sorted(HUMAN_TOOLS),
        optional_tools=sorted(HUMAN_TOOLS),
        policy=RuntimePolicy(allow_human_tools=True),
    )
    assert hidden.tool_specs == []
    assert [spec.name for spec in exposed.tool_specs] == sorted(HUMAN_TOOLS)


def _array_paths_missing_items(schema: dict, path: str) -> list[str]:
    missing = []
    if schema.get("type") == "array":
        if "items" not in schema:
            missing.append(path)
        else:
            items = schema["items"]
            if isinstance(items, dict):
                missing.extend(_array_paths_missing_items(items, f"{path}.items"))
    properties = schema.get("properties", {})
    if isinstance(properties, dict):
        for name, property_schema in properties.items():
            if isinstance(property_schema, dict):
                missing.extend(_array_paths_missing_items(property_schema, f"{path}.properties.{name}"))
    return missing


def _tool_fixtures(tmp_path: Path) -> dict[str, dict]:
    text_path = tmp_path / "paper.md"
    text_path.write_text("# Abstract\nAlpha beta.\n\n# Results\nGamma beta.\n", encoding="utf-8")
    csv_path = tmp_path / "table.csv"
    csv_path.write_text("name,count,sequence\nalpha,2,ACGTACGT\nbeta,3,GGGGTTTT\n", encoding="utf-8")
    xlsx_path = tmp_path / "workbook.xlsx"
    _write_minimal_xlsx(xlsx_path)
    schema = {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}
    return {
        "detect_table_header": {"rows": [["name", "count"], ["alpha", "2"]]},
        "extract_sections": {"path": str(text_path)},
        "inspect_excel_workbook": {"path": str(xlsx_path)},
        "inspect_file_metadata": {"path": str(text_path)},
        "inspect_table": {"path": str(csv_path)},
        "json_schema_validate": {"schema": schema, "instance": {"name": "alpha"}},
        "list_files": {"root": str(tmp_path), "recursive": False},
        "normalize_table": {"rows": [["Name", " Count "], [" alpha ", "2"]]},
        "profile_table": {"path": str(csv_path)},
        "read_excel_sheet": {"path": str(xlsx_path), "sheet_name": "Sheet1"},
        "read_table_slice": {"path": str(csv_path), "start_row": 1},
        "read_text": {"path": str(text_path), "max_chars": 100},
        "search_text": {"path": str(text_path), "query": "beta"},
        "write_jsonl": {"records": [{"name": "alpha"}], "artifact_name": "records.jsonl"},
        "write_report": {"content": {"ok": True}, "format": "json", "artifact_name": "report.json"},
    }


def _write_minimal_xlsx(path: Path) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "")
        archive.writestr(
            "xl/workbook.xml",
            """<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
                <sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>""",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            """<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
                <Relationship Id="rId1" Target="worksheets/sheet1.xml"
                 Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"/>
                </Relationships>""",
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            """<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
                <sheetData>
                <row r="1"><c r="A1" t="inlineStr"><is><t>name</t></is></c><c r="B1" t="inlineStr"><is><t>count</t></is></c></row>
                <row r="2"><c r="A2" t="inlineStr"><is><t>alpha</t></is></c><c r="B2"><v>2</v></c></row>
                </sheetData></worksheet>""",
        )
