import os

from evolab.contracts.common import RuntimePolicy
from evolab.contracts.tools import ToolCall
from evolab.tools.files import list_files
from evolab.tools.files import register_file_tools
from evolab.tools.runtime import ToolRegistry, ToolRuntime
from evolab.tools.text import register_text_tools


def test_file_and_text_tools_smoke(tmp_path):
    paper = tmp_path / "paper.md"
    paper.write_text("# Abstract\nAlpha beta.\n\n# Methods\nGamma beta.\n", encoding="utf-8")
    registry = ToolRegistry()
    register_file_tools(registry)
    register_text_tools(registry)
    runtime = ToolRuntime(registry)
    runtime.prepare(
        required_tools=["list_files", "read_text", "inspect_file_metadata", "search_text", "extract_sections"],
        allowed_tools=["list_files", "read_text", "inspect_file_metadata", "search_text", "extract_sections"],
        policy=RuntimePolicy(),
    )

    listed = runtime.execute(ToolCall(call_id="list", name="list_files", arguments={"root": str(tmp_path)}))
    read = runtime.execute(ToolCall(call_id="read", name="read_text", arguments={"path": str(paper), "max_chars": 20}))
    metadata = runtime.execute(ToolCall(call_id="meta", name="inspect_file_metadata", arguments={"path": str(paper)}))
    search = runtime.execute(ToolCall(call_id="search", name="search_text", arguments={"path": str(paper), "query": "beta"}))
    sections = runtime.execute(ToolCall(call_id="sections", name="extract_sections", arguments={"path": str(paper)}))

    assert listed.status == "ok"
    assert listed.metadata["files"][0]["relative_path"] == "paper.md"
    assert read.status == "ok"
    assert read.metadata["truncated"] is True
    assert metadata.status == "ok"
    assert metadata.metadata["is_file"] is True
    assert search.status == "ok"
    assert search.metadata["match_count"] == 2
    assert sections.status == "ok"
    assert [section["title"] for section in sections.metadata["sections"]] == ["Abstract", "Methods"]


def test_list_files_double_star_pattern_includes_root_files(tmp_path):
    (tmp_path / "main_text.md").write_text("main", encoding="utf-8")
    nested = tmp_path / "supplementary"
    nested.mkdir()
    (nested / "table.md").write_text("table", encoding="utf-8")

    registry = ToolRegistry()
    register_file_tools(registry)
    runtime = ToolRuntime(registry)
    runtime.prepare(
        required_tools=["list_files"],
        allowed_tools=["list_files"],
        policy=RuntimePolicy(),
    )

    result = runtime.execute(
        ToolCall(
            call_id="list",
            name="list_files",
            arguments={"root": str(tmp_path), "recursive": True, "include_patterns": ["**/*"]},
        )
    )

    assert result.status == "ok"
    assert [item["relative_path"] for item in result.metadata["files"]] == [
        "main_text.md",
        "supplementary/table.md",
    ]


def test_list_files_accepts_absolute_include_patterns_under_root(tmp_path):
    (tmp_path / "main_text.md").write_text("main", encoding="utf-8")
    nested = tmp_path / "supplementary"
    nested.mkdir()
    (nested / "table.md").write_text("table", encoding="utf-8")

    registry = ToolRegistry()
    register_file_tools(registry)
    runtime = ToolRuntime(registry)
    runtime.prepare(required_tools=["list_files"], allowed_tools=["list_files"], policy=RuntimePolicy())

    result = runtime.execute(
        ToolCall(
            call_id="list",
            name="list_files",
            arguments={
                "root": str(tmp_path),
                "recursive": True,
                "include_patterns": [f"{tmp_path}/**/main_text.md", f"{tmp_path}/**/*.md"],
            },
        )
    )

    assert result.status == "ok"
    assert [item["relative_path"] for item in result.metadata["files"]] == [
        "main_text.md",
        "supplementary/table.md",
    ]


def test_list_files_absolute_include_patterns_do_not_scan_unrelated_root(tmp_path, monkeypatch):
    target = tmp_path / "target"
    target.mkdir()
    (target / "main_text.md").write_text("main", encoding="utf-8")
    unrelated_root = tmp_path.parent
    original_rglob = type(tmp_path).rglob

    def blocked_root_rglob(self, pattern):
        if self == unrelated_root:
            raise PermissionError("unrelated root should not be scanned")
        return original_rglob(self, pattern)

    monkeypatch.setattr(type(tmp_path), "rglob", blocked_root_rglob)

    result = list_files(
        {
            "root": str(unrelated_root),
            "recursive": True,
            "include_patterns": [f"{target}/**"],
        }
    )

    assert result.status == "ok"
    assert [item["path"] for item in result.metadata["files"]] == [str(target / "main_text.md")]


def test_list_files_reports_and_skips_unreadable_paths(tmp_path, monkeypatch):
    (tmp_path / "main_text.md").write_text("main", encoding="utf-8")

    def fake_walk(top, topdown=True, onerror=None):
        yield str(top), [], ["main_text.md"]
        if onerror is not None:
            onerror(PermissionError("blocked"))

    monkeypatch.setattr(os, "walk", fake_walk)

    result = list_files({"root": str(tmp_path), "recursive": True})

    assert result.status == "ok"
    assert [item["relative_path"] for item in result.metadata["files"]] == ["main_text.md"]
    assert "blocked" in result.metadata["warnings"][0]
