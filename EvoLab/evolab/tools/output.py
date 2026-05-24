from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evolab.contracts.common import ArtifactRef
from evolab.contracts.tools import ToolResult, ToolSpec
from evolab.tools.runtime import ToolRegistry


def register_output_tools(registry: ToolRegistry, *, artifact_root: str | Path | None = None) -> None:
    root = Path(artifact_root) if artifact_root is not None else Path("artifacts")
    _register_if_missing(registry, _write_jsonl_spec(), lambda arguments: write_jsonl(arguments, artifact_root=root))
    _register_if_missing(registry, _write_report_spec(), lambda arguments: write_report(arguments, artifact_root=root))


def write_jsonl(arguments: dict[str, Any], *, artifact_root: Path | None = None) -> ToolResult:
    used_default_empty_records = "records" not in arguments
    records = arguments.get("records", [])
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError("records must be a list of objects; use records: [] for an explicit empty dataset")
    path = _output_path(arguments, artifact_root or Path("artifacts"), default_name="records.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    warnings = ["records argument missing; wrote explicit empty JSONL dataset"] if used_default_empty_records else []
    if not records and path.exists() and path.stat().st_size > 0:
        existing_count = _count_jsonl_records(path)
        warnings.append(
            "preserved existing non-empty JSONL artifact; empty records would have overwritten prior output"
        )
        artifact = ArtifactRef(
            uri=str(path),
            type="dataset",
            metadata={"record_count": existing_count, "format": "jsonl", "warnings": warnings},
        )
        return ToolResult(
            call_id="write_jsonl",
            status="ok",
            content=f"preserved {existing_count} existing records at {path}",
            artifact_refs=[artifact],
            metadata={"path": str(path), "record_count": existing_count, "warnings": warnings},
        )
    lines = [json.dumps(record, sort_keys=True) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    artifact = ArtifactRef(
        uri=str(path),
        type="dataset",
        metadata={"record_count": len(records), "format": "jsonl", "warnings": warnings},
    )
    return ToolResult(
        call_id="write_jsonl",
        status="ok",
        content=f"wrote {len(records)} records to {path}",
        artifact_refs=[artifact],
        metadata={"path": str(path), "record_count": len(records), "warnings": warnings},
    )


def write_report(arguments: dict[str, Any], *, artifact_root: Path | None = None) -> ToolResult:
    report_format = str(arguments.get("format") or "markdown")
    content = arguments.get("content", "")
    if isinstance(content, dict):
        text = json.dumps(content, indent=2, sort_keys=True)
        if report_format == "markdown":
            text = f"```json\n{text}\n```\n"
    else:
        text = str(content)
    suffix = {"markdown": ".md", "json": ".json", "text": ".txt"}.get(report_format, ".txt")
    path = _output_path(arguments, artifact_root or Path("artifacts"), default_name=f"report{suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    artifact = ArtifactRef(uri=str(path), type="log", metadata={"format": report_format})
    return ToolResult(
        call_id="write_report",
        status="ok",
        content=f"wrote report to {path}",
        artifact_refs=[artifact],
        metadata={"path": str(path), "format": report_format},
    )


def _output_path(arguments: dict[str, Any], artifact_root: Path, *, default_name: str) -> Path:
    raw_path = arguments.get("path")
    if isinstance(raw_path, str) and raw_path:
        requested_path = Path(raw_path).expanduser()
        if requested_path.is_absolute():
            return artifact_root / _safe_filename(requested_path.name)
        return artifact_root / _safe_relative_path(requested_path, default_name=default_name)
    artifact_name = arguments.get("artifact_name")
    name = str(artifact_name) if artifact_name else default_name
    return artifact_root / _safe_filename(name)


def _count_jsonl_records(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return 0


def _safe_filename(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in value)
    return safe or "artifact"


def _safe_relative_path(path: Path, *, default_name: str) -> Path:
    parts = [
        _safe_filename(part)
        for part in path.parts
        if part not in {"", ".", ".."}
    ]
    if not parts:
        return Path(_safe_filename(default_name))
    return Path(*parts)


def _write_jsonl_spec() -> ToolSpec:
    return ToolSpec(
        name="write_jsonl",
        description=(
            "Write deterministic JSONL records and return a dataset artifact reference. "
            "Pass records: [] when the intended output is an explicit empty dataset."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "records": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Records to write as JSONL. Use an empty list for no accepted records.",
                },
                "path": {"type": "string", "description": "Optional output path."},
                "artifact_name": {"type": "string", "description": "Optional artifact filename under artifact_root."},
            },
            "required": ["records"],
        },
    )


def _write_report_spec() -> ToolSpec:
    return ToolSpec(
        name="write_report",
        description="Write a markdown, JSON, or text report and return an artifact reference.",
        parameters_schema={
            "type": "object",
            "properties": {
                "content": {"description": "Report content as text or JSON-compatible data."},
                "format": {
                    "type": "string",
                    "enum": ["markdown", "json", "text"],
                    "description": "Output format. Defaults to markdown.",
                },
                "path": {"type": "string", "description": "Optional output path."},
                "artifact_name": {"type": "string", "description": "Optional artifact filename under artifact_root."},
            },
            "required": ["content"],
        },
    )


def _register_if_missing(registry: ToolRegistry, spec: ToolSpec, handler: Any) -> None:
    if registry.get_spec(spec.name) is None:
        registry.register(spec, handler)
