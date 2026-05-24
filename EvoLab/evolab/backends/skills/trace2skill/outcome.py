from __future__ import annotations

from typing import Any

from evolab.backends.skills.trace2skill.schema import TraceOutcome, TraceRecord


def classify_trace_outcome(trace: TraceRecord | dict[str, Any]) -> TraceOutcome:
    payload = _dict(trace)
    explicit = payload.get("final_status")
    if isinstance(explicit, str) and explicit in {
        "runtime_success",
        "runtime_failure",
        "evaluation_success",
        "evaluation_failure",
        "partial_success",
        "unknown",
    }:
        return explicit  # type: ignore[return-value]

    metrics = _dict(payload.get("evaluation_metrics"))
    if metrics:
        passed = metrics.get("passed")
        if passed is True:
            return "evaluation_success"
        if passed is False:
            return "evaluation_failure"
        score = metrics.get("score")
        threshold = metrics.get("threshold", 0.8)
        if isinstance(score, int | float):
            return "evaluation_success" if score >= threshold else "evaluation_failure"

    if detect_missing_tools(payload) or payload.get("error_summary"):
        return "runtime_failure"

    workflow_status = _dict(payload.get("metadata")).get("workflow_status")
    if workflow_status == "completed":
        return "runtime_success"
    if workflow_status == "failed":
        return "runtime_failure"
    if workflow_status == "partial":
        return "partial_success"

    status = _first_string(_dict(payload.get("metadata")).get("status"), payload.get("status"))
    if status:
        normalized = status.casefold()
        if normalized in {"success", "succeeded", "completed", "ok"}:
            return "runtime_success"
        if normalized in {"failure", "failed", "error"}:
            return "runtime_failure"
        if normalized in {"partial", "partially_completed"}:
            return "partial_success"

    if payload.get("compact_execution_summary"):
        return "runtime_success"
    return "unknown"


def summarize_failure(trace: TraceRecord | dict[str, Any]) -> str:
    payload = _dict(trace)
    if isinstance(payload.get("error_summary"), str) and payload["error_summary"]:
        return _bounded(payload["error_summary"], 400)
    missing = detect_missing_tools(payload)
    if missing:
        return f"Missing required tools: {', '.join(missing)}"
    if detect_low_retrieval_coverage(payload):
        return "Low retrieval coverage for task request."
    return "Run did not produce a successful completion signal."


def summarize_success(trace: TraceRecord | dict[str, Any]) -> str:
    payload = _dict(trace)
    tools = ", ".join(_strings(payload.get("tools_used")))
    artifacts = len(_list(payload.get("artifacts")))
    summary = payload.get("compact_execution_summary") or "Run completed successfully."
    if tools:
        summary = f"{summary} Tools used: {tools}."
    if artifacts:
        summary = f"{summary} Artifacts produced: {artifacts}."
    return _bounded(str(summary), 500)


def detect_missing_tools(trace: TraceRecord | dict[str, Any]) -> list[str]:
    payload = _dict(trace)
    tools: list[str] = []
    tools.extend(_strings(payload.get("missing_tools")))
    metadata = _dict(payload.get("metadata"))
    tools.extend(_strings(metadata.get("missing_tools")))
    tools.extend(_strings(metadata.get("missing_required_tools")))
    for call in _list(metadata.get("tool_calls")):
        if not isinstance(call, dict):
            continue
        result = _dict(call.get("result"))
        error_type = _dict(result.get("metadata")).get("error_type")
        if error_type in {"missing_required_tool", "missing_required_tools", "unprepared_tool"}:
            name = _dict(call.get("tool_call")).get("name")
            if isinstance(name, str) and name:
                tools.append(name)
            tools.extend(_strings(_dict(result.get("metadata")).get("missing_tools")))
    return _dedupe(tools)


def detect_low_retrieval_coverage(trace: TraceRecord | dict[str, Any]) -> bool:
    payload = _dict(trace)
    if not _strings(payload.get("selected_skill_ids")):
        return True
    metadata = _dict(payload.get("metadata"))
    coverage = _dict(metadata.get("coverage_report"))
    if coverage.get("sufficient") is False:
        return True
    if metadata.get("missing_skill_report"):
        return True
    return False


def _dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _strings(value: Any) -> list[str]:
    return [item for item in _list(value) if isinstance(item, str) and item]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _first_string(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def _bounded(value: str, max_chars: int) -> str:
    return value if len(value) <= max_chars else value[: max_chars - 3] + "..."
