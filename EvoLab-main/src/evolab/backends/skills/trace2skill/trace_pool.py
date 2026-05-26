from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from evolab.backends.skills.trace2skill.outcome import (
    classify_trace_outcome,
    detect_low_retrieval_coverage,
    detect_missing_tools,
    summarize_failure,
    summarize_success,
)
from evolab.backends.skills.trace2skill.schema import TracePool, TraceRecord


class TracePoolBuilder:
    def build_trace_pool(
        self,
        traces: Iterable[TraceRecord | dict[str, Any]],
        *,
        task_type: str | None = None,
        target_skill_ids: Iterable[str] | None = None,
        max_traces: int | None = None,
    ) -> TracePool:
        normalized = [self.normalize_trace(trace) for trace in traces]
        normalized = self.deduplicate_traces(normalized)
        if task_type:
            normalized = self.filter_by_task_type(normalized, task_type)
        if target_skill_ids:
            for skill_id in target_skill_ids:
                normalized = self.filter_by_skill_id(normalized, skill_id)
        if max_traces:
            normalized = self.sample_traces(normalized, max_traces)
        return _pool(normalized, task_type=task_type, target_skill_ids=list(target_skill_ids or []))

    def build_from_observations(
        self,
        observations: Iterable[Any],
        *,
        task_type: str | None = None,
        target_skill_ids: Iterable[str] | None = None,
        max_traces: int | None = None,
    ) -> TracePool:
        traces = [self._from_observation(observation) for observation in observations]
        return self.build_trace_pool(
            traces,
            task_type=task_type,
            target_skill_ids=target_skill_ids,
            max_traces=max_traces,
        )

    def build_from_run_records(
        self,
        run_records: Iterable[Any],
        *,
        task_type: str | None = None,
        target_skill_ids: Iterable[str] | None = None,
        max_traces: int | None = None,
    ) -> TracePool:
        traces = [self._from_run_record(record) for record in run_records]
        return self.build_trace_pool(
            traces,
            task_type=task_type,
            target_skill_ids=target_skill_ids,
            max_traces=max_traces,
        )

    def build_from_backend_state(
        self,
        state_root: str | Path,
        *,
        task_type: str | None = None,
        target_skill_ids: Iterable[str] | None = None,
        max_traces: int | None = None,
    ) -> TracePool:
        root = Path(state_root)
        observations_path = root / "observations.jsonl"
        observations: list[dict[str, Any]] = []
        if observations_path.exists():
            observations.extend(_read_jsonl(observations_path))
        return self.build_trace_pool(
            [self.normalize_trace({"metadata": observation, **observation}) for observation in observations],
            task_type=task_type,
            target_skill_ids=target_skill_ids,
            max_traces=max_traces,
        )

    def normalize_trace(self, trace: TraceRecord | dict[str, Any]) -> TraceRecord:
        if isinstance(trace, TraceRecord):
            trace = trace.model_dump(mode="json")
        payload = _dict(trace)
        outcome = classify_trace_outcome(payload)
        missing_tools = _dedupe([*_strings(payload.get("missing_tools")), *detect_missing_tools(payload)])
        metadata = _dict(payload.get("metadata"))
        if detect_low_retrieval_coverage(payload):
            metadata["low_retrieval_coverage"] = True
        if "coverage_report" not in metadata and payload.get("coverage_report"):
            metadata["coverage_report"] = payload.get("coverage_report")
        trace_id = payload.get("trace_id")
        if not isinstance(trace_id, str) or not trace_id:
            trace_id = _stable_id(
                "trace",
                [
                    payload.get("task_id"),
                    payload.get("run_record_ref"),
                    payload.get("observation_id"),
                    payload.get("task_summary"),
                    payload.get("selected_skill_ids"),
                ],
            )
        error_summary = payload.get("error_summary")
        if not error_summary and outcome in {"runtime_failure", "evaluation_failure"}:
            error_summary = summarize_failure({**payload, "missing_tools": missing_tools, "metadata": metadata})
        compact_summary = payload.get("compact_execution_summary")
        if not compact_summary and outcome in {"runtime_success", "evaluation_success"}:
            compact_summary = summarize_success(payload)
        return TraceRecord(
            trace_id=trace_id,
            task_id=_optional_str(payload.get("task_id")),
            task_summary=_optional_str(payload.get("task_summary")),
            task_type=_optional_str(payload.get("task_type")),
            target_skill_ids=_strings(payload.get("target_skill_ids")),
            retrieved_skill_ids=_strings(payload.get("retrieved_skill_ids")),
            selected_skill_ids=_strings(payload.get("selected_skill_ids")),
            tools_used=_dedupe(_strings(payload.get("tools_used"))),
            missing_tools=missing_tools,
            artifacts=_list(payload.get("artifacts")),
            final_status=outcome,
            evaluation_metrics=_dict(payload.get("evaluation_metrics")),
            error_summary=_optional_str(error_summary),
            compact_execution_summary=_optional_str(compact_summary),
            observation_id=_optional_str(payload.get("observation_id")),
            run_record_ref=_optional_str(payload.get("run_record_ref")),
            created_at=_optional_str(payload.get("created_at")) or _utc_now(),
            metadata=metadata,
        )

    def deduplicate_traces(self, traces: Iterable[TraceRecord]) -> list[TraceRecord]:
        by_id: dict[str, TraceRecord] = {}
        for trace in traces:
            by_id.setdefault(trace.trace_id, trace)
        return [by_id[trace_id] for trace_id in sorted(by_id)]

    def filter_by_skill_id(self, traces: Iterable[TraceRecord], skill_id: str) -> list[TraceRecord]:
        return [
            trace
            for trace in traces
            if skill_id in trace.target_skill_ids
            or skill_id in trace.selected_skill_ids
            or skill_id in trace.retrieved_skill_ids
        ]

    def filter_by_task_type(self, traces: Iterable[TraceRecord], task_type: str) -> list[TraceRecord]:
        return [trace for trace in traces if trace.task_type == task_type]

    def filter_by_outcome(self, traces: Iterable[TraceRecord], outcome: str) -> list[TraceRecord]:
        return [trace for trace in traces if trace.final_status == outcome]

    def filter_by_time_window(
        self,
        traces: Iterable[TraceRecord],
        *,
        start_at: str | None = None,
        end_at: str | None = None,
    ) -> list[TraceRecord]:
        return [
            trace
            for trace in traces
            if (start_at is None or trace.created_at >= start_at) and (end_at is None or trace.created_at <= end_at)
        ]

    def sample_traces(self, traces: Iterable[TraceRecord], max_traces: int) -> list[TraceRecord]:
        return sorted(traces, key=lambda trace: (trace.created_at, trace.trace_id))[:max_traces]

    def _from_observation(self, observation: Any) -> dict[str, Any]:
        payload = _dict(observation)
        retrieval_request = _dict(payload.get("retrieval_request"))
        skill_bundle = _dict(payload.get("skill_bundle"))
        skills = _list(skill_bundle.get("skills"))
        metadata = _dict(payload.get("metadata"))
        tool_trace = _dict(payload.get("tool_trace"))
        calls = _list(tool_trace.get("calls"))
        bundle_metadata = _dict(skill_bundle.get("metadata"))
        retrieval_trace = _dict(bundle_metadata.get("retrieval_trace"))
        selected_skill_ids = [
            skill.get("skill_id")
            for skill in skills
            if isinstance(skill, dict) and isinstance(skill.get("skill_id"), str)
        ]
        return {
            "trace_id": _stable_id("trace", [payload.get("task_id"), payload.get("run_ref"), retrieval_request.get("query")]),
            "task_id": payload.get("task_id"),
            "task_summary": retrieval_request.get("query") or payload.get("final_answer"),
            "task_type": retrieval_request.get("role") or payload.get("role"),
            "target_skill_ids": _strings(metadata.get("target_skill_ids")),
            "retrieved_skill_ids": _strings(retrieval_trace.get("returned_skill_ids")) or selected_skill_ids,
            "selected_skill_ids": selected_skill_ids,
            "tools_used": _tool_names(calls),
            "missing_tools": _strings(metadata.get("missing_tools")) + _strings(metadata.get("missing_required_tools")),
            "artifacts": _artifact_refs(payload, calls),
            "evaluation_metrics": _dict(metadata.get("evaluation_metrics")),
            "error_summary": _optional_str(metadata.get("failure_reason") or metadata.get("error")),
            "compact_execution_summary": _compact_summary(payload, calls),
            "observation_id": _stable_id("observation", [payload.get("task_id"), payload.get("run_ref")]),
            "run_record_ref": payload.get("run_ref"),
            "created_at": _utc_now(),
            "metadata": {
                "status": metadata.get("status") or metadata.get("run_status"),
                "coverage_report": _coverage_report(skill_bundle),
                "missing_skill_report": bundle_metadata.get("missing_skill_report"),
                "workflow_status": _dict(metadata.get("plan_execution_trace")).get("status"),
                "tool_calls": calls,
                "skill_bundle": skill_bundle,
                "retrieval_request": retrieval_request,
            },
        }

    def _from_run_record(self, record: Any) -> dict[str, Any]:
        payload = _dict(record)
        retrieval_request = _dict(payload.get("retrieval_request"))
        skill_bundle = _dict(payload.get("skill_bundle"))
        metadata = _dict(payload.get("metadata"))
        tool_calls = _list(payload.get("tool_calls")) or _list(_dict(metadata.get("tool_trace")).get("calls"))
        skills = _list(skill_bundle.get("skills"))
        selected_skill_ids = [
            skill.get("skill_id")
            for skill in skills
            if isinstance(skill, dict) and isinstance(skill.get("skill_id"), str)
        ]
        return {
            "trace_id": _stable_id("trace", [payload.get("task_id"), payload.get("run_ref")]),
            "task_id": payload.get("task_id"),
            "task_summary": retrieval_request.get("query") or payload.get("instruction"),
            "task_type": payload.get("role"),
            "target_skill_ids": _strings(metadata.get("target_skill_ids")),
            "retrieved_skill_ids": selected_skill_ids,
            "selected_skill_ids": selected_skill_ids,
            "tools_used": _tool_names(tool_calls),
            "missing_tools": _strings(metadata.get("missing_tools")) + _strings(metadata.get("missing_required_tools")),
            "artifacts": _list(payload.get("artifact_refs")),
            "evaluation_metrics": _dict(metadata.get("evaluation_metrics")),
            "error_summary": _optional_str(metadata.get("failure_reason") or metadata.get("error")),
            "compact_execution_summary": _compact_outputs(payload),
            "observation_id": _optional_str(metadata.get("observation_id")),
            "run_record_ref": payload.get("run_ref"),
            "created_at": str(payload.get("created_at") or _utc_now()),
            "metadata": {
                "status": metadata.get("status"),
                "coverage_report": _coverage_report(skill_bundle),
                "workflow_status": _dict(metadata.get("plan_execution_trace")).get("status"),
                "tool_calls": tool_calls,
                "skill_observation_request": metadata.get("skill_observation_request"),
            },
        }


def _pool(traces: list[TraceRecord], *, task_type: str | None, target_skill_ids: list[str]) -> TracePool:
    success = [trace for trace in traces if trace.final_status in {"runtime_success", "evaluation_success"}]
    failure = [trace for trace in traces if trace.final_status in {"runtime_failure", "evaluation_failure"}]
    mixed = [trace for trace in traces if trace.final_status in {"partial_success", "unknown"}]
    stats = {
        "trace_count": len(traces),
        "success_count": len(success),
        "failure_count": len(failure),
        "mixed_count": len(mixed),
        "missing_tool_count": sum(1 for trace in traces if trace.missing_tools),
        "low_coverage_count": sum(1 for trace in traces if trace.metadata.get("low_retrieval_coverage")),
    }
    pool_id = _stable_id("trace-pool", [trace.trace_id for trace in traces])
    return TracePool(
        pool_id=pool_id,
        traces=traces,
        success_traces=success,
        failure_traces=failure,
        mixed_traces=mixed,
        task_type=task_type,
        target_skill_ids=target_skill_ids,
        stats=stats,
        created_at=_utc_now(),
    )


def _tool_names(calls: list[Any]) -> list[str]:
    names = []
    for call in calls:
        payload = _dict(call)
        tool_call = _dict(payload.get("tool_call"))
        name = tool_call.get("name") or payload.get("tool_name")
        if isinstance(name, str):
            names.append(name)
    return _dedupe(names)


def _artifact_refs(payload: dict[str, Any], calls: list[Any]) -> list[Any]:
    refs = _list(_dict(payload.get("metadata")).get("artifact_refs"))
    for call in calls:
        refs.extend(_list(_dict(_dict(call).get("result")).get("artifact_refs")))
    return refs[:50]


def _coverage_report(skill_bundle: dict[str, Any]) -> dict[str, Any]:
    metadata = _dict(skill_bundle.get("metadata"))
    graph_summary = _dict(metadata.get("graph_context_summary"))
    retrieval_trace = _dict(metadata.get("retrieval_trace"))
    return _dict(graph_summary.get("coverage_report") or retrieval_trace.get("coverage_report"))


def _compact_summary(payload: dict[str, Any], calls: list[Any]) -> str | None:
    final_answer = payload.get("final_answer")
    if isinstance(final_answer, str) and final_answer:
        return final_answer[:500]
    if calls:
        return f"{len(calls)} tool calls recorded."
    return None


def _compact_outputs(payload: dict[str, Any]) -> str | None:
    messages = _list(payload.get("output_messages"))
    content = []
    for message in messages:
        item = _dict(message)
        if isinstance(item.get("content"), str):
            content.append(item["content"])
    return " ".join(content)[:500] if content else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _stable_id(prefix: str, parts: Iterable[Any]) -> str:
    encoded = json.dumps(_json_compatible(list(parts)), sort_keys=True, separators=(",", ":"))
    return f"{prefix}-{sha256(encoded.encode('utf-8')).hexdigest()[:16]}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _json_compatible(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_compatible(item) for item in value]
    if isinstance(value, tuple):
        return [_json_compatible(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)
