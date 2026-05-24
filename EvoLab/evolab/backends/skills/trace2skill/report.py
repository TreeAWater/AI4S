from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from evolab.backends.skills.trace2skill.evaluation import before_after_metrics
from evolab.backends.skills.trace2skill.schema import Trace2SkillRunResult, TracePool


class Trace2SkillReportWriter:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_run_report(self, result: Trace2SkillRunResult, *, trace_pool: TracePool | None = None) -> dict[str, str]:
        paths: dict[str, str] = {}
        paths["trace2skill_run_summary"] = self._write_json(
            "trace2skill_run_summary.json",
            {
                "run_id": result.run_id,
                "status": result.status,
                "changed_library": result.changed_library,
                "trace_pool_summary": result.trace_pool_summary,
                "lesson_count": len(result.lessons),
                "local_patch_count": len(result.local_patches),
                "consolidated_patch_count": len(result.consolidation_result.consolidated_patches),
                "staged_update_count": len(result.staged_updates),
                "rejected_update_count": len(result.rejected_updates),
                "before_graph_hash": result.before_graph_hash,
                "after_graph_hash": result.after_graph_hash,
                "regression_gate_result": result.regression_gate_result,
                "blocked_by_regression": result.blocked_by_regression,
                "before_metrics": result.before_metrics,
                "after_metrics": result.after_metrics,
                "changed_skill_ids": result.retrieval_change_summary.get("changed_skill_ids", []),
                "retrieval_change_summary": result.retrieval_change_summary,
                "analysis_failures": result.metadata.get("analysis_failures", []),
                "llm_audit_events": result.metadata.get("llm_audit_events", []),
            },
        )
        if trace_pool is not None:
            paths["trace_pool_stats"] = self._write_json("trace_pool_stats.json", trace_pool.stats)
        paths["lessons"] = self._write_jsonl("lessons.jsonl", result.lessons)
        paths["local_patch_proposals"] = self._write_jsonl("local_patch_proposals.jsonl", result.local_patches)
        paths["consolidated_patches"] = self._write_json(
            "consolidated_patches.json",
            result.consolidation_result.model_dump(mode="json"),
        )
        paths["conflict_report"] = self._write_json(
            "conflict_report.json",
            {
                "conflicts": [conflict.model_dump(mode="json") for conflict in result.consolidation_result.conflicts],
                "validation_conflicts": []
                if result.validation_result is None
                else [conflict.model_dump(mode="json") for conflict in result.validation_result.conflicts],
            },
        )
        paths["validation_report"] = self._write_json(
            "validation_report.json",
            {} if result.validation_result is None else result.validation_result.model_dump(mode="json"),
        )
        paths["converted_skill_update_proposals"] = self._write_jsonl(
            "converted_skill_update_proposals.jsonl",
            result.converted_skill_update_proposals,
        )
        paths["applied_transactions"] = self._write_jsonl("applied_transactions.jsonl", result.applied_transactions)
        paths["policy_decisions"] = self._write_jsonl("policy_decisions.jsonl", result.policy_decisions)
        paths["regression_gate_report"] = self._write_json(
            "regression_gate_report.json",
            result.regression_gate_result or {"status": "not_run"},
        )
        paths["before_after_metrics"] = self._write_json(
            "before_after_metrics.json",
            {
                **before_after_metrics(
                    before_graph_hash=result.before_graph_hash,
                    after_graph_hash=result.after_graph_hash,
                    validation_result=result.validation_result,
                ),
                "before_metrics": result.before_metrics,
                "after_metrics": result.after_metrics,
                "regression_gate_result": result.regression_gate_result,
                "retrieval_change_summary": result.retrieval_change_summary,
            },
        )
        paths["trace2skill_audit_report"] = self._write_text("trace2skill_audit_report.md", self._markdown_report(result))
        return paths

    def _write_json(self, filename: str, payload: dict[str, Any]) -> str:
        path = self.output_dir / filename
        path.write_text(json.dumps(_json_compatible(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return str(path)

    def _write_jsonl(self, filename: str, rows: list[Any]) -> str:
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(_json_compatible(row), sort_keys=True) + "\n")
        return str(path)

    def _write_text(self, filename: str, content: str) -> str:
        path = self.output_dir / filename
        path.write_text(content, encoding="utf-8")
        return str(path)

    def _markdown_report(self, result: Trace2SkillRunResult) -> str:
        validation = result.validation_result
        valid_count = 0 if validation is None else len(validation.valid_patches)
        invalid_count = 0 if validation is None else len(validation.invalid_patches)
        return "\n".join(
            [
                "# Trace2Skill Audit Report",
                "",
                f"- run_id: `{result.run_id}`",
                f"- status: `{result.status}`",
                f"- trace_count: `{result.trace_pool_summary.get('trace_count', 0)}`",
                f"- lessons: `{len(result.lessons)}`",
                f"- local_patches: `{len(result.local_patches)}`",
                f"- consolidated_patches: `{len(result.consolidation_result.consolidated_patches)}`",
                f"- valid_patches: `{valid_count}`",
                f"- invalid_patches: `{invalid_count}`",
                f"- policy_decisions: `{len(result.policy_decisions)}`",
                f"- applied_transactions: `{len(result.applied_transactions)}`",
                f"- changed_library: `{result.changed_library}`",
                f"- staged_updates: `{len(result.staged_updates)}`",
                f"- rejected_updates: `{len(result.rejected_updates)}`",
                f"- regression_gate: `{(result.regression_gate_result or {}).get('status', 'not_run')}`",
                f"- blocked_by_regression: `{result.blocked_by_regression}`",
                f"- before_graph_hash: `{result.before_graph_hash}`",
                f"- after_graph_hash: `{result.after_graph_hash}`",
                f"- changed_skill_ids: `{', '.join(result.retrieval_change_summary.get('changed_skill_ids', []))}`",
                "",
                "Stable library mutation is policy-gated. Candidate skills and relationship updates are applied only when explicit deterministic policy flags allow them.",
                "No human review path is invoked by this phase.",
                "",
            ]
        )


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
