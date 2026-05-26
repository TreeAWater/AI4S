from __future__ import annotations

from typing import Any

from evolab.backends.skills.trace2skill.schema import PatchValidationResult, TracePool


def trace_pool_metrics(pool: TracePool) -> dict[str, Any]:
    return {
        "trace_count": len(pool.traces),
        "success_count": len(pool.success_traces),
        "failure_count": len(pool.failure_traces),
        "mixed_count": len(pool.mixed_traces),
        "missing_tool_count": pool.stats.get("missing_tool_count", 0),
        "low_coverage_count": pool.stats.get("low_coverage_count", 0),
    }


def before_after_metrics(
    *,
    before_graph_hash: str | None,
    after_graph_hash: str | None,
    validation_result: PatchValidationResult | None,
) -> dict[str, Any]:
    return {
        "before_graph_hash": before_graph_hash,
        "after_graph_hash": after_graph_hash,
        "changed_library": before_graph_hash is not None
        and after_graph_hash is not None
        and before_graph_hash != after_graph_hash,
        "valid_patch_count": 0 if validation_result is None else len(validation_result.valid_patches),
        "invalid_patch_count": 0 if validation_result is None else len(validation_result.invalid_patches),
    }
