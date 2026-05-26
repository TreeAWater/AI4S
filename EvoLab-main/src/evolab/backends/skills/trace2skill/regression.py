from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import Field

from evolab.contracts.common import StrictBaseModel
from evolab.contracts.retrieval import RetrievalRequest


RegressionGateStatus = Literal["pass", "fail_regression", "inconclusive", "skipped_no_benchmark"]
BenchmarkRunStatus = Literal["completed", "failed", "inconclusive"]


class BenchmarkTask(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    task_id: str
    query: str
    expected_skill_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkRunResult(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    status: BenchmarkRunStatus = "completed"
    task_count: int = 0
    metrics: dict[str, float] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkRunner(Protocol):
    def run(
        self,
        tasks: list[BenchmarkTask],
        *,
        snapshot_ref: str,
        graph_backend: Any | None = None,
    ) -> BenchmarkRunResult:
        ...


class ReplayBenchmarkRunner:
    """Small deterministic runner used by tests or offline replay hooks.

    Production callers can provide any object matching BenchmarkRunner. This
    runner deliberately does not claim improvement; it only returns configured
    metrics for explicit snapshot refs.
    """

    def __init__(self, metrics_by_snapshot_ref: dict[str, dict[str, float]] | None = None):
        self.metrics_by_snapshot_ref = metrics_by_snapshot_ref or {}

    def run(
        self,
        tasks: list[BenchmarkTask],
        *,
        snapshot_ref: str,
        graph_backend: Any | None = None,
    ) -> BenchmarkRunResult:
        metrics = self.metrics_by_snapshot_ref.get(snapshot_ref)
        if metrics is None:
            return BenchmarkRunResult(
                status="inconclusive",
                task_count=len(tasks),
                failures=[f"no metrics configured for snapshot {snapshot_ref!r}"],
            )
        return BenchmarkRunResult(status="completed", task_count=len(tasks), metrics=metrics)


class RegressionGateResult(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    status: RegressionGateStatus
    before_metrics: dict[str, float] = Field(default_factory=dict)
    after_metrics: dict[str, float] = Field(default_factory=dict)
    checked_metrics: list[str] = Field(default_factory=list)
    regressions: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_sanity: dict[str, Any] = Field(default_factory=dict)
    reason: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillEvolutionRegressionGate:
    def __init__(
        self,
        *,
        benchmark_runner: BenchmarkRunner | None = None,
        metrics: list[str] | None = None,
        no_regression_threshold: float = 0.0,
        graph_backend: Any | None = None,
    ):
        self.benchmark_runner = benchmark_runner
        self.metrics = metrics or ["accuracy"]
        self.no_regression_threshold = no_regression_threshold
        self.graph_backend = graph_backend

    def evaluate(
        self,
        tasks: list[BenchmarkTask] | None = None,
        *,
        before_snapshot_ref: str,
        after_snapshot_ref: str,
        graph_backend: Any | None = None,
    ) -> RegressionGateResult:
        tasks = tasks or []
        backend = graph_backend if graph_backend is not None else self.graph_backend
        retrieval_sanity = self._retrieval_sanity(tasks, backend)
        if retrieval_sanity.get("status") == "failed":
            return RegressionGateResult(
                status="inconclusive",
                retrieval_sanity=retrieval_sanity,
                reason="retrieval sanity check failed",
            )

        if self.benchmark_runner is None:
            return RegressionGateResult(
                status="skipped_no_benchmark",
                retrieval_sanity=retrieval_sanity,
                reason="no benchmark runner configured",
            )

        before = self.benchmark_runner.run(tasks, snapshot_ref=before_snapshot_ref, graph_backend=backend)
        after = self.benchmark_runner.run(tasks, snapshot_ref=after_snapshot_ref, graph_backend=backend)
        if before.status != "completed" or after.status != "completed":
            return RegressionGateResult(
                status="inconclusive",
                before_metrics=before.metrics,
                after_metrics=after.metrics,
                checked_metrics=list(self.metrics),
                retrieval_sanity=retrieval_sanity,
                reason="benchmark runner did not return completed before/after results",
                metadata={"before": before.model_dump(mode="json"), "after": after.model_dump(mode="json")},
            )

        regressions = []
        for metric in self.metrics:
            before_value = before.metrics.get(metric)
            after_value = after.metrics.get(metric)
            if before_value is None or after_value is None:
                return RegressionGateResult(
                    status="inconclusive",
                    before_metrics=before.metrics,
                    after_metrics=after.metrics,
                    checked_metrics=list(self.metrics),
                    retrieval_sanity=retrieval_sanity,
                    reason=f"missing benchmark metric {metric!r}",
                )
            if after_value + self.no_regression_threshold < before_value:
                regressions.append(
                    {
                        "metric": metric,
                        "before": before_value,
                        "after": after_value,
                        "threshold": self.no_regression_threshold,
                    }
                )

        if regressions:
            return RegressionGateResult(
                status="fail_regression",
                before_metrics=before.metrics,
                after_metrics=after.metrics,
                checked_metrics=list(self.metrics),
                regressions=regressions,
                retrieval_sanity=retrieval_sanity,
                reason="one or more benchmark metrics regressed",
            )
        return RegressionGateResult(
            status="pass",
            before_metrics=before.metrics,
            after_metrics=after.metrics,
            checked_metrics=list(self.metrics),
            retrieval_sanity=retrieval_sanity,
            reason="no configured benchmark metric regressed",
        )

    def _retrieval_sanity(self, tasks: list[BenchmarkTask], graph_backend: Any | None) -> dict[str, Any]:
        if graph_backend is None or not tasks:
            return {"status": "skipped", "reason": "no graph backend or benchmark tasks configured"}
        checked = []
        try:
            for task in tasks:
                bundle = graph_backend.get(
                    RetrievalRequest(task_id=task.task_id, role="regression_gate", query=task.query)
                )
                returned_ids = [skill.skill_id for skill in bundle.skills]
                missing = [skill_id for skill_id in task.expected_skill_ids if skill_id not in returned_ids]
                checked.append(
                    {
                        "task_id": task.task_id,
                        "returned_skill_ids": returned_ids,
                        "missing_expected_skill_ids": missing,
                    }
                )
                if missing:
                    return {"status": "failed", "checked": checked}
        except Exception as exc:  # pragma: no cover - defensive boundary
            return {"status": "failed", "error": str(exc), "checked": checked}
        return {"status": "ok", "checked": checked}


__all__ = [
    "BenchmarkRunResult",
    "BenchmarkRunner",
    "BenchmarkTask",
    "RegressionGateResult",
    "ReplayBenchmarkRunner",
    "SkillEvolutionRegressionGate",
]
