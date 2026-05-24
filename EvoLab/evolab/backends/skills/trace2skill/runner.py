from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any

from evolab.backends.skills.trace2skill.analysts import (
    CoverageAnalyst,
    ErrorAnalyst,
    PatchProposalAnalyst,
    SuccessAnalyst,
)
from evolab.backends.skills.trace2skill.schema import AnalystRunResult, Trace2SkillRunConfig, TracePool, TraceRecord


class ParallelAnalystRunner:
    def __init__(
        self,
        *,
        execution_mode: str = "sequential",
        max_workers: int = 4,
        timeout_seconds: float | None = None,
        lesson_analysts: list[Any] | None = None,
        patch_analyst: Any | None = None,
        llm_extractor: Any | None = None,
    ):
        self.execution_mode = execution_mode
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds
        self.lesson_analysts = lesson_analysts
        self.patch_analyst = patch_analyst
        self.llm_extractor = llm_extractor

    def run(self, pool: TracePool, *, config: Trace2SkillRunConfig) -> AnalystRunResult:
        if config.enable_llm_analysts and self.llm_extractor is not None:
            lessons, patches = self.llm_extractor.extract_lessons_and_patches(pool, config=config)
            return AnalystRunResult(
                lessons=_sort_lessons(lessons),
                patches=_sort_patches(patches),
                failures=[],
                metadata={
                    "execution_mode": "llm",
                    "max_workers": self.max_workers,
                    "llm_audit_events": list(getattr(self.llm_extractor, "audit_events", [])),
                },
            )

        jobs = self._jobs(pool, config)
        if self.execution_mode == "thread" or config.analyst_execution_mode == "thread":
            result = self._run_threaded(jobs, config=config)
        else:
            result = self._run_sequential(jobs)
        patch_analyst = self.patch_analyst or PatchProposalAnalyst()
        patches = patch_analyst.analyze(result.lessons)
        return result.model_copy(
            update={
                "patches": _sort_patches(patches),
                "metadata": {
                    **result.metadata,
                    "execution_mode": self.execution_mode,
                    "max_workers": self.max_workers,
                    "job_count": len(jobs),
                },
            }
        )

    def _jobs(self, pool: TracePool, config: Trace2SkillRunConfig) -> list[tuple[Any, TraceRecord]]:
        if self.lesson_analysts is not None:
            traces = _mode_traces(pool, config.mode)
            return [(analyst, trace) for analyst in self.lesson_analysts for trace in traces]

        jobs: list[tuple[Any, TraceRecord]] = []
        if config.mode in {"error_only", "combined", "mixed", "skill_deepening", "skill_creation_from_scratch"}:
            jobs.extend((ErrorAnalyst(), trace) for trace in pool.failure_traces)
        if config.mode in {"success_only", "combined", "mixed", "skill_deepening"}:
            jobs.extend((SuccessAnalyst(), trace) for trace in pool.success_traces)
        if config.mode in {"combined", "mixed", "skill_creation_from_scratch"}:
            jobs.extend((CoverageAnalyst(), trace) for trace in [*pool.failure_traces, *pool.mixed_traces])
        return jobs

    def _run_sequential(self, jobs: list[tuple[Any, TraceRecord]]) -> AnalystRunResult:
        lessons = []
        failures = []
        for analyst, trace in jobs:
            try:
                lessons.extend(analyst.analyze([trace]))
            except Exception as exc:  # pragma: no cover - exercised through tests
                failures.append(_failure(analyst, trace, exc))
        return AnalystRunResult(lessons=_sort_lessons(lessons), failures=failures)

    def _run_threaded(self, jobs: list[tuple[Any, TraceRecord]], *, config: Trace2SkillRunConfig) -> AnalystRunResult:
        lessons = []
        failures = []
        timeout = self.timeout_seconds if self.timeout_seconds is not None else config.analyst_timeout_seconds
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_jobs = [
                (executor.submit(analyst.analyze, [trace]), analyst, trace)
                for analyst, trace in jobs
            ]
            for future, analyst, trace in future_jobs:
                try:
                    lessons.extend(future.result(timeout=timeout))
                except TimeoutError as exc:
                    failures.append(_failure(analyst, trace, exc, failure_type="timeout"))
                except Exception as exc:  # pragma: no cover - exercised through tests
                    failures.append(_failure(analyst, trace, exc))
        return AnalystRunResult(lessons=_sort_lessons(lessons), failures=failures)


def _mode_traces(pool: TracePool, mode: str) -> list[TraceRecord]:
    if mode == "success_only":
        return list(pool.success_traces)
    if mode == "error_only":
        return list(pool.failure_traces)
    return [*pool.failure_traces, *pool.success_traces, *pool.mixed_traces]


def _failure(analyst: Any, trace: TraceRecord, exc: BaseException, *, failure_type: str = "exception") -> dict[str, Any]:
    return {
        "analyst": getattr(analyst, "name", analyst.__class__.__name__),
        "trace_id": trace.trace_id,
        "failure_type": failure_type,
        "error": str(exc),
    }


def _sort_lessons(lessons):
    by_id = {lesson.lesson_id: lesson for lesson in lessons}
    return [by_id[key] for key in sorted(by_id)]


def _sort_patches(patches):
    by_id = {patch.patch_id: patch for patch in patches}
    return [by_id[key] for key in sorted(by_id)]


__all__ = ["ParallelAnalystRunner"]
