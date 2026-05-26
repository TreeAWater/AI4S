from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import Any

from evolab.backends.skills.trace2skill.outcome import (
    detect_low_retrieval_coverage,
    detect_missing_tools,
    summarize_failure,
    summarize_success,
)
from evolab.backends.skills.trace2skill.schema import SkillPatchProposal, Trace2SkillRunConfig, TracePool, TraceRecord, TrajectoryLesson


class ErrorAnalyst:
    name = "ErrorAnalyst"

    def __init__(self, *, llm_extractor: Any | None = None, config: Trace2SkillRunConfig | None = None):
        self.llm_extractor = llm_extractor
        self.config = config

    def analyze(self, traces: Iterable[TraceRecord]) -> list[TrajectoryLesson]:
        trace_list = list(traces)
        llm_lessons = _llm_lessons(self.llm_extractor, self.config, trace_list, mode="error_only")
        if llm_lessons is not None:
            return llm_lessons
        lessons: list[TrajectoryLesson] = []
        for trace in trace_list:
            missing_tools = detect_missing_tools(trace)
            target = _first(trace.selected_skill_ids or trace.retrieved_skill_ids or trace.target_skill_ids)
            if missing_tools:
                lessons.append(
                    _lesson(
                        lesson_type="tool_lesson",
                        trace=trace,
                        target_skill_id=target,
                        evidence_summary=summarize_failure(trace),
                        reusable_principle="Missing tool coverage should be surfaced as a staged skill/tool contract patch.",
                        proposed_delta={"missing_tools": missing_tools},
                        confidence=0.78,
                    )
                )
            if detect_low_retrieval_coverage(trace):
                lessons.append(
                    _lesson(
                        lesson_type="coverage_lesson",
                        trace=trace,
                        target_skill_id=None,
                        evidence_summary=summarize_failure(trace),
                        reusable_principle="Low retrieval coverage indicates a potential missing reusable capability.",
                        proposed_delta={
                            "missing_capability": trace.task_summary or trace.error_summary or "uncovered task capability",
                            "suggested_required_tools": missing_tools,
                        },
                        confidence=0.62,
                    )
                )
            if not missing_tools:
                lessons.append(
                    _lesson(
                        lesson_type="error_lesson",
                        trace=trace,
                        target_skill_id=target,
                        evidence_summary=summarize_failure(trace),
                        reusable_principle="A bounded failure case should be retained for future precondition and repair guidance.",
                        proposed_delta={"failure_case": summarize_failure(trace)},
                        confidence=0.68,
                    )
                )
        return _dedupe_lessons(lessons)


class SuccessAnalyst:
    name = "SuccessAnalyst"

    def __init__(self, *, llm_extractor: Any | None = None, config: Trace2SkillRunConfig | None = None):
        self.llm_extractor = llm_extractor
        self.config = config

    def analyze(self, traces: Iterable[TraceRecord]) -> list[TrajectoryLesson]:
        trace_list = list(traces)
        llm_lessons = _llm_lessons(self.llm_extractor, self.config, trace_list, mode="success_only")
        if llm_lessons is not None:
            return llm_lessons
        lessons: list[TrajectoryLesson] = []
        for trace in trace_list:
            target = _first(trace.selected_skill_ids or trace.retrieved_skill_ids or trace.target_skill_ids)
            summary = summarize_success(trace)
            lessons.append(
                _lesson(
                    lesson_type="success_lesson",
                    trace=trace,
                    target_skill_id=target,
                    evidence_summary=summary,
                    reusable_principle="Successful compact traces can be staged as example memory for the selected skill.",
                    proposed_delta={
                        "example_summary": summary,
                        "tools_used": trace.tools_used,
                        "artifacts": trace.artifacts[:10],
                    },
                    confidence=0.72,
                    metadata={"lesson_subtype": "example_memory"},
                )
            )
            if trace.tools_used and target:
                lessons.append(
                    _lesson(
                        lesson_type="success_lesson",
                        trace=trace,
                        target_skill_id=target,
                        evidence_summary=f"Effective tool sequence: {', '.join(trace.tools_used)}",
                        reusable_principle="Repeated successful tool sequences are candidates for procedure guidance.",
                        proposed_delta={"procedure_steps": [f"Use tool sequence: {', '.join(trace.tools_used)}."]},
                        confidence=0.65,
                        metadata={"lesson_subtype": "tool_sequence"},
                    )
                )
        return _dedupe_lessons(lessons)


class CoverageAnalyst:
    name = "CoverageAnalyst"

    def __init__(self, *, llm_extractor: Any | None = None, config: Trace2SkillRunConfig | None = None):
        self.llm_extractor = llm_extractor
        self.config = config

    def analyze(self, traces: Iterable[TraceRecord]) -> list[TrajectoryLesson]:
        trace_list = list(traces)
        llm_lessons = _llm_lessons(self.llm_extractor, self.config, trace_list, mode="skill_creation_from_scratch")
        if llm_lessons is not None:
            return llm_lessons
        lessons: list[TrajectoryLesson] = []
        for trace in trace_list:
            if not detect_low_retrieval_coverage(trace):
                continue
            missing = trace.task_summary or trace.error_summary or "uncovered task capability"
            lessons.append(
                _lesson(
                    lesson_type="coverage_lesson",
                    trace=trace,
                    target_skill_id=None,
                    evidence_summary=f"Trace had insufficient skill coverage for: {missing}",
                    reusable_principle="Uncovered recurring tasks should be staged as candidate reusable skills.",
                    proposed_delta={
                        "missing_capability": missing,
                        "suggested_required_tools": trace.missing_tools,
                        "task_type": trace.task_type,
                    },
                    confidence=0.64,
                )
            )
        return _dedupe_lessons(lessons)


class PatchProposalAnalyst:
    name = "PatchProposalAnalyst"

    def __init__(
        self,
        *,
        llm_extractor: Any | None = None,
        config: Trace2SkillRunConfig | None = None,
        trace_pool: TracePool | None = None,
    ):
        self.llm_extractor = llm_extractor
        self.config = config
        self.trace_pool = trace_pool

    def analyze(self, lessons: Iterable[TrajectoryLesson]) -> list[SkillPatchProposal]:
        lesson_list = list(lessons)
        if (
            self.llm_extractor is not None
            and self.config is not None
            and self.config.enable_llm_analysts
            and self.trace_pool is not None
        ):
            patches = self.llm_extractor.extract_patches(
                self.trace_pool,
                lessons=lesson_list,
                mode=self.config.mode,
                config=self.config,
            )
            if patches:
                return patches
        patches: list[SkillPatchProposal] = []
        for lesson in lesson_list:
            patch_type = _patch_type_for_lesson(lesson)
            candidate_id = None
            if patch_type == "skill_create_patch":
                candidate_id = _stable_id("candidate-skill", [lesson.lesson_id, lesson.proposed_delta])
            patches.append(
                SkillPatchProposal(
                    patch_id=_stable_id("local-patch", [lesson.lesson_id, patch_type, lesson.target_skill_id]),
                    patch_type=patch_type,
                    target_skill_id=lesson.target_skill_id,
                    candidate_skill_id=candidate_id,
                    source_lesson_ids=[lesson.lesson_id],
                    source_trace_ids=lesson.source_trace_ids,
                    patch_content=_patch_content(lesson, patch_type, candidate_id),
                    evidence_summary=lesson.evidence_summary,
                    confidence=lesson.confidence,
                    support_count=max(lesson.support_count, len(lesson.source_trace_ids), 1),
                    risk_level=_risk_for_patch_type(patch_type),
                    created_at=_utc_now(),
                )
            )
        return _dedupe_patches(patches)


def run_deterministic_analysts(pool: TracePool, *, mode: str = "combined") -> tuple[list[TrajectoryLesson], list[SkillPatchProposal]]:
    lessons: list[TrajectoryLesson] = []
    if mode in {"error_only", "combined", "mixed", "skill_deepening", "skill_creation_from_scratch"}:
        lessons.extend(ErrorAnalyst().analyze(pool.failure_traces))
    if mode in {"success_only", "combined", "mixed", "skill_deepening"}:
        lessons.extend(SuccessAnalyst().analyze(pool.success_traces))
    if mode in {"combined", "mixed", "skill_creation_from_scratch"}:
        lessons.extend(CoverageAnalyst().analyze([*pool.failure_traces, *pool.mixed_traces]))
    patches = PatchProposalAnalyst().analyze(lessons)
    return lessons, patches


def _llm_lessons(
    llm_extractor: Any | None,
    config: Trace2SkillRunConfig | None,
    traces: list[TraceRecord],
    *,
    mode: str,
) -> list[TrajectoryLesson] | None:
    if llm_extractor is None or config is None or not config.enable_llm_analysts:
        return None
    lessons = llm_extractor.extract_lessons(traces, mode=mode, config=config)
    return lessons if lessons else None


def _lesson(
    *,
    lesson_type: str,
    trace: TraceRecord,
    target_skill_id: str | None,
    evidence_summary: str,
    reusable_principle: str,
    proposed_delta: dict[str, Any],
    confidence: float,
    metadata: dict[str, Any] | None = None,
) -> TrajectoryLesson:
    return TrajectoryLesson(
        lesson_id=_stable_id(
            "lesson",
            [lesson_type, trace.trace_id, target_skill_id, evidence_summary, proposed_delta],
        ),
        source_trace_ids=[trace.trace_id],
        lesson_type=lesson_type,  # type: ignore[arg-type]
        target_skill_id=target_skill_id,
        evidence_summary=evidence_summary[:600],
        reusable_principle=reusable_principle,
        proposed_delta=proposed_delta,
        confidence=confidence,
        support_count=1,
        metadata=metadata or {},
    )


def _patch_type_for_lesson(lesson: TrajectoryLesson) -> str:
    if lesson.lesson_type == "coverage_lesson":
        return "skill_create_patch"
    if lesson.lesson_type == "tool_lesson":
        return "required_tools_patch"
    if lesson.lesson_type == "validation_lesson":
        return "validation_rule_patch"
    if lesson.lesson_type == "error_lesson":
        return "failure_case_patch"
    if lesson.metadata.get("lesson_subtype") == "tool_sequence":
        return "procedure_step_patch"
    return "example_memory_patch"


def _patch_content(lesson: TrajectoryLesson, patch_type: str, candidate_id: str | None) -> dict[str, Any]:
    if patch_type == "required_tools_patch":
        return {"required_tools": _strings(lesson.proposed_delta.get("missing_tools"))}
    if patch_type == "skill_create_patch":
        missing = str(lesson.proposed_delta.get("missing_capability") or "uncovered capability")
        return {
            "candidate_id": candidate_id,
            "proposed_name": _title_from_text(missing),
            "missing_capability_description": missing,
            "suggested_required_inputs": ["task goal", "available evidence"],
            "suggested_expected_outputs": ["validated task output"],
            "suggested_required_tools": _strings(lesson.proposed_delta.get("suggested_required_tools")),
        }
    if patch_type == "procedure_step_patch":
        return {"procedure_steps": _strings(lesson.proposed_delta.get("procedure_steps"))}
    if patch_type == "failure_case_patch":
        return {"failure_reason": str(lesson.proposed_delta.get("failure_case") or lesson.evidence_summary)}
    if patch_type == "validation_rule_patch":
        return {"validation_rules": _strings(lesson.proposed_delta.get("validation_rules"))}
    return {"example_summary": str(lesson.proposed_delta.get("example_summary") or lesson.evidence_summary)}


def _risk_for_patch_type(patch_type: str) -> str:
    if patch_type in {"example_memory_patch", "metadata_patch", "failure_case_patch"}:
        return "low"
    if patch_type in {"required_tools_patch", "procedure_step_patch", "precondition_patch", "validation_rule_patch"}:
        return "medium"
    return "high"


def _dedupe_lessons(lessons: list[TrajectoryLesson]) -> list[TrajectoryLesson]:
    by_id = {lesson.lesson_id: lesson for lesson in lessons}
    return [by_id[key] for key in sorted(by_id)]


def _dedupe_patches(patches: list[SkillPatchProposal]) -> list[SkillPatchProposal]:
    by_id = {patch.patch_id: patch for patch in patches}
    return [by_id[key] for key in sorted(by_id)]


def _first(values: list[str]) -> str | None:
    return values[0] if values else None


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list | tuple):
        return [item for item in value if isinstance(item, str) and item]
    return []


def _title_from_text(value: str) -> str:
    words = [word.strip("._-/") for word in value.split() if word.strip("._-/")]
    return " ".join(word[:1].upper() + word[1:] for word in words[:8]) or "Uncovered Capability"


def _stable_id(prefix: str, parts: Iterable[Any]) -> str:
    encoded = json.dumps(_json_compatible(list(parts)), sort_keys=True, separators=(",", ":"))
    return f"{prefix}-{sha256(encoded.encode('utf-8')).hexdigest()[:16]}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
