from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import Any

from pydantic import ValidationError

from evolab.backends.llm.base import LLMBackend
from evolab.backends.skills.trace2skill.analysts import run_deterministic_analysts
from evolab.backends.skills.trace2skill.schema import (
    SkillPatchProposal,
    Trace2SkillRunConfig,
    TracePool,
    TraceRecord,
    TrajectoryLesson,
)
from evolab.contracts.common import Message
from evolab.contracts.llm import LLMGenerationConfig


class Trace2SkillLLMExtractor:
    """Bridge Trace2Skill extraction to EvoLab's existing LLM runtime API."""

    def __init__(self, llm_client: Any | None = None):
        self.llm_client = llm_client
        self._runtime = None
        self.audit_events: list[dict[str, Any]] = []

    def extract_lessons(
        self,
        traces: TracePool | Iterable[TraceRecord],
        *,
        mode: str,
        config: Trace2SkillRunConfig | None = None,
        target_skill_context: dict[str, Any] | None = None,
        retrieved_skill_metadata: dict[str, Any] | None = None,
    ) -> list[TrajectoryLesson]:
        config = config or Trace2SkillRunConfig(mode=mode)
        pool = traces if isinstance(traces, TracePool) else _pool_from_traces(list(traces))
        extracted = self._extract_with_llm(
            pool,
            mode=mode,
            config=config,
            target_skill_context=target_skill_context,
            retrieved_skill_metadata=retrieved_skill_metadata,
        )
        if extracted is None:
            return run_deterministic_analysts(pool, mode=mode)[0] if config.enable_deterministic_fallback else []
        return extracted[0]

    def extract_patches(
        self,
        traces: TracePool | Iterable[TraceRecord],
        *,
        lessons: list[TrajectoryLesson] | None = None,
        mode: str,
        config: Trace2SkillRunConfig | None = None,
        target_skill_context: dict[str, Any] | None = None,
        retrieved_skill_metadata: dict[str, Any] | None = None,
    ) -> list[SkillPatchProposal]:
        config = config or Trace2SkillRunConfig(mode=mode)
        pool = traces if isinstance(traces, TracePool) else _pool_from_traces(list(traces))
        extracted = self._extract_with_llm(
            pool,
            mode=mode,
            config=config,
            target_skill_context=target_skill_context,
            retrieved_skill_metadata=retrieved_skill_metadata,
        )
        if extracted is None:
            return run_deterministic_analysts(pool, mode=mode)[1] if config.enable_deterministic_fallback else []
        return extracted[1]

    def extract_lessons_and_patches(
        self,
        pool: TracePool,
        *,
        config: Trace2SkillRunConfig | None = None,
        mode: str | None = None,
        enable_llm_analysts: bool | None = None,
        enable_deterministic_fallback: bool | None = None,
    ) -> tuple[list[TrajectoryLesson], list[SkillPatchProposal]]:
        config = config or Trace2SkillRunConfig(
            mode=mode or "combined",
            enable_llm_analysts=bool(enable_llm_analysts),
            enable_deterministic_fallback=True
            if enable_deterministic_fallback is None
            else enable_deterministic_fallback,
        )
        if not config.enable_llm_analysts:
            return run_deterministic_analysts(pool, mode=config.mode) if config.enable_deterministic_fallback else ([], [])
        extracted = self._extract_with_llm(pool, mode=config.mode, config=config)
        if extracted is not None:
            return extracted
        if not config.enable_deterministic_fallback:
            return [], []
        return run_deterministic_analysts(pool, mode=config.mode)

    def lesson_output_schema(self) -> dict[str, Any]:
        delta_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "missing_tools": {"type": "array", "items": {"type": "string"}},
                "suggested_required_tools": {"type": "array", "items": {"type": "string"}},
                "example_summary": {"type": "string"},
                "failure_case": {"type": "string"},
                "failure_reason": {"type": "string"},
                "procedure_steps": {"type": "array", "items": {"type": "string"}},
                "preconditions": {"type": "array", "items": {"type": "string"}},
                "validation_rules": {"type": "array", "items": {"type": "string"}},
                "missing_capability": {"type": "string"},
                "metadata_note": {"type": "string"},
            },
            "required": [
                "missing_tools",
                "suggested_required_tools",
                "example_summary",
                "failure_case",
                "failure_reason",
                "procedure_steps",
                "preconditions",
                "validation_rules",
                "missing_capability",
                "metadata_note",
            ],
        }
        lesson_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "lesson_type": {
                    "type": "string",
                    "enum": [
                        "error_lesson",
                        "success_lesson",
                        "coverage_lesson",
                        "tool_lesson",
                        "validation_lesson",
                    ],
                },
                "target_skill_id": {"type": ["string", "null"]},
                "evidence_summary": {"type": "string"},
                "reusable_principle": {"type": "string"},
                "proposed_delta": delta_schema,
                "confidence": {"type": "number"},
                "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
                "source_trace_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "lesson_type",
                "target_skill_id",
                "evidence_summary",
                "reusable_principle",
                "proposed_delta",
                "confidence",
                "risk_level",
                "source_trace_ids",
            ],
        }
        patch_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "patch_type": {
                    "type": "string",
                    "enum": [
                        "skill_deepen_patch",
                        "skill_create_patch",
                        "required_tools_patch",
                        "example_memory_patch",
                        "procedure_step_patch",
                        "precondition_patch",
                        "failure_case_patch",
                        "validation_rule_patch",
                        "relationship_patch",
                        "metadata_patch",
                    ],
                },
                "target_skill_id": {"type": ["string", "null"]},
                "candidate_skill_id": {"type": ["string", "null"]},
                "evidence_summary": {"type": "string"},
                "proposed_delta": delta_schema,
                "confidence": {"type": "number"},
                "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
                "source_trace_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "patch_type",
                "target_skill_id",
                "candidate_skill_id",
                "evidence_summary",
                "proposed_delta",
                "confidence",
                "risk_level",
                "source_trace_ids",
            ],
        }
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "lessons": {"type": "array", "items": lesson_schema},
                "patches": {"type": "array", "items": patch_schema},
            },
            "required": ["lessons", "patches"],
        }

    def bounded_trace_payload(self, pool: TracePool, *, max_chars: int = 1200, max_traces: int = 20) -> list[dict[str, Any]]:
        rows = []
        for trace in pool.traces[:max_traces]:
            rows.append(
                {
                    "trace_id": trace.trace_id,
                    "task_summary": _bounded(trace.task_summary or "", max_chars),
                    "target_skill_ids": trace.target_skill_ids,
                    "retrieved_skill_ids": trace.retrieved_skill_ids,
                    "selected_skill_ids": trace.selected_skill_ids,
                    "tools_used": trace.tools_used,
                    "missing_tools": trace.missing_tools,
                    "final_status": trace.final_status,
                    "error_summary": _bounded(trace.error_summary or "", max_chars),
                    "compact_execution_summary": _bounded(trace.compact_execution_summary or "", max_chars),
                    "evaluation_metrics": trace.evaluation_metrics,
                }
            )
        return rows

    def _extract_with_llm(
        self,
        pool: TracePool,
        *,
        mode: str,
        config: Trace2SkillRunConfig,
        target_skill_context: dict[str, Any] | None = None,
        retrieved_skill_metadata: dict[str, Any] | None = None,
    ) -> tuple[list[TrajectoryLesson], list[SkillPatchProposal]] | None:
        runtime = self._resolve_runtime()
        if runtime is None:
            self._audit("llm_unavailable", "no LLM runtime configured")
            return None

        prompt_payload = {
            "mode": mode,
            "trace_records": self.bounded_trace_payload(pool, max_chars=config.max_trace_summary_chars),
            "target_skill_context": target_skill_context or {},
            "retrieved_skill_metadata": retrieved_skill_metadata or {},
            "instructions": {
                "json_only": True,
                "do_not_include_chain_of_thought": True,
                "empty_unused_delta_fields": True,
            },
        }
        last_error = None
        attempts = config.max_llm_retries + 1
        for attempt in range(attempts):
            try:
                content = self._call_runtime(runtime, prompt_payload, config=config, repair_error=last_error)
                return self._parse_output(content, pool=pool)
            except Exception as exc:  # parsing and model failures share bounded fallback path
                last_error = str(exc)
                self._audit(
                    "llm_extract_failed",
                    last_error,
                    metadata={"attempt": attempt + 1, "max_attempts": attempts},
                )
        return None

    def _resolve_runtime(self):
        if self.llm_client is None:
            return None
        if self._runtime is not None:
            return self._runtime
        if isinstance(self.llm_client, LLMBackend) or hasattr(self.llm_client, "instantiate"):
            self._runtime = self.llm_client.instantiate(None)
            return self._runtime
        if hasattr(self.llm_client, "generate"):
            self._runtime = self.llm_client
            return self._runtime
        return None

    def _call_runtime(
        self,
        runtime,
        prompt_payload: dict[str, Any],
        *,
        config: Trace2SkillRunConfig,
        repair_error: str | None,
    ) -> str:
        system = (
            "You extract reusable Trace2Skill lessons and patch proposals. "
            "Return JSON only. Do not include private reasoning, chain_of_thought, hidden rationale, or tool calls."
        )
        if repair_error:
            system += f" Previous output was invalid: {repair_error}. Return only valid JSON matching the schema."
        response = runtime.generate(
            [
                Message(role="system", content=system),
                Message(role="user", content=json.dumps(prompt_payload, sort_keys=True)),
            ],
            [],
            LLMGenerationConfig(
                model=config.llm_config_ref or getattr(runtime, "model", "trace2skill-llm"),
                temperature=config.llm_temperature,
                response_json_schema=self.lesson_output_schema(),
                metadata={"trace2skill_mode": config.mode, "llm_config_ref": config.llm_config_ref},
            ),
        )
        if response.action.action != "final_answer" or not response.action.content:
            raise ValueError(f"LLM returned unsupported action {response.action.action!r}")
        return response.action.content

    def _parse_output(
        self,
        content: str,
        *,
        pool: TracePool,
    ) -> tuple[list[TrajectoryLesson], list[SkillPatchProposal]]:
        loaded = json.loads(content)
        if not isinstance(loaded, dict):
            raise ValueError("LLM output must be a JSON object")
        lessons = [self._lesson(row, pool=pool) for row in _list(loaded.get("lessons"))]
        patches = [self._patch(row, pool=pool) for row in _list(loaded.get("patches"))]
        return _sort_lessons(lessons), _sort_patches(patches)

    def _lesson(self, row: Any, *, pool: TracePool) -> TrajectoryLesson:
        if not isinstance(row, dict):
            raise ValueError("lesson row must be an object")
        if any(key in row for key in _PRIVATE_KEYS):
            raise ValueError("LLM lesson output included private reasoning field")
        source_trace_ids = _source_trace_ids(row, pool)
        proposed_delta = _clean_delta(_dict(row.get("proposed_delta")))
        try:
            return TrajectoryLesson(
                lesson_id=_stable_id(
                    "llm-lesson",
                    [
                        row.get("lesson_type"),
                        row.get("target_skill_id"),
                        row.get("evidence_summary"),
                        proposed_delta,
                        source_trace_ids,
                    ],
                ),
                source_trace_ids=source_trace_ids,
                lesson_type=row.get("lesson_type"),
                target_skill_id=_optional_str(row.get("target_skill_id")),
                evidence_summary=str(row.get("evidence_summary") or "")[:600],
                reusable_principle=str(row.get("reusable_principle") or "")[:600],
                proposed_delta=proposed_delta,
                confidence=float(row.get("confidence", 0.5)),
                support_count=max(len(source_trace_ids), 1),
                metadata={"analyst": "llm", "risk_level": row.get("risk_level", "medium")},
            )
        except (TypeError, ValueError, ValidationError) as exc:
            raise ValueError(f"invalid LLM lesson schema: {exc}") from exc

    def _patch(self, row: Any, *, pool: TracePool) -> SkillPatchProposal:
        if not isinstance(row, dict):
            raise ValueError("patch row must be an object")
        if any(key in row for key in _PRIVATE_KEYS):
            raise ValueError("LLM patch output included private reasoning field")
        source_trace_ids = _source_trace_ids(row, pool)
        patch_content = _clean_delta(_dict(row.get("patch_content")) or _dict(row.get("proposed_delta")))
        try:
            patch_type = row.get("patch_type")
            return SkillPatchProposal(
                patch_id=_stable_id(
                    "llm-patch",
                    [
                        patch_type,
                        row.get("target_skill_id"),
                        row.get("candidate_skill_id"),
                        patch_content,
                        source_trace_ids,
                    ],
                ),
                patch_type=patch_type,
                target_skill_id=_optional_str(row.get("target_skill_id")),
                candidate_skill_id=_optional_str(row.get("candidate_skill_id")),
                source_lesson_ids=[],
                source_trace_ids=source_trace_ids,
                patch_content=_patch_content_for_type(str(patch_type), patch_content),
                evidence_summary=str(row.get("evidence_summary") or "")[:600],
                confidence=float(row.get("confidence", 0.5)),
                support_count=max(len(source_trace_ids), 1),
                risk_level=row.get("risk_level", "medium"),
                created_at=_utc_now(),
            )
        except (TypeError, ValueError, ValidationError) as exc:
            raise ValueError(f"invalid LLM patch schema: {exc}") from exc

    def _audit(self, event_type: str, message: str, *, metadata: dict[str, Any] | None = None) -> None:
        self.audit_events.append(
            {
                "event_type": event_type,
                "message": message,
                "created_at": _utc_now(),
                "metadata": metadata or {},
            }
        )


_PRIVATE_KEYS = {"chain_of_thought", "hidden_reasoning", "private_reasoning", "rationale_cot"}


def _pool_from_traces(traces: list[TraceRecord]) -> TracePool:
    success = [trace for trace in traces if trace.final_status in {"runtime_success", "evaluation_success"}]
    failure = [trace for trace in traces if trace.final_status in {"runtime_failure", "evaluation_failure"}]
    mixed = [trace for trace in traces if trace.final_status in {"partial_success", "unknown"}]
    return TracePool(
        pool_id=_stable_id("llm-trace-pool", [trace.trace_id for trace in traces]),
        traces=traces,
        success_traces=success,
        failure_traces=failure,
        mixed_traces=mixed,
        stats={
            "trace_count": len(traces),
            "success_count": len(success),
            "failure_count": len(failure),
            "mixed_count": len(mixed),
        },
        created_at=_utc_now(),
    )


def _patch_content_for_type(patch_type: str, content: dict[str, Any]) -> dict[str, Any]:
    if patch_type == "required_tools_patch":
        return {"required_tools": _strings(content.get("required_tools") or content.get("missing_tools"))}
    if patch_type == "example_memory_patch":
        return {"example_summary": str(content.get("example_summary") or content.get("metadata_note") or "")}
    if patch_type == "procedure_step_patch":
        return {"procedure_steps": _strings(content.get("procedure_steps"))}
    if patch_type == "precondition_patch":
        return {"preconditions": _strings(content.get("preconditions"))}
    if patch_type == "failure_case_patch":
        return {
            "failure_reason": str(
                content.get("failure_reason") or content.get("failure_case") or content.get("metadata_note") or ""
            )
        }
    if patch_type == "validation_rule_patch":
        return {"validation_rules": _strings(content.get("validation_rules"))}
    if patch_type == "skill_create_patch":
        missing = str(content.get("missing_capability") or content.get("metadata_note") or "uncovered capability")
        return {
            "proposed_name": _title(missing),
            "missing_capability_description": missing,
            "suggested_required_tools": _strings(content.get("suggested_required_tools") or content.get("missing_tools")),
        }
    return content


def _clean_delta(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in _PRIVATE_KEYS and item not in ("", [], {}, None)
    }


def _source_trace_ids(row: dict[str, Any], pool: TracePool) -> list[str]:
    source = _strings(row.get("source_trace_ids"))
    valid_ids = {trace.trace_id for trace in pool.traces}
    if source:
        return [trace_id for trace_id in source if trace_id in valid_ids] or source
    return [trace.trace_id for trace in pool.traces[:1]]


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list | tuple):
        return [item for item in value if isinstance(item, str) and item]
    return []


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _bounded(value: str, max_chars: int) -> str:
    return value if len(value) <= max_chars else value[: max_chars - 3] + "..."


def _title(value: str) -> str:
    words = [word.strip("._-/") for word in value.split() if word.strip("._-/")]
    return " ".join(word[:1].upper() + word[1:] for word in words[:8]) or "Uncovered Capability"


def _sort_lessons(lessons: list[TrajectoryLesson]) -> list[TrajectoryLesson]:
    by_id = {lesson.lesson_id: lesson for lesson in lessons}
    return [by_id[key] for key in sorted(by_id)]


def _sort_patches(patches: list[SkillPatchProposal]) -> list[SkillPatchProposal]:
    by_id = {patch.patch_id: patch for patch in patches}
    return [by_id[key] for key in sorted(by_id)]


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


__all__ = ["Trace2SkillLLMExtractor"]
