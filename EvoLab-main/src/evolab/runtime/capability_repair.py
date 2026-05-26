from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from evolab.contracts.common import Message, RuntimePolicy
from evolab.contracts.generated_tools import GeneratedToolCapabilityGrant
from evolab.contracts.repair import (
    FailureSignal,
    PromotionCandidate,
    RepairPlan,
    RepairValidationResult,
    SkillOverlayPatch,
    ToolOverlayPatch,
)
from evolab.contracts.retrieval import SkillBundle, SkillRef
from evolab.contracts.tools import ToolCall, ToolCallRecord, ToolResult, ToolSpec
from evolab.runtime.skill_retrieval import build_skill_context


@dataclass
class TaskLocalToolOverlay:
    patches: list[ToolOverlayPatch] = field(default_factory=list)


@dataclass
class TaskLocalSkillOverlay:
    patches: list[SkillOverlayPatch] = field(default_factory=list)
    runtime_skills: list[SkillRef] = field(default_factory=list)

    def apply(self, skill_bundle: SkillBundle) -> SkillBundle:
        skills = [skill.model_copy(deep=True) for skill in skill_bundle.skills]
        runtime_skill_ids: list[str] = []
        applied_patches: list[dict[str, Any]] = []
        for patch in self.patches:
            matched = False
            for index, skill in enumerate(skills):
                if _skill_matches_patch(skill, patch):
                    matched = True
                    skills[index] = _patched_skill(skill, patch)
            if not matched:
                runtime_skill_id = f"runtime-skill-{patch.patch_id}"
                runtime_skill_ids.append(runtime_skill_id)
                skills.append(
                    SkillRef(
                        skill_id=runtime_skill_id,
                        name=f"Runtime repair: {patch.patch_id}",
                        content=_skill_overlay_content(patch),
                        required_tools=[],
                        metadata={"runtime_skill": True, "patch_id": patch.patch_id},
                    )
                )
            applied_patches.append(patch.model_dump(mode="json"))

        for runtime_skill in self.runtime_skills:
            runtime_skill_ids.append(runtime_skill.skill_id)
            skills.append(runtime_skill)

        metadata = dict(skill_bundle.metadata)
        metadata["runtime_skill_overlay"] = {
            "patches": applied_patches,
            "runtime_skill_ids": runtime_skill_ids,
        }
        return skill_bundle.model_copy(update={"skills": skills, "metadata": metadata})


class FailureDetector:
    def detect_tool_failure(
        self,
        *,
        task_id: str,
        subagent_id: str,
        step_id: str,
        tool_call: ToolCall,
        tool_result: ToolResult,
        active_skill_bundle: SkillBundle,
        task_context: dict[str, Any],
    ) -> FailureSignal | None:
        arguments = tool_call.arguments
        metadata = tool_result.metadata
        failure_type: str | None = None
        severity = "medium"
        evidence: list[str] = []
        suspected_cause = ""
        actions: list[str] = []

        start_row = _int_or_none(arguments.get("start_row"))
        end_row = _int_or_none(arguments.get("end_row"))
        block = metadata.get("plain_text_table_block") or task_context.get("plain_text_table_block")
        row_count = _row_count(tool_result)

        if tool_call.name == "read_table_slice" and start_row is not None and end_row is not None and start_row > end_row:
            failure_type = "invalid_tool_arguments"
            severity = "high"
            suspected_cause = "start_row_greater_than_end_row"
            evidence.append(f"start_row={start_row} end_row={end_row}")
            actions = ["retry_with_corrected_tool_arguments"]
        elif tool_call.name == "read_table_slice" and _looks_like_coordinate_mismatch(
            start_row=start_row,
            end_row=end_row,
            row_count=row_count,
            block=block,
            rows=metadata.get("rows"),
        ):
            failure_type = "coordinate_system_mismatch"
            severity = "high"
            suspected_cause = "source_line_as_row_index"
            evidence.append(f"row_count={row_count}")
            evidence.append(f"table_block={block}")
            actions = ["apply_tool_overlay", "apply_skill_overlay", "retry_failed_step"]
        elif tool_result.status == "error" and metadata.get("error_type") == "unprepared_tool":
            failure_type = "missing_tool_capability"
            severity = "high"
            suspected_cause = "tool_not_prepared"
            evidence.append(tool_result.content)
            actions = ["add_runtime_tool", "retry_failed_step"]
        elif tool_result.status == "error" and _is_retryable_api_failure(tool_result):
            failure_type = "retryable_api_failure"
            severity = "medium"
            suspected_cause = "transient_external_failure"
            evidence.append(tool_result.content)
            actions = ["retry_same_call"]
        elif tool_result.status == "error" and _looks_like_schema_mismatch(tool_result):
            failure_type = "schema_mismatch"
            severity = "medium"
            suspected_cause = "schema_validation_failed"
            evidence.append(tool_result.content)
            actions = ["add_runtime_skill_policy", "retry_failed_step"]
        elif tool_result.status == "ok" and isinstance(metadata.get("rows"), list) and not metadata.get("rows"):
            failure_type = "empty_tool_result"
            severity = "medium"
            suspected_cause = "empty_rows_for_recoverable_input"
            evidence.append(tool_result.content)
            if row_count is not None:
                evidence.append(f"expected_row_count={row_count}")
            actions = ["apply_tool_overlay", "retry_failed_step"]
        elif not active_skill_bundle.skills:
            failure_type = "missing_skill"
            severity = "medium"
            suspected_cause = "no_active_skill_bundle"
            evidence.append("active skill bundle is empty")
            actions = ["create_runtime_skill"]
        elif tool_call.name == "read_table_slice" and not _skill_mentions_table_coordinate_policy(active_skill_bundle):
            failure_type = "missing_skill_policy"
            severity = "low"
            suspected_cause = "missing_coordinate_policy"
            evidence.append("active skills do not mention table coordinate handling")
            actions = ["apply_skill_overlay"]

        if failure_type is None:
            return None

        return FailureSignal(
            failure_id=f"failure-{uuid4()}",
            task_id=task_id,
            subagent_id=subagent_id,
            step_id=step_id,
            failure_type=failure_type,
            severity=severity,
            failed_tool_call=tool_call,
            failed_tool_result=tool_result,
            active_skill_ids=[skill.skill_id for skill in active_skill_bundle.skills],
            evidence=evidence,
            suspected_cause=suspected_cause,
            suggested_repair_actions=actions,
        )


class CapabilityRepairPlanner:
    def plan(self, failure_signal: FailureSignal) -> RepairPlan:
        skill_target = failure_signal.active_skill_ids[0] if failure_signal.active_skill_ids else None
        if failure_signal.failure_type == "coordinate_system_mismatch":
            return RepairPlan(
                repair_id=f"repair-{uuid4()}",
                failure_id=failure_signal.failure_id,
                diagnosis="skill_policy_gap",
                repair_action="apply_tool_overlay_and_retry",
                rationale="read_table_slice appears to have received source-file line coordinates instead of table-relative row indexes",
                skill_overlay_patch=SkillOverlayPatch(
                    patch_id=f"skill-patch-{uuid4()}",
                    target_skill_id=skill_target,
                    principles=[
                        "distinguish source-file line coordinates from table-relative row coordinates",
                    ],
                    failure_modes=["source_line_as_row_index"],
                    recovery_strategies=[
                        "normalize source line ranges to table-relative row bounds before retry",
                        "retry full caption read when a recoverable table returns zero rows",
                    ],
                    tool_use_policies=[
                        "use read_table_slice_safe when table metadata includes source line boundaries",
                    ],
                ),
                tool_overlay_patch=ToolOverlayPatch(
                    patch_id=f"tool-patch-{uuid4()}",
                    name="read_table_slice_safe",
                    base_tool_name="read_table_slice",
                    strategy="safe_read_table_slice_wrapper",
                    description="Validate and normalize table slice arguments before calling read_table_slice",
                ),
                retry_plan={"tool_name": "read_table_slice_safe", "reuse_failed_arguments": True},
                validation_plan={
                    "smoke_tests": [
                        "failed_call_returns_non_empty_rows_or_clear_diagnostics",
                        "normal_row_index_behavior_still_works",
                    ]
                },
                promotion_candidate_policy={"emit_candidate": True, "allow_global_mutation": False},
            )

        if failure_signal.failure_type == "invalid_tool_arguments":
            return RepairPlan(
                repair_id=f"repair-{uuid4()}",
                failure_id=failure_signal.failure_id,
                diagnosis="tool_misuse",
                repair_action="retry_with_corrected_tool_arguments",
                rationale="tool arguments contain descending slice bounds",
                retry_plan={"tool_name": "read_table_slice", "normalize_descending_bounds": True},
                validation_plan={"smoke_tests": ["corrected_bounds_return_non_error_result"]},
                promotion_candidate_policy={"emit_candidate": True, "allow_global_mutation": False},
            )

        if failure_signal.failure_type == "retryable_api_failure":
            return RepairPlan(
                repair_id=f"repair-{uuid4()}",
                failure_id=failure_signal.failure_id,
                diagnosis="external_retryable_failure",
                repair_action="retry_same_call",
                rationale="tool or backend failure looks transient",
                retry_plan={"reuse_failed_arguments": True},
                validation_plan={"smoke_tests": ["retry_does_not_repeat_same_error"]},
                promotion_candidate_policy={"emit_candidate": False, "allow_global_mutation": False},
            )

        if failure_signal.failure_type == "missing_tool_capability":
            return RepairPlan(
                repair_id=f"repair-{uuid4()}",
                failure_id=failure_signal.failure_id,
                diagnosis="missing_tool",
                repair_action="create_runtime_tool",
                rationale="requested tool is not currently prepared for the run",
                promotion_candidate_policy={"emit_candidate": True, "allow_global_mutation": False},
            )

        return RepairPlan(
            repair_id=f"repair-{uuid4()}",
            failure_id=failure_signal.failure_id,
            diagnosis="unknown_failure",
            repair_action="record_failure_only",
            rationale="no safe automated repair is available",
            promotion_candidate_policy={"emit_candidate": False, "allow_global_mutation": False},
        )


class RepairValidator:
    def validate_repair_plan(
        self,
        *,
        plan: RepairPlan,
        tool_runtime: Any,
        failed_tool_call: ToolCall,
    ) -> RepairValidationResult:
        before_result = tool_runtime.execute_registered_tool_name(
            call_id=f"before-{uuid4()}",
            name=failed_tool_call.name,
            arguments=failed_tool_call.arguments,
        )
        warnings: list[str] = []
        if plan.tool_overlay_patch is not None:
            tool_runtime.apply_runtime_tool_overlay(TaskLocalToolOverlay(patches=[plan.tool_overlay_patch]))
        retry_call = _retry_tool_call(plan, failed_tool_call)
        after_result = tool_runtime.execute_tool_name(
            call_id=f"after-{uuid4()}",
            name=retry_call.name,
            arguments=retry_call.arguments,
        )
        normal_call = failed_tool_call.model_copy(
            update={
                "call_id": f"normal-{uuid4()}",
                "arguments": {
                    **failed_tool_call.arguments,
                    "start_row": 0,
                    "end_row": min(2, _row_count(before_result) or 2),
                },
            }
        )
        normal_result = tool_runtime.execute_registered_tool_name(
            call_id=normal_call.call_id,
            name=failed_tool_call.name,
            arguments=normal_call.arguments,
        )
        valid = _row_count(after_result) > _row_count(before_result) or (
            before_result.status == "error" and after_result.status == "ok"
        )
        if not valid:
            warnings.append("repair validation did not improve the failed call output")
        return RepairValidationResult(
            valid=valid,
            status="passed" if valid else "failed",
            warnings=warnings,
            before_summary=_tool_result_summary(before_result),
            after_summary=_tool_result_summary(after_result),
            normal_behavior_ok=normal_result.status == "ok" and _row_count(normal_result) > 0,
        )


@dataclass
class RepairRuntimeOutcome:
    retry_record: ToolCallRecord | None = None
    repair_messages: list[Message] = field(default_factory=list)
    updated_skill_bundle: SkillBundle | None = None
    updated_skill_context: dict[str, Any] | None = None
    repair_entry: dict[str, Any] | None = None
    promotion_candidates: list[dict[str, Any]] = field(default_factory=list)


class CapabilityRepairRuntime:
    def __init__(
        self,
        *,
        detector: FailureDetector | None = None,
        planner: CapabilityRepairPlanner | None = None,
        validator: RepairValidator | None = None,
    ) -> None:
        self.detector = detector or FailureDetector()
        self.planner = planner or CapabilityRepairPlanner()
        self.validator = validator or RepairValidator()

    def maybe_repair(
        self,
        *,
        task_id: str,
        run_ref: str,
        step_id: str,
        role: str,
        task_goal: str,
        tool_call: ToolCall,
        tool_result: ToolResult,
        active_skill_bundle: SkillBundle,
        tool_runtime: Any,
        generated_tool_runtime: Any | None = None,
        generated_tool_package: Any | None = None,
        generated_tool_builder: Any | None = None,
        role_pool_templates: list[Any] | None = None,
        trajectory_collector: Any,
        runtime_policy: RuntimePolicy,
        repair_log: list[dict[str, Any]] | None = None,
    ) -> RepairRuntimeOutcome | None:
        signal = self.detector.detect_tool_failure(
            task_id=task_id,
            subagent_id=run_ref,
            step_id=step_id,
            tool_call=tool_call,
            tool_result=tool_result,
            active_skill_bundle=active_skill_bundle,
            task_context={"task_goal": task_goal},
        )
        if signal is None:
            return None
        trajectory_collector.record_event(
            event_type="repair_detected",
            subject_type="repair",
            subject_ref=signal.failure_id,
            task_id=task_id,
            run_ref=run_ref,
            metadata={"failure_signal": signal.model_dump(mode="json")},
        )
        plan = self.planner.plan(signal)
        trajectory_collector.record_event(
            event_type="repair_planned",
            subject_type="repair",
            subject_ref=plan.repair_id,
            task_id=task_id,
            run_ref=run_ref,
            parent_ref=signal.failure_id,
            metadata={"repair_plan": plan.model_dump(mode="json")},
        )
        repair_log = repair_log if repair_log is not None else []
        repair_entry: dict[str, Any] = {
            "failure_signal": signal.model_dump(mode="json"),
            "repair_plan": plan.model_dump(mode="json"),
            "validation_result": None,
            "retry_result": None,
        }
        repair_log.append(repair_entry)
        updated_skill_bundle = active_skill_bundle
        if runtime_policy.allow_runtime_skill_overlay and plan.skill_overlay_patch is not None:
            updated_skill_bundle = TaskLocalSkillOverlay(patches=[plan.skill_overlay_patch]).apply(
                active_skill_bundle
            )
        generated_registered_name: str | None = None
        if plan.repair_action == "create_runtime_tool" and runtime_policy.allow_runtime_tool_creation:
            package = generated_tool_package if generated_tool_package is not None else plan.new_runtime_tool
            if package is None and generated_tool_builder is not None:
                try:
                    package = generated_tool_builder.build(
                        task_id=task_id,
                        task_goal=task_goal,
                        run_ref=run_ref,
                        built_in_tool_specs=_tool_specs_for_runtime(tool_runtime),
                        generated_tool_specs=_generated_tool_specs_for_runtime(tool_runtime),
                        role_pool_templates=role_pool_templates or [],
                        artifact_root=getattr(generated_tool_runtime, "artifact_root", ""),
                        capability_grant=GeneratedToolCapabilityGrant(),
                        failure_signal=signal,
                        requested_tool_name=tool_call.name,
                    )
                except Exception as exc:
                    registration_json = {
                        "status": "rejected",
                        "failure_reason": str(exc),
                        "errors": [str(exc)],
                    }
                    repair_entry["generated_tool_registration"] = registration_json
                    trajectory_collector.record_event(
                        event_type="generated_tool_rejected",
                        subject_type="generated_tool",
                        subject_ref=tool_call.name,
                        task_id=task_id,
                        run_ref=run_ref,
                        parent_ref=plan.repair_id,
                        metadata=registration_json,
                    )
            if generated_tool_runtime is not None and package is not None:
                registration_json: dict[str, Any]
                try:
                    registration = generated_tool_runtime.register_package(
                        package=package,
                        task_id=task_id,
                        run_ref=run_ref,
                        context={"task_id": task_id, "goal": task_goal, "repair_id": plan.repair_id},
                    )
                    registration_json = registration.model_dump(mode="json")
                    repair_entry["generated_tool_registration"] = registration_json
                    if registration.validation.valid:
                        generated_registered_name = registration.registered_tool_name
                        plan = plan.model_copy(
                            update={
                                "retry_plan": {
                                    **plan.retry_plan,
                                    "tool_name": generated_registered_name,
                                }
                            }
                        )
                        _mark_tool_prepared(tool_runtime, generated_registered_name)
                        repair_entry["repair_plan"] = plan.model_dump(mode="json")
                        trajectory_collector.record_event(
                            event_type="generated_tool_registered",
                            subject_type="generated_tool",
                            subject_ref=generated_registered_name,
                            task_id=task_id,
                            run_ref=run_ref,
                            parent_ref=plan.repair_id,
                            metadata={"registration": registration_json},
                        )
                    else:
                        trajectory_collector.record_event(
                            event_type="generated_tool_rejected",
                            subject_type="generated_tool",
                            subject_ref=registration.registered_tool_name,
                            task_id=task_id,
                            run_ref=run_ref,
                            parent_ref=plan.repair_id,
                            metadata={"registration": registration_json},
                        )
                except Exception as exc:
                    registration_json = {
                        "status": "rejected",
                        "failure_reason": str(exc),
                        "errors": [str(exc)],
                    }
                    repair_entry["generated_tool_registration"] = registration_json
                    trajectory_collector.record_event(
                        event_type="generated_tool_rejected",
                        subject_type="generated_tool",
                        subject_ref=getattr(package, "tool_name", None),
                        task_id=task_id,
                        run_ref=run_ref,
                        parent_ref=plan.repair_id,
                        metadata=registration_json,
                    )

        validation = self.validate_and_retry(
            plan=plan,
            tool_runtime=tool_runtime,
            failed_tool_call=tool_call,
        )
        repair_entry["validation_result"] = validation.model_dump(mode="json")
        trajectory_collector.record_event(
            event_type="repair_validated",
            subject_type="repair",
            subject_ref=plan.repair_id,
            task_id=task_id,
            run_ref=run_ref,
            metadata={"validation_result": validation.model_dump(mode="json")},
        )
        retry_record: ToolCallRecord | None = None
        promotion_candidates: list[dict[str, Any]] = []
        if validation.valid:
            retry_call = _retry_tool_call(plan, tool_call)
            retry_result = tool_runtime.execute_tool_name(
                call_id=retry_call.call_id,
                name=retry_call.name,
                arguments=retry_call.arguments,
            )
            retry_record = ToolCallRecord(tool_call=retry_call, result=retry_result)
            repair_entry["retry_result"] = retry_result.model_dump(mode="json")
            trajectory_collector.record_event(
                event_type="repair_retried",
                subject_type="repair",
                subject_ref=plan.repair_id,
                task_id=task_id,
                run_ref=run_ref,
                metadata={
                    "retry_tool_call": retry_call.model_dump(mode="json"),
                    "retry_result": retry_result.model_dump(mode="json"),
                },
            )
        should_emit_candidate = bool(plan.promotion_candidate_policy.get("emit_candidate"))
        if plan.repair_action == "create_runtime_tool" and generated_registered_name is None:
            should_emit_candidate = False
        if validation.valid and should_emit_candidate:
            if generated_registered_name is not None:
                candidate_type = "new_tool"
                target_id = generated_registered_name
                affected_ids = [generated_registered_name]
            else:
                candidate_type = "tool_patch" if plan.tool_overlay_patch is not None else "skill_patch"
                target_id = (
                    plan.tool_overlay_patch.base_tool_name
                    if plan.tool_overlay_patch is not None
                    else (plan.skill_overlay_patch.target_skill_id if plan.skill_overlay_patch is not None else None)
                )
                affected_ids = [
                    value
                    for value in [
                        plan.tool_overlay_patch.name if plan.tool_overlay_patch is not None else None,
                        plan.skill_overlay_patch.target_skill_id if plan.skill_overlay_patch is not None else None,
                    ]
                    if isinstance(value, str)
                ]
            promotion_candidate = PromotionCandidate(
                candidate_id=f"candidate-{uuid4()}",
                candidate_type=candidate_type,
                target_id=target_id,
                supporting_evidence=signal.evidence,
                validation_result=validation,
                affected_ids=affected_ids,
                recommended_decision="review",
                metadata={"repair_plan": plan.model_dump(mode="json")},
            )
            promotion_candidates.append(promotion_candidate.model_dump(mode="json"))
            trajectory_collector.record_event(
                event_type="repair_promotion_candidate",
                subject_type="repair",
                subject_ref=promotion_candidate.candidate_id,
                task_id=task_id,
                run_ref=run_ref,
                metadata={"promotion_candidate": promotion_candidate.model_dump(mode="json")},
            )
        updated_skill_context = build_skill_context(updated_skill_bundle)
        return RepairRuntimeOutcome(
            retry_record=retry_record,
            repair_messages=[
                Message(
                    role="user",
                    content=(
                        "Runtime repair applied. Distinguish source-file line coordinates from table-relative "
                        "row indexes and prefer read_table_slice_safe for recoverable caption blocks."
                    ),
                )
            ],
            updated_skill_bundle=updated_skill_bundle,
            updated_skill_context=updated_skill_context,
            repair_entry=repair_entry,
            promotion_candidates=promotion_candidates,
        )

    def validate_and_retry(self, *args, **kwargs) -> RepairValidationResult:
        return self.validator.validate_repair_plan(*args, **kwargs)


def _skill_matches_patch(skill: SkillRef, patch: SkillOverlayPatch) -> bool:
    if patch.target_skill_id and skill.skill_id == patch.target_skill_id:
        return True
    if patch.target_skill_name and skill.name == patch.target_skill_name:
        return True
    return False


def _patched_skill(skill: SkillRef, patch: SkillOverlayPatch) -> SkillRef:
    content = "\n".join(
        [
            skill.content,
            "",
            "Runtime repair overlay:",
            *[f"- Principle: {item}" for item in patch.principles],
            *[f"- Failure mode: {item}" for item in patch.failure_modes],
            *[f"- Recovery: {item}" for item in patch.recovery_strategies],
            *[f"- Tool policy: {item}" for item in patch.tool_use_policies],
        ]
    ).strip()
    metadata = dict(skill.metadata)
    metadata.setdefault("runtime_skill_overlay_patches", []).append(patch.patch_id)
    return skill.model_copy(update={"content": content, "metadata": metadata})


def _skill_overlay_content(patch: SkillOverlayPatch) -> str:
    lines = [
        "Runtime repair overlay:",
        *[f"- Principle: {item}" for item in patch.principles],
        *[f"- Failure mode: {item}" for item in patch.failure_modes],
        *[f"- Recovery: {item}" for item in patch.recovery_strategies],
        *[f"- Tool policy: {item}" for item in patch.tool_use_policies],
    ]
    return "\n".join(lines)


def _tool_result_summary(result: ToolResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "row_count": _row_count(result),
        "warnings": result.metadata.get("warnings", []),
        "error_type": result.metadata.get("error_type"),
    }


def _retry_tool_call(plan: RepairPlan, failed_tool_call: ToolCall) -> ToolCall:
    tool_name = plan.retry_plan.get("tool_name", failed_tool_call.name)
    arguments = dict(failed_tool_call.arguments)
    if plan.retry_plan.get("normalize_descending_bounds"):
        start_row = _int_or_none(arguments.get("start_row"))
        end_row = _int_or_none(arguments.get("end_row"))
        if start_row is not None and end_row is not None and start_row > end_row:
            arguments["start_row"], arguments["end_row"] = end_row, start_row
    return ToolCall(
        call_id=f"{failed_tool_call.call_id}-repair-1",
        name=tool_name,
        arguments=arguments,
    )


def _mark_tool_prepared(tool_runtime: Any, registered_name: str) -> None:
    mark_tool_prepared = getattr(tool_runtime, "mark_tool_prepared", None)
    if callable(mark_tool_prepared):
        mark_tool_prepared(registered_name)


def _tool_specs_for_runtime(tool_runtime: Any) -> list[ToolSpec]:
    prepared_tool_specs = getattr(tool_runtime, "prepared_tool_specs", None)
    if callable(prepared_tool_specs):
        generated_names = set(_generated_tool_names_for_runtime(tool_runtime))
        return [
            spec
            for spec in prepared_tool_specs()
            if isinstance(spec, ToolSpec) and spec.name not in generated_names
        ]
    specs: list[ToolSpec] = []
    prepared_names = getattr(tool_runtime, "_prepared_tool_names", None)
    if isinstance(prepared_names, set):
        for name in sorted(str(item) for item in prepared_names):
            effective_spec = getattr(tool_runtime, "_get_effective_spec", None)
            spec = effective_spec(name) if callable(effective_spec) else None
            if isinstance(spec, ToolSpec):
                specs.append(spec)
        generated_names = set(_generated_tool_names_for_runtime(tool_runtime))
        return [spec for spec in specs if spec.name not in generated_names]
    registry = getattr(tool_runtime, "_registry", None)
    registry_specs = getattr(registry, "_specs", None)
    if isinstance(registry_specs, dict):
        return [spec for spec in registry_specs.values() if isinstance(spec, ToolSpec)]
    return specs


def _generated_tool_specs_for_runtime(tool_runtime: Any) -> list[ToolSpec]:
    generated_tool_specs = getattr(tool_runtime, "generated_tool_specs", None)
    if callable(generated_tool_specs):
        return [spec for spec in generated_tool_specs() if isinstance(spec, ToolSpec)]
    generated_specs = getattr(tool_runtime, "_generated_specs", None)
    if not isinstance(generated_specs, dict):
        return []
    return [spec for spec in generated_specs.values() if isinstance(spec, ToolSpec)]


def _generated_tool_names_for_runtime(tool_runtime: Any) -> list[str]:
    generated_tool_names = getattr(tool_runtime, "generated_tool_names", None)
    if callable(generated_tool_names):
        return [str(name) for name in generated_tool_names()]
    generated_specs = getattr(tool_runtime, "_generated_specs", None)
    if not isinstance(generated_specs, dict):
        return []
    return [str(name) for name in generated_specs]


def _row_count(result: ToolResult) -> int:
    if isinstance(result.metadata.get("row_count"), int):
        return int(result.metadata["row_count"])
    rows = result.metadata.get("rows")
    if isinstance(rows, list):
        return len(rows)
    return 0


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _looks_like_coordinate_mismatch(
    *,
    start_row: int | None,
    end_row: int | None,
    row_count: int | None,
    block: Any,
    rows: Any,
) -> bool:
    if start_row is None or end_row is None or not isinstance(rows, list) or rows:
        return False
    if not isinstance(block, dict):
        return False
    source_start = block.get("start_line")
    source_end = block.get("end_line")
    if not isinstance(source_start, int) or not isinstance(source_end, int):
        return False
    if row_count is not None and row_count > 0 and start_row < row_count and end_row <= row_count:
        return False
    return not (end_row <= source_start or start_row > source_end)


def _skill_mentions_table_coordinate_policy(skill_bundle: SkillBundle) -> bool:
    for skill in skill_bundle.skills:
        content = skill.content.casefold()
        if "source-file line" in content or "table-relative" in content or "coordinate" in content:
            return True
    return False


def _is_retryable_api_failure(tool_result: ToolResult) -> bool:
    content = tool_result.content.casefold()
    return any(token in content for token in ("429", "502", "503", "504", "timeout", "bad gateway"))


def _looks_like_schema_mismatch(tool_result: ToolResult) -> bool:
    if tool_result.metadata.get("error_type") == "schema_validation_failed":
        return True
    return "schema" in tool_result.content.casefold()
