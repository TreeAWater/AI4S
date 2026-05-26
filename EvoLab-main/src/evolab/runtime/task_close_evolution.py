from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from evolab.contracts.evolution import LLMEvolutionMode, LLMEvolutionRequest
from evolab.contracts.records import SubagentRunRecord
from evolab.contracts.task import ProposerInputRef, TaskRequest
from evolab.lab.layout import LabLayout
from evolab.lab.queue import FileWorkQueue
from evolab.registries.backend_state import BackendStateRegistry
from evolab.registries.trajectory import TrajectoryRegistry


class TaskCloseEvolutionScheduler:
    def __init__(
        self,
        *,
        layout: LabLayout,
        evolve_queue: FileWorkQueue,
        trajectory_registry: TrajectoryRegistry,
        backend_state_registry: BackendStateRegistry,
        worker_id: str,
        evolution_backend_ids: set[str] | None = None,
    ) -> None:
        self.layout = layout
        self.evolve_queue = evolve_queue
        self.trajectory_registry = trajectory_registry
        self.backend_state_registry = backend_state_registry
        self.worker_id = worker_id
        self.evolution_backend_ids = evolution_backend_ids

    def run(
        self,
        request: TaskRequest,
        task_result: Mapping[str, Any],
    ) -> list[str]:
        run_refs: list[str] = []
        for subagent_run in self._subagent_runs_for_task_close(request, task_result):
            if (
                self.evolution_backend_ids is not None
                and subagent_run.llm_backend_id not in self.evolution_backend_ids
            ):
                continue
            run_refs.append(self._schedule_subagent_evolution(request, subagent_run, task_result))
        return run_refs

    def _subagent_runs_for_task_close(
        self,
        request: TaskRequest,
        task_result: Mapping[str, Any],
    ) -> list[SubagentRunRecord]:
        run_refs = _run_refs_from_result(task_result)
        if run_refs:
            requested = set(run_refs)
            records_by_ref = {
                record.run_ref: record
                for record in self.trajectory_registry.query_subagent_runs({"task_id": request.task_id})
                if record.run_ref in requested
            }
            return [records_by_ref[run_ref] for run_ref in run_refs if run_ref in records_by_ref]
        return self.trajectory_registry.query_subagent_runs({"task_id": request.task_id})

    def _schedule_subagent_evolution(
        self,
        request: TaskRequest,
        subagent_run: SubagentRunRecord,
        task_result: Mapping[str, Any],
    ) -> str:
        run_ref = f"evo-{uuid4()}"
        evolution_request = self._build_request(request, subagent_run, run_ref, task_result)
        request_path = self._write_request(run_ref, evolution_request)
        self.evolve_queue.enqueue(
            run_ref,
            {
                "request_payload_uri": str(request_path),
                "task_id": request.task_id,
                "backend_id": evolution_request.backend_id,
                "trigger_trajectory_ref": subagent_run.run_ref,
            },
        )
        return run_ref

    def _build_request(
        self,
        task_request: TaskRequest,
        subagent_run: SubagentRunRecord,
        run_ref: str,
        task_result: Mapping[str, Any],
    ) -> LLMEvolutionRequest:
        previous_state_ref = (
            subagent_run.llm_backend_state_ref
            or self.backend_state_registry.resolve_active_state(
                subagent_run.llm_backend_id,
                role=subagent_run.role,
            )
        )
        reflector_feedback = _reflector_feedback_for_evolution(task_result)
        proposer_input_refs = [
            ProposerInputRef(
                ref_type="trajectory",
                ref_id=subagent_run.run_ref,
                role=subagent_run.role,
                summary=_summary_from_subagent_run(subagent_run),
                metadata={"task_id": task_request.task_id},
            )
        ]
        if reflector_feedback is not None:
            proposer_input_refs.append(
                ProposerInputRef(
                    ref_type="reflector_feedback",
                    ref_id=f"{task_request.task_id}:reflector",
                    role="reflector",
                    summary=_reflector_feedback_summary(reflector_feedback),
                    metadata={
                        "task_id": task_request.task_id,
                        "reflector_feedback": reflector_feedback,
                    },
                )
            )
        return LLMEvolutionRequest(
            mode=LLMEvolutionMode.BASICS,
            backend_id=subagent_run.llm_backend_id,
            previous_state_ref=previous_state_ref,
            artifact_root_uri=str(self.layout.evolution_run_dir(run_ref)),
            trigger_trajectory_ref=subagent_run.run_ref,
            proposer_input_refs=proposer_input_refs,
            metadata={
                "task_id": task_request.task_id,
                "role": subagent_run.role,
                "source_worker_id": self.worker_id,
                "trigger": "task_close",
                **({"reflector_feedback": reflector_feedback} if reflector_feedback is not None else {}),
            },
        )

    def _write_request(self, run_ref: str, request: LLMEvolutionRequest) -> Path:
        path = self.layout.evolution_run_dir(run_ref) / "request.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(request.model_dump_json(), encoding="utf-8")
        os.replace(tmp, path)
        return path


def _run_refs_from_result(task_result: Mapping[str, Any]) -> list[str]:
    raw_run_refs = task_result.get("run_refs")
    if isinstance(raw_run_refs, list):
        return [value for value in raw_run_refs if isinstance(value, str)]
    raw_run_ref = task_result.get("run_ref")
    if isinstance(raw_run_ref, str):
        return [raw_run_ref]
    return []


def _summary_from_subagent_run(subagent_run: SubagentRunRecord) -> str | None:
    if not subagent_run.output_messages:
        return None
    content = subagent_run.output_messages[-1].content.strip()
    if not content:
        return None
    return content[:240]


def _reflector_feedback_for_evolution(task_result: Mapping[str, Any]) -> dict[str, Any] | None:
    evaluation = task_result.get("reflector_evaluation")
    if not isinstance(evaluation, Mapping):
        return None
    feedback: dict[str, Any] = {}
    for key in (
        "score",
        "passed",
        "metrics",
        "metric_source",
        "summary",
        "errors",
        "credit_assignment",
        "evolution_recommendations",
        "specific_evolution_instructions",
    ):
        value = evaluation.get(key)
        if value is not None:
            feedback[key] = _sanitize_reflector_value(value)
    sequence_error_analysis = evaluation.get("sequence_error_analysis")
    if isinstance(sequence_error_analysis, Mapping):
        feedback["sequence_error_analysis"] = _sanitized_sequence_error_analysis(sequence_error_analysis)
    if not feedback:
        return None
    feedback["ground_truth_policy"] = (
        "Post-run evaluator feedback may guide reusable skill or prompt changes, but raw ground-truth "
        "answers are not included in evolution requests and must not be memorized into extraction prompts."
    )
    return feedback


def _sanitized_sequence_error_analysis(sequence_error_analysis: Mapping[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key in (
        "matched_example_count",
        "false_positive_example_count",
        "false_negative_example_count",
        "example_limits",
    ):
        value = sequence_error_analysis.get(key)
        if value is not None:
            sanitized[key] = _sanitize_reflector_value(value)
    false_positive_examples = sequence_error_analysis.get("false_positive_examples")
    if isinstance(false_positive_examples, list):
        sanitized["false_positive_examples"] = [
            _sanitize_reflector_value(example) for example in false_positive_examples[:10]
        ]
    false_negative_examples = sequence_error_analysis.get("false_negative_examples")
    if isinstance(false_negative_examples, list):
        sanitized["false_negative_example_count_available"] = len(false_negative_examples)
        sanitized["false_negative_examples_redacted"] = True
    return sanitized


def _sanitize_reflector_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.casefold() in {"sequence", "ground_truth_sequence", "gt_sequence"}:
                sanitized[key_text] = "[redacted]"
                continue
            sanitized[key_text] = _sanitize_reflector_value(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_reflector_value(item) for item in value]
    if isinstance(value, str):
        return value[:2_000]
    return value


def _reflector_feedback_summary(feedback: Mapping[str, Any]) -> str:
    metrics = feedback.get("metrics")
    summary = feedback.get("summary")
    pieces: list[str] = []
    if isinstance(metrics, Mapping):
        precision = metrics.get("precision")
        recall = metrics.get("recall")
        f1 = metrics.get("f1")
        pieces.append(f"metrics precision={precision} recall={recall} f1={f1}")
    if isinstance(summary, str) and summary:
        pieces.append(summary[:240])
    instructions = feedback.get("specific_evolution_instructions")
    if isinstance(instructions, list) and instructions:
        pieces.append(f"{len(instructions)} specific evolution instruction(s)")
    return "; ".join(pieces)[:500] or "post-run reflector feedback"
