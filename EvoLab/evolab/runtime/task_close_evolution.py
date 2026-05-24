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
            run_refs.append(self._schedule_subagent_evolution(request, subagent_run))
        return run_refs

    def _subagent_runs_for_task_close(
        self,
        request: TaskRequest,
        task_result: Mapping[str, Any],
    ) -> list[SubagentRunRecord]:
        run_refs = _run_refs_from_result(task_result)
        if run_refs:
            records = []
            for run_ref in run_refs:
                record = self.trajectory_registry.get_subagent_run(run_ref)
                if record is not None and record.task_id == request.task_id:
                    records.append(record)
            return records
        return self.trajectory_registry.query_subagent_runs({"task_id": request.task_id})

    def _schedule_subagent_evolution(
        self,
        request: TaskRequest,
        subagent_run: SubagentRunRecord,
    ) -> str:
        run_ref = f"evo-{uuid4()}"
        evolution_request = self._build_request(request, subagent_run, run_ref)
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
    ) -> LLMEvolutionRequest:
        previous_state_ref = (
            subagent_run.llm_backend_state_ref
            or self.backend_state_registry.resolve_active_state(
                subagent_run.llm_backend_id,
                role=subagent_run.role,
            )
        )
        return LLMEvolutionRequest(
            mode=LLMEvolutionMode.BASICS,
            backend_id=subagent_run.llm_backend_id,
            previous_state_ref=previous_state_ref,
            artifact_root_uri=str(self.layout.evolution_run_dir(run_ref)),
            trigger_trajectory_ref=subagent_run.run_ref,
            proposer_input_refs=[
                ProposerInputRef(
                    ref_type="trajectory",
                    ref_id=subagent_run.run_ref,
                    role=subagent_run.role,
                    summary=_summary_from_subagent_run(subagent_run),
                    metadata={"task_id": task_request.task_id},
                )
            ],
            metadata={
                "task_id": task_request.task_id,
                "role": subagent_run.role,
                "source_worker_id": self.worker_id,
                "trigger": "task_close",
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
