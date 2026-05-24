from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from evolab.contracts.lab_state import (
    LabStateDigest,
    LabStateIndex,
    LabStateIndexTaskSummary,
    LabStateTrajectorySummary,
)
from evolab.contracts.task import TaskRequest
from evolab.registries.backend_state import BackendStateRegistry
from evolab.registries.lab_state import FileLabStateRegistry
from evolab.registries.task import FileTaskRegistry
from evolab.registries.trajectory import FileTrajectoryRegistry


class LabStateBuilder:
    def __init__(
        self,
        *,
        task_registry: FileTaskRegistry | None = None,
        trajectory_registry: FileTrajectoryRegistry | None = None,
        backend_state_registry: BackendStateRegistry | None = None,
        lab_state_registry: FileLabStateRegistry | None = None,
    ) -> None:
        self.task_registry = task_registry
        self.trajectory_registry = trajectory_registry
        self.backend_state_registry = backend_state_registry
        self.lab_state_registry = lab_state_registry

    def build_for_meta_agent(
        self,
        *,
        request: TaskRequest,
        requested_detail_refs: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        requested_detail_refs = requested_detail_refs or {}
        index = self.build_index(request)
        digest = self.build_digest(index)
        if self.lab_state_registry is not None:
            self.lab_state_registry.save_index(index)
            self.lab_state_registry.save_digest(digest)
        return {
            "index": index.model_dump(mode="json"),
            "digest": digest.model_dump(mode="json"),
            "requested_details": self.resolve_details(requested_detail_refs, task_id=request.task_id),
        }

    def build_index(self, request: TaskRequest) -> LabStateIndex:
        reports = self.lab_state_registry.list_subagent_reports(request.task_id) if self.lab_state_registry else []
        artifacts = self.lab_state_registry.list_artifacts(request.task_id) if self.lab_state_registry else []
        training_samples = self.lab_state_registry.list_training_samples(request.task_id) if self.lab_state_registry else []
        backend_states = self._backend_states_for_task(request.task_id)
        return LabStateIndex(
            index_ref=f"lab-index-{uuid4()}",
            task=LabStateIndexTaskSummary(
                task_id=request.task_id,
                task_goal=request.goal,
                task_request_ref=self._task_request_ref(request.task_id),
                status=self._task_status(request.task_id),
            ),
            trajectory=self._trajectory_summary(request.task_id),
            subagent_reports=[
                {
                    "report_ref": report.report_ref,
                    "run_ref": report.run_ref,
                    "role": report.role,
                    "status": report.status,
                    "summary": _shorten(report.summary),
                    "artifact_count": len(report.artifact_refs),
                    "created_at": report.created_at.isoformat(),
                    "completion_contract": report.metadata.get("completion_contract"),
                }
                for report in reports[-20:]
            ],
            artifacts=[
                {
                    "artifact_ref": artifact.artifact_ref,
                    "uri": artifact.uri,
                    "artifact_type": artifact.artifact_type,
                    "role": artifact.role,
                    "status": artifact.status,
                    "producer_run_ref": artifact.producer_run_ref,
                }
                for artifact in artifacts[-50:]
            ],
            backend_states=[
                {
                    "state_ref": state.state_ref,
                    "backend_id": state.backend_id,
                    "backend_type": state.backend_type,
                    "role": state.metadata.get("role"),
                    "parent_state_refs": state.parent_state_refs,
                }
                for state in backend_states[-50:]
            ],
            training_samples=[
                {
                    "sample_ref": sample.sample_ref,
                    "source_run_ref": sample.source_run_ref,
                    "sample_kind": sample.sample_kind,
                    "quality_label": sample.quality_label,
                    "llm_call_count": len(sample.source_llm_call_refs),
                }
                for sample in training_samples[-50:]
            ],
            detail_refs={
                "subagent_reports": [report.report_ref for report in reports[-20:]],
                "artifacts": [artifact.artifact_ref for artifact in artifacts[-50:]],
                "training_samples": [sample.sample_ref for sample in training_samples[-50:]],
                "backend_states": [state.state_ref for state in backend_states[-50:]],
            },
            metadata={
                "built_at": datetime.utcnow().isoformat(),
                "progressive_disclosure": True,
            },
        )

    def build_digest(self, index: LabStateIndex) -> LabStateDigest:
        trajectory = index.trajectory
        summary = (
            f"Lab contains {len(index.subagent_reports)} subagent report(s), "
            f"{len(index.artifacts)} artifact index item(s), "
            f"{trajectory.llm_call_count} LLM call(s), "
            f"{trajectory.tool_call_count} tool call(s), and "
            f"{len(index.backend_states)} backend state record(s) for task {index.task.task_id}."
        )
        return LabStateDigest(
            digest_ref=f"lab-digest-{uuid4()}",
            index_ref=index.index_ref,
            task_id=index.task.task_id,
            summary=summary,
            sections={
                "recent_subagent_reports": index.subagent_reports[-5:],
                "recent_artifacts": index.artifacts[-10:],
                "trajectory": trajectory.model_dump(mode="json"),
            },
            detail_refs=index.detail_refs,
            metadata={"progressive_disclosure": True},
        )

    def resolve_details(self, requested_detail_refs: dict[str, list[str]], task_id: str | None = None) -> dict[str, Any]:
        if not requested_detail_refs or self.lab_state_registry is None:
            return {}
        details: dict[str, Any] = {}
        reports = {report.report_ref: report for report in self.lab_state_registry.list_subagent_reports(task_id)}
        artifacts = {artifact.artifact_ref: artifact for artifact in self.lab_state_registry.list_artifacts(task_id)}
        training = {sample.sample_ref: sample for sample in self.lab_state_registry.list_training_samples(task_id)}
        if requested_detail_refs.get("subagent_reports"):
            details["subagent_reports"] = [
                reports[ref].model_dump(mode="json")
                for ref in requested_detail_refs["subagent_reports"]
                if ref in reports
            ]
        if requested_detail_refs.get("artifacts"):
            details["artifacts"] = [
                artifacts[ref].model_dump(mode="json")
                for ref in requested_detail_refs["artifacts"]
                if ref in artifacts
            ]
        if requested_detail_refs.get("training_samples"):
            details["training_samples"] = [
                training[ref].model_dump(mode="json")
                for ref in requested_detail_refs["training_samples"]
                if ref in training
            ]
        if requested_detail_refs.get("backend_states") and self.backend_state_registry is not None:
            backend_states = {state.state_ref: state for state in self._backend_states_for_task(task_id)}
            details["backend_states"] = [
                state.model_dump(mode="json")
                for ref in requested_detail_refs["backend_states"]
                if (state := backend_states.get(ref)) is not None
            ]
        return details

    def _backend_states_for_task(self, task_id: str | None) -> list[Any]:
        if self.backend_state_registry is None:
            return []
        records = self.backend_state_registry.list_states()
        if task_id is None:
            return records
        return [record for record in records if _record_matches_task(record, task_id)]

    def _trajectory_summary(self, task_id: str) -> LabStateTrajectorySummary:
        if self.trajectory_registry is None:
            return LabStateTrajectorySummary()
        meta_agent_runs = _task_records(self.trajectory_registry.list_meta_agent_runs(), task_id)
        subagent_runs = _task_records(self.trajectory_registry.list_subagent_runs(), task_id)
        run_refs = {record.run_ref for record in [*meta_agent_runs, *subagent_runs]}
        llm_calls = [
            record
            for record in self.trajectory_registry.list_llm_calls()
            if _record_matches_task(record, task_id) or record.run_ref in run_refs
        ]
        tool_calls = _task_records(self.trajectory_registry.list_tool_call_records(), task_id)
        events = _task_records(self.trajectory_registry.list_events(), task_id)
        evolution_runs = _task_records(self.trajectory_registry.list_evolution_runs(), task_id)
        return LabStateTrajectorySummary(
            meta_agent_run_count=len(meta_agent_runs),
            subagent_run_count=len(subagent_runs),
            llm_call_count=len(llm_calls),
            tool_call_count=len(tool_calls),
            event_count=len(events),
            evolution_run_count=len(evolution_runs),
        )

    def _task_request_ref(self, task_id: str) -> str | None:
        if self.task_registry is None:
            return None
        path = self.task_registry.root / f"{task_id}.json"
        return str(path) if path.exists() else None

    def _task_status(self, task_id: str) -> str | None:
        if self.lab_state_registry is None:
            return None
        ledger = self.lab_state_registry.get_run_ledger(task_id)
        return ledger.status if ledger is not None else None


def _shorten(value: str, limit: int = 240) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _record_matches_task(record: Any, task_id: str) -> bool:
    record_task_id = getattr(record, "task_id", None)
    if record_task_id == task_id:
        return True
    created_from_task_id = getattr(record, "created_from_task_id", None)
    if created_from_task_id == task_id:
        return True
    metadata = getattr(record, "metadata", None)
    return isinstance(metadata, dict) and metadata.get("task_id") == task_id


def _task_records(records: list[Any], task_id: str) -> list[Any]:
    return [record for record in records if _record_matches_task(record, task_id)]
