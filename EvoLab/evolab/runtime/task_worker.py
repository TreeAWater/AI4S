from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from collections.abc import Callable
from typing import Any
from urllib.parse import unquote, urlparse

from evolab.config.task_config import BackendBinding, TaskConfig
from evolab.contracts.common import ArtifactRef, Message
from evolab.contracts.lab_state import ArtifactIndexRecord, RunLedgerRecord, SubagentReportRecord
from evolab.contracts.records import SubagentRunRecord
from evolab.contracts.retrieval import MemoryBundle, RetrievalRequest, SkillBundle
from evolab.contracts.task import TaskOrigin, TaskPurpose, TaskRequest
from evolab.lab.layout import LabLayout
from evolab.lab.queue import ClaimedJob, FileWorkQueue
from evolab.lab.resolver import LabResolver
from evolab.registries.backend_state import FileBackendStateRegistry
from evolab.registries.lab_state import FileLabStateRegistry
from evolab.registries.task import FileTaskRegistry
from evolab.registries.trajectory import FileTrajectoryRegistry
from evolab.runtime.prompt_builder import PromptBuilder
from evolab.runtime.task_close_evolution import TaskCloseEvolutionScheduler
from evolab.runtime.task_runtime import TaskRuntime
from evolab.runtime.trajectory_collector import TrajectoryCollector
from evolab.tools.runtime import ToolRegistry, ToolRuntime


class TaskWorker:
    def __init__(
        self,
        layout: LabLayout,
        worker_id: str,
        *,
        task_config: TaskConfig | None = None,
        task_queue: FileWorkQueue | None = None,
        evolve_queue: FileWorkQueue | None = None,
        task_runtime: Any | None = None,
        task_registry: FileTaskRegistry | None = None,
        backend_state_registry: FileBackendStateRegistry | None = None,
        lab_state_registry: FileLabStateRegistry | None = None,
        trajectory_registry: FileTrajectoryRegistry | None = None,
        tool_registry: ToolRegistry | None = None,
        tool_runtime: ToolRuntime | None = None,
        prompt_builder: PromptBuilder | None = None,
        llm_backends: dict[str, Any] | None = None,
        embedding_backends: dict[str, Any] | None = None,
        memory_backends: dict[str, Any] | None = None,
        skill_backends: dict[str, Any] | None = None,
        llm_backend_bindings: list[BackendBinding] | None = None,
        memory_backend_bindings: list[BackendBinding] | None = None,
        skill_backend_bindings: list[BackendBinding] | None = None,
        llm_runtimes: dict[str, Any] | None = None,
        embedding_runtimes: dict[str, Any] | None = None,
        memory_runtimes: dict[str, Any] | None = None,
        skill_runtimes: dict[str, Any] | None = None,
        evolution_backends: dict[str, Any] | None = None,
        evolution_backend_ids: set[str] | None = None,
        task_close_evolution_scheduler: TaskCloseEvolutionScheduler | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.layout = layout
        self.worker_id = worker_id
        self.task_config = task_config

        self.task_queue = task_queue
        self.evolve_queue = evolve_queue
        self.task_runtime = task_runtime
        self.task_registry = task_registry
        self.backend_state_registry = backend_state_registry
        self.lab_state_registry = lab_state_registry
        self.trajectory_registry = trajectory_registry
        self.tool_registry = tool_registry
        self.tool_runtime = tool_runtime
        self.prompt_builder = prompt_builder

        self.llm_backends = llm_backends or {}
        self.embedding_backends = embedding_backends or {}
        self.memory_backends = memory_backends or {}
        self.skill_backends = skill_backends or {}
        self.llm_backend_bindings = llm_backend_bindings
        self.memory_backend_bindings = memory_backend_bindings
        self.skill_backend_bindings = skill_backend_bindings
        self.llm_runtimes = llm_runtimes or {}
        self.embedding_runtimes = embedding_runtimes or {}
        self.memory_runtimes = memory_runtimes or {}
        self.skill_runtimes = skill_runtimes or {}
        self.evolution_backends = evolution_backends or {}
        self.evolution_backend_ids = evolution_backend_ids
        self.task_close_evolution_scheduler = task_close_evolution_scheduler
        self.trajectory_collector = TrajectoryCollector(self.trajectory_registry)
        self.progress_callback = progress_callback

    def startup(self) -> None:
        self.layout.ensure()
        resolver = LabResolver(self.layout)

        if self.task_queue is None:
            self.task_queue = resolver.task_queue()
        self.task_queue.ensure()
        if self.evolve_queue is None:
            self.evolve_queue = resolver.evolve_queue()
        self.evolve_queue.ensure()

        if self.task_registry is None:
            self.task_registry = resolver.task_registry()
        if self.backend_state_registry is None:
            self.backend_state_registry = resolver.backend_state_registry()
        if self.lab_state_registry is None:
            self.lab_state_registry = resolver.lab_state_registry()
        if self.trajectory_registry is None:
            self.trajectory_registry = resolver.trajectory_registry()
        self.trajectory_collector = TrajectoryCollector(self.trajectory_registry)

        if self.tool_registry is None:
            self.tool_registry = ToolRegistry()
        if self.tool_runtime is None:
            self.tool_runtime = ToolRuntime(self.tool_registry)
        if self.prompt_builder is None:
            self.prompt_builder = PromptBuilder()

        llm_backend_bindings = self.llm_backend_bindings
        if llm_backend_bindings is None and self.task_config is not None:
            llm_backend_bindings = _task_config_llm_bindings(self.task_config)
        self._initialize_backend_map(
            self.llm_backends,
            llm_backend_bindings or [],
            self.llm_runtimes,
            use_active_state=True,
        )
        self._initialize_required_backend_ids(
            self.llm_backends,
            _memory_llm_backend_ids(self.memory_backends.values()),
            self.llm_runtimes,
            use_active_state=True,
        )
        self._initialize_backend_map(
            self.embedding_backends,
            None,
            self.embedding_runtimes,
        )
        for backend in self.memory_backends.values():
            bind = getattr(backend, "bind_runtimes", None)
            if bind is not None:
                bind(llm_runtimes=self.llm_runtimes, embedding_runtimes=self.embedding_runtimes)
        self._initialize_backend_map(
            self.memory_backends,
            self.memory_backend_bindings,
            self.memory_runtimes,
        )
        self._initialize_backend_map(
            self.skill_backends,
            self.skill_backend_bindings,
            self.skill_runtimes,
        )

        if self.task_runtime is None:
            self.task_runtime = TaskRuntime(
                task_config=self.task_config,
                prompt_builder=self.prompt_builder,
                tool_runtime=self.tool_runtime,
                task_registry=self.task_registry,
                trajectory_registry=self.trajectory_registry,
                backend_state_registry=self.backend_state_registry,
                lab_state_registry=self.lab_state_registry,
                tool_artifact_root_factory=lambda request, run_ref: self.layout.task_dir(request.task_id)
                / "artifacts"
                / run_ref,
                llm_runtimes=self.llm_runtimes,
                memory_runtimes=self.memory_runtimes,
                skill_runtimes=self.skill_runtimes,
                trajectory_collector=self.trajectory_collector,
                progress_callback=self.progress_callback,
            )

        evolution_backend_ids = self.evolution_backend_ids
        if evolution_backend_ids is None and self.evolution_backends:
            evolution_backend_ids = set(self.evolution_backends)
        if self.task_close_evolution_scheduler is None and evolution_backend_ids:
            self.task_close_evolution_scheduler = TaskCloseEvolutionScheduler(
                layout=self.layout,
                evolve_queue=self.evolve_queue,
                trajectory_registry=self.trajectory_registry,
                backend_state_registry=self.backend_state_registry,
                worker_id=self.worker_id,
                evolution_backend_ids=evolution_backend_ids,
            )

    def run_once(self) -> dict[str, Any] | None:
        if self.task_queue is None or self.task_runtime is None:
            raise RuntimeError("TaskWorker.startup() must be called before run_once()")

        job = self.task_queue.claim(self.worker_id)
        if job is None:
            return None

        request: TaskRequest | None = None
        try:
            request = self._load_request(job)
            self._progress(f"task started: {request.task_id}")
            self._save_run_ledger(
                request,
                status="running",
                metadata={
                    "worker_id": self.worker_id,
                    "queue_path": str(job.path),
                    "claimed_at": job.payload.get("claimed_at"),
                    "claimed_by": job.payload.get("claimed_by"),
                },
            )
            self.trajectory_collector.record_event(
                event_type="task_started",
                subject_type="task",
                subject_ref=job.job_id,
                task_id=request.task_id,
                metadata={
                    "worker_id": self.worker_id,
                    "queue_path": str(job.path),
                    "claimed_at": job.payload.get("claimed_at"),
                    "claimed_by": job.payload.get("claimed_by"),
                },
            )
            result = self.task_runtime.run(request)
            self._report_result_progress(result)
            if self.task_close_evolution_scheduler is not None:
                evolution_run_refs = self.task_close_evolution_scheduler.run(request, result)
                if evolution_run_refs:
                    result = dict(result)
                    result["evolution_run_refs"] = evolution_run_refs
            terminal_status = _terminal_status_from_result(result)
            event_type = {
                "completed": "task_completed",
                "failed": "task_failed",
                "interrupted": "task_interrupted",
            }.get(terminal_status, "task_completed")
            final_artifacts = self._write_final_artifact_index(
                request=request,
                status=terminal_status,
                result=result,
                failure_reason=result.get("failure_reason") if isinstance(result.get("failure_reason"), str) else None,
            )
            self._reconcile_work_item_lifecycles(
                request,
                terminal_status=terminal_status,
                failure_reason=result.get("failure_reason") if isinstance(result.get("failure_reason"), str) else None,
            )
            self.trajectory_collector.record_event(
                event_type=event_type,
                subject_type="task",
                subject_ref=job.job_id,
                task_id=request.task_id,
                metadata={"worker_id": self.worker_id, "result": result},
            )
            self._save_run_ledger(
                request,
                status=terminal_status,
                result=result,
                final_artifact_refs=final_artifacts,
                failure_reason=result.get("failure_reason") if isinstance(result.get("failure_reason"), str) else None,
                metadata={"worker_id": self.worker_id},
            )
            if terminal_status == "failed":
                self.task_queue.mark_failed(job, result.get("failure_reason", "task runtime returned failed status"))
            elif terminal_status == "interrupted":
                self.task_queue.mark_interrupted(job, result.get("failure_reason", "task runtime returned interrupted status"))
            else:
                self.task_queue.mark_done(job)
            self._progress(f"task {terminal_status}: {request.task_id}")
            return result
        except KeyboardInterrupt as exc:
            reason = "KeyboardInterrupt"
            if request is not None:
                self._record_open_subagents_interrupted(request.task_id, reason)
            self.trajectory_collector.record_event(
                event_type="task_interrupted",
                subject_type="task",
                subject_ref=job.job_id,
                task_id=request.task_id if request is not None else None,
                metadata={"worker_id": self.worker_id, "error": reason},
            )
            if request is not None:
                final_artifacts = self._write_final_artifact_index(
                    request=request,
                    status="interrupted",
                    result=None,
                    failure_reason=reason,
                )
                self._save_run_ledger(
                    request,
                    status="interrupted",
                    failure_reason=reason,
                    final_artifact_refs=final_artifacts,
                    metadata={"worker_id": self.worker_id},
                )
                self._reconcile_work_item_lifecycles(
                    request,
                    terminal_status="interrupted",
                    failure_reason=reason,
                )
            self.task_queue.mark_interrupted(job, reason)
            self._progress(f"task interrupted: {request.task_id if request is not None else job.job_id}; {reason}")
            return None
        except Exception as exc:
            if request is not None:
                self._record_open_subagents_interrupted(request.task_id, str(exc))
            self.trajectory_collector.record_event(
                event_type="task_failed",
                subject_type="task",
                subject_ref=job.job_id,
                task_id=request.task_id if request is not None else None,
                metadata={"worker_id": self.worker_id, "error": str(exc)},
            )
            if request is not None:
                final_artifacts = self._write_final_artifact_index(
                    request=request,
                    status="failed",
                    result=None,
                    failure_reason=str(exc),
                )
                self._save_run_ledger(
                    request,
                    status="failed",
                    failure_reason=str(exc),
                    final_artifact_refs=final_artifacts,
                    metadata={"worker_id": self.worker_id},
                )
                self._reconcile_work_item_lifecycles(
                    request,
                    terminal_status="failed",
                    failure_reason=str(exc),
                )
            self.task_queue.mark_failed(job, str(exc))
            self._progress(f"task failed: {request.task_id if request is not None else job.job_id}; {exc}")
            return None

    def _progress(self, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(f"[EvoLab] {message}")

    def _report_result_progress(self, result: dict[str, Any]) -> None:
        for run in result.get("runs", []) if isinstance(result.get("runs"), list) else []:
            if not isinstance(run, dict):
                continue
            role = run.get("role") or run.get("generic_agent_type") or "unknown"
            status = run.get("status") or "unknown"
            artifact_count = len(run.get("artifact_refs", [])) if isinstance(run.get("artifact_refs"), list) else 0
            self._progress(f"subagent {role} {status}; artifacts={artifact_count}")

    def _load_request(self, job: ClaimedJob) -> TaskRequest:
        request_payload_uri = job.payload["request_payload_uri"]
        path = _local_path_from_uri(request_payload_uri)
        if path is None:
            raise ValueError(f"unsupported request_payload_uri {request_payload_uri!r}")
        return TaskRequest.model_validate_json(path.read_text(encoding="utf-8"))

    def _open_subagent_run_refs(self, task_id: str) -> list[str]:
        if self.trajectory_registry is None:
            return []
        started: list[str] = []
        closed: set[str] = set()
        for event in self.trajectory_registry.list_events():
            if event.task_id != task_id or event.subject_type != "subagent" or event.run_ref is None:
                continue
            if event.event_type == "subagent_started":
                started.append(event.run_ref)
            if event.event_type in {"subagent_completed", "subagent_failed", "subagent_interrupted"}:
                closed.add(event.run_ref)
        return [run_ref for run_ref in started if run_ref not in closed]

    def _record_open_subagents_interrupted(self, task_id: str, reason: str) -> None:
        open_run_refs = self._open_subagent_run_refs(task_id)
        for run_ref in open_run_refs:
            started_event = self._subagent_started_event(task_id, run_ref)
            metadata = started_event.metadata if started_event is not None else {}
            self.trajectory_collector.record_event(
                event_type="subagent_interrupted",
                subject_type="subagent",
                subject_ref=run_ref,
                task_id=task_id,
                run_ref=run_ref,
                metadata={"worker_id": self.worker_id, "error": reason},
            )
            if self.lab_state_registry is not None:
                role = str(metadata.get("role") or metadata.get("generic_agent_type") or "unknown")
                assigned_task = str(metadata.get("assigned_task") or "")
                self._save_interrupted_subagent_run(
                    task_id=task_id,
                    run_ref=run_ref,
                    role=role,
                    assigned_task=assigned_task,
                    reason=reason,
                    metadata=metadata,
                )
                self.lab_state_registry.save_subagent_report(
                    SubagentReportRecord(
                        report_ref=f"report-{run_ref}-interrupted",
                        task_id=task_id,
                        run_ref=run_ref,
                        role=role,
                        status="interrupted",
                        assigned_task=assigned_task,
                        summary=f"Subagent interrupted before normal completion: {reason}",
                        failures=[{"reason": reason}],
                        metadata={
                            "worker_id": self.worker_id,
                            "partial": True,
                            "stage_index": metadata.get("stage_index"),
                            "generic_agent_type": metadata.get("generic_agent_type"),
                            "meta_workflow_node_id": metadata.get("meta_workflow_node_id"),
                        },
                    )
                )

    def _save_interrupted_subagent_run(
        self,
        *,
        task_id: str,
        run_ref: str,
        role: str,
        assigned_task: str,
        reason: str,
        metadata: dict[str, Any],
    ) -> None:
        if self.trajectory_registry is None:
            return
        if self.trajectory_registry.get_subagent_run(run_ref) is not None:
            return
        request = self._request_for_task_id(task_id)
        if request is None:
            request = TaskRequest(
                task_id=task_id,
                origin=TaskOrigin.HUMAN,
                purpose=TaskPurpose.SCIENCE,
                goal=assigned_task or "Interrupted task.",
            )
        retrieval_request = RetrievalRequest(
            task_id=task_id,
            role=role,
            query=assigned_task or request.goal,
            task_origin=request.origin,
            task_purpose=request.purpose,
            filters={},
            metadata={"partial": True, "interrupted": True},
        )
        self.trajectory_registry.save_subagent_run(
            SubagentRunRecord(
                run_ref=run_ref,
                task_id=task_id,
                task_origin=request.origin,
                task_purpose=request.purpose,
                producer_ref=request.producer_ref,
                round_id=request.round_id,
                stage_index=int(metadata.get("stage_index") or 0),
                role=role,
                instruction=assigned_task,
                retrieval_request=retrieval_request,
                memory_bundle=MemoryBundle(backend_id="partial", items=[], metadata={"partial": True}),
                skill_bundle=SkillBundle(backend_id="partial", skills=[], metadata={"partial": True}),
                prompt_messages=[],
                llm_call_refs=[],
                llm_backend_id=str(metadata.get("llm_backend_id") or "unknown"),
                llm_backend_config_ref=metadata.get("llm_backend_config_ref")
                if isinstance(metadata.get("llm_backend_config_ref"), str)
                else None,
                llm_backend_state_ref=metadata.get("llm_backend_state_ref")
                if isinstance(metadata.get("llm_backend_state_ref"), str)
                else None,
                tool_calls=[],
                output_messages=[Message(role="assistant", content=f"Subagent interrupted before normal completion: {reason}")],
                artifact_refs=[],
                metadata={
                    "status": "interrupted",
                    "failure_reason": reason,
                    "partial": True,
                    "worker_id": self.worker_id,
                    "generic_agent_type": metadata.get("generic_agent_type") or role,
                    "meta_workflow_node_id": metadata.get("meta_workflow_node_id"),
                    "assigned_task": assigned_task,
                    "stage_index": metadata.get("stage_index"),
                    "interrupted_before_complete_record": True,
                },
            )
        )

    def _request_for_task_id(self, task_id: str) -> TaskRequest | None:
        request_path = self.layout.root / "requests" / f"{task_id}.json"
        if not request_path.exists():
            return None
        try:
            return TaskRequest.model_validate_json(request_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _subagent_started_event(self, task_id: str, run_ref: str):
        if self.trajectory_registry is None:
            return None
        for event in reversed(self.trajectory_registry.list_events()):
            if (
                event.task_id == task_id
                and event.run_ref == run_ref
                and event.subject_type == "subagent"
                and event.event_type == "subagent_started"
            ):
                return event
        return None

    def _write_final_artifact_index(
        self,
        *,
        request: TaskRequest,
        status: str,
        result: dict[str, Any] | None,
        failure_reason: str | None,
    ) -> list[ArtifactRef]:
        if self.lab_state_registry is None:
            return []
        entries = _final_artifact_index_entries(
            result=result,
            lab_artifacts=self.lab_state_registry.list_artifacts(request.task_id),
        )
        warnings = [] if entries else ["no artifacts were recorded for this task run"]
        payload = {
            "schema_version": "v1",
            "task_id": request.task_id,
            "status": status,
            "failure_reason": failure_reason,
            "created_at": datetime.utcnow().isoformat(),
            "warnings": warnings,
            "artifacts": entries,
        }
        self.lab_state_registry.save_final_artifact_index(request.task_id, payload)
        return [
            ArtifactRef(uri=entry["artifact_path"], type=entry["artifact_type"], metadata=entry["metadata"])
            for entry in entries
            if entry.get("is_final") is True
        ]

    def _reconcile_work_item_lifecycles(
        self,
        request: TaskRequest,
        *,
        terminal_status: str,
        failure_reason: str | None,
    ) -> None:
        if self.lab_state_registry is None:
            return
        root = self.lab_state_registry.root / "work_items" / _safe_state_ref(request.task_id)
        if not root.exists():
            return
        reconciled_status = {
            "completed": "completed",
            "interrupted": "interrupted",
        }.get(terminal_status, "failed")
        for path in sorted(root.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            current_status = payload.get("status")
            if current_status not in {"pending", "claimed", "running"}:
                continue
            history = payload.get("history")
            if not isinstance(history, list):
                history = []
            history.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "event": "task_terminal_reconcile",
                    "previous_status": current_status,
                    "status": reconciled_status,
                    "terminal_task_status": terminal_status,
                    "failure_reason": failure_reason,
                }
            )
            payload.update(
                {
                    "schema_version": "v1",
                    "task_id": request.task_id,
                    "status": reconciled_status,
                    "updated_at": datetime.utcnow().isoformat(),
                    "history": history,
                }
            )
            path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _save_run_ledger(
        self,
        request: TaskRequest,
        *,
        status: str,
        result: dict[str, Any] | None = None,
        failure_reason: str | None = None,
        metadata: dict[str, Any] | None = None,
        final_artifact_refs: list[ArtifactRef] | None = None,
    ) -> None:
        if self.lab_state_registry is None:
            return
        prior = self.lab_state_registry.get_run_ledger(request.task_id)
        result = result or {}
        metadata_payload = {**(prior.metadata if prior is not None else {}), **(metadata or {})}
        if result:
            metadata_payload["result_summary"] = _json_compatible(result)
        recorded_meta_run_refs = self._recorded_meta_run_refs(request.task_id)
        recorded_subagent_run_refs = self._recorded_subagent_run_refs(request.task_id)
        result_meta_run_refs = _string_list(result.get("meta_run_refs"))
        result_subagent_run_refs = _string_list(result.get("run_refs"))
        started_at = prior.started_at if prior is not None else datetime.utcnow()
        completed_at = None if status == "running" else datetime.utcnow()
        self.lab_state_registry.save_run_ledger(
            RunLedgerRecord(
                task_run_id=request.task_id,
                task_id=request.task_id,
                task_goal=request.goal,
                status=status,  # type: ignore[arg-type]
                config_ref=request.task_config_ref,
                started_at=started_at,
                completed_at=completed_at,
                final_answer=result.get("final_answer") if isinstance(result.get("final_answer"), str) else None,
                failure_reason=failure_reason,
                meta_run_refs=_dedupe_strings([*result_meta_run_refs, *recorded_meta_run_refs]),
                subagent_run_refs=_dedupe_strings([*result_subagent_run_refs, *recorded_subagent_run_refs]),
                final_artifact_refs=final_artifact_refs or [],
                metadata=metadata_payload,
            )
        )

    def _recorded_meta_run_refs(self, task_id: str) -> list[str]:
        if self.trajectory_registry is None:
            return []
        return [record.run_ref for record in self.trajectory_registry.list_meta_agent_runs() if record.task_id == task_id]

    def _recorded_subagent_run_refs(self, task_id: str) -> list[str]:
        if self.trajectory_registry is None:
            return []
        return [record.run_ref for record in self.trajectory_registry.list_subagent_runs() if record.task_id == task_id]

    def _initialize_backend_map(
        self,
        backends: dict[str, Any],
        bindings: list[BackendBinding] | None,
        runtimes: dict[str, Any],
        *,
        use_active_state: bool = False,
    ) -> None:
        if not backends:
            return

        active_bindings = bindings or [
            BackendBinding(backend_id=backend_id) for backend_id in sorted(backends)
        ]
        for binding in active_bindings:
            backend = backends.get(binding.backend_id)
            if backend is None:
                raise ValueError(f"backend {binding.backend_id!r} is not configured")
            state_ref = binding.state_ref
            if state_ref is None and use_active_state and self.backend_state_registry is not None:
                state_ref = self.backend_state_registry.resolve_active_state(binding.backend_id)
            runtimes[binding.backend_id] = _instantiate_backend(backend, state_ref)

    def _initialize_unbound_backend_map(
        self,
        backends: dict[str, Any],
        runtimes: dict[str, Any],
        *,
        use_active_state: bool = False,
    ) -> None:
        for backend_id in sorted(backends):
            if backend_id in runtimes:
                continue
            state_ref = None
            if use_active_state and self.backend_state_registry is not None:
                state_ref = self.backend_state_registry.resolve_active_state(backend_id)
            runtimes[backend_id] = _instantiate_backend(backends[backend_id], state_ref)

    def _initialize_required_backend_ids(
        self,
        backends: dict[str, Any],
        backend_ids: set[str],
        runtimes: dict[str, Any],
        *,
        use_active_state: bool = False,
    ) -> None:
        for backend_id in sorted(backend_ids):
            if backend_id in runtimes:
                continue
            backend = backends.get(backend_id)
            if backend is None:
                raise ValueError(f"backend {backend_id!r} is not configured")
            state_ref = None
            if use_active_state and self.backend_state_registry is not None:
                state_ref = self.backend_state_registry.resolve_active_state(backend_id)
            runtimes[backend_id] = _instantiate_backend(backend, state_ref)


def _instantiate_backend(backend: Any, state_ref: str | None) -> Any:
    instantiate = getattr(backend, "instantiate", None)
    if callable(instantiate):
        return instantiate(state_ref)
    return backend


def _task_config_llm_bindings(task_config: TaskConfig) -> list[BackendBinding]:
    bindings = [role.llm_backend for role in task_config.roles.values()]
    if task_config.meta_agent is not None:
        bindings.append(task_config.meta_agent.llm_backend)
    if task_config.dynamic_subagents is not None and task_config.dynamic_subagents.enabled:
        dynamic = task_config.dynamic_subagents
        if dynamic.planner_backend is not None:
            bindings.append(BackendBinding(backend_id=dynamic.planner_backend.backend_id))
        if dynamic.default_worker_backend is not None:
            bindings.append(BackendBinding(backend_id=dynamic.default_worker_backend.backend_id))
        bindings.extend(BackendBinding(backend_id=backend_id) for backend_id in dynamic.allowed_worker_backend_ids)
    return _unique_backend_bindings(bindings)


def _unique_backend_bindings(bindings: list[BackendBinding]) -> list[BackendBinding]:
    unique: dict[str, BackendBinding] = {}
    for binding in bindings:
        if binding.backend_id not in unique:
            unique[binding.backend_id] = binding
    return list(unique.values())


def _memory_llm_backend_ids(memory_backends: Any) -> set[str]:
    backend_ids: set[str] = set()
    for backend in memory_backends:
        backend_ids.update(_llm_backend_ids_from_object(backend))
        method = getattr(backend, "method", None)
        if method is not None:
            backend_ids.update(_llm_backend_ids_from_object(method))
    return backend_ids


def _llm_backend_ids_from_object(value: Any) -> set[str]:
    backend_id = getattr(value, "llm_backend_id", None)
    if isinstance(backend_id, str) and backend_id:
        return {backend_id}
    return set()


def _terminal_status_from_result(result: dict[str, Any]) -> str:
    status = result.get("status")
    if status in {"failed", "budget_exceeded"}:
        return "failed"
    if status == "interrupted":
        return "interrupted"
    return "completed"


def _final_artifact_index_entries(
    *,
    result: dict[str, Any] | None,
    lab_artifacts: list[ArtifactIndexRecord],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None]] = set()
    if isinstance(result, dict):
        for run in result.get("runs", []):
            if not isinstance(run, dict):
                continue
            producer = run.get("role") if isinstance(run.get("role"), str) else run.get("generic_agent_type")
            step = run.get("meta_workflow_node_id") if isinstance(run.get("meta_workflow_node_id"), str) else None
            for raw_ref in run.get("artifact_refs", []):
                if not isinstance(raw_ref, dict):
                    continue
                artifact = _artifact_ref_from_payload(raw_ref)
                if artifact is None:
                    continue
                key = (artifact.uri, str(producer) if producer is not None else None)
                if key in seen:
                    continue
                seen.add(key)
                entries.append(_artifact_index_entry(artifact, producer=producer, step=step))
    for record in lab_artifacts:
        artifact = ArtifactRef(uri=record.uri, type=record.artifact_type, metadata=record.metadata)
        key = (artifact.uri, record.role)
        if key in seen:
            continue
        seen.add(key)
        entries.append(
            _artifact_index_entry(
                artifact,
                producer=record.role or record.metadata.get("producer_role"),
                step=record.metadata.get("workflow_node_id"),
                status=record.status,
                created_at=record.created_at.isoformat(),
            )
        )
    return entries


def _artifact_ref_from_payload(payload: dict[str, Any]) -> ArtifactRef | None:
    try:
        return ArtifactRef.model_validate(payload)
    except Exception:
        return None


def _artifact_index_entry(
    artifact: ArtifactRef,
    *,
    producer: Any,
    step: Any,
    status: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    metadata = _json_compatible(artifact.metadata)
    artifact_status = status or str(metadata.get("status") or "intermediate")
    return {
        "artifact_path": artifact.uri,
        "artifact_type": artifact.type,
        "producing_agent": str(producer) if producer is not None else None,
        "producing_step": str(step) if step is not None else None,
        "created_timestamp": created_at,
        "validation_status": metadata.get("validation_status"),
        "is_final": artifact_status == "final",
        "metadata": metadata,
    }


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


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item]


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _local_path_from_uri(uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        if parsed.scheme == "file" and parsed.netloc not in ("", "localhost"):
            return None
        path = unquote(parsed.path if parsed.scheme == "file" else uri)
        return Path(path)
    return None


def _safe_state_ref(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in value)
    if not safe or safe in {".", ".."}:
        raise ValueError(f"unsafe state ref: {value!r}")
    return safe
