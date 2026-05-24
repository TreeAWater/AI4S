from __future__ import annotations

import json
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4

from evolab.config.task_config import BackendBinding, MetaAgentSpec, TaskConfig
from evolab.contracts.common import ArtifactRef, Message
from evolab.contracts.dispatch import DispatchAction, DispatchDecision
from evolab.contracts.lab_state import ArtifactIndexRecord, SubagentReportRecord, TrainingIndexRecord
from evolab.contracts.llm import LLMGenerationConfig, LLMRuntimeResponse, SubAgentAction
from evolab.contracts.records import LLMCallRecord, MetaAgentRunRecord, SubagentRunRecord
from evolab.contracts.retrieval import (
    MemoryBundle,
    MemoryItem,
    RetrievalRequest,
    SkillBundle,
    SkillRef,
    SkillObservationRequest,
    SkillUpdateResult,
)
from evolab.contracts.state import BackendStateRecord
from evolab.contracts.task import TaskRequest
from evolab.contracts.tools import ToolCall, ToolCallRecord, ToolResult, ToolTrace
from evolab.contracts.workflow import NodeExecutionRecord, PlanExecutionTrace, WorkflowNode, WorkflowPlan
from evolab.registries.backend_state import BackendStateRegistry
from evolab.registries.lab_state import FileLabStateRegistry
from evolab.registries.task import FileTaskRegistry
from evolab.registries.trajectory import FileTrajectoryRegistry
from evolab.runtime.prompt_builder import PromptBuilder
from evolab.runtime.capability_repair import CapabilityRepairRuntime, RepairRuntimeOutcome
from evolab.runtime.lab_state import LabStateBuilder
from evolab.runtime.dynamic_workflow import (
    DynamicSubAgentFactory,
    DynamicWorkflowPlanner,
    DynamicWorkflowTrace,
    dynamic_workflow_node_order,
    persist_dynamic_workflow_artifacts,
)
from evolab.runtime.skill_retrieval import prepare_skill_runtime_context
from evolab.runtime.trajectory_collector import TrajectoryCollector
from evolab.runtime.workflow_planner import SkillWorkflowPlanner
from evolab.tools.runtime import ToolRuntime

TaskDispatchLoop = Callable[[TaskRequest], dict[str, Any]]
ToolArtifactRootFactory = Callable[[TaskRequest, str], Path | str]


@dataclass
class _RoleExecutionPayload:
    prompt_messages: list[Message]
    output_message: Message
    tool_trace_records: list[ToolCallRecord]
    tool_trace: ToolTrace
    artifact_refs: list[ArtifactRef]
    final_answer: str
    skill_bundle: SkillBundle | None = None
    skill_context: dict[str, Any] = field(default_factory=dict)
    repair_trajectory: list[dict[str, Any]] = field(default_factory=list)
    promotion_candidates: list[dict[str, Any]] = field(default_factory=list)
    llm_call_refs: list[str] = field(default_factory=list)
    workflow_plan: WorkflowPlan | None = None
    plan_execution_trace: PlanExecutionTrace | None = None
    node_execution_records: list[NodeExecutionRecord] = field(default_factory=list)
    status: str = "completed"
    failure_reason: str | None = None
    memory_update_messages: list[Message] | None = None
    budget: dict[str, Any] = field(default_factory=dict)


@dataclass
class _ToolExecutionOutcome:
    record: ToolCallRecord
    counts_against_budget: bool = True
    repair_messages: list[Message] = field(default_factory=list)
    updated_skill_bundle: SkillBundle | None = None
    updated_skill_context: dict[str, Any] | None = None
    repair_entry: dict[str, Any] | None = None
    promotion_candidates: list[dict[str, Any]] = field(default_factory=list)


class _SubagentBudgetExceeded(RuntimeError):
    def __init__(self, message: str, *, metadata: dict[str, Any]):
        super().__init__(message)
        self.metadata = metadata


@dataclass
class _SubagentBudgetTracker:
    role_name: str
    started_at_monotonic: float
    max_llm_calls: int | None = None
    max_tool_calls: int | None = None
    max_runtime_seconds: float | None = None
    llm_calls: int = 0
    tool_calls: int = 0
    warnings: list[str] = field(default_factory=list)

    def check(self, checkpoint: str) -> None:
        if self.max_runtime_seconds is not None and self.elapsed_seconds() >= self.max_runtime_seconds:
            self._raise_budget_exceeded(
                "max_subagent_runtime_seconds",
                checkpoint,
                self.elapsed_seconds(),
                self.max_runtime_seconds,
            )
        if self.max_llm_calls is not None and self.llm_calls >= self.max_llm_calls:
            self._raise_budget_exceeded("max_subagent_llm_calls", checkpoint, self.llm_calls, self.max_llm_calls)
        if self.max_tool_calls is not None and self.tool_calls >= self.max_tool_calls:
            self._raise_budget_exceeded("max_subagent_tool_calls", checkpoint, self.tool_calls, self.max_tool_calls)

    def note_llm_call(self) -> None:
        self.llm_calls += 1

    def note_tool_call(self) -> None:
        self.tool_calls += 1

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at_monotonic

    def metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": self.role_name,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "elapsed_seconds": self.elapsed_seconds(),
            "warnings": list(self.warnings),
        }
        if self.max_llm_calls is not None:
            payload["max_subagent_llm_calls"] = self.max_llm_calls
        if self.max_tool_calls is not None:
            payload["max_subagent_tool_calls"] = self.max_tool_calls
        if self.max_runtime_seconds is not None:
            payload["max_subagent_runtime_seconds"] = self.max_runtime_seconds
        return payload

    def _raise_budget_exceeded(self, budget_name: str, checkpoint: str, observed: float | int, limit: float | int) -> None:
        message = (
            f"budget_exceeded: {budget_name} exceeded for {self.role_name} "
            f"at {checkpoint} ({observed} >= {limit})"
        )
        raise _SubagentBudgetExceeded(
            message,
            metadata={
                **self.metadata(),
                "budget_name": budget_name,
                "checkpoint": checkpoint,
                "observed": observed,
                "limit": limit,
            },
        )


class TaskRuntime:
    def __init__(
        self,
        *,
        task_config: TaskConfig | None = None,
        prompt_builder: PromptBuilder | None = None,
        tool_runtime: ToolRuntime | None = None,
        task_registry: FileTaskRegistry | None = None,
        trajectory_registry: FileTrajectoryRegistry | None = None,
        backend_state_registry: BackendStateRegistry | None = None,
        lab_state_registry: FileLabStateRegistry | None = None,
        tool_artifact_registrar: Callable[[ToolResult], None] | None = None,
        tool_artifact_root_factory: ToolArtifactRootFactory | None = None,
        llm_runtimes: dict[str, Any] | None = None,
        memory_runtimes: dict[str, Any] | None = None,
        skill_runtimes: dict[str, Any] | None = None,
        dispatch_loop: TaskDispatchLoop | None = None,
        trajectory_collector: TrajectoryCollector | None = None,
        capability_repair_runtime: CapabilityRepairRuntime | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.task_config = task_config
        self.prompt_builder = prompt_builder
        self.tool_runtime = tool_runtime
        self.task_registry = task_registry
        self.trajectory_registry = trajectory_registry
        self.trajectory_collector = trajectory_collector or TrajectoryCollector(trajectory_registry)
        self.backend_state_registry = backend_state_registry
        self.lab_state_registry = lab_state_registry
        self.tool_artifact_registrar = tool_artifact_registrar
        self.tool_artifact_root_factory = tool_artifact_root_factory
        self.llm_runtimes = llm_runtimes or {}
        self.memory_runtimes = memory_runtimes or {}
        self.skill_runtimes = skill_runtimes or {}
        self.dispatch_loop = dispatch_loop
        self.capability_repair_runtime = capability_repair_runtime or CapabilityRepairRuntime()
        self.progress_callback = progress_callback
        self._pending_lab_state_detail_requests: dict[str, list[str]] = {}
        self.lab_root = _infer_lab_root(
            trajectory_registry=self.trajectory_registry,
            lab_state_registry=self.lab_state_registry,
            task_registry=self.task_registry,
        )

    def run(self, request: TaskRequest) -> dict[str, Any]:
        if self.dispatch_loop is not None:
            return self.dispatch_loop(request)
        dynamic_result = self._maybe_run_dynamic_subagents(request)
        if dynamic_result is not None:
            return dynamic_result
        if self.task_config is not None and self.task_config.meta_agent is not None:
            return self._run_meta_agent_dispatch(request)

        repair_budget = {"task_attempts": 0}
        # V0 demo behavior runs all configured roles in order; revisit when dispatch owns role selection.
        role_results = [
            self._run_role(request, role, stage_index, repair_budget=repair_budget)
            for stage_index, role in enumerate(self._roles())
        ]
        final_result = role_results[-1]
        if final_result.get("status", "completed") != "completed" and final_result.get("status") != "budget_exceeded":
            raise RuntimeError(_failed_subagent_result_message("task runtime", final_result))
        return {
            "task_id": request.task_id,
            "status": final_result.get("status", "completed"),
            "failure_reason": final_result.get("failure_reason") if isinstance(final_result.get("failure_reason"), str) else None,
            "run_ref": final_result["run_ref"],
            "run_refs": [result["run_ref"] for result in role_results],
            "runs": role_results,
            "role": final_result["role"],
            "final_answer": final_result["final_answer"],
        }

    def _maybe_run_dynamic_subagents(self, request: TaskRequest) -> dict[str, Any] | None:
        if self.task_config is None or self.task_config.dynamic_subagents is None:
            return None
        dynamic_config = self.task_config.dynamic_subagents
        if not dynamic_config.enabled or dynamic_config.mode == "static":
            return None
        if dynamic_config.planner_backend is None or dynamic_config.default_worker_backend is None:
            raise RuntimeError("dynamic_subagents.enabled requires planner_backend and default_worker_backend")
        planner_run_ref = f"dynamic-planner-{uuid4()}"
        planner_llm = self._llm_runtime(dynamic_config.planner_backend.backend_id)
        skill_backend = self._first_runtime(self.skill_runtimes, "skill")
        if self.tool_runtime is None:
            raise RuntimeError("dynamic_subagents requires ToolRuntime")
        available_backend_ids = set(self.llm_runtimes)
        planner = DynamicWorkflowPlanner(
            planner_llm=planner_llm,
            config=dynamic_config,
            tool_runtime=self.tool_runtime,
            skill_backend=skill_backend,
            available_llm_backend_ids=available_backend_ids,
            llm_call_recorder=lambda messages, generation_config, response, metadata: self._save_llm_call(
                run_ref=planner_run_ref,
                backend_id=dynamic_config.planner_backend.backend_id,
                llm=planner_llm,
                generation_config=generation_config,
                input_messages=_copy_messages(messages),
                tool_specs=[],
                response=response,
                metadata={"task_id": request.task_id, "role": "DynamicWorkflowPlanner", **metadata},
            ),
        )
        work_items = _dynamic_work_items_for_request(request, self.task_config.runtime_policy.metadata, dynamic_config.scope)
        static_subagents = [
            {"name": role.name, "system_prompt": role.system_prompt, "allowed_tools": list(role.allowed_tools)}
            for role in self.task_config.roles.values()
        ]
        all_runs: list[dict[str, Any]] = []
        all_run_refs: list[str] = []
        workflow_results: list[dict[str, Any]] = []
        for work_item in work_items:
            outcome = planner.plan(request=request, work_item=work_item, static_subagents=static_subagents)
            if outcome.spec is None:
                if self.lab_root is not None:
                    persist_dynamic_workflow_artifacts(
                        lab_root=self.lab_root,
                        task_id=request.task_id,
                        spec=None,
                        validation_report=outcome.validation_report,
                        fallback_reason=outcome.fallback_reason,
                    )
                if dynamic_config.fallback_to_static:
                    self.trajectory_collector.record_event(
                        event_type="dynamic_workflow_fallback",
                        subject_type="task",
                        subject_ref=request.task_id,
                        task_id=request.task_id,
                        run_ref=planner_run_ref,
                        metadata=outcome.fallback_reason or {},
                    )
                    if all_runs:
                        failed_run_ref = f"dynamic-planning-failed-{uuid4()}"
                        failure_reason = json.dumps(
                            outcome.fallback_reason or outcome.validation_report.model_dump(mode="json"),
                            sort_keys=True,
                        )
                        failed_run = {
                            "task_id": request.task_id,
                            "status": "failed",
                            "failure_reason": failure_reason,
                            "execution_mode": "dynamic",
                            "run_ref": failed_run_ref,
                            "run_refs": [failed_run_ref],
                            "runs": [],
                            "role": "DynamicWorkflowPlanner",
                            "final_answer": failure_reason,
                            "metadata": {
                                "work_item_id": _dynamic_work_item_id(work_item),
                                "fallback_to_static": True,
                                "fallback_suppressed_after_partial_dynamic": True,
                            },
                        }
                        all_runs.append(failed_run)
                        all_run_refs.append(failed_run_ref)
                        workflow_results.append(
                            {
                                "workflow_id": None,
                                "work_item_id": _dynamic_work_item_id(work_item),
                                "status": "failed",
                                "failure_reason": failure_reason,
                                "runs": [failed_run],
                                "run_refs": [failed_run_ref],
                                "fallback_reason": outcome.fallback_reason,
                            }
                        )
                        continue
                    return None
                raise RuntimeError(json.dumps(outcome.fallback_reason or outcome.validation_report.model_dump(mode="json")))
            workflow_result = self._execute_dynamic_workflow_spec(
                request=request,
                spec=outcome.spec,
                validation_report=outcome.validation_report,
                work_item=work_item,
                planner_backend_id=dynamic_config.planner_backend.backend_id,
                default_worker_backend_id=dynamic_config.default_worker_backend.backend_id,
                available_backend_ids=available_backend_ids,
                skill_backend=skill_backend,
            )
            workflow_results.append(workflow_result)
            all_runs.extend(workflow_result["runs"])
            all_run_refs.extend(workflow_result["run_refs"])
        self._write_dynamic_aggregate_final_records(request)
        if not all_runs:
            if dynamic_config.fallback_to_static:
                return None
            raise RuntimeError("dynamic_subagents produced no runs")
        failed = [run for run in all_runs if run.get("status") not in {"completed"}]
        final_result = all_runs[-1]
        status = "failed" if failed else "completed"
        return {
            "task_id": request.task_id,
            "status": status,
            "failure_reason": _dynamic_failure_reason(failed) if failed else None,
            "execution_mode": "dynamic",
            "run_ref": final_result["run_ref"],
            "run_refs": all_run_refs,
            "runs": all_runs,
            "role": final_result["role"],
            "final_answer": final_result["final_answer"],
            "dynamic_workflows": workflow_results,
        }

    def _write_dynamic_aggregate_final_records(self, request: TaskRequest) -> None:
        if self.lab_root is None or self.lab_state_registry is None:
            return
        if not _scientific_handoff_bootstrap_enabled(self.task_config.runtime_policy.metadata):
            return
        records = _final_records_for_write_bootstrap(self.lab_state_registry, task_id=request.task_id)
        root = self.lab_root / "artifacts" / "tools"
        root.mkdir(parents=True, exist_ok=True)
        for filename in ("biology_component_records.jsonl", "final_records.jsonl"):
            path = root / filename
            path.write_text(
                "".join(json.dumps(_json_compatible(record), sort_keys=True) + "\n" for record in records),
                encoding="utf-8",
            )

    def _execute_dynamic_workflow_spec(
        self,
        *,
        request: TaskRequest,
        spec: Any,
        validation_report: Any,
        work_item: dict[str, Any] | None,
        planner_backend_id: str,
        default_worker_backend_id: str,
        available_backend_ids: set[str],
        skill_backend: Any,
    ) -> dict[str, Any]:
        dynamic_config = self.task_config.dynamic_subagents
        assert dynamic_config is not None
        factory = DynamicSubAgentFactory(
            config=dynamic_config,
            tool_runtime=self.tool_runtime,
            skill_backend=skill_backend,
            available_llm_backend_ids=available_backend_ids,
            runtime_policy=self.task_config.runtime_policy,
            agent_memory_backend=_dynamic_agent_memory_backend(self.task_config),
        )
        agent_by_id = {agent.subagent_id: agent for agent in spec.dynamic_subagents}
        node_by_id = {node.node_id: node for node in spec.workflow_nodes}
        ordered_node_ids = dynamic_workflow_node_order(spec)
        results_by_node: dict[str, dict[str, Any]] = {}
        runs: list[dict[str, Any]] = []
        repair_budget = {"task_attempts": 0}
        produced_artifact_names = _dynamic_declared_output_artifact_names(spec)
        available_artifact_names: set[str] = set()
        work_item_context = _json_compatible(work_item or {})
        for stage_index, node_id in enumerate(ordered_node_ids):
            node = node_by_id[node_id]
            blocked = [dependency for dependency in node.dependencies if results_by_node.get(dependency, {}).get("status") != "completed"]
            missing_inputs = _dynamic_missing_input_artifacts(
                node.input_artifacts,
                produced_artifact_names=produced_artifact_names,
                available_artifact_names=available_artifact_names,
            )
            if blocked or missing_inputs:
                reason_parts: list[str] = []
                if blocked:
                    reason_parts.append(f"failed dependencies: {', '.join(blocked)}")
                if missing_inputs:
                    reason_parts.append(f"missing dynamic input artifacts: {', '.join(missing_inputs)}")
                failure_reason = "; ".join(reason_parts)
                result = {
                    "task_id": request.task_id,
                    "run_ref": f"dynamic-skipped-{uuid4()}",
                    "role": agent_by_id[node.subagent_id].role_name,
                    "generic_agent_type": agent_by_id[node.subagent_id].role_name,
                    "assigned_task": agent_by_id[node.subagent_id].goal,
                    "final_answer": f"dynamic node skipped: {failure_reason}",
                    "stage_index": stage_index,
                    "status": "failed",
                    "failure_reason": failure_reason,
                    "llm_call_refs": [],
                    "tool_call_count": 0,
                    "artifact_refs": [],
                    "dispatch_metadata": {
                        "execution_mode": "dynamic",
                        "dynamic_workflow_id": spec.workflow_id,
                        "dynamic_workflow_node": node.model_dump(mode="json"),
                    },
                    "completion_contract": {},
                    "budget": {},
                }
            else:
                runtime_agent = factory.create(
                    spec=agent_by_id[node.subagent_id],
                    task_id=request.task_id,
                    work_item_id=spec.work_item_id,
                    workflow_id=spec.workflow_id,
                    planner_backend_id=planner_backend_id,
                    static_fallback_available=dynamic_config.fallback_to_static,
                )
                expected_output_artifacts = _dynamic_node_expected_output_artifacts(node, agent_by_id[node.subagent_id])
                instruction = json.dumps(
                    {
                        "dynamic_workflow_id": spec.workflow_id,
                        "work_item_id": spec.work_item_id,
                        "work_item_context": work_item_context,
                        "task_summary": spec.task_summary,
                        "article_context_summary": spec.article_context_summary,
                        "node": node.model_dump(mode="json"),
                        "subagent_goal": runtime_agent.spec.goal,
                        "input_artifacts": node.input_artifacts,
                        "available_input_artifacts": sorted(available_artifact_names),
                        "expected_output_artifacts": expected_output_artifacts,
                        "output_artifact_policy": _dynamic_output_artifact_policy(expected_output_artifacts),
                        "input_schema": runtime_agent.spec.input_schema,
                        "output_schema": runtime_agent.spec.output_schema,
                        "acceptance_criteria": runtime_agent.spec.acceptance_criteria,
                        "constraints": runtime_agent.spec.constraints,
                    },
                    indent=2,
                    sort_keys=True,
                )
                try:
                    dynamic_dispatch_metadata = {
                        "execution_mode": "dynamic",
                        "dynamic_workflow_id": spec.workflow_id,
                        "work_item_id": spec.work_item_id,
                        "work_item_context": work_item_context,
                        "dynamic_workflow_node": node.model_dump(mode="json"),
                        "dynamic_subagent_spec": runtime_agent.spec.model_dump(mode="json"),
                        "dynamic_subagent_provenance": runtime_agent.provenance,
                        "planner_backend_id": planner_backend_id,
                        "default_worker_backend_id": default_worker_backend_id,
                        "artifact_contracts": spec.artifact_contracts,
                        "disable_internal_workflow_planning": True,
                        "expected_outputs": _dynamic_expected_output_contracts(expected_output_artifacts),
                    }
                    if isinstance(work_item, dict):
                        for key in ("lab_path", "article_package", "article_path", "work_item_path", "root"):
                            if isinstance(work_item.get(key), str) and work_item[key]:
                                dynamic_dispatch_metadata.setdefault(key, work_item[key])
                        for key in ("source_files", "exact_source_files"):
                            if isinstance(work_item.get(key), list):
                                dynamic_dispatch_metadata.setdefault(key, work_item[key])
                    result = self._run_role(
                        request,
                        runtime_agent.role,
                        stage_index,
                        instruction=instruction,
                        retrieval_query=runtime_agent.spec.goal,
                        dispatch_metadata=dynamic_dispatch_metadata,
                        return_failed_result=True,
                        repair_budget=repair_budget,
                    )
                    result = _enforce_dynamic_node_output_contract(
                        result,
                        expected_output_artifacts=expected_output_artifacts,
                    )
                except Exception as exc:
                    result = {
                        "task_id": request.task_id,
                        "run_ref": f"dynamic-failed-{uuid4()}",
                        "role": runtime_agent.role.name,
                        "generic_agent_type": runtime_agent.role.name,
                        "assigned_task": instruction,
                        "final_answer": str(exc),
                        "stage_index": stage_index,
                        "status": "failed",
                        "failure_reason": str(exc),
                        "llm_call_refs": [],
                        "tool_call_count": 0,
                        "artifact_refs": [],
                        "dispatch_metadata": {
                            "execution_mode": "dynamic",
                            "dynamic_workflow_id": spec.workflow_id,
                            "dynamic_workflow_node": node.model_dump(mode="json"),
                            "dynamic_subagent_spec": runtime_agent.spec.model_dump(mode="json"),
                            "dynamic_subagent_provenance": runtime_agent.provenance,
                        },
                        "completion_contract": {},
                        "budget": {},
                    }
            results_by_node[node_id] = result
            runs.append(result)
            if result.get("status") == "completed":
                available_artifact_names.update(
                    _dynamic_available_output_names(
                        result.get("artifact_refs", []),
                        expected_output_artifacts=_dynamic_node_expected_output_artifacts(
                            node,
                            agent_by_id[node.subagent_id],
                        ),
                    )
                )
        trace_status = "completed" if all(run.get("status") == "completed" for run in runs) else "failed"
        trace = DynamicWorkflowTrace(
            workflow_id=spec.workflow_id,
            work_item_id=spec.work_item_id,
            execution_mode="dynamic",
            status=trace_status,
            planner_backend_id=planner_backend_id,
            default_worker_backend_id=default_worker_backend_id,
            run_refs=[run["run_ref"] for run in runs],
            node_results=[
                {
                    "node_id": node_id,
                    "run_ref": result.get("run_ref"),
                    "role": result.get("role"),
                    "status": result.get("status"),
                    "artifact_refs": result.get("artifact_refs", []),
                    "failure_reason": result.get("failure_reason"),
                }
                for node_id, result in results_by_node.items()
            ],
            metadata={"task_id": request.task_id},
        )
        artifact_paths = {}
        if self.lab_root is not None:
            artifact_paths = persist_dynamic_workflow_artifacts(
                lab_root=self.lab_root,
                task_id=request.task_id,
                spec=spec,
                validation_report=validation_report,
                trace=trace,
            )
        return {
            "workflow_id": spec.workflow_id,
            "work_item_id": spec.work_item_id,
            "status": trace_status,
            "run_refs": [run["run_ref"] for run in runs],
            "runs": runs,
            "artifact_paths": artifact_paths,
            "validation_report": validation_report.model_dump(mode="json"),
        }

    def _run_meta_agent_dispatch(self, request: TaskRequest) -> dict[str, Any]:
        assert self.task_config is not None
        meta_agent = self.task_config.meta_agent
        assert meta_agent is not None
        meta_llm = self._llm_runtime(meta_agent.llm_backend.backend_id)
        meta_memory = self._meta_memory_runtime(meta_agent)
        role_results: list[dict[str, Any]] = []
        meta_run_refs: list[str] = []
        repair_budget = {"task_attempts": 0}
        self._initialize_work_item_lifecycle(request)
        for step_index in range(self.task_config.max_dispatch_steps):
            meta_run_ref = f"meta-{uuid4()}"
            decision, llm_call_ref, meta_memory_request, meta_memory_bundle = self._next_dispatch_decision(
                request=request,
                meta_agent=meta_agent,
                meta_llm=meta_llm,
                meta_memory=meta_memory,
                run_ref=meta_run_ref,
                step_index=step_index,
                role_results=role_results,
            )
            decision = self._apply_generic_work_item_retry_policy(
                request=request,
                decision=decision,
                role_results=role_results,
            )
            meta_memory_update_result = self._update_meta_agent_memory(
                request=request,
                meta_agent=meta_agent,
                meta_memory=meta_memory,
                meta_memory_bundle=meta_memory_bundle,
                decision=decision,
                run_ref=meta_run_ref,
                step_index=step_index,
                role_results=role_results,
                llm_call_ref=llm_call_ref,
            )
            self._save_meta_agent_run(
                request=request,
                run_ref=meta_run_ref,
                decision=decision,
                step_index=step_index,
                role_results=role_results,
                llm_call_ref=llm_call_ref,
                meta_memory_request=meta_memory_request,
                meta_memory_bundle=meta_memory_bundle,
                meta_memory_update_result=meta_memory_update_result,
            )
            meta_run_refs.append(meta_run_ref)
            if decision.action == DispatchAction.RUN_SUBAGENT:
                _validate_no_progress_dispatch(
                    decision=decision,
                    role_results=role_results,
                    runtime_metadata=self.task_config.runtime_policy.metadata,
                )
                preflight_result = self._preflight_work_item_dispatch(
                    request=request,
                    decision=decision,
                    stage_index=len(role_results),
                )
                if preflight_result is not None:
                    role_results.append(preflight_result)
                    self._record_work_item_lifecycle_from_run(request, preflight_result)
                    continue
                role = self._role_by_name(decision.target_role)
                self._progress(
                    f"MetaAgent dispatch step {step_index}: {role.name} "
                    f"({decision.metadata.get('meta_workflow_node_id') or 'unscoped-node'})"
                )
                run_result = self._run_role(
                    request,
                    role,
                    len(role_results),
                    instruction=decision.instruction,
                    retrieval_query=decision.retrieval_query,
                    dispatch_metadata=_dispatch_metadata_with_expected_outputs(decision),
                    return_failed_result=True,
                    repair_budget=repair_budget,
                )
                role_results.append(run_result)
                self._record_work_item_lifecycle_from_run(request, run_result)
                continue
            if decision.action == DispatchAction.FINISH_TASK:
                self._progress(f"MetaAgent dispatch step {step_index}: END")
                return self._meta_agent_result(request, role_results, meta_run_refs, decision)
            if decision.action == DispatchAction.ASK_HUMAN:
                raise NotImplementedError("meta-agent ask_human dispatch is not implemented in V0")
            if decision.action == DispatchAction.ABORT:
                raise RuntimeError(decision.instruction or "meta-agent aborted task")
        raise RuntimeError("meta-agent exceeded max_dispatch_steps before finish_task")

    def _next_dispatch_decision(
        self,
        *,
        request: TaskRequest,
        meta_agent: MetaAgentSpec,
        meta_llm: Any,
        run_ref: str,
        step_index: int,
        role_results: list[dict[str, Any]],
        meta_memory: Any | None = None,
    ) -> tuple[DispatchDecision, str | None, RetrievalRequest | None, MemoryBundle | None]:
        meta_memory_request, meta_memory_bundle = self._search_meta_agent_memory(
            request=request,
            meta_agent=meta_agent,
            meta_memory=meta_memory,
            step_index=step_index,
            role_results=role_results,
        )
        messages = [
            Message(role="system", content=meta_agent.system_prompt),
            *self._meta_agent_instruction_messages(meta_agent),
            Message(
                role="user",
                content=json.dumps(
                    {
                        "task_id": request.task_id,
                        "goal": request.goal,
                        "step_index": step_index,
                        "available_roles": [
                            {"name": role.name, "system_prompt": role.system_prompt}
                            for role in self._roles()
                        ],
                        "routing_state": _meta_agent_routing_state(
                            role_results,
                            self.task_config.runtime_policy.metadata,
                        ),
                        "lab_state": self._meta_agent_lab_state(request),
                        "meta_memory": _meta_memory_prompt_payload(meta_memory_bundle),
                        "completed_runs": role_results,
                        "required_response": (
                            "Return one JSON object with route set to one available role name or END."
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                ),
            ),
        ]
        allowed_roles = [role.name for role in self._roles()]
        max_retries = _meta_dispatch_parse_retries(self.task_config.runtime_policy.metadata)
        last_error: Exception | None = None
        last_raw_output = ""
        last_llm_call_ref: str | None = None
        for attempt_index in range(max_retries + 1):
            attempt_messages = messages
            if attempt_index > 0:
                attempt_messages = [
                    *messages,
                    _meta_dispatch_repair_message(
                        raw_output=last_raw_output,
                        error=last_error,
                        expected_schema=_meta_dispatch_expected_schema(allowed_roles),
                        retry_count=attempt_index,
                    ),
                ]
            generation_config = LLMGenerationConfig(model="")
            input_messages = _copy_messages(attempt_messages)
            response = meta_llm.generate(attempt_messages, [], generation_config)
            llm_call_ref = self._save_llm_call(
                run_ref=run_ref,
                backend_id=meta_agent.llm_backend.backend_id,
                llm=meta_llm,
                generation_config=generation_config,
                input_messages=input_messages,
                tool_specs=[],
                response=response,
                metadata={
                    "task_id": request.task_id,
                    "role": meta_agent.name,
                    "runtime_stage": "meta_agent_dispatch",
                    "step_index": step_index,
                    "parse_retry_index": attempt_index,
                },
            )
            last_llm_call_ref = llm_call_ref
            if response.action.action != "final_answer":
                raise RuntimeError(f"meta-agent returned unsupported action {response.action.action!r}")
            last_raw_output = response.action.content or ""
            try:
                decision = _parse_dispatch_decision(last_raw_output, completed_runs=role_results)
                _validate_meta_dispatch_decision(
                    decision,
                    allowed_roles,
                    role_results,
                    runtime_metadata=self.task_config.runtime_policy.metadata,
                )
                return decision, llm_call_ref, meta_memory_request, meta_memory_bundle
            except Exception as exc:
                last_error = exc
                if attempt_index >= max_retries:
                    raise RuntimeError(
                        _meta_dispatch_failure_message(
                            raw_output=last_raw_output,
                            error=exc,
                            expected_schema=_meta_dispatch_expected_schema(allowed_roles),
                            retry_count=max_retries,
                            task_id=request.task_id,
                            step_index=step_index,
                        )
                    ) from exc
        raise RuntimeError("MetaAgent dispatch parsing failed unexpectedly")

    def _meta_memory_runtime(self, meta_agent: MetaAgentSpec) -> Any | None:
        if meta_agent.memory_backend is None:
            return None
        return self._memory_runtime_for_binding(meta_agent.memory_backend)

    def _search_meta_agent_memory(
        self,
        *,
        request: TaskRequest,
        meta_agent: MetaAgentSpec,
        meta_memory: Any | None,
        step_index: int,
        role_results: list[dict[str, Any]],
    ) -> tuple[RetrievalRequest | None, MemoryBundle | None]:
        if meta_memory is None:
            return None, None
        retrieval_request = RetrievalRequest(
            task_id=request.task_id,
            role=meta_agent.name,
            query=_meta_agent_memory_query(request=request, step_index=step_index, role_results=role_results),
            task_origin=request.origin,
            task_purpose=request.purpose,
            filters={
                "memory_scope": "agent",
                "memory_scope_id": _meta_agent_memory_scope_id(meta_agent),
            },
            metadata={"runtime_stage": "meta_agent_dispatch", "memory_consumer": "meta_agent"},
        )
        return retrieval_request, meta_memory.search(retrieval_request)

    def _update_meta_agent_memory(
        self,
        *,
        request: TaskRequest,
        meta_agent: MetaAgentSpec,
        meta_memory: Any | None,
        meta_memory_bundle: MemoryBundle | None,
        decision: DispatchDecision,
        run_ref: str,
        step_index: int,
        role_results: list[dict[str, Any]],
        llm_call_ref: str | None,
    ) -> Any | None:
        if meta_memory is None or meta_memory_bundle is None:
            return None
        update_result = meta_memory.add(
            request.task_id,
            meta_agent.name,
            _meta_agent_memory_update_messages(
                request=request,
                meta_agent=meta_agent,
                decision=decision,
                step_index=step_index,
                role_results=role_results,
                llm_call_ref=llm_call_ref,
            ),
        )
        _register_memory_state_update(
            registry=self.backend_state_registry,
            task_id=request.task_id,
            run_ref=run_ref,
            role=meta_agent.name,
            memory_scope="agent",
            memory_scope_id=_meta_agent_memory_scope_id(meta_agent),
            memory_bundle=meta_memory_bundle,
            update_result=update_result,
        )
        return update_result

    def _meta_agent_instruction_messages(self, meta_agent: MetaAgentSpec) -> list[Message]:
        if meta_agent.instruction_ref is None:
            return []
        path = Path(meta_agent.instruction_ref)
        return [
            Message(
                role="system",
                content=f"Meta-agent instructions from {meta_agent.instruction_ref}:\n{path.read_text(encoding='utf-8')}",
            )
        ]

    def _meta_agent_lab_state(self, request: TaskRequest) -> dict[str, Any]:
        payload = LabStateBuilder(
            task_registry=self.task_registry,
            trajectory_registry=self.trajectory_registry,
            backend_state_registry=self.backend_state_registry,
            lab_state_registry=self.lab_state_registry,
        ).build_for_meta_agent(
            request=request,
            requested_detail_refs=self._pending_lab_state_detail_requests,
        )
        self._pending_lab_state_detail_requests = {}
        return payload

    def _save_meta_agent_run(
        self,
        *,
        request: TaskRequest,
        run_ref: str,
        decision: DispatchDecision,
        step_index: int,
        role_results: list[dict[str, Any]],
        llm_call_ref: str | None = None,
        meta_memory_request: RetrievalRequest | None = None,
        meta_memory_bundle: MemoryBundle | None = None,
        meta_memory_update_result: Any | None = None,
    ) -> None:
        if not self.trajectory_collector.enabled:
            return
        metadata = {
            "step_index": step_index,
            "completed_run_refs": [result["run_ref"] for result in role_results],
            "dispatch_metadata": _json_compatible(decision.metadata),
        }
        workflow_dag = _agent_workflow_dag(decision.metadata)
        if workflow_dag is not None:
            metadata["agent_level_workflow_dag"] = workflow_dag
        if llm_call_ref is not None:
            metadata["llm_call_refs"] = [llm_call_ref]
        if meta_memory_request is not None:
            metadata["meta_memory_retrieval_request"] = _json_compatible(meta_memory_request)
        if meta_memory_bundle is not None:
            metadata["meta_memory_bundle"] = _json_compatible(meta_memory_bundle)
        if meta_memory_update_result is not None:
            metadata["meta_memory_update_result"] = _json_compatible(meta_memory_update_result)
        detail_requests = _lab_state_detail_requests(decision.metadata)
        if detail_requests:
            metadata["lab_state_detail_requests"] = detail_requests
            self._pending_lab_state_detail_requests = detail_requests
        self.trajectory_collector.save_meta_agent_run(
            MetaAgentRunRecord(
                run_ref=run_ref,
                task_id=request.task_id,
                decision=decision,
                metadata=metadata,
            )
        )

    def _meta_agent_result(
        self,
        request: TaskRequest,
        role_results: list[dict[str, Any]],
        meta_run_refs: list[str],
        decision: DispatchDecision,
    ) -> dict[str, Any]:
        if not role_results:
            raise RuntimeError("meta-agent finished without running a subagent")
        final_result = role_results[-1]
        if final_result.get("status", "completed") != "completed" and not _run_has_handoff_artifact(
            final_result,
            {"final_records"},
        ):
            raise RuntimeError(_failed_subagent_result_message("meta-agent finished with", final_result))
        return {
            "task_id": request.task_id,
            "status": "completed",
            "run_ref": final_result["run_ref"],
            "run_refs": [result["run_ref"] for result in role_results],
            "meta_run_refs": meta_run_refs,
            "runs": role_results,
            "role": final_result["role"],
            "final_answer": _final_answer_from_dispatch(decision, final_result),
        }

    def _preflight_work_item_dispatch(
        self,
        *,
        request: TaskRequest,
        decision: DispatchDecision,
        stage_index: int,
    ) -> dict[str, Any] | None:
        if decision.action != DispatchAction.RUN_SUBAGENT or decision.target_role is None:
            return None
        policy = _work_item_routing_policy(self.task_config.runtime_policy.metadata)
        if policy is None:
            return None
        work_item_id = _work_item_id_from_metadata(decision.metadata, policy["work_item_id_field"])
        if work_item_id is None:
            return None
        context = _scientific_work_item_context(
            role_instruction=decision.instruction or "",
            task_goal=request.goal,
            dispatch_metadata=decision.metadata,
        )
        issues = _work_item_preflight_issues(context)
        if not issues:
            return None
        run_ref = f"preflight-{uuid4()}"
        failure_reason = "work item preflight failed: " + "; ".join(issues)
        meta_workflow_node_id = _meta_workflow_node_id(decision.metadata, decision.target_role, stage_index)
        dispatch_metadata = {
            **decision.metadata,
            policy["work_item_id_field"]: work_item_id,
            "preflight_status": "failed",
            "preflight_issues": issues,
            "preflight_context": _json_compatible(context),
        }
        self.trajectory_collector.record_event(
            event_type="work_item_preflight_failed",
            subject_type="work_item",
            subject_ref=work_item_id,
            task_id=request.task_id,
            run_ref=run_ref,
            metadata={
                "role": decision.target_role,
                "stage_index": stage_index,
                "failure_reason": failure_reason,
                "dispatch_metadata": _json_compatible(dispatch_metadata),
            },
        )
        return {
            "task_id": request.task_id,
            "run_ref": run_ref,
            "role": decision.target_role,
            "generic_agent_type": decision.target_role,
            "meta_workflow_node_id": meta_workflow_node_id,
            "assigned_task": decision.instruction or "",
            "final_answer": failure_reason,
            "stage_index": stage_index,
            "status": "failed",
            "failure_reason": failure_reason,
            "llm_call_refs": [],
            "tool_call_count": 0,
            "artifact_refs": [],
            "dispatch_metadata": _json_compatible(dispatch_metadata),
            "completion_contract": {
                "assigned_task_complete": False,
                "ready_for_task_end": False,
                "blocking_issues": issues,
                "evidence": {
                    "status": "failed",
                    "failure_reason": failure_reason,
                    "tool_call_count": 0,
                    "artifact_count": 0,
                },
            },
            "budget": {},
        }

    def _apply_generic_work_item_retry_policy(
        self,
        *,
        request: TaskRequest,
        decision: DispatchDecision,
        role_results: list[dict[str, Any]],
    ) -> DispatchDecision:
        if decision.action != DispatchAction.RUN_SUBAGENT or decision.target_role is None:
            return decision
        policy = _work_item_routing_policy(self.task_config.runtime_policy.metadata)
        if policy is None:
            return decision
        work_item_id = _work_item_id_from_metadata(decision.metadata, policy["work_item_id_field"])
        if work_item_id is None or decision.target_role not in policy["executor_roles"]:
            return decision
        failed_attempts = _failed_executor_attempt_count(role_results, policy, work_item_id)
        if failed_attempts <= 0:
            return decision

        survey_role = _first_available_recovery_role(
            [role.name for role in self._roles()],
            preferred_names=("SurveyAgent", "Survey", "DiscoveryAgent"),
            excluded_roles=policy["executor_roles"] | policy["reviewer_roles"] | policy["finalizer_roles"],
        )
        if survey_role is not None and not _role_attempted_for_work_item(role_results, policy, survey_role, work_item_id):
            metadata = {
                **decision.metadata,
                policy["work_item_id_field"]: work_item_id,
                "route": survey_role,
                "generic_agent_type": survey_role,
                "assigned_task": (
                    f"Recover work item {work_item_id!r} before another executor attempt. "
                    "Discover source documents, supplementary files, sheets, tables, candidate rows, "
                    "and evidence locations. Produce explicit intermediate candidate artifacts when tools allow."
                ),
                "recovery_strategy": "survey_before_retry",
                "recovered_from_role": decision.target_role,
                "recovered_after_failed_attempts": failed_attempts,
                "expected_intermediate_artifacts": _generic_scientific_extraction_artifact_contracts(),
            }
            return DispatchDecision(
                action=DispatchAction.RUN_SUBAGENT,
                target_role=survey_role,
                instruction=(
                    f"Recover work item {work_item_id!r} before another executor attempt. "
                    "Discover source documents, supplementary files, sheets, tables, candidate rows, "
                    "and evidence locations. Produce explicit intermediate candidate artifacts when tools allow."
                ),
                retrieval_query=f"survey discovery candidate sources for work item {work_item_id}",
                metadata=metadata,
            )

        max_attempts = _max_failed_executor_attempts_per_work_item(policy)
        if failed_attempts >= max_attempts:
            self._save_work_item_lifecycle_event(
                request=request,
                work_item_id=work_item_id,
                status="failed",
                event={
                    "event": "retry_budget_exhausted",
                    "role": decision.target_role,
                    "failed_attempts": failed_attempts,
                    "max_failed_executor_attempts_per_work_item": max_attempts,
                    "reason": "executor retry budget exhausted before current repeated route",
                },
            )
            next_work_item_id = _next_unresolved_work_item_id(
                role_results,
                policy,
                current_work_item_id=work_item_id,
            )
            if next_work_item_id is not None:
                metadata = {
                    **decision.metadata,
                    policy["work_item_id_field"]: next_work_item_id,
                    "route": decision.target_role,
                    "generic_agent_type": decision.target_role,
                    "assigned_task": (
                        f"Process next independent work item {next_work_item_id!r}. "
                        f"The prior work item {work_item_id!r} exhausted its generic retry budget and is marked failed."
                    ),
                    "recovery_strategy": "advance_to_next_work_item",
                    "skipped_failed_work_item_id": work_item_id,
                    "previous_failed_attempts": failed_attempts,
                    "expected_intermediate_artifacts": _generic_scientific_extraction_artifact_contracts(),
                }
                return DispatchDecision(
                    action=DispatchAction.RUN_SUBAGENT,
                    target_role=decision.target_role,
                    instruction=(
                        f"Process next independent work item {next_work_item_id!r}. "
                        f"The prior work item {work_item_id!r} exhausted its generic retry budget and is marked failed."
                    ),
                    retrieval_query=decision.retrieval_query,
                    metadata=metadata,
                )
            raise RuntimeError(
                "work-item retry budget exhausted and no unresolved configured work items remain: "
                f"{work_item_id}"
            )
        return decision

    def _initialize_work_item_lifecycle(self, request: TaskRequest) -> None:
        policy = _work_item_routing_policy(self.task_config.runtime_policy.metadata)
        if policy is None:
            return
        for work_item_id in sorted(policy["required_work_item_ids"]):
            self._save_work_item_lifecycle_event(
                request=request,
                work_item_id=work_item_id,
                status="pending",
                event={"event": "initialized", "status": "pending"},
                append_if_missing_only=True,
            )

    def _record_work_item_lifecycle_from_run(self, request: TaskRequest, run_result: dict[str, Any]) -> None:
        policy = _work_item_routing_policy(self.task_config.runtime_policy.metadata)
        if policy is None:
            return
        metadata = run_result.get("dispatch_metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        work_item_id = _work_item_id_from_metadata(metadata, policy["work_item_id_field"])
        if work_item_id is None:
            return
        role = run_result.get("role")
        role = role if isinstance(role, str) else ""
        status = _work_item_lifecycle_status_for_run(run_result, role, policy)
        self._save_work_item_lifecycle_event(
            request=request,
            work_item_id=work_item_id,
            status=status,
            event={
                "event": "subagent_run",
                "run_ref": run_result.get("run_ref"),
                "role": role,
                "status": status,
                "subagent_status": run_result.get("status", "completed"),
                "failure_reason": run_result.get("failure_reason"),
                "artifact_count": len(run_result.get("artifact_refs", []))
                if isinstance(run_result.get("artifact_refs"), list)
                else 0,
                "metadata": _json_compatible(metadata),
            },
        )

    def _save_work_item_lifecycle_event(
        self,
        *,
        request: TaskRequest,
        work_item_id: str,
        status: str,
        event: dict[str, Any],
        append_if_missing_only: bool = False,
    ) -> None:
        if self.lab_state_registry is None:
            return
        root = self.lab_state_registry.root / "work_items" / _safe_state_ref(request.task_id)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{_safe_state_ref(work_item_id)}.json"
        if append_if_missing_only and path.exists():
            return
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            payload = {
                "schema_version": "v1",
                "task_id": request.task_id,
                "work_item_id": work_item_id,
                "status": "pending",
                "history": [],
            }
        history = payload.get("history")
        if not isinstance(history, list):
            history = []
        history.append({"timestamp": _utc_now(), **_json_compatible(event)})
        payload.update(
            {
                "schema_version": "v1",
                "task_id": request.task_id,
                "work_item_id": work_item_id,
                "status": status,
                "updated_at": _utc_now(),
                "history": history,
            }
        )
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _run_role(
        self,
        request: TaskRequest,
        role: Any,
        stage_index: int,
        *,
        instruction: str | None = None,
        retrieval_query: str | None = None,
        dispatch_metadata: dict[str, Any] | None = None,
        return_failed_result: bool = False,
        repair_budget: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        run_ref = f"subagent-{uuid4()}"
        dispatch_metadata = dispatch_metadata or {}
        meta_workflow_node_id = _meta_workflow_node_id(dispatch_metadata, role.name, stage_index)
        role_instruction = instruction or request.goal
        self._progress(f"subagent started: {role.name} stage={stage_index} run_ref={run_ref}")
        self.trajectory_collector.record_event(
            event_type="subagent_started",
            subject_type="subagent",
            subject_ref=run_ref,
            task_id=request.task_id,
            run_ref=run_ref,
            metadata={
                "role": role.name,
                "generic_agent_type": role.name,
                "meta_workflow_node_id": meta_workflow_node_id,
                "assigned_task": role_instruction,
                "stage_index": stage_index,
                "llm_backend_id": role.llm_backend.backend_id,
                "llm_backend_config_ref": role.llm_backend.config_ref,
                "llm_backend_state_ref": role.llm_backend.state_ref,
                "dispatch_metadata": _json_compatible(dispatch_metadata),
            },
        )
        llm = self._llm_runtime(role.llm_backend.backend_id)
        agent_memory, task_memory = self._memory_runtimes_for_scopes(role)
        skill = self._first_runtime(self.skill_runtimes, "skill")
        if self.prompt_builder is None:
            raise RuntimeError("task runtime requires prompt_builder for default dispatch")

        query = retrieval_query or role_instruction
        agent_memory_scope_id = f"agent:{role.name}"
        task_memory_scope_id = f"task:{request.task_id}"
        agent_retrieval_request = RetrievalRequest(
            task_id=request.task_id,
            role=role.name,
            query=query,
            task_origin=request.origin,
            task_purpose=request.purpose,
            filters={
                "memory_scope": "agent",
                "memory_scope_id": agent_memory_scope_id,
            },
            metadata=_generic_role_retrieval_metadata(role.name),
        )
        task_retrieval_request = RetrievalRequest(
            task_id=request.task_id,
            role="task",
            query=query,
            task_origin=request.origin,
            task_purpose=request.purpose,
            filters={
                "memory_scope": "task",
                "memory_scope_id": task_memory_scope_id,
            },
        )
        agent_memory_bundle = agent_memory.search(agent_retrieval_request)
        task_memory_bundle = task_memory.search(task_retrieval_request)
        memory_bundle = _combined_memory_bundle(agent_memory_bundle, task_memory_bundle)
        runtime_policy_for_dispatch = self.task_config.runtime_policy.model_copy(
            update={
                "metadata": _completion_policy_metadata_for_dispatch(
                    self.task_config.runtime_policy.metadata,
                    dispatch_metadata,
                )
            }
        )

        prepared_skills = prepare_skill_runtime_context(
            retrieval_request=agent_retrieval_request,
            skill_backend=skill,
            tool_runtime=self.tool_runtime,
            allowed_tools=role.allowed_tools,
            policy=runtime_policy_for_dispatch,
            role_name=role.name,
        )
        prepared_skills = _apply_subagent_skill_budget(
            prepared_skills,
            role_name=role.name,
            policy_metadata=runtime_policy_for_dispatch.metadata,
        )
        prepared_skills = _prepare_dynamic_subagent_tool_context(
            prepared_skills,
            role=role,
            tool_runtime=self.tool_runtime,
            policy=runtime_policy_for_dispatch,
            dispatch_metadata=dispatch_metadata,
        )
        budget_tracker = _subagent_budget_tracker(
            policy_metadata=runtime_policy_for_dispatch.metadata,
            role_name=role.name,
        )
        partial_repair_trajectory: list[dict[str, Any]] = []
        partial_promotion_candidates: list[dict[str, Any]] = []
        try:
            if self.task_config.runtime_policy.enable_workflow_planning and not _dispatch_requests_flat_execution(
                dispatch_metadata,
                role_instruction,
            ):
                execution = self._execute_workflow_agent(
                    request=request,
                    role=role,
                    role_instruction=role_instruction,
                    memory_bundle=memory_bundle,
                    prepared_skills=prepared_skills,
                    llm=llm,
                    run_ref=run_ref,
                    repair_budget=repair_budget,
                    repair_trajectory=partial_repair_trajectory,
                    promotion_candidates=partial_promotion_candidates,
                    dispatch_metadata=dispatch_metadata,
                    budget_tracker=budget_tracker,
                )
            else:
                execution = self._execute_flat_agent(
                    request=request,
                    role=role,
                    role_instruction=role_instruction,
                    memory_bundle=memory_bundle,
                    prepared_skills=prepared_skills,
                    llm=llm,
                    run_ref=run_ref,
                    repair_budget=repair_budget,
                    repair_trajectory=partial_repair_trajectory,
                    promotion_candidates=partial_promotion_candidates,
                    dispatch_metadata=dispatch_metadata,
                    budget_tracker=budget_tracker,
                )
        except _SubagentBudgetExceeded as exc:
            execution = _RoleExecutionPayload(
                prompt_messages=[],
                output_message=Message(role="assistant", content=str(exc)),
                tool_trace_records=[],
                tool_trace=ToolTrace(run_ref=run_ref, calls=[]),
                artifact_refs=[],
                final_answer=str(exc),
                skill_bundle=prepared_skills.skill_bundle,
                skill_context=prepared_skills.skill_context,
                repair_trajectory=partial_repair_trajectory,
                promotion_candidates=partial_promotion_candidates,
                status="budget_exceeded",
                failure_reason=str(exc),
                budget=exc.metadata,
            )
        except Exception as exc:
            self._record_partial_subagent_postmortem(
                request=request,
                run_ref=run_ref,
                role=role.name,
                stage_index=stage_index,
                instruction=role_instruction,
                retrieval_request=agent_retrieval_request,
                memory_bundle=memory_bundle,
                skill_bundle=prepared_skills.skill_bundle,
                llm_backend_id=role.llm_backend.backend_id,
                llm_backend_config_ref=role.llm_backend.config_ref,
                llm_backend_state_ref=role.llm_backend.state_ref,
                metadata={
                    "error": str(exc),
                    "generic_agent_type": role.name,
                    "meta_workflow_node_id": meta_workflow_node_id,
                    "assigned_task": role_instruction,
                    "stage_index": stage_index,
                    "dispatch_metadata": _json_compatible(dispatch_metadata),
                    "repair_trajectory": _json_compatible(partial_repair_trajectory),
                    "promotion_candidates": _json_compatible(partial_promotion_candidates),
                    "skill_context": _json_compatible(prepared_skills.skill_context),
                },
            )
            raise
        skill_bundle = execution.skill_bundle or prepared_skills.skill_bundle
        skill_context = execution.skill_context or prepared_skills.skill_context
        prompt_messages = execution.prompt_messages
        output_message = execution.output_message
        tool_trace_records = execution.tool_trace_records
        artifact_refs = execution.artifact_refs
        final_answer = execution.final_answer
        tool_trace = execution.tool_trace
        status = execution.status
        failure_reason = execution.failure_reason
        completion_policy_metadata = _completion_policy_metadata_for_dispatch(
            self.task_config.runtime_policy.metadata,
            dispatch_metadata,
        )
        completion_contract = _subagent_completion_contract(
            status=status,
            failure_reason=failure_reason,
            artifact_refs=artifact_refs,
            node_records=execution.node_execution_records,
            role=role.name,
            assigned_task=role_instruction,
            expected_outputs=_expected_outputs_for_completion(dispatch_metadata),
            tool_trace_records=tool_trace_records,
            final_answer=final_answer,
            policy_metadata=completion_policy_metadata,
        )
        if status == "completed" and completion_contract["blocking_issues"]:
            status = "guard_failed"
            failure_reason = _completion_contract_failure_reason(completion_contract)
            completion_contract = {
                **completion_contract,
                "assigned_task_complete": False,
                "ready_for_task_end": False,
                "evidence": {
                    **completion_contract["evidence"],
                    "status": status,
                },
            }
        memory_update_messages = execution.memory_update_messages or _messages_for_memory_update(
            prompt_messages,
            output_message,
        )
        agent_memory_update_result = agent_memory.add(
            request.task_id,
            role.name,
            memory_update_messages,
        )
        task_memory_update_result = task_memory.add(
            request.task_id,
            "task",
            memory_update_messages,
        )
        _register_memory_state_update(
            registry=self.backend_state_registry,
            task_id=request.task_id,
            run_ref=run_ref,
            role=role.name,
            memory_scope="agent",
            memory_scope_id=agent_memory_scope_id,
            memory_bundle=agent_memory_bundle,
            update_result=agent_memory_update_result,
        )
        _register_memory_state_update(
            registry=self.backend_state_registry,
            task_id=request.task_id,
            run_ref=run_ref,
            role=role.name,
            memory_scope="task",
            memory_scope_id=task_memory_scope_id,
            memory_bundle=task_memory_bundle,
            update_result=task_memory_update_result,
        )
        memory_update_result = agent_memory_update_result
        observation_metadata = {
            "agent_retrieval_request": _json_compatible(agent_retrieval_request),
            "task_retrieval_request": _json_compatible(task_retrieval_request),
            "memory_state_ref": memory_bundle.state_ref,
            "agent_memory_state_ref": agent_memory_bundle.state_ref,
            "task_memory_state_ref": task_memory_bundle.state_ref,
            "repair_trajectory": _json_compatible(execution.repair_trajectory),
            "promotion_candidates": _json_compatible(execution.promotion_candidates),
        }
        if execution.workflow_plan is not None:
            observation_metadata.update(
                {
                    "workflow_plan": _json_compatible(execution.workflow_plan),
                    "plan_execution_trace": _json_compatible(execution.plan_execution_trace),
                    "node_execution_records": _json_compatible(execution.node_execution_records),
                    "workflow_topological_order": execution.workflow_plan.metadata.get("topological_order", []),
                    "artifact_refs": _json_compatible(artifact_refs),
                    "skill_context": _json_compatible(skill_context),
                }
            )
        skill_observation_request = SkillObservationRequest(
            task_id=request.task_id,
            run_ref=run_ref,
            role=role.name,
            retrieval_request=agent_retrieval_request,
            skill_bundle=skill_bundle,
            graph_version_ref=skill_bundle.graph_version_ref,
            skill_state_ref=skill_bundle.skill_state_ref,
            tool_trace=tool_trace,
            final_answer=final_answer,
            metadata=observation_metadata,
        )
        skill_update_result = skill.look_at(skill_observation_request.model_dump(mode="json"))
        parsed_skill_update_result = _parse_skill_update_result(skill_update_result)
        _register_skill_state_update(
            registry=self.backend_state_registry,
            request=request,
            run_ref=run_ref,
            skill_bundle=skill_bundle,
            update_result=parsed_skill_update_result,
        )

        if self.trajectory_collector.enabled:
            run_metadata = {
                "parent_task_id": request.parent_task_id,
                "status": status,
                "failure_reason": failure_reason,
                "generic_agent_type": role.name,
                "meta_workflow_node_id": meta_workflow_node_id,
                "assigned_task": role_instruction,
                "stage_index": stage_index,
                "run_ref": run_ref,
                "dispatch_metadata": _json_compatible(dispatch_metadata),
                "tool_bundle": _json_compatible(prepared_skills.tool_bundle),
                "tool_trace": _json_compatible(tool_trace),
                "skill_context": _json_compatible(skill_context),
                "skill_observation_request": _json_compatible(skill_observation_request),
                "agent_memory_bundle": _json_compatible(agent_memory_bundle),
                "task_memory_bundle": _json_compatible(task_memory_bundle),
                "agent_memory_update_result": _json_compatible(agent_memory_update_result),
                "task_memory_update_result": _json_compatible(task_memory_update_result),
                "memory_update_result": _json_compatible(memory_update_result),
                "skill_update_result": _json_compatible(skill_update_result),
                "repair_trajectory": _json_compatible(execution.repair_trajectory),
                "promotion_candidates": _json_compatible(execution.promotion_candidates),
                "completion_contract": _json_compatible(completion_contract),
                "budget": _json_compatible(execution.budget),
            }
            if execution.workflow_plan is not None:
                run_metadata.update(
                    {
                        "workflow_plan": _json_compatible(execution.workflow_plan),
                        "plan_execution_trace": _json_compatible(execution.plan_execution_trace),
                        "node_execution_records": _json_compatible(execution.node_execution_records),
                    }
                )
            self.trajectory_collector.save_subagent_run(
                SubagentRunRecord(
                    run_ref=run_ref,
                    task_id=request.task_id,
                    task_origin=request.origin,
                    task_purpose=request.purpose,
                    producer_ref=request.producer_ref,
                    round_id=request.round_id,
                    human_anchor_task_refs=_human_anchor_task_refs(request),
                    human_anchor_trajectory_refs=_human_anchor_trajectory_refs(request),
                    proposed_relation_type=_proposed_relation_type(request),
                    expected_transfer=_expected_transfer(request),
                    stage_index=stage_index,
                    role=role.name,
                    instruction=role_instruction,
                    retrieval_request=agent_retrieval_request,
                    memory_bundle=memory_bundle,
                    skill_bundle=skill_bundle,
                    prompt_messages=prompt_messages,
                    llm_call_refs=execution.llm_call_refs,
                    llm_backend_id=role.llm_backend.backend_id,
                    llm_backend_config_ref=role.llm_backend.config_ref,
                    llm_backend_state_ref=role.llm_backend.state_ref,
                    tool_calls=tool_trace_records,
                    output_messages=[output_message],
                    artifact_refs=artifact_refs,
                    metadata=run_metadata,
                )
            )
        self.trajectory_collector.record_event(
            event_type="subagent_completed" if status == "completed" else "subagent_failed",
            subject_type="subagent",
            subject_ref=run_ref,
            task_id=request.task_id,
            run_ref=run_ref,
            metadata={
                "role": role.name,
                "generic_agent_type": role.name,
                "meta_workflow_node_id": meta_workflow_node_id,
                "assigned_task": role_instruction,
                "stage_index": stage_index,
                "status": status,
                "failure_reason": failure_reason,
                "llm_call_refs": execution.llm_call_refs,
                "tool_call_count": len(tool_trace_records),
                "artifact_refs": [ref.model_dump(mode="json") for ref in artifact_refs],
                "completion_contract": _json_compatible(completion_contract),
            },
        )
        self._progress(
            f"subagent {status}: {role.name} stage={stage_index} "
            f"tools={len(tool_trace_records)} artifacts={len(artifact_refs)}"
        )
        self._record_lab_state_subagent_outputs(
            request=request,
            run_ref=run_ref,
            role=role.name,
            status=status,
            assigned_task=role_instruction,
            final_answer=final_answer,
            stage_index=stage_index,
            artifact_refs=artifact_refs,
            llm_call_refs=execution.llm_call_refs,
            tool_call_count=len(tool_trace_records),
            failure_reason=failure_reason,
            meta_workflow_node_id=meta_workflow_node_id,
            dispatch_metadata=dispatch_metadata,
            execution=execution,
        )

        result = {
            "task_id": request.task_id,
            "run_ref": run_ref,
            "role": role.name,
            "generic_agent_type": role.name,
            "meta_workflow_node_id": meta_workflow_node_id,
            "assigned_task": role_instruction,
            "final_answer": final_answer,
            "stage_index": stage_index,
            "status": status,
            "failure_reason": failure_reason,
            "llm_call_refs": execution.llm_call_refs,
            "tool_call_count": len(tool_trace_records),
            "artifact_refs": [ref.model_dump(mode="json") for ref in artifact_refs],
            "dispatch_metadata": _json_compatible(dispatch_metadata),
            "completion_contract": _json_compatible(completion_contract),
            "budget": _json_compatible(execution.budget),
        }
        agent_level_workflow_dag = _agent_workflow_dag(dispatch_metadata)
        if agent_level_workflow_dag is not None:
            result["agent_level_workflow_dag"] = agent_level_workflow_dag
        if status != "completed" and status != "budget_exceeded" and not return_failed_result:
            raise RuntimeError(_failed_subagent_result_message("task runtime", result))
        return result

    def _progress(self, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(f"[EvoLab] {message}")

    def _upstream_outputs_for_prompt(self, request: TaskRequest) -> dict[str, Any]:
        if self.lab_state_registry is None:
            return {"reports": [], "artifacts": []}
        reports = self.lab_state_registry.list_subagent_reports(request.task_id)[-20:]
        artifacts = self.lab_state_registry.list_artifacts(request.task_id)[-50:]
        return {
            "reports": [
                {
                    "report_ref": report.report_ref,
                    "run_ref": report.run_ref,
                    "role": report.role,
                    "status": report.status,
                    "assigned_task": _truncate_text(report.assigned_task, 1_000),
                    "summary": _truncate_text(report.summary, 2_000),
                    "artifact_refs": [ref.model_dump(mode="json") for ref in report.artifact_refs],
                    "coverage": _json_compatible(report.coverage),
                    "failures": _json_compatible(report.failures),
                    "completion_contract": _json_compatible(report.metadata.get("completion_contract")),
                }
                for report in reports
            ],
            "artifacts": [
                {
                    "artifact_ref": artifact.artifact_ref,
                    "uri": artifact.uri,
                    "artifact_type": artifact.artifact_type,
                    "status": artifact.status,
                    "producer_run_ref": artifact.producer_run_ref,
                    "producer_role": artifact.metadata.get("producer_role") or artifact.role,
                    "source_report_ref": artifact.metadata.get("source_report_ref"),
                    "metadata": _json_compatible(artifact.metadata),
                }
                for artifact in artifacts
            ],
        }

    def _bootstrap_scientific_handoff_artifacts(
        self,
        *,
        request: TaskRequest,
        role: str,
        role_instruction: str,
        run_ref: str,
        artifact_refs: list[ArtifactRef],
        tool_trace_records: list[ToolCallRecord],
        dispatch_metadata: dict[str, Any] | None = None,
        runtime_stage: str | None = None,
    ) -> list[ToolCallRecord]:
        if self.tool_runtime is None:
            return []
        if not _scientific_handoff_bootstrap_enabled(self.task_config.runtime_policy.metadata):
            return []
        tool_calls = _scientific_handoff_bootstrap_calls(
            role=role,
            role_instruction=role_instruction,
            task_goal=request.goal,
            dispatch_metadata=dispatch_metadata or {},
            runtime_metadata=self.task_config.runtime_policy.metadata,
            lab_state_registry=self.lab_state_registry,
            task_id=request.task_id,
            artifact_root=(self.lab_root / "artifacts" / "tools") if self.lab_root is not None else Path("artifacts/tools"),
        )
        bootstrapped: list[ToolCallRecord] = []
        for index, tool_call in enumerate(tool_calls):
            if not _tool_runtime_has_registered_tool(self.tool_runtime, tool_call["name"]):
                break
            call = ToolCall(
                call_id=f"bootstrap-{index + 1}-{tool_call['name']}",
                name=tool_call["name"],
                arguments=tool_call["arguments"],
            )
            result = self.tool_runtime.execute_registered_tool_name(
                call_id=call.call_id,
                name=call.name,
                arguments=call.arguments,
            )
            result = _manage_tool_result_artifacts(
                result=result,
                request=request,
                run_ref=run_ref,
                artifact_root_factory=self.tool_artifact_root_factory,
            )
            artifact_refs.extend(result.artifact_refs)
            _register_tool_result_artifacts(result, self.tool_artifact_registrar)
            record = ToolCallRecord(tool_call=call, result=result)
            tool_trace_records.append(record)
            bootstrapped.append(record)
            self.trajectory_collector.record_tool_call(
                run_ref=run_ref,
                task_id=request.task_id,
                record=record,
                role=role,
                runtime_stage=runtime_stage,
                metadata={"scientific_handoff_bootstrap": True},
            )
            if result.status != "ok":
                break
        return bootstrapped

    def _record_lab_state_subagent_outputs(
        self,
        *,
        request: TaskRequest,
        run_ref: str,
        role: str,
        status: str,
        assigned_task: str,
        final_answer: str,
        stage_index: int,
        artifact_refs: list[ArtifactRef],
        llm_call_refs: list[str],
        tool_call_count: int,
        failure_reason: str | None,
        meta_workflow_node_id: str,
        dispatch_metadata: dict[str, Any],
        execution: _RoleExecutionPayload,
    ) -> None:
        if self.lab_state_registry is None:
            return
        normalized_status = status if status in {"completed", "failed", "guard_failed", "interrupted", "partial"} else "partial"
        report_ref = f"report-{run_ref}"
        report_coverage = _coverage_from_subagent_summary(final_answer)
        report_failures = [{"reason": failure_reason}] if failure_reason else _failures_from_node_records(execution.node_execution_records)
        completion_contract = _subagent_completion_contract(
            status=status,
            failure_reason=failure_reason,
            artifact_refs=artifact_refs,
            node_records=execution.node_execution_records,
            role=role,
            assigned_task=assigned_task,
            expected_outputs=_expected_outputs_for_completion(dispatch_metadata),
            tool_trace_records=execution.tool_trace_records,
            final_answer=final_answer,
            policy_metadata=_completion_policy_metadata_for_dispatch(
                self.task_config.runtime_policy.metadata,
                dispatch_metadata,
            ),
        )
        self.lab_state_registry.save_subagent_report(
            SubagentReportRecord(
                report_ref=report_ref,
                task_id=request.task_id,
                run_ref=run_ref,
                role=role,
                status=normalized_status,  # type: ignore[arg-type]
                assigned_task=assigned_task,
                summary=final_answer,
                artifact_refs=artifact_refs,
                coverage=report_coverage,
                failures=report_failures,
                skipped_items=_skipped_items_from_node_records(execution.node_execution_records),
                metadata={
                    "stage_index": stage_index,
                    "generic_agent_type": role,
                    "meta_workflow_node_id": meta_workflow_node_id,
                    "tool_call_count": tool_call_count,
                    "llm_call_refs": llm_call_refs,
                    "dispatch_metadata": _json_compatible(dispatch_metadata),
                    "internal_dag": _internal_dag_summary(execution.workflow_plan),
                    "retrieved_skills": _retrieved_skill_summary(execution.skill_bundle),
                    "prepared_tools_by_node": _prepared_tools_by_node(execution.node_execution_records),
                    "tool_calls": _tool_call_summary(execution.tool_trace_records),
                    "artifacts": [ref.model_dump(mode="json") for ref in artifact_refs],
                    "completion_contract": _json_compatible(completion_contract),
                    "budget": _json_compatible(execution.budget),
                },
            )
        )
        for index, artifact in enumerate(artifact_refs):
            self.lab_state_registry.save_artifact_index_record(
                ArtifactIndexRecord(
                    artifact_ref=f"artifact-{run_ref}-{index + 1:03d}",
                    task_id=request.task_id,
                    producer_run_ref=run_ref,
                    uri=artifact.uri,
                    artifact_type=artifact.type,
                    role=_artifact_role(artifact) or role,
                    status=_artifact_index_status(artifact),
                    metadata={
                        **_json_compatible(artifact.metadata),
                        "source_report_ref": report_ref,
                        "producer_role": role,
                        "semantic_kind": _artifact_semantic_kind(artifact),
                    },
                )
            )
        if llm_call_refs:
            self.lab_state_registry.save_training_index_record(
                TrainingIndexRecord(
                    sample_ref=f"training-{run_ref}",
                    task_id=request.task_id,
                    source_run_ref=run_ref,
                    source_llm_call_refs=llm_call_refs,
                    sample_kind="subagent_trace",
                    quality_label="accepted" if status == "completed" else "candidate",
                    metadata={
                        "role": role,
                        "stage_index": stage_index,
                        "tool_call_count": tool_call_count,
                        "artifact_count": len(artifact_refs),
                    },
                )
            )

    def _execute_flat_agent(
        self,
        *,
        request: TaskRequest,
        role: Any,
        role_instruction: str,
        memory_bundle: MemoryBundle,
        prepared_skills: Any,
        llm: Any,
        run_ref: str,
        repair_budget: dict[str, int] | None = None,
        repair_trajectory: list[dict[str, Any]] | None = None,
        promotion_candidates: list[dict[str, Any]] | None = None,
        dispatch_metadata: dict[str, Any] | None = None,
        budget_tracker: _SubagentBudgetTracker | None = None,
    ) -> _RoleExecutionPayload:
        assert self.prompt_builder is not None
        active_skill_bundle = prepared_skills.skill_bundle
        active_skill_context = prepared_skills.skill_context
        if repair_trajectory is None:
            repair_trajectory = []
        if promotion_candidates is None:
            promotion_candidates = []
        policy_metadata = _completion_policy_metadata_for_dispatch(
            self.task_config.runtime_policy.metadata,
            dispatch_metadata or {},
        )
        expected_outputs = (
            _expected_outputs_for_completion(dispatch_metadata or {})
            if _dispatch_is_dynamic(dispatch_metadata)
            else []
        )
        tool_trace_records: list[ToolCallRecord] = []
        artifact_refs: list[ArtifactRef] = []
        bootstrap_records = self._bootstrap_scientific_handoff_artifacts(
            request=request,
            role=role.name,
            role_instruction=role_instruction,
            run_ref=run_ref,
            artifact_refs=artifact_refs,
            tool_trace_records=tool_trace_records,
            dispatch_metadata=dispatch_metadata,
            runtime_stage="subagent_flat_bootstrap",
        )
        prompt_messages = self.prompt_builder.build(
            role,
            role_instruction,
            memory_bundle,
            active_skill_bundle,
            skill_context={
                **active_skill_context,
                "upstream_outputs": self._upstream_outputs_for_prompt(request),
                "bootstrap_artifact_refs": [ref.model_dump(mode="json") for ref in artifact_refs],
                "bootstrap_tool_calls": _tool_call_summary(bootstrap_records),
            },
        )
        tool_specs = [spec.model_dump(mode="json") for spec in prepared_skills.tool_bundle.tool_specs]
        llm_call_refs: list[str] = []
        response = None
        status = "completed"
        failure_reason = None
        tool_call_budget = self.task_config.runtime_policy.max_tool_steps
        tool_calls_used = 0
        role_completion_guard_satisfied = False
        completion_guard_violations = 0
        max_completion_guard_violations = _completion_guard_violation_limit(
            policy_metadata,
            role.name,
        )
        repeated_suppression_violations = 0
        max_repeated_suppression_violations = _repeated_suppression_violation_limit(
            self.task_config.runtime_policy.metadata,
        )
        for step_index in range(self.task_config.runtime_policy.max_tool_steps + 1):
            try:
                if budget_tracker is not None:
                    budget_tracker.check(f"flat_step:{step_index}:before_llm_call")
                generation_config = LLMGenerationConfig(model="")
                input_messages = _copy_messages(prompt_messages)
                response = llm.generate(prompt_messages, tool_specs, generation_config)
                if budget_tracker is not None:
                    budget_tracker.note_llm_call()
                    budget_tracker.check(f"flat_step:{step_index}:after_llm_call")
            except _SubagentBudgetExceeded as exc:
                status = "budget_exceeded"
                failure_reason = str(exc)
                response = LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content=str(exc)))
                break
            call_ref = self._save_llm_call(
                run_ref=run_ref,
                backend_id=role.llm_backend.backend_id,
                llm=llm,
                generation_config=generation_config,
                input_messages=input_messages,
                tool_specs=tool_specs,
                response=response,
                metadata={
                    "task_id": request.task_id,
                    "role": role.name,
                    "runtime_stage": "subagent_flat",
                    "step_index": step_index,
                },
            )
            if call_ref is not None:
                llm_call_refs.append(call_ref)
            action_name = response.action.action
            if action_name == "final_answer":
                rejection_reason = _final_answer_rejection_reason(
                    policy_metadata,
                    tool_trace_records,
                    role.name,
                    expected_outputs=expected_outputs,
                    artifact_refs=artifact_refs,
                    final_answer=response.action.content or "",
                )
                if rejection_reason:
                    if step_index >= self.task_config.runtime_policy.max_tool_steps:
                        status = "guard_failed"
                        failure_reason = rejection_reason
                        prompt_messages.append(Message(role="assistant", content=response.action.content or ""))
                        prompt_messages.append(_final_answer_rejection_message(rejection_reason))
                        break
                    prompt_messages.append(Message(role="assistant", content=response.action.content or ""))
                    prompt_messages.append(_final_answer_rejection_message(rejection_reason))
                    continue
                break
            if action_name != "tool_call":
                raise NotImplementedError(f"task runtime action is not implemented: {action_name}")
            tool_calls = _tool_calls_from_action(response.action)
            if not tool_calls:
                raise RuntimeError("tool_call action did not include any ToolCall")
            for tool_call_index, tool_call in enumerate(tool_calls):
                if tool_calls_used >= tool_call_budget:
                    status = "failed"
                    failure_reason = "task runtime exceeded max_tool_steps before final answer"
                    response = LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content=failure_reason))
                    break
                if budget_tracker is not None:
                    budget_tracker.check(f"flat_step:{step_index}:tool_call:{tool_call_index}:before_tool_call")
                outcome = self._execute_tool_call(
                    tool_call,
                    request=request,
                    run_ref=run_ref,
                    artifact_refs=artifact_refs,
                    tool_trace_records=tool_trace_records,
                    role=role.name,
                    runtime_stage="subagent_flat",
                    step_index=step_index,
                    task_goal=request.goal,
                    active_skill_bundle=active_skill_bundle,
                    repair_budget=repair_budget,
                    step_key=f"flat:{step_index}:{tool_call.call_id}",
                    repair_log=repair_trajectory,
                    remaining_tool_budget=tool_call_budget - tool_calls_used,
                    policy_metadata=policy_metadata,
                )
                if outcome.counts_against_budget:
                    tool_calls_used += 1
                if _is_completion_guard_violation(outcome.record):
                    completion_guard_violations += 1
                    if completion_guard_violations >= max_completion_guard_violations:
                        status = "guard_failed"
                        failure_reason = _completion_guard_violation_failure_reason(
                            outcome.record,
                            violation_count=completion_guard_violations,
                            violation_limit=max_completion_guard_violations,
                        )
                        response = LLMRuntimeResponse(
                            action=SubAgentAction(action="final_answer", content=failure_reason)
                        )
                else:
                    completion_guard_violations = 0
                if _is_repeated_tool_call_suppression(outcome.record):
                    repeated_suppression_violations += 1
                    if repeated_suppression_violations >= max_repeated_suppression_violations:
                        status = "failed"
                        failure_reason = _repeated_suppression_failure_reason(
                            outcome.record,
                            violation_count=repeated_suppression_violations,
                            violation_limit=max_repeated_suppression_violations,
                        )
                        response = LLMRuntimeResponse(
                            action=SubAgentAction(action="final_answer", content=failure_reason)
                        )
                if budget_tracker is not None:
                    if outcome.counts_against_budget:
                        budget_tracker.note_tool_call()
                    try:
                        budget_tracker.check(f"flat_step:{step_index}:tool_call:{tool_call_index}:after_tool_call")
                    except _SubagentBudgetExceeded as exc:
                        status = "budget_exceeded"
                        failure_reason = str(exc)
                        response = LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content=str(exc)))
                if outcome.updated_skill_bundle is not None:
                    active_skill_bundle = outcome.updated_skill_bundle
                    active_skill_context = outcome.updated_skill_context or active_skill_context
                if outcome.promotion_candidates:
                    promotion_candidates.extend(outcome.promotion_candidates)
                prompt_messages.extend(outcome.repair_messages)
                prompt_messages.append(
                    _tool_result_message(
                        outcome.record.tool_call.name,
                        outcome.record.result,
                        policy_metadata=policy_metadata,
                    )
                )
                if _role_completion_guards_satisfied(
                    policy_metadata,
                    role.name,
                    tool_trace_records,
                ):
                    role_completion_guard_satisfied = True
                    response = LLMRuntimeResponse(
                        action=SubAgentAction(
                            action="final_answer",
                            content="role completion guards satisfied after successful required tool calls",
                        )
                    )
                    break
                if status == "budget_exceeded":
                    break
                if status == "guard_failed":
                    break
                if status == "failed":
                    break
            if status == "failed":
                break
            if status == "budget_exceeded":
                break
            if status == "guard_failed":
                break
            if role_completion_guard_satisfied:
                break

        if response is None:
            raise RuntimeError("task runtime did not receive an LLM response")
        final_answer = response.action.content or ""
        output_message = Message(role="assistant", content=final_answer)
        tool_trace = ToolTrace(run_ref=run_ref, calls=tool_trace_records)
        memory_update_messages = _flat_memory_update_messages(
            role=role.name,
            role_instruction=role_instruction,
            tool_trace_records=tool_trace_records,
            artifact_refs=artifact_refs,
            final_answer=final_answer,
            status=status,
        )
        return _RoleExecutionPayload(
            prompt_messages=prompt_messages,
            output_message=output_message,
            tool_trace_records=tool_trace_records,
            tool_trace=tool_trace,
            artifact_refs=artifact_refs,
            final_answer=final_answer,
            skill_bundle=active_skill_bundle,
            skill_context=active_skill_context,
            repair_trajectory=repair_trajectory,
            promotion_candidates=promotion_candidates,
            llm_call_refs=llm_call_refs,
            status=status,
            failure_reason=failure_reason,
            memory_update_messages=memory_update_messages,
            budget=budget_tracker.metadata() if budget_tracker is not None else {},
        )

    def _execute_workflow_agent(
        self,
        *,
        request: TaskRequest,
        role: Any,
        role_instruction: str,
        memory_bundle: MemoryBundle,
        prepared_skills: Any,
        llm: Any,
        run_ref: str,
        repair_budget: dict[str, int] | None = None,
        repair_trajectory: list[dict[str, Any]] | None = None,
        promotion_candidates: list[dict[str, Any]] | None = None,
        dispatch_metadata: dict[str, Any] | None = None,
        budget_tracker: _SubagentBudgetTracker | None = None,
    ) -> _RoleExecutionPayload:
        policy = self.task_config.runtime_policy
        dispatch_metadata = dispatch_metadata or {}
        policy_metadata = _completion_policy_metadata_for_dispatch(
            policy.metadata,
            dispatch_metadata,
        )
        active_skill_bundle = prepared_skills.skill_bundle
        if repair_trajectory is None:
            repair_trajectory = []
        if promotion_candidates is None:
            promotion_candidates = []
        workflow_plan = SkillWorkflowPlanner().plan(
            task_id=request.task_id,
            task_goal=request.goal,
            role=role.name,
            skill_bundle=active_skill_bundle,
        )
        workflow_plan, active_skill_bundle = _ensure_workflow_has_fallback_node(
            workflow_plan=workflow_plan,
            skill_bundle=active_skill_bundle,
            role_name=role.name,
            role_instruction=role_instruction,
            required_tools=prepared_skills.skill_bundle.required_tools,
        )
        if dispatch_metadata:
            workflow_plan.metadata["meta_dispatch_metadata"] = _json_compatible(dispatch_metadata)
            workflow_plan.metadata["meta_workflow_node_id"] = _meta_workflow_node_id(dispatch_metadata, role.name, 0)
        skill_context = {
            **prepared_skills.skill_context,
            "workflow_plan": workflow_plan.model_dump(mode="json"),
        }
        node_by_id = {node.node_id: node for node in workflow_plan.nodes}
        ordered_node_ids = [
            node_id
            for node_id in workflow_plan.metadata.get("topological_node_order", [])
            if isinstance(node_id, str) and node_id in node_by_id
        ] or [node.node_id for node in workflow_plan.nodes]
        role_budget = _subagent_budget(policy.metadata, role.name)
        max_nodes = _optional_positive_int(role_budget.get("max_internal_dag_nodes")) or policy.max_workflow_nodes
        executable_node_ids = ordered_node_ids[:max_nodes]
        skipped_node_ids = ordered_node_ids[max_nodes:]
        if skipped_node_ids:
            warnings = workflow_plan.metadata.setdefault("planning_warnings", [])
            warnings.extend([item for item in active_skill_bundle.metadata.get("budget_warnings", []) if isinstance(item, str)])
            warnings.append(
                f"workflow execution limited to {max_nodes} nodes; skipped {len(skipped_node_ids)} nodes"
            )
        elif active_skill_bundle.metadata.get("budget_warnings"):
            workflow_plan.metadata.setdefault("planning_warnings", []).extend(
                [item for item in active_skill_bundle.metadata.get("budget_warnings", []) if isinstance(item, str)]
            )

        all_prompt_messages: list[Message] = []
        tool_trace_records: list[ToolCallRecord] = []
        artifact_refs: list[ArtifactRef] = []
        bootstrap_records = self._bootstrap_scientific_handoff_artifacts(
            request=request,
            role=role.name,
            role_instruction=role_instruction,
            run_ref=run_ref,
            artifact_refs=artifact_refs,
            tool_trace_records=tool_trace_records,
            dispatch_metadata=dispatch_metadata,
            runtime_stage="subagent_workflow_bootstrap",
        )
        if bootstrap_records:
            skill_context["bootstrap_artifact_refs"] = [ref.model_dump(mode="json") for ref in artifact_refs]
            skill_context["bootstrap_tool_calls"] = _tool_call_summary(bootstrap_records)
        node_records: list[NodeExecutionRecord] = []
        llm_call_refs: list[str] = []
        previous_summaries: list[dict[str, str]] = []
        failed = False
        budget_failure_reason: str | None = None
        guard_failure_reason: str | None = None
        completion_guard_stop_reason: str | None = None

        for node_id in executable_node_ids:
            node = node_by_id[node_id]
            if completion_guard_stop_reason is not None:
                node_records.append(_skipped_node_record(node, completion_guard_stop_reason))
                continue
            if budget_failure_reason is not None:
                node_records.append(_skipped_node_record(node, budget_failure_reason))
                continue
            if failed:
                node_records.append(_skipped_node_record(node, "skipped after prior node failure"))
                continue
            try:
                if budget_tracker is not None:
                    budget_tracker.check(f"workflow_node:{node.node_id}:before_node")
            except _SubagentBudgetExceeded as exc:
                budget_failure_reason = str(exc)
                node_records.append(_skipped_node_record(node, budget_failure_reason))
                continue
            node_tool_bundle = _workflow_node_tool_bundle(
                tool_runtime=self.tool_runtime,
                node=node,
                role_allowed_tools=role.allowed_tools,
                policy=policy,
                fallback_tool_bundle=prepared_skills.tool_bundle,
                role_name=role.name,
            )
            node_tool_specs = [spec.model_dump(mode="json") for spec in node_tool_bundle.tool_specs]
            node_messages = self._workflow_node_prompt_messages(
                role=role,
                role_instruction=role_instruction,
                request=request,
                memory_bundle=memory_bundle,
                prepared_skills=prepared_skills,
                skill_context=skill_context,
                node=node,
                previous_summaries=previous_summaries,
                artifact_refs=artifact_refs,
                available_tool_names=[spec["name"] for spec in node_tool_specs],
            )
            all_prompt_messages.extend(node_messages)
            started_at = _utc_now()
            node_tool_records: list[ToolCallRecord] = []
            node_tool_calls_used = 0
            completion_guard_violations = 0
            max_completion_guard_violations = _completion_guard_violation_limit(policy_metadata, role.name)
            repeated_suppression_violations = 0
            max_repeated_suppression_violations = _repeated_suppression_violation_limit(policy.metadata)
            artifact_start = len(artifact_refs)
            output_summary: str | None = None
            status = "completed"
            node_should_stop = False
            max_steps = policy.max_tool_steps_per_node
            if max_steps is None:
                max_steps = policy.max_tool_steps
            response = None
            for step_index in range(max_steps + 1):
                try:
                    if budget_tracker is not None:
                        budget_tracker.check(f"workflow_node:{node.node_id}:step:{step_index}:before_llm_call")
                    generation_config = LLMGenerationConfig(model="")
                    input_messages = _copy_messages(node_messages)
                    response = llm.generate(node_messages, node_tool_specs, generation_config)
                    if budget_tracker is not None:
                        budget_tracker.note_llm_call()
                        budget_tracker.check(f"workflow_node:{node.node_id}:step:{step_index}:after_llm_call")
                except _SubagentBudgetExceeded as exc:
                    status = "failed"
                    budget_failure_reason = str(exc)
                    output_summary = budget_failure_reason
                    response = LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content=budget_failure_reason))
                    break
                call_ref = self._save_llm_call(
                    run_ref=run_ref,
                    backend_id=role.llm_backend.backend_id,
                    llm=llm,
                    generation_config=generation_config,
                    input_messages=input_messages,
                    tool_specs=node_tool_specs,
                    response=response,
                    metadata={
                        "task_id": request.task_id,
                        "role": role.name,
                        "generic_agent_type": role.name,
                        "runtime_stage": "workflow_node",
                        "workflow_node_id": node.node_id,
                        "workflow_skill_id": node.skill_id,
                        "workflow_node_name": node.name,
                        "prepared_tool_names": [spec["name"] for spec in node_tool_specs],
                        "step_index": step_index,
                    },
                )
                if call_ref is not None:
                    llm_call_refs.append(call_ref)
                action_name = response.action.action
                if action_name == "final_answer":
                    output_summary = response.action.content or ""
                    break
                if action_name != "tool_call":
                    status = "failed"
                    output_summary = f"unsupported workflow node action: {action_name}"
                    break
                tool_calls = _tool_calls_from_action(response.action)
                if not tool_calls:
                    status = "failed"
                    output_summary = "tool_call action did not include any ToolCall"
                    break
                for tool_call_index, tool_call in enumerate(tool_calls):
                    if node_tool_calls_used >= max_steps:
                        output_summary = "workflow node exceeded max tool calls before final answer"
                        if _workflow_node_can_continue_after_local_budget(
                            policy.metadata,
                            role.name,
                            tool_trace_records,
                            available_tool_names=[spec["name"] for spec in node_tool_specs],
                        ):
                            status = "completed"
                            output_summary = (
                                "workflow node exceeded local max tool calls before final_answer; "
                                "continuing with gathered context"
                            )
                            node_should_stop = True
                        else:
                            status = "failed"
                        break
                    try:
                        if budget_tracker is not None:
                            budget_tracker.check(
                                f"workflow_node:{node.node_id}:step:{step_index}:"
                                f"tool_call:{tool_call_index}:before_tool_call"
                            )
                    except _SubagentBudgetExceeded as exc:
                        status = "failed"
                        budget_failure_reason = str(exc)
                        output_summary = budget_failure_reason
                        break
                    outcome = self._execute_tool_call(
                        tool_call,
                        request=request,
                        run_ref=run_ref,
                        artifact_refs=artifact_refs,
                        tool_trace_records=tool_trace_records,
                        role=role.name,
                        runtime_stage="workflow_node",
                        step_index=step_index,
                        workflow_node_id=node.node_id,
                        task_goal=request.goal,
                        active_skill_bundle=active_skill_bundle,
                        repair_budget=repair_budget,
                        step_key=f"workflow:{node.node_id}:{step_index}:{tool_call.call_id}",
                        repair_log=repair_trajectory,
                        remaining_tool_budget=max_steps - node_tool_calls_used,
                        completion_guard_available_tools=[spec["name"] for spec in node_tool_specs],
                        policy_metadata=policy_metadata,
                    )
                    if budget_tracker is not None:
                        if outcome.counts_against_budget:
                            budget_tracker.note_tool_call()
                        try:
                            budget_tracker.check(
                                f"workflow_node:{node.node_id}:step:{step_index}:"
                                f"tool_call:{tool_call_index}:after_tool_call"
                            )
                        except _SubagentBudgetExceeded as exc:
                            status = "failed"
                            budget_failure_reason = str(exc)
                            output_summary = budget_failure_reason
                    if outcome.counts_against_budget:
                        node_tool_calls_used += 1
                    if _is_completion_guard_violation(outcome.record):
                        completion_guard_violations += 1
                        if completion_guard_violations >= max_completion_guard_violations:
                            status = "failed"
                            guard_failure_reason = _completion_guard_violation_failure_reason(
                                outcome.record,
                                violation_count=completion_guard_violations,
                                violation_limit=max_completion_guard_violations,
                            )
                            output_summary = guard_failure_reason
                    else:
                        completion_guard_violations = 0
                    if _is_repeated_tool_call_suppression(outcome.record):
                        repeated_suppression_violations += 1
                        if repeated_suppression_violations >= max_repeated_suppression_violations:
                            repeated_reason = _repeated_suppression_failure_reason(
                                outcome.record,
                                violation_count=repeated_suppression_violations,
                                violation_limit=max_repeated_suppression_violations,
                            )
                            if _workflow_node_can_continue_after_local_budget(
                                policy.metadata,
                                role.name,
                                tool_trace_records,
                                available_tool_names=[spec["name"] for spec in node_tool_specs],
                            ):
                                status = "completed"
                                output_summary = (
                                    f"{repeated_reason}; continuing with gathered context after "
                                    "suppressed repeated tool calls"
                                )
                                node_should_stop = True
                            else:
                                status = "failed"
                                guard_failure_reason = repeated_reason
                                output_summary = repeated_reason
                    if outcome.updated_skill_bundle is not None:
                        active_skill_bundle = outcome.updated_skill_bundle
                        skill_context = {
                            **(outcome.updated_skill_context or skill_context),
                            "workflow_plan": workflow_plan.model_dump(mode="json"),
                        }
                    if outcome.promotion_candidates:
                        promotion_candidates.extend(outcome.promotion_candidates)
                    node_tool_records.append(outcome.record)
                    for repair_message in outcome.repair_messages:
                        node_messages.append(repair_message)
                        all_prompt_messages.append(repair_message)
                    tool_message = _tool_result_message(
                        outcome.record.tool_call.name,
                        outcome.record.result,
                        policy_metadata=policy_metadata,
                    )
                    node_messages.append(tool_message)
                    all_prompt_messages.append(tool_message)
                    if _role_completion_guards_satisfied(
                        policy_metadata,
                        role.name,
                        tool_trace_records,
                    ):
                        output_summary = (
                            "role completion guards satisfied after successful required tool calls"
                        )
                        break
                    if budget_failure_reason is not None:
                        break
                    if guard_failure_reason is not None:
                        break
                    if node_should_stop:
                        break
                if budget_failure_reason is not None:
                    break
                if guard_failure_reason is not None:
                    break
                if node_should_stop:
                    break
                if status == "failed":
                    break
                if output_summary == "role completion guards satisfied after successful required tool calls":
                    break

            if response is None:
                status = "failed"
                output_summary = "workflow node did not receive an LLM response"
            if (
                status == "completed"
                and output_summary
                and "continuing with gathered context" in output_summary
            ):
                output_summary = _with_recent_successful_tool_context(output_summary, node_tool_records)
            node_artifacts = artifact_refs[artifact_start:]
            node_record = NodeExecutionRecord(
                node_id=node.node_id,
                skill_id=node.skill_id,
                status=status,
                started_at=started_at,
                ended_at=_utc_now(),
                tool_calls=[record.model_dump(mode="json") for record in node_tool_records],
                artifact_refs=[ref.model_dump(mode="json") for ref in node_artifacts],
                output_summary=output_summary,
                metadata={
                    "node_name": node.name,
                    "prepared_tool_names": [spec["name"] for spec in node_tool_specs],
                    "retrieved_skill_id": node.skill_id,
                    "internal_dag_node": node.model_dump(mode="json"),
                    "fallback_node": bool(node.metadata.get("fallback_node")),
                },
            )
            node_records.append(node_record)
            if status == "failed":
                failed = True
                if budget_failure_reason is not None:
                    failed = False
            else:
                previous_summaries.append(
                    {
                        "node_id": node.node_id,
                        "skill_id": node.skill_id,
                        "name": node.name,
                        "summary": _truncate_text(output_summary or "", 2_000),
                    }
                )
                if _role_completion_guards_satisfied(
                    policy_metadata,
                    role.name,
                    tool_trace_records,
                ):
                    completion_guard_stop_reason = "skipped after role completion guards were satisfied"

        for node_id in skipped_node_ids:
            node_records.append(_skipped_node_record(node_by_id[node_id], "skipped by max_workflow_nodes"))

        final_answer = _workflow_final_answer(node_records, node_by_id)
        status = (
            "budget_exceeded"
            if budget_failure_reason is not None
            else "guard_failed"
            if guard_failure_reason is not None
            else "completed"
        )
        failure_reason = budget_failure_reason or guard_failure_reason
        output_message = Message(role="assistant", content=final_answer)
        tool_trace = ToolTrace(run_ref=run_ref, calls=tool_trace_records)
        plan_status = "partial" if budget_failure_reason is not None else _plan_status(node_records)
        plan_trace = PlanExecutionTrace(
            plan_id=workflow_plan.plan_id,
            run_ref=run_ref,
            status=plan_status,
            node_records=node_records,
            tool_trace=tool_trace.model_dump(mode="json"),
            artifact_refs=[ref.model_dump(mode="json") for ref in artifact_refs],
            metadata={
                "topological_order": workflow_plan.metadata.get("topological_order", []),
                "topological_node_order": workflow_plan.metadata.get("topological_node_order", []),
            },
        )
        memory_update_messages = _workflow_memory_update_messages(
            role=role.name,
            role_instruction=role_instruction,
            workflow_plan=workflow_plan,
            node_records=node_records,
            artifact_refs=artifact_refs,
            final_answer=final_answer,
            status=plan_status,
        )
        return _RoleExecutionPayload(
            prompt_messages=all_prompt_messages,
            output_message=output_message,
            tool_trace_records=tool_trace_records,
            tool_trace=tool_trace,
            artifact_refs=artifact_refs,
            final_answer=final_answer,
            skill_bundle=active_skill_bundle,
            skill_context=skill_context,
            repair_trajectory=repair_trajectory,
            promotion_candidates=promotion_candidates,
            llm_call_refs=llm_call_refs,
            workflow_plan=workflow_plan,
            plan_execution_trace=plan_trace,
            node_execution_records=node_records,
            memory_update_messages=memory_update_messages,
            status=status,
            failure_reason=failure_reason,
            budget=budget_tracker.metadata() if budget_tracker is not None else {},
        )

    def _workflow_node_prompt_messages(
        self,
        *,
        role: Any,
        role_instruction: str,
        request: TaskRequest,
        memory_bundle: MemoryBundle,
        prepared_skills: Any,
        skill_context: dict[str, Any],
        node: WorkflowNode,
        previous_summaries: list[dict[str, str]],
        artifact_refs: list[ArtifactRef],
        available_tool_names: list[str],
    ) -> list[Message]:
        skill_by_id = {skill.skill_id: skill for skill in prepared_skills.skill_bundle.skills}
        skill = skill_by_id.get(node.skill_id)
        max_tool_call_rounds = self.task_config.runtime_policy.max_tool_steps_per_node
        if max_tool_call_rounds is None:
            max_tool_call_rounds = self.task_config.runtime_policy.max_tool_steps
        payload = {
            "instruction": role_instruction,
            "task_goal": request.goal,
            "workflow_node": node.model_dump(mode="json"),
            "skill": skill.model_dump(mode="json") if skill is not None else None,
            "skill_context": _json_compatible(skill_context),
            "lab_context": _lab_context_for_prompt(
                lab_root=self.lab_root,
                task_goal="\n".join(
                    part
                    for part in [
                        request.goal,
                        self.task_config.goal if self.task_config is not None else None,
                    ]
                    if isinstance(part, str) and part
                ),
            ),
            "previous_node_summaries": _compact_previous_node_summaries(previous_summaries),
            "artifact_refs": [ref.model_dump(mode="json") for ref in artifact_refs],
            "generic_scientific_extraction_artifact_contracts": _generic_scientific_extraction_artifact_contracts(),
            "candidate_artifact_refs": _candidate_artifacts_for_prompt(artifact_refs),
            "output_node_contract": _workflow_output_node_contract(
                node=node,
                available_tool_names=available_tool_names,
                artifact_refs=artifact_refs,
                previous_summaries=previous_summaries,
            ),
            "memory": _compact_memory_items_for_prompt(memory_bundle.items),
            "upstream_outputs": self._upstream_outputs_for_prompt(request),
            "tool_step_budget": {
                "max_tool_calls": max(0, max_tool_call_rounds),
                "budget_rule": "Use the fewest tool calls needed. Before the tool-call budget is exhausted, return final_answer with a concise node result.",
            },
            "required_response": (
                "Execute only this workflow node. Return tool_call actions as needed, then final_answer. "
                "If the assigned task explicitly asks for a report, artifact, records, audit, or other written output "
                "and an output-writing tool is available, call the appropriate output tool before final_answer. "
                "Record-construction and final-output nodes must use concrete candidate artifacts, tables, rows, "
                "columns, and source references when available; if those candidates are missing, request upstream "
                "Survey/Discovery instead of constructing records from vague summaries."
            ),
        }
        return [
            Message(role="system", content=role.system_prompt),
            Message(role="user", content=json.dumps(payload, indent=2, sort_keys=True)),
        ]

    def _save_llm_call(
        self,
        *,
        run_ref: str,
        backend_id: str,
        llm: Any,
        generation_config: LLMGenerationConfig,
        input_messages: list[Message],
        tool_specs: list[dict[str, Any]],
        response: LLMRuntimeResponse,
        metadata: dict[str, Any],
    ) -> str | None:
        if not self.trajectory_collector.enabled:
            return None
        call_ref = f"llm-call-{uuid4()}"
        self.trajectory_collector.save_llm_call(
            LLMCallRecord(
                call_ref=call_ref,
                run_ref=run_ref,
                backend_id=backend_id,
                model=_llm_model_name(llm, generation_config),
                input_messages=input_messages,
                output_messages=_llm_output_messages(response),
                metadata={
                    **metadata,
                    "generation_config": generation_config.model_dump(mode="json"),
                    "tool_specs": _json_compatible(tool_specs),
                    "action": response.action.action,
                    "raw_response": _json_compatible(response.raw_response),
                },
            )
        )
        return call_ref

    def _execute_tool_call(
        self,
        tool_call: Any,
        *,
        request: TaskRequest,
        run_ref: str,
        artifact_refs: list[ArtifactRef],
        tool_trace_records: list[ToolCallRecord],
        role: str | None = None,
        runtime_stage: str | None = None,
        step_index: int | None = None,
        workflow_node_id: str | None = None,
        task_goal: str | None = None,
        active_skill_bundle: SkillBundle | None = None,
        repair_budget: dict[str, int] | None = None,
        step_key: str | None = None,
        repair_log: list[dict[str, Any]] | None = None,
        remaining_tool_budget: int | None = None,
        completion_guard_available_tools: list[str] | None = None,
        policy_metadata: dict[str, Any] | None = None,
    ) -> _ToolExecutionOutcome:
        effective_policy_metadata = policy_metadata or self.task_config.runtime_policy.metadata
        guard_reserved_result = _completion_guard_budget_reserved_result(
            tool_call,
            tool_trace_records,
            policy_metadata=effective_policy_metadata,
            role_name=role,
            remaining_tool_budget=remaining_tool_budget,
            available_tool_names=completion_guard_available_tools,
        )
        if guard_reserved_result is not None:
            record = ToolCallRecord(tool_call=tool_call, result=guard_reserved_result)
            tool_trace_records.append(record)
            self.trajectory_collector.record_tool_call(
                run_ref=run_ref,
                task_id=request.task_id,
                record=record,
                role=role,
                runtime_stage=runtime_stage,
                step_index=step_index,
                workflow_node_id=workflow_node_id,
                metadata={"completion_guard_budget_reserved": True},
            )
            return _ToolExecutionOutcome(record=record, counts_against_budget=False)
        repeated_result = _repeated_tool_call_suppression_result(
            tool_call,
            tool_trace_records,
            policy_metadata=effective_policy_metadata,
        )
        if repeated_result is not None:
            record = ToolCallRecord(tool_call=tool_call, result=repeated_result)
            tool_trace_records.append(record)
            self.trajectory_collector.record_tool_call(
                run_ref=run_ref,
                task_id=request.task_id,
                record=record,
                role=role,
                runtime_stage=runtime_stage,
                step_index=step_index,
                workflow_node_id=workflow_node_id,
                metadata={"repeated_tool_call_suppressed": True},
            )
            return _ToolExecutionOutcome(record=record, counts_against_budget=False)
        record = self._perform_tool_call(
            tool_call,
            request=request,
            run_ref=run_ref,
            artifact_refs=artifact_refs,
            tool_trace_records=tool_trace_records,
            role=role,
            runtime_stage=runtime_stage,
            step_index=step_index,
            workflow_node_id=workflow_node_id,
        )
        outcome = _ToolExecutionOutcome(record=record)
        policy = self.task_config.runtime_policy
        if (
            not policy.enable_runtime_capability_repair
            or self.capability_repair_runtime is None
            or active_skill_bundle is None
            or task_goal is None
        ):
            return outcome
        if repair_budget is None:
            repair_budget = {"task_attempts": 0}
        step_budget_key = step_key or f"step:{step_index or 0}"
        per_step_key = f"{step_budget_key}:repair_attempts"
        if repair_budget.get("task_attempts", 0) >= policy.max_repair_attempts_per_task:
            return outcome
        if repair_budget.get(per_step_key, 0) >= policy.max_repair_attempts_per_step:
            return outcome
        repair_result = self.capability_repair_runtime.maybe_repair(
            task_id=request.task_id,
            run_ref=run_ref,
            step_id=step_budget_key,
            role=role or "",
            task_goal=task_goal,
            tool_call=record.tool_call,
            tool_result=record.result,
            active_skill_bundle=active_skill_bundle,
            tool_runtime=self.tool_runtime,
            trajectory_collector=self.trajectory_collector,
            runtime_policy=policy,
            repair_log=repair_log,
        )
        if repair_result is None:
            return outcome
        repair_budget["task_attempts"] = repair_budget.get("task_attempts", 0) + 1
        repair_budget[per_step_key] = repair_budget.get(per_step_key, 0) + 1
        if repair_result.retry_record is not None:
            retried_record = repair_result.retry_record
            retried_record = ToolCallRecord(
                tool_call=retried_record.tool_call,
                result=_manage_tool_result_artifacts(
                    result=retried_record.result,
                    request=request,
                    run_ref=run_ref,
                    artifact_root_factory=self.tool_artifact_root_factory,
                ),
            )
            artifact_refs.extend(retried_record.result.artifact_refs)
            _register_tool_result_artifacts(retried_record.result, self.tool_artifact_registrar)
            tool_trace_records.append(retried_record)
            self.trajectory_collector.record_tool_call(
                run_ref=run_ref,
                task_id=request.task_id,
                record=retried_record,
                role=role,
                runtime_stage=runtime_stage,
                step_index=step_index,
                workflow_node_id=workflow_node_id,
                metadata={"repair_retry": True},
            )
            outcome.record = retried_record
        outcome.repair_messages = repair_result.repair_messages
        outcome.updated_skill_bundle = repair_result.updated_skill_bundle
        outcome.updated_skill_context = repair_result.updated_skill_context
        outcome.repair_entry = repair_result.repair_entry
        outcome.promotion_candidates = repair_result.promotion_candidates
        return outcome

    def _perform_tool_call(
        self,
        tool_call: Any,
        *,
        request: TaskRequest,
        run_ref: str,
        artifact_refs: list[ArtifactRef],
        tool_trace_records: list[ToolCallRecord],
        role: str | None = None,
        runtime_stage: str | None = None,
        step_index: int | None = None,
        workflow_node_id: str | None = None,
    ) -> ToolCallRecord:
        if self.tool_runtime is None:
            raise RuntimeError("task runtime received tool_call but no tool_runtime is configured")
        tool_result = self.tool_runtime.execute(tool_call)
        tool_result = _manage_tool_result_artifacts(
            result=tool_result,
            request=request,
            run_ref=run_ref,
            artifact_root_factory=self.tool_artifact_root_factory,
        )
        artifact_refs.extend(tool_result.artifact_refs)
        _register_tool_result_artifacts(tool_result, self.tool_artifact_registrar)
        record = ToolCallRecord(tool_call=tool_call, result=tool_result)
        tool_trace_records.append(record)
        self.trajectory_collector.record_tool_call(
            run_ref=run_ref,
            task_id=request.task_id,
            record=record,
            role=role,
            runtime_stage=runtime_stage,
            step_index=step_index,
            workflow_node_id=workflow_node_id,
        )
        return record

    def _roles(self) -> list[Any]:
        if self.task_config is None:
            raise RuntimeError("task_config is required for default task runtime dispatch")
        if not self.task_config.roles:
            raise RuntimeError("task_config requires at least one role for default task runtime dispatch")
        return list(self.task_config.roles.values())

    def _role_by_name(self, role_name: str | None) -> Any:
        if not role_name:
            raise RuntimeError("dispatch decision did not include target_role")
        if self.task_config is None:
            raise RuntimeError("task_config is required for default task runtime dispatch")
        try:
            return self.task_config.roles[role_name]
        except KeyError as exc:
            raise RuntimeError(f"meta-agent dispatched unknown role {role_name!r}") from exc

    def _llm_runtime(self, backend_id: str) -> Any:
        try:
            return self.llm_runtimes[backend_id]
        except KeyError as exc:
            raise RuntimeError(f"missing LLM runtime for backend_id={backend_id!r}") from exc

    @staticmethod
    def _first_runtime(runtimes: dict[str, Any], runtime_type: str) -> Any:
        if not runtimes:
            raise RuntimeError(f"missing {runtime_type} runtime for default task runtime dispatch")
        return next(iter(runtimes.values()))

    def _memory_runtime_for_binding(self, binding: Any | None) -> Any:
        if binding is None:
            return self._first_runtime(self.memory_runtimes, "memory")
        try:
            return self.memory_runtimes[binding.backend_id]
        except KeyError as exc:
            raise RuntimeError(f"missing memory runtime for backend_id={binding.backend_id!r}") from exc

    def _memory_runtimes_for_scopes(self, role: Any) -> tuple[Any, Any]:
        agent_binding = role.agent_memory_backend
        task_binding = self.task_config.task_memory_backend
        if agent_binding is None and task_binding is None:
            memory = self._first_runtime(self.memory_runtimes, "memory")
            return memory, memory
        if agent_binding is None:
            raise RuntimeError("role.agent_memory_backend is required when task_memory_backend is configured")
        if task_binding is None:
            raise RuntimeError("task_config.task_memory_backend is required when agent_memory_backend is configured")
        return self._memory_runtime_for_binding(agent_binding), self._memory_runtime_for_binding(task_binding)

    def _record_partial_subagent_postmortem(
        self,
        *,
        request: TaskRequest,
        run_ref: str,
        role: str,
        stage_index: int,
        instruction: str,
        retrieval_request: RetrievalRequest,
        memory_bundle: MemoryBundle,
        skill_bundle: SkillBundle,
        llm_backend_id: str,
        llm_backend_config_ref: str | None,
        llm_backend_state_ref: str | None,
        metadata: dict[str, Any],
    ) -> None:
        self.trajectory_collector.record_event(
            event_type="subagent_postmortem",
            subject_type="subagent",
            subject_ref=run_ref,
            task_id=request.task_id,
            run_ref=run_ref,
            metadata={
                "role": role,
                "stage_index": stage_index,
                "instruction": instruction,
                "retrieval_request": _json_compatible(retrieval_request),
                "memory_bundle": _json_compatible(memory_bundle),
                "skill_bundle": _json_compatible(skill_bundle),
                "llm_backend_id": llm_backend_id,
                "llm_backend_config_ref": llm_backend_config_ref,
                "llm_backend_state_ref": llm_backend_state_ref,
                **metadata,
            },
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dynamic_work_items_for_request(
    request: TaskRequest,
    policy_metadata: dict[str, Any],
    scope: str,
) -> list[dict[str, Any] | None]:
    if scope != "per_work_item":
        return [None]
    raw_items = request.metadata.get("work_items")
    items: list[dict[str, Any] | None] = []
    if isinstance(raw_items, list):
        for index, raw_item in enumerate(raw_items):
            if isinstance(raw_item, dict):
                work_item_id = raw_item.get("work_item_id") or raw_item.get("id") or raw_item.get("article_id")
                items.append({"work_item_id": str(work_item_id or f"work-item-{index + 1}"), **_json_compatible(raw_item)})
            elif isinstance(raw_item, str) and raw_item:
                items.append({"work_item_id": raw_item})
    if items:
        return items
    routing = _work_item_routing_policy(policy_metadata)
    if routing is not None:
        ids = sorted(routing.get("required_work_item_ids", []))
        if ids:
            return [_dynamic_work_item_from_task_goal(request.goal, work_item_id) for work_item_id in ids]
    return [None]


def _dynamic_work_item_id(work_item: dict[str, Any] | None) -> str | None:
    if not isinstance(work_item, dict):
        return None
    value = work_item.get("work_item_id") or work_item.get("id") or work_item.get("article_id")
    return str(value) if value not in (None, "") else None


def _dynamic_work_item_from_task_goal(task_goal: str, work_item_id: str) -> dict[str, Any]:
    item: dict[str, Any] = {"work_item_id": work_item_id}
    block = _work_item_block(task_goal, work_item_id)
    if not block:
        return item
    article_package = _article_package_from_text(block)
    source_files = _source_files_from_text(block)
    if article_package:
        item["article_package"] = article_package
        item["lab_path"] = article_package
    if source_files:
        item["exact_source_files"] = source_files
        item["source_files"] = source_files
    return item


def _dynamic_agent_memory_backend(task_config: TaskConfig) -> BackendBinding | None:
    for role in task_config.roles.values():
        if role.agent_memory_backend is not None:
            return role.agent_memory_backend
    return None


def _dynamic_failure_reason(failed_runs: list[dict[str, Any]]) -> str:
    parts = []
    for run in failed_runs[:5]:
        role = run.get("role") or run.get("generic_agent_type") or "dynamic_subagent"
        reason = run.get("failure_reason") or run.get("status") or "failed"
        parts.append(f"{role}: {reason}")
    suffix = "" if len(failed_runs) <= 5 else f"; {len(failed_runs) - 5} additional failures"
    return "; ".join(parts) + suffix


def _skipped_node_record(node: WorkflowNode, reason: str) -> NodeExecutionRecord:
    return NodeExecutionRecord(
        node_id=node.node_id,
        skill_id=node.skill_id,
        status="skipped",
        output_summary=reason,
        metadata={"node_name": node.name, "reason": reason, "internal_dag_node": node.model_dump(mode="json")},
    )


def _workflow_node_tool_bundle(
    *,
    tool_runtime: ToolRuntime | None,
    node: WorkflowNode,
    role_allowed_tools: list[str],
    policy: Any,
    fallback_tool_bundle: Any,
    role_name: str | None = None,
) -> Any:
    required_tools = _workflow_node_required_tools(node, role_name)
    if tool_runtime is None or not required_tools:
        return fallback_tool_bundle
    return tool_runtime.prepare(
        required_tools=required_tools,
        allowed_tools=role_allowed_tools,
        policy=policy,
        optional_tools=policy.allowed_human_tools if policy.allow_human_tools else None,
    )


def _workflow_node_required_tools(node: WorkflowNode, role_name: str | None) -> list[str]:
    role_support_tools = _role_support_tools_for_workflow_node(role_name)
    return _dedupe([*node.required_tools, *role_support_tools])


def _role_support_tools_for_workflow_node(role_name: str | None) -> list[str]:
    role = (role_name or "").casefold()
    if role == "execagent":
        return [
            "list_files",
            "read_text",
            "search_text",
            "inspect_file_metadata",
            "inspect_table",
            "read_table_slice",
            "inspect_excel_workbook",
            "read_excel_sheet",
            "detect_table_header",
            "normalize_table",
            "profile_table",
            "build_document_inventory",
            "discover_candidate_source_files",
            "discover_candidate_tables",
            "extract_candidate_rows",
            "build_candidate_records",
            "json_schema_validate",
            "write_report",
        ]
    if role == "surveyagent":
        return [
            "list_files",
            "read_text",
            "search_text",
            "inspect_file_metadata",
            "extract_sections",
            "inspect_excel_workbook",
            "read_excel_sheet",
            "build_document_inventory",
            "discover_candidate_source_files",
            "discover_candidate_tables",
            "write_report",
        ]
    if role == "criticagent":
        return [
            "read_text",
            "search_text",
            "inspect_table",
            "read_table_slice",
            "normalize_table",
            "profile_table",
            "validate_candidate_records",
            "json_schema_validate",
            "write_report",
        ]
    if role == "writeagent":
        return ["read_text", "serialize_final_records", "json_schema_validate", "write_jsonl", "write_report"]
    return []


def _workflow_node_can_continue_after_local_budget(
    policy_metadata: dict[str, Any],
    role_name: str | None,
    tool_trace_records: list[ToolCallRecord],
    *,
    available_tool_names: list[str],
) -> bool:
    guard_metadata = _completion_guard_metadata(policy_metadata, role_name)
    due_tools = _completion_guard_due_tools(guard_metadata, tool_trace_records)
    if not due_tools:
        return False
    return due_tools[0] not in {name for name in available_tool_names if isinstance(name, str)}


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _ensure_workflow_has_fallback_node(
    *,
    workflow_plan: WorkflowPlan,
    skill_bundle: SkillBundle,
    role_name: str,
    role_instruction: str,
    required_tools: list[str],
) -> tuple[WorkflowPlan, SkillBundle]:
    if workflow_plan.nodes:
        return workflow_plan, skill_bundle
    if not required_tools:
        return workflow_plan, skill_bundle
    fallback_skill = SkillRef(
        skill_id=f"runtime.assigned_task.{role_name}",
        name=f"{role_name} Assigned Task",
        content=role_instruction,
        required_tools=required_tools,
        metadata={"runtime_fallback": True},
    )
    fallback_node = WorkflowNode(
        node_id=f"node-001-runtime-{role_name}",
        skill_id=fallback_skill.skill_id,
        name=fallback_skill.name,
        purpose="Execute the assigned task using role-compatible tools because no retrieved skill node matched.",
        required_inputs=["assigned_task"],
        expected_outputs=["assigned_task_result"],
        required_tools=required_tools,
        metadata={"fallback_node": True},
    )
    updated_plan = workflow_plan.model_copy(
        update={
            "nodes": [fallback_node],
            "required_tools": required_tools,
            "expected_artifacts": ["assigned_task_result"],
            "metadata": {
                **workflow_plan.metadata,
                "topological_order": [fallback_skill.skill_id],
                "topological_node_order": [fallback_node.node_id],
                "fallback_node_created": True,
            },
        }
    )
    updated_bundle = skill_bundle.model_copy(
        update={
            "skills": [*skill_bundle.skills, fallback_skill],
            "required_tools": required_tools,
            "metadata": {
                **skill_bundle.metadata,
                "runtime_fallback_skill_id": fallback_skill.skill_id,
            },
        }
    )
    return updated_plan, updated_bundle


def _meta_workflow_node_id(dispatch_metadata: dict[str, Any], role_name: str, stage_index: int) -> str:
    for key in ("meta_workflow_node_id", "workflow_node_id", "node_id"):
        value = dispatch_metadata.get(key)
        if isinstance(value, str) and value:
            return value
    return f"dispatch-{stage_index + 1:03d}-{role_name}"


def _agent_workflow_dag(dispatch_metadata: dict[str, Any]) -> Any | None:
    for key in ("agent_level_workflow_dag", "workflow_dag", "selected_workflow_dag"):
        value = dispatch_metadata.get(key)
        if isinstance(value, dict | list):
            return _json_compatible(value)
    return None


def _dispatch_requests_flat_execution(dispatch_metadata: dict[str, Any], instruction: str) -> bool:
    mode = dispatch_metadata.get("execution_mode") or dispatch_metadata.get("mode")
    if isinstance(mode, str) and mode.casefold() in {"direct", "flat", "no_workflow", "no_dag"}:
        return True
    for key in ("disable_internal_workflow_planning", "disable_workflow_planning", "use_direct_execution"):
        if dispatch_metadata.get(key) is True:
            return True
    lowered = instruction.casefold()
    return any(
        phrase in lowered
        for phrase in (
            "do not use any dag",
            "do not use the skill dag",
            "without using the dag",
            "bypass internal workflow",
            "simpler direct instruction",
            "direct execution",
            "direct mode",
            "internal dag node exceeded tool budget",
            "internal dag node exceeding tool budget",
        )
    )


def _generic_role_retrieval_metadata(role_name: str | None) -> dict[str, Any]:
    role = (role_name or "").casefold()
    if role == "surveyagent":
        return {
            "scientific_process_capability": "Literature",
            "target_category": "task.scientific_document_intake",
            "task_types": ["intake", "document", "artifact", "discovery", "table"],
        }
    if role == "designagent":
        return {
            "scientific_process_capability": "Analysis",
            "task_types": ["schema", "mapping", "planning", "validation"],
        }
    if role == "execagent":
        return {
            "scientific_process_capability": "Analysis",
            "task_types": ["extraction", "record", "field", "schema", "table"],
        }
    if role == "criticagent":
        return {
            "scientific_process_capability": "Validation",
            "task_types": ["validation", "deduplication", "evaluation"],
        }
    if role == "writeagent":
        return {
            "scientific_process_capability": "Analysis",
            "task_types": ["record", "artifact", "reporting"],
        }
    return {}


def _lab_state_detail_requests(dispatch_metadata: dict[str, Any]) -> dict[str, list[str]]:
    raw = dispatch_metadata.get("lab_state_detail_requests")
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, list[str]] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, list):
            continue
        refs = [item for item in value if isinstance(item, str) and item]
        if refs:
            normalized[key] = refs
    return normalized


def _artifact_role(artifact: ArtifactRef) -> str | None:
    role = artifact.metadata.get("role")
    return role if isinstance(role, str) and role else None


def _artifact_index_status(artifact: ArtifactRef) -> str:
    status = artifact.metadata.get("status")
    allowed = {"intermediate", "final", "validation", "audit", "candidate", "rejected"}
    if isinstance(status, str) and status in allowed:
        return status
    inferred = _artifact_semantic_kind(artifact)
    if inferred in allowed:
        return inferred
    return "intermediate"


def _artifact_semantic_kind(artifact: ArtifactRef) -> str:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    for key in ("artifact_kind", "artifact_type", "status"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            value = value.casefold()
            if value in {"candidate", "records", "record", "record_set", "dataset"}:
                return "candidate"
            if value in {"audit", "validation", "report"}:
                return "audit" if value != "validation" else "validation"
            if value in {"final", "rejected", "intermediate"}:
                return value
    candidates = [
        artifact.uri,
        metadata.get("filename"),
        metadata.get("name"),
        metadata.get("source_uri"),
    ]
    name_text = " ".join(str(item) for item in candidates if isinstance(item, str)).casefold()
    if "candidate" in name_text or "record" in name_text or name_text.endswith(".jsonl"):
        return "candidate"
    if "audit" in name_text or "validation" in name_text or "report" in name_text:
        return "audit"
    return "intermediate"


def _expected_outputs_for_completion(dispatch_metadata: dict[str, Any]) -> list[dict[str, Any]]:
    raw = dispatch_metadata.get("expected_outputs")
    if not isinstance(raw, list):
        return []
    outputs: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            outputs.append({"name": item, "description": item, "required": True})
        elif isinstance(item, dict):
            name = item.get("name") or item.get("id") or item.get("type") or item.get("description")
            if isinstance(name, str) and name.strip():
                outputs.append(
                    {
                        "name": name,
                        "description": item.get("description") if isinstance(item.get("description"), str) else name,
                        "required": item.get("required", True) is not False,
                        "metadata": _json_compatible(item.get("metadata", {})),
                    }
                )
    return outputs


def _dynamic_expected_output_contracts(output_artifact_names: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "description": f"dynamic workflow output artifact {name}",
            "required": True,
            "metadata": {"dynamic_output_artifact": True, "requires_artifact": True},
        }
        for name in _dedupe([name for name in output_artifact_names if isinstance(name, str) and name])
    ]


def _dynamic_output_artifact_policy(output_artifact_names: list[str]) -> dict[str, Any]:
    expected = _dedupe([name for name in output_artifact_names if isinstance(name, str) and name])
    if not expected:
        return {
            "required": False,
            "instruction": "This dynamic node has no declared output artifact contract.",
        }
    return {
        "required": True,
        "expected_output_artifacts": expected,
        "instruction": (
            "Write every expected_output_artifact before final_answer. If source evidence is incomplete, "
            "a tool path is not recoverable, or a repeated tool call is suppressed, do not retry the same "
            "tool target indefinitely. Use the evidence already gathered and write the required artifact "
            "with explicit diagnostics; for record outputs with no supported records, write an empty records "
            "list plus warnings instead of leaving the artifact missing."
        ),
    }


def _dispatch_metadata_with_expected_outputs(decision: DispatchDecision) -> dict[str, Any]:
    metadata = dict(decision.metadata)
    if "expected_outputs" not in metadata and decision.expected_outputs:
        metadata["expected_outputs"] = [output.model_dump(mode="json") for output in decision.expected_outputs]
    return metadata


def _completion_contract_failure_reason(contract: dict[str, Any]) -> str:
    issues = contract.get("blocking_issues")
    if not isinstance(issues, list) or not issues:
        return "incomplete completed subagent outputs"
    messages = [issue.get("message") for issue in issues if isinstance(issue, dict) and isinstance(issue.get("message"), str)]
    suffix = "; ".join(messages[:3])
    return "incomplete completed subagent outputs" + (f": {suffix}" if suffix else "")


def _role_completion_guards_satisfied(
    policy_metadata: dict[str, Any],
    role_name: str | None,
    tool_trace_records: list[ToolCallRecord],
) -> bool:
    guard_metadata = _completion_guard_metadata(policy_metadata, role_name)
    if not guard_metadata or not any(
        key in guard_metadata
        for key in ("required_tool_calls_before_final", "minimum_jsonl_records_before_final")
    ):
        return False
    return _final_answer_rejection_reason(policy_metadata, tool_trace_records, role_name) is None


def _missing_expected_outputs(
    expected_outputs: list[dict[str, Any]],
    artifact_refs: list[ArtifactRef],
    final_answer: str,
) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for output in expected_outputs:
        if output.get("required", True) is False:
            continue
        name = str(output.get("name") or output.get("description") or "").strip()
        if not name:
            continue
        if _expected_output_requires_artifact(output):
            if _artifact_refs_satisfy_expected_name(name, artifact_refs):
                continue
            missing.append(output)
            continue
        if _expected_output_satisfied(name, artifact_refs, final_answer):
            continue
        missing.append(output)
    return missing


def _expected_output_requires_artifact(output: dict[str, Any]) -> bool:
    metadata = output.get("metadata")
    return isinstance(metadata, dict) and metadata.get("requires_artifact") is True


def _artifact_refs_satisfy_expected_name(name: str, artifact_refs: list[Any]) -> bool:
    expected = _normalize_artifact_name(name)
    if not expected:
        return bool(artifact_refs)
    return any(expected in _artifact_ref_name_candidates(ref) for ref in artifact_refs)


def _expected_output_satisfied(name: str, artifact_refs: list[ArtifactRef], final_answer: str) -> bool:
    expected_kind = _expected_output_kind(name)
    if expected_kind == "artifact":
        return bool(artifact_refs) or bool(final_answer.strip())
    for artifact in artifact_refs:
        if _artifact_satisfies_expected_kind(artifact, expected_kind):
            return True
    if expected_kind == "records":
        return False
    if expected_kind in {"summary", "plan", "inventory", "audit"}:
        return bool(final_answer.strip())
    return bool(artifact_refs) or bool(final_answer.strip())


def _expected_output_kind(name: str) -> str:
    lowered = name.casefold()
    if any(token in lowered for token in ("record", "jsonl", "dataset", "candidate")):
        return "records"
    if any(token in lowered for token in ("report", "audit", "validation", "critique")):
        return "audit"
    if any(token in lowered for token in ("plan", "design", "strategy")):
        return "plan"
    if any(token in lowered for token in ("inventory", "survey", "coverage")):
        return "inventory"
    return "artifact"


def _artifact_satisfies_expected_kind(artifact: ArtifactRef, expected_kind: str) -> bool:
    kind = _artifact_semantic_kind(artifact)
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    text = " ".join(
        str(item)
        for item in [artifact.uri, metadata.get("filename"), metadata.get("name"), metadata.get("source_uri")]
        if isinstance(item, str)
    ).casefold()
    if expected_kind == "records":
        return kind in {"candidate", "final"} and ("record" in text or "jsonl" in text or "candidate" in text)
    if expected_kind == "audit":
        return kind in {"audit", "validation", "final"}
    if expected_kind in {"plan", "inventory"}:
        return bool(text) and kind in {"intermediate", "audit", "validation", "final", "candidate"}
    return bool(text)


def _normalize_artifact_name(value: str | None) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    return Path(value.strip()).name.casefold()


def _artifact_ref_name_candidates(ref: Any) -> set[str]:
    if hasattr(ref, "model_dump"):
        payload = ref.model_dump(mode="json")
    elif isinstance(ref, dict):
        payload = ref
    else:
        payload = {"uri": str(ref)}
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    values = [
        payload.get("uri"),
        metadata.get("filename"),
        metadata.get("name"),
        metadata.get("path"),
        metadata.get("source_uri"),
    ]
    result = {_normalize_artifact_name(str(value)) for value in values if value not in (None, "")}
    return {value for value in result if value}


def _dynamic_declared_output_artifact_names(spec: Any) -> set[str]:
    agent_by_id = {agent.subagent_id: agent for agent in spec.dynamic_subagents}
    names: set[str] = set()
    for node in spec.workflow_nodes:
        names.update(_dynamic_node_expected_output_artifacts(node, agent_by_id[node.subagent_id]))
    return names


def _dynamic_node_expected_output_artifacts(node: Any, agent: Any) -> list[str]:
    outputs = [item for item in getattr(node, "output_artifacts", []) if isinstance(item, str) and item]
    if outputs:
        return _dedupe(outputs)
    return _dedupe([item for item in getattr(agent, "artifact_outputs", []) if isinstance(item, str) and item])


def _dynamic_missing_input_artifacts(
    input_artifacts: list[str],
    *,
    produced_artifact_names: set[str],
    available_artifact_names: set[str],
) -> list[str]:
    return [
        name
        for name in input_artifacts
        if isinstance(name, str) and name in produced_artifact_names and name not in available_artifact_names
    ]


def _dynamic_available_output_names(
    artifact_refs: list[Any],
    *,
    expected_output_artifacts: list[str],
) -> set[str]:
    available: set[str] = set()
    for artifact in artifact_refs:
        available.update(_artifact_ref_name_candidates(artifact))
    for expected in expected_output_artifacts:
        if _artifact_refs_satisfy_expected_name(expected, artifact_refs):
            available.add(expected)
    return available


def _enforce_dynamic_node_output_contract(
    result: dict[str, Any],
    *,
    expected_output_artifacts: list[str],
) -> dict[str, Any]:
    missing = [
        name
        for name in expected_output_artifacts
        if not _artifact_refs_satisfy_expected_name(name, result.get("artifact_refs", []))
    ]
    if not missing and result.get("status") != "completed":
        completion_contract = result.get("completion_contract")
        if not isinstance(completion_contract, dict):
            completion_contract = {}
        if _dynamic_result_has_required_outputs(result, expected_output_artifacts, completion_contract):
            warnings = list(completion_contract.get("warnings") or [])
            original_status = result.get("status")
            original_failure_reason = result.get("failure_reason")
            warnings.append(
                {
                    "type": "dynamic_output_contract_satisfied_after_noncritical_failure",
                    "message": (
                        "dynamic node produced all expected output artifacts; allowing downstream "
                        "nodes to continue while preserving the original failure as a warning"
                    ),
                    "original_status": original_status,
                    "original_failure_reason": original_failure_reason,
                }
            )
            return {
                **result,
                "status": "completed",
                "failure_reason": None,
                "completion_contract": {
                    **completion_contract,
                    "assigned_task_complete": True,
                    "produced_required_outputs": True,
                    "ready_for_task_end": True,
                    "blocking_issues": [],
                    "warnings": warnings,
                },
                "dynamic_output_contract_override": {
                    "original_status": original_status,
                    "original_failure_reason": original_failure_reason,
                    "expected_output_artifacts": expected_output_artifacts,
                },
            }
    if not missing or result.get("status") != "completed":
        return result
    failure_reason = f"missing dynamic output artifacts: {', '.join(missing)}"
    completion_contract = result.get("completion_contract")
    if not isinstance(completion_contract, dict):
        completion_contract = {}
    blocking_issues = list(completion_contract.get("blocking_issues") or [])
    blocking_issues.append(
        {
            "type": "missing_dynamic_output_artifact",
            "severity": "blocking",
            "message": failure_reason,
            "expected_artifacts": missing,
        }
    )
    return {
        **result,
        "status": "guard_failed",
        "failure_reason": failure_reason,
        "completion_contract": {
            **completion_contract,
            "assigned_task_complete": False,
            "produced_required_outputs": False,
            "ready_for_task_end": False,
            "blocking_issues": blocking_issues,
        },
    }


def _dynamic_result_has_required_outputs(
    result: dict[str, Any],
    expected_output_artifacts: list[str],
    completion_contract: dict[str, Any],
) -> bool:
    if not expected_output_artifacts:
        return False
    if completion_contract.get("produced_required_outputs") is True:
        return True
    return all(
        _artifact_refs_satisfy_expected_name(name, result.get("artifact_refs", []))
        for name in expected_output_artifacts
    )


def _assigned_task_can_complete_without_artifact(assigned_task: str, final_answer: str) -> bool:
    task = assigned_task.casefold()
    if any(
        token in task
        for token in (
            "write",
            "writing",
            "produce",
            "output",
            "artifact",
            "report",
            "records",
            "jsonl",
            "extract candidate",
        )
    ):
        return False
    return bool(final_answer.strip())


def _workflow_final_answer(
    node_records: list[NodeExecutionRecord],
    node_by_id: dict[str, WorkflowNode],
) -> str:
    completed = [record for record in node_records if record.status == "completed"]
    if not completed:
        failed = [record for record in node_records if record.status == "failed"]
        if failed:
            return "\n".join(
                f"{node_by_id[record.node_id].name}: {record.output_summary or 'failed'}"
                for record in failed
                if record.node_id in node_by_id
            )
        return "Workflow completed with no executed nodes."
    return "\n".join(
        f"{node_by_id[record.node_id].name}: {record.output_summary or ''}"
        for record in completed
        if record.node_id in node_by_id
    )


def _with_recent_successful_tool_context(summary: str, node_tool_records: list[ToolCallRecord]) -> str:
    snippets: list[str] = []
    for record in node_tool_records:
        if record.result.status != "ok":
            continue
        content = str(record.result.content or "").strip()
        if not content:
            continue
        snippets.append(f"{record.tool_call.name}: {_truncate_text(content, 700)}")
    if not snippets:
        return summary
    context = "\n".join(snippets[-5:])
    return f"{summary}\n\nRecent successful tool context:\n{context}"


def _apply_subagent_skill_budget(prepared_skills: Any, *, role_name: str, policy_metadata: dict[str, Any]) -> Any:
    role_budget = _subagent_budget(policy_metadata, role_name)
    max_skills = _optional_positive_int(role_budget.get("max_retrieved_skills"))
    if max_skills is None:
        return prepared_skills
    skill_bundle = prepared_skills.skill_bundle
    if len(skill_bundle.skills) <= max_skills:
        return prepared_skills
    kept = skill_bundle.skills[:max_skills]
    pruned = skill_bundle.skills[max_skills:]
    metadata = {
        **skill_bundle.metadata,
        "budget_warnings": [
            *[item for item in skill_bundle.metadata.get("budget_warnings", []) if isinstance(item, str)],
            f"retrieved skill budget limited {role_name} to {max_skills} skill(s); pruned {len(pruned)}",
        ],
        "pruned_skill_ids_by_budget": [skill.skill_id for skill in pruned],
    }
    trace = metadata.get("retrieval_trace")
    kept_ids = {skill.skill_id for skill in kept}
    if isinstance(trace, dict):
        metadata["retrieval_trace"] = _filter_retrieval_trace_for_budget(trace, kept_ids)
    updated_bundle = skill_bundle.model_copy(
        update={
            "skills": kept,
            "required_tools": _dedupe(tool for skill in kept for tool in skill.required_tools),
            "metadata": metadata,
        }
    )
    return prepared_skills.__class__(
        skill_bundle=updated_bundle,
        tool_bundle=prepared_skills.tool_bundle,
        skill_context=build_budgeted_skill_context(prepared_skills.skill_context, updated_bundle),
    )


def _subagent_budget_tracker(*, policy_metadata: dict[str, Any], role_name: str) -> _SubagentBudgetTracker:
    budget = _subagent_budget(policy_metadata, role_name)
    default_budget = policy_metadata.get("subagent_budget")
    default_budget = default_budget if isinstance(default_budget, dict) else {}
    return _SubagentBudgetTracker(
        role_name=role_name,
        started_at_monotonic=time.monotonic(),
        max_llm_calls=_optional_positive_int(
            budget.get("max_subagent_llm_calls", default_budget.get("max_subagent_llm_calls"))
        ),
        max_tool_calls=_optional_positive_int(
            budget.get("max_subagent_tool_calls", default_budget.get("max_subagent_tool_calls"))
        ),
        max_runtime_seconds=_optional_positive_float(
            budget.get("max_subagent_runtime_seconds", default_budget.get("max_subagent_runtime_seconds"))
        ),
    )


def build_budgeted_skill_context(skill_context: dict[str, Any], skill_bundle: SkillBundle) -> dict[str, Any]:
    selected = [
        {
            "skill_id": skill.skill_id,
            "name": skill.name,
            "required_tools": list(skill.required_tools),
            "retrieval": _json_compatible(skill.metadata.get("retrieval", {})),
        }
        for skill in skill_bundle.skills
    ]
    context = {**skill_context, "selected_skills": selected, "required_tools": skill_bundle.required_tools}
    for key in ("budget_warnings", "pruned_skill_ids_by_budget"):
        if key in skill_bundle.metadata:
            context[key] = _json_compatible(skill_bundle.metadata[key])
    return context


def _filter_retrieval_trace_for_budget(trace: dict[str, Any], kept_ids: set[str]) -> dict[str, Any]:
    filtered = dict(trace)
    for key in ("returned_skill_ids", "directly_matched_skill_ids", "dependency_added_skill_ids", "optional_expanded_skill_ids"):
        value = filtered.get(key)
        if isinstance(value, list):
            filtered[key] = [item for item in value if item in kept_ids]
    return filtered


def _subagent_budget(policy_metadata: dict[str, Any], role_name: str | None) -> dict[str, Any]:
    budgets = policy_metadata.get("subagent_budgets_by_role")
    if isinstance(budgets, dict) and isinstance(role_name, str):
        role_budget = budgets.get(role_name)
        if isinstance(role_budget, dict):
            return role_budget
    default_budget = policy_metadata.get("subagent_budget")
    return default_budget if isinstance(default_budget, dict) else {}


def _coverage_from_subagent_summary(summary: str) -> dict[str, Any]:
    payload = _json_object_from_text(summary)
    if not isinstance(payload, dict):
        return {}
    coverage_keys = {
        "processed_article_count",
        "processed_file_count",
        "processed_table_count",
        "candidate_count",
        "accepted_record_count",
        "rejected_record_count",
        "skipped_article_count",
        "skipped_table_count",
        "failure_count",
        "record_count",
    }
    return {key: payload[key] for key in coverage_keys if key in payload and _is_scalar_json_value(payload[key])}


def _json_object_from_text(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = _first_json_object_from_text(stripped)
        if payload is None:
            return None
    return payload if isinstance(payload, dict) else None


def _first_json_object_from_text(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _is_scalar_json_value(value: Any) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def _internal_dag_summary(workflow_plan: WorkflowPlan | None) -> dict[str, Any] | None:
    if workflow_plan is None:
        return None
    return {
        "plan_id": workflow_plan.plan_id,
        "role": workflow_plan.role,
        "node_count": len(workflow_plan.nodes),
        "edge_count": len(workflow_plan.edges),
        "topological_node_order": workflow_plan.metadata.get("topological_node_order", []),
        "nodes": [
            {
                "node_id": node.node_id,
                "skill_id": node.skill_id,
                "name": node.name,
                "purpose": _truncate_text(node.purpose, 500),
                "required_tools": list(node.required_tools),
                "expected_outputs": list(node.expected_outputs),
                "status": node.status,
            }
            for node in workflow_plan.nodes
        ],
        "edges": [
            {
                "source_node_id": edge.source_node_id,
                "target_node_id": edge.target_node_id,
                "relation": edge.relation,
            }
            for edge in workflow_plan.edges
        ],
    }


def _retrieved_skill_summary(skill_bundle: SkillBundle | None) -> list[dict[str, Any]]:
    if skill_bundle is None:
        return []
    return [
        {
            "skill_id": skill.skill_id,
            "name": skill.name,
            "required_tools": list(skill.required_tools),
            "metadata": _json_compatible(skill.metadata),
        }
        for skill in skill_bundle.skills
    ]


def _prepared_tools_by_node(node_records: list[NodeExecutionRecord]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for record in node_records:
        tool_names = record.metadata.get("prepared_tool_names")
        if not isinstance(tool_names, list):
            tool_names = []
        prepared.append(
            {
                "node_id": record.node_id,
                "skill_id": record.skill_id,
                "status": record.status,
                "tool_names": [name for name in tool_names if isinstance(name, str)],
            }
        )
    return prepared


def _tool_call_summary(tool_trace_records: list[ToolCallRecord]) -> list[dict[str, Any]]:
    return [
        {
            "call_id": record.tool_call.call_id,
            "tool_name": record.tool_call.name,
            "status": record.result.status,
            "artifact_count": len(record.result.artifact_refs),
            "metadata": _json_compatible(record.result.metadata),
        }
        for record in tool_trace_records
    ]


def _failures_from_node_records(node_records: list[NodeExecutionRecord]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for record in node_records:
        if record.status != "failed":
            continue
        failures.append(
            {
                "node_id": record.node_id,
                "skill_id": record.skill_id,
                "reason": record.output_summary or "node failed",
            }
        )
    return failures


def _skipped_items_from_node_records(node_records: list[NodeExecutionRecord]) -> list[dict[str, Any]]:
    skipped: list[dict[str, Any]] = []
    for record in node_records:
        if record.status != "skipped":
            continue
        skipped.append(
            {
                "node_id": record.node_id,
                "skill_id": record.skill_id,
                "reason": record.output_summary or "node skipped",
            }
        )
    return skipped


def _plan_status(node_records: list[NodeExecutionRecord]) -> str:
    statuses = {record.status for record in node_records}
    if "failed" in statuses:
        return "failed"
    if "skipped" in statuses:
        return "partial"
    return "completed"


def _human_anchor_task_refs(request: TaskRequest) -> list[str]:
    if request.proposed_task_relation is None:
        return []
    return request.proposed_task_relation.human_anchor_task_refs


def _human_anchor_trajectory_refs(request: TaskRequest) -> list[str]:
    if request.proposed_task_relation is None:
        return []
    return request.proposed_task_relation.human_anchor_trajectory_refs


def _proposed_relation_type(request: TaskRequest) -> Any:
    if request.proposed_task_relation is None:
        return None
    return request.proposed_task_relation.relation_type


def _expected_transfer(request: TaskRequest) -> str | None:
    if request.proposed_task_relation is None:
        return None
    return request.proposed_task_relation.expected_transfer


def _parse_dispatch_decision(content: str, completed_runs: list[dict[str, Any]] | None = None) -> DispatchDecision:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        extracted = _extract_single_json_object(content)
        if extracted is None:
            raise RuntimeError(f"meta-agent returned malformed dispatch JSON: {exc}") from exc
        try:
            payload = json.loads(extracted)
        except json.JSONDecodeError as repair_exc:
            raise RuntimeError(f"meta-agent returned malformed dispatch JSON: {repair_exc}") from repair_exc
    if isinstance(payload, dict) and isinstance(payload.get("dispatch_decision"), dict):
        payload = _dispatch_from_decision_wrapper(payload["dispatch_decision"])
    if isinstance(payload, dict) and isinstance(payload.get("route"), str):
        payload = _dispatch_from_route_decision(payload)
    if isinstance(payload, dict) and isinstance(payload.get("run_subagent"), dict):
        payload = _dispatch_from_run_subagent_wrapper(payload)
    if isinstance(payload, dict) and payload.get("finish") is True:
        payload = _dispatch_from_finish_alias(payload)
    if isinstance(payload, dict) and isinstance(payload.get("selected_subagents"), list):
        payload = _dispatch_from_selected_subagent_dag(payload, completed_runs=completed_runs or [])
    if isinstance(payload, dict) and isinstance(payload.get("dispatch"), list):
        payload = _dispatch_from_dispatch_list(payload, completed_runs=completed_runs or [])
    if isinstance(payload, dict) and payload.get("decision") == "run_subagent":
        payload = _dispatch_from_single_node_alias(payload)
    if isinstance(payload, dict) and isinstance(payload.get("action"), str):
        try:
            payload["action"] = DispatchAction(payload["action"])
        except ValueError as exc:
            raise RuntimeError("meta-agent returned unknown dispatch action") from exc
    try:
        return DispatchDecision.model_validate(payload)
    except ValueError as exc:
        raise RuntimeError("meta-agent returned invalid dispatch decision") from exc


def _extract_single_json_object(content: str) -> str | None:
    stripped = _strip_markdown_json_fence(content)
    if stripped != content:
        try:
            json.loads(stripped)
            return stripped
        except json.JSONDecodeError:
            pass
    decoder = json.JSONDecoder()
    candidates: list[tuple[int, int, str]] = []
    for index, char in enumerate(content):
        if char not in "{[":
            continue
        try:
            _, end = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        candidate = content[index : index + end]
        candidates.append((index, index + end, candidate))
    if len(candidates) == 1:
        return candidates[0][2]
    return None


def _strip_markdown_json_fence(content: str) -> str:
    stripped = content.strip()
    if not stripped.startswith("```"):
        return content
    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return content


def _validate_meta_dispatch_decision(
    decision: DispatchDecision,
    allowed_roles: list[str],
    completed_runs: list[dict[str, Any]] | None = None,
    runtime_metadata: dict[str, Any] | None = None,
) -> None:
    completed_runs = completed_runs or []
    runtime_metadata = runtime_metadata or {}
    if decision.action == DispatchAction.RUN_SUBAGENT:
        if decision.target_role not in allowed_roles:
            raise RuntimeError(
                "invalid target_role "
                f"{decision.target_role!r}; allowed target_role values: {', '.join(allowed_roles)}"
            )
        _validate_work_item_run_subagent_decision(decision, completed_runs, runtime_metadata)
    if decision.action == DispatchAction.FINISH_TASK:
        _validate_work_item_finish_decision(completed_runs, runtime_metadata)
        pending_nodes = _pending_selected_workflow_nodes(decision, completed_runs)
        if pending_nodes:
            raise RuntimeError(
                "finish_task rejected because selected workflow has pending selected workflow nodes: "
                + json.dumps(pending_nodes, sort_keys=True)
            )
        incomplete_runs = _incomplete_completed_runs(completed_runs)
        if incomplete_runs:
            raise RuntimeError(
                "finish_task rejected because completed subagent outputs are incomplete: "
                + json.dumps(incomplete_runs, sort_keys=True)
            )
        missing_artifacts = _missing_required_final_artifacts(runtime_metadata, completed_runs)
        if missing_artifacts:
            raise RuntimeError(
                "finish_task rejected because missing required final artifact(s): "
                + ", ".join(missing_artifacts)
            )
        missing_artifact_groups = _missing_required_final_artifact_groups(runtime_metadata, completed_runs)
        if missing_artifact_groups:
            raise RuntimeError(
                "finish_task rejected because missing required final artifact group(s): "
                + "; ".join(missing_artifact_groups)
            )
        invalid_artifacts = _invalid_required_final_artifacts(runtime_metadata, completed_runs)
        if invalid_artifacts:
            raise RuntimeError(
                "finish_task rejected because invalid required final artifact(s): "
                + "; ".join(invalid_artifacts)
            )


def _validate_work_item_run_subagent_decision(
    decision: DispatchDecision,
    completed_runs: list[dict[str, Any]],
    runtime_metadata: dict[str, Any],
) -> None:
    policy = _work_item_routing_policy(runtime_metadata)
    if policy is None:
        return
    target_role = decision.target_role
    if target_role is None:
        return
    field_name = policy["work_item_id_field"]
    if _work_item_final_records_written(completed_runs):
        raise RuntimeError("work-item routing requires END after final records are written")
    pending = _pending_work_item_reviews(completed_runs, policy)
    if _work_item_routing_needs_finalizer(completed_runs, policy):
        if target_role not in policy["finalizer_roles"]:
            raise RuntimeError(
                "work-item routing requires finalizer dispatch before continuing: "
                "all required work items are resolved and final records are not written"
            )
    if pending:
        if target_role not in policy["reviewer_roles"]:
            pending_ids = ", ".join(item["work_item_id"] for item in pending)
            raise RuntimeError(
                "work-item routing requires reviewer dispatch before continuing: "
                + pending_ids
            )
        decision_work_item_id = _work_item_id_from_metadata(decision.metadata, field_name)
        expected_ids = {item["work_item_id"] for item in pending}
        if decision_work_item_id not in expected_ids:
            raise RuntimeError(
                "work-item routing reviewer must target pending work item "
                + ", ".join(sorted(expected_ids))
            )
    if target_role in policy["reviewer_roles"]:
        decision_work_item_id = _work_item_id_from_metadata(decision.metadata, field_name)
        if decision_work_item_id is None:
            raise RuntimeError(
                f"work-item routing reviewer role {target_role!r} requires metadata.{field_name}"
            )
        pending_ids = {item["work_item_id"] for item in pending}
        if decision_work_item_id not in pending_ids:
            if pending_ids:
                raise RuntimeError(
                    "work-item routing reviewer must target pending work item "
                    + ", ".join(sorted(pending_ids))
                )
            raise RuntimeError(
                "work-item routing reviewer rejected because no pending completed executor work item "
                f"matches {field_name}={decision_work_item_id!r}"
            )
    if target_role in policy["executor_roles"]:
        work_item_id = _work_item_id_from_metadata(decision.metadata, field_name)
        if work_item_id is None:
            raise RuntimeError(
                f"work-item routing executor role {target_role!r} requires metadata.{field_name}"
            )
        reviewed = _reviewed_work_item_ids(completed_runs, policy)
        if work_item_id in reviewed:
            raise RuntimeError(
                f"work-item routing executor rejected because work item {work_item_id!r} is already reviewed"
            )
    if target_role in policy["finalizer_roles"]:
        missing = _missing_required_unresolved_work_items(completed_runs, policy)
        if missing:
            raise RuntimeError(
                "work-item routing finalizer rejected because required work item(s) are unresolved: "
                + ", ".join(missing)
            )


def _validate_work_item_finish_decision(
    completed_runs: list[dict[str, Any]],
    runtime_metadata: dict[str, Any],
) -> None:
    policy = _work_item_routing_policy(runtime_metadata)
    if policy is None:
        return
    pending = _pending_work_item_reviews(completed_runs, policy)
    if not pending:
        missing = _missing_required_reviewed_work_items(completed_runs, policy)
        if not missing:
            if _work_item_routing_needs_finalizer(completed_runs, policy):
                raise RuntimeError(
                    "finish_task rejected because work-item routing requires finalizer dispatch before finish: "
                    "final records are not written"
                )
            return
        raise RuntimeError(
            "finish_task rejected because required work item(s) are not reviewed: "
            + ", ".join(missing)
        )
    pending_ids = ", ".join(item["work_item_id"] for item in pending)
    raise RuntimeError(
        "finish_task rejected because work-item routing requires reviewer dispatch before finish: "
        + pending_ids
    )


def _meta_agent_routing_state(
    completed_runs: list[dict[str, Any]],
    runtime_metadata: dict[str, Any],
) -> dict[str, Any]:
    policy = _work_item_routing_policy(runtime_metadata)
    if policy is None:
        return {}
    pending = _pending_work_item_reviews(completed_runs, policy)
    reviewed = sorted(_reviewed_work_item_ids(completed_runs, policy))
    missing = _missing_required_reviewed_work_items(completed_runs, policy)
    retry_exhausted = sorted(_retry_budget_exhausted_work_item_ids(completed_runs, policy))
    failed = sorted(_failed_work_item_ids(completed_runs, policy))
    unresolved_missing = _missing_required_unresolved_work_items(completed_runs, policy)
    work_item_state: dict[str, Any] = {
        "enabled": True,
        "work_item_id_field": policy["work_item_id_field"],
        "executor_roles": sorted(policy["executor_roles"]),
        "reviewer_roles": sorted(policy["reviewer_roles"]),
        "finalizer_roles": sorted(policy["finalizer_roles"]),
        "required_work_item_ids": sorted(policy["required_work_item_ids"]),
        "reviewed_work_item_ids": reviewed,
        "failed_work_item_ids": failed,
        "retry_budget_exhausted_work_item_ids": retry_exhausted,
        "max_failed_executor_attempts_per_work_item": _max_failed_executor_attempts_per_work_item(policy),
        "pending_reviews": pending,
        "missing_required_reviewed_work_item_ids": missing,
        "missing_required_unresolved_work_item_ids": unresolved_missing,
        "generic_scientific_extraction_artifact_contracts": _generic_scientific_extraction_artifact_contracts(),
    }
    if pending:
        work_item_state["required_next_action"] = {
            "route_one_of": sorted(policy["reviewer_roles"]),
            "metadata": {policy["work_item_id_field"]: pending[0]["work_item_id"]},
            "reason": "review pending work item before any executor, finalizer, or END route",
        }
    elif unresolved_missing:
        work_item_state["blocked_routes"] = sorted([*policy["finalizer_roles"], "END"])
        work_item_state["required_next_action"] = {
            "route_one_of": sorted(policy["executor_roles"]),
            "metadata": {policy["work_item_id_field"]: unresolved_missing[0]},
            "reason": "process the next unresolved required work item before finalizer or END",
        }
        work_item_state["reason"] = "required work items must be reviewed or exhaust retry budget before finalizer or END"
    elif _work_item_routing_needs_finalizer(completed_runs, policy):
        work_item_state["blocked_routes"] = sorted([*policy["executor_roles"], *policy["reviewer_roles"], "END"])
        work_item_state["required_next_action"] = {
            "route_one_of": sorted(policy["finalizer_roles"]),
            "reason": "write final artifacts after resolved work items before any executor, reviewer, or END route",
        }
        work_item_state["reason"] = "required work items are resolved and final records are not yet written"
    elif retry_exhausted:
        work_item_state["reason"] = "all missing required work items exhausted retry budget"
    return {"work_item_routing": work_item_state}


def _work_item_routing_policy(runtime_metadata: dict[str, Any]) -> dict[str, Any] | None:
    raw = runtime_metadata.get("work_item_routing")
    if not isinstance(raw, dict) or raw.get("enabled") is not True:
        return None
    return {
        "executor_roles": _metadata_string_set(raw.get("executor_roles")),
        "reviewer_roles": _metadata_string_set(raw.get("reviewer_roles")),
        "finalizer_roles": _metadata_string_set(raw.get("finalizer_roles")),
        "required_work_item_ids": _metadata_string_set(raw.get("required_work_item_ids")),
        "work_item_id_field": raw.get("work_item_id_field")
        if isinstance(raw.get("work_item_id_field"), str) and raw.get("work_item_id_field")
        else "work_item_id",
        "max_failed_executor_attempts_per_work_item": raw.get("max_failed_executor_attempts_per_work_item"),
    }


def _metadata_string_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, str) and item}


def _pending_work_item_reviews(
    completed_runs: list[dict[str, Any]],
    policy: dict[str, Any],
) -> list[dict[str, str]]:
    pending: dict[str, dict[str, str]] = {}
    field_name = policy["work_item_id_field"]
    for run in completed_runs:
        role = run.get("role")
        if not isinstance(role, str):
            continue
        metadata = run.get("dispatch_metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        work_item_id = _work_item_id_from_metadata(metadata, field_name)
        if role in policy["executor_roles"] and _work_item_execution_succeeded(run):
            pending[work_item_id or "<missing>"] = {
                "work_item_id": work_item_id or "<missing>",
                "run_ref": str(run.get("run_ref") or ""),
                "role": role,
            }
        elif role in policy["reviewer_roles"] and work_item_id is not None and _work_item_review_succeeded(run):
            pending.pop(work_item_id, None)
    return list(pending.values())


def _missing_required_reviewed_work_items(
    completed_runs: list[dict[str, Any]],
    policy: dict[str, Any],
) -> list[str]:
    required = policy.get("required_work_item_ids")
    if not required:
        return []
    reviewed = _reviewed_work_item_ids(completed_runs, policy)
    return sorted(item for item in required if item not in reviewed)


def _missing_required_unresolved_work_items(
    completed_runs: list[dict[str, Any]],
    policy: dict[str, Any],
) -> list[str]:
    missing = _missing_required_reviewed_work_items(completed_runs, policy)
    exhausted = _retry_budget_exhausted_work_item_ids(completed_runs, policy)
    return sorted(item for item in missing if item not in exhausted)


def _reviewed_work_item_ids(completed_runs: list[dict[str, Any]], policy: dict[str, Any]) -> set[str]:
    field_name = policy["work_item_id_field"]
    reviewed: set[str] = set()
    for run in completed_runs:
        role = run.get("role")
        if role not in policy["reviewer_roles"]:
            continue
        metadata = run.get("dispatch_metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        work_item_id = _work_item_id_from_metadata(metadata, field_name)
        if work_item_id is not None and _work_item_review_succeeded(run):
            reviewed.add(work_item_id)
    return reviewed


def _work_item_review_succeeded(run: dict[str, Any]) -> bool:
    return _work_item_role_succeeded(run) or _budget_exceeded_with_handoff_artifact(
        run,
        {"validated_records", "final_records"},
    )


def _work_item_execution_succeeded(run: dict[str, Any]) -> bool:
    return _work_item_role_succeeded(run) or _budget_exceeded_with_handoff_artifact(
        run,
        {"candidate_records", "validated_records", "final_records"},
    )


def _work_item_role_succeeded(run: dict[str, Any]) -> bool:
    if run.get("status", "completed") != "completed":
        return False
    contract = run.get("completion_contract")
    if not isinstance(contract, dict):
        return True
    blocking = contract.get("blocking_issues")
    if isinstance(blocking, list) and blocking:
        return False
    if contract.get("assigned_task_complete") is False:
        return False
    if contract.get("ready_for_task_end") is False:
        return False
    return True


def _budget_exceeded_with_handoff_artifact(run: dict[str, Any], artifact_kinds: set[str]) -> bool:
    if run.get("status") not in {"budget_exceeded", "guard_failed", "failed"}:
        return False
    return _run_has_handoff_artifact(run, artifact_kinds)


def _run_has_handoff_artifact(run: dict[str, Any], artifact_kinds: set[str]) -> bool:
    artifact_refs = run.get("artifact_refs")
    if not isinstance(artifact_refs, list):
        return False
    filenames_by_kind = {
        "candidate_records": {"candidate_records.json"},
        "validated_records": {"validated_records.json"},
        "final_records": {"final_records.jsonl", "biology_component_records.jsonl"},
    }
    allowed_filenames = {name for kind in artifact_kinds for name in filenames_by_kind.get(kind, set())}
    for ref in artifact_refs:
        if not isinstance(ref, dict):
            continue
        metadata = ref.get("metadata") if isinstance(ref.get("metadata"), dict) else {}
        names = {
            str(metadata.get("artifact_kind") or ""),
            str(metadata.get("filename") or ""),
            Path(str(ref.get("uri") or "")).name,
        }
        if names & artifact_kinds or names & allowed_filenames:
            return True
    return False


def _work_item_lifecycle_status_for_run(
    run: dict[str, Any],
    role: str,
    policy: dict[str, Any],
) -> str:
    executor_roles = set(policy.get("executor_roles") or [])
    reviewer_roles = set(policy.get("reviewer_roles") or [])
    finalizer_roles = set(policy.get("finalizer_roles") or [])
    status = run.get("status", "completed")
    if status == "interrupted":
        return "interrupted"
    if role in finalizer_roles:
        if _work_item_role_succeeded(run) or _run_has_handoff_artifact(run, {"final_records"}):
            return "completed"
        if status == "budget_exceeded":
            return "budget_exceeded"
        return "failed"
    if role in reviewer_roles and _work_item_review_succeeded(run):
        return "completed"
    if role in executor_roles and _work_item_execution_succeeded(run):
        return "claimed" if reviewer_roles else "completed"
    if status == "budget_exceeded":
        return "budget_exceeded"
    if status != "completed":
        return "failed"
    if role in executor_roles | reviewer_roles:
        return "failed"
    return "claimed"


def _failed_executor_attempt_count(
    completed_runs: list[dict[str, Any]],
    policy: dict[str, Any],
    work_item_id: str,
) -> int:
    field_name = policy["work_item_id_field"]
    count = 0
    for run in completed_runs:
        role = run.get("role")
        if role not in policy["executor_roles"]:
            continue
        metadata = run.get("dispatch_metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        if _work_item_id_from_metadata(metadata, field_name) != work_item_id:
            continue
        if not _work_item_execution_succeeded(run):
            count += 1
    return count


def _failed_work_item_ids(completed_runs: list[dict[str, Any]], policy: dict[str, Any]) -> set[str]:
    failed: set[str] = set()
    for work_item_id in policy.get("required_work_item_ids") or set():
        if _failed_executor_attempt_count(completed_runs, policy, work_item_id) > 0:
            failed.add(work_item_id)
    return failed


def _retry_budget_exhausted_work_item_ids(completed_runs: list[dict[str, Any]], policy: dict[str, Any]) -> set[str]:
    exhausted: set[str] = set()
    max_attempts = _max_failed_executor_attempts_per_work_item(policy)
    for work_item_id in policy.get("required_work_item_ids") or set():
        if _failed_executor_attempt_count(completed_runs, policy, work_item_id) >= max_attempts:
            exhausted.add(work_item_id)
    return exhausted


def _work_item_final_records_written(completed_runs: list[dict[str, Any]]) -> bool:
    return any(_run_has_handoff_artifact(run, {"final_records"}) for run in completed_runs)


def _work_item_routing_needs_finalizer(completed_runs: list[dict[str, Any]], policy: dict[str, Any]) -> bool:
    if not policy["finalizer_roles"]:
        return False
    if not policy["required_work_item_ids"]:
        return False
    if _work_item_final_records_written(completed_runs):
        return False
    if _pending_work_item_reviews(completed_runs, policy):
        return False
    return not _missing_required_unresolved_work_items(completed_runs, policy)


def _role_attempted_for_work_item(
    completed_runs: list[dict[str, Any]],
    policy: dict[str, Any],
    role_name: str,
    work_item_id: str,
) -> bool:
    field_name = policy["work_item_id_field"]
    for run in completed_runs:
        if run.get("role") != role_name:
            continue
        metadata = run.get("dispatch_metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        if _work_item_id_from_metadata(metadata, field_name) == work_item_id:
            return True
    return False


def _max_failed_executor_attempts_per_work_item(policy: dict[str, Any]) -> int:
    value = _optional_positive_int(policy.get("max_failed_executor_attempts_per_work_item"))
    return value if value is not None else 2


def _next_unresolved_work_item_id(
    completed_runs: list[dict[str, Any]],
    policy: dict[str, Any],
    *,
    current_work_item_id: str,
) -> str | None:
    required = sorted(policy.get("required_work_item_ids") or [])
    if not required:
        return None
    reviewed = _reviewed_work_item_ids(completed_runs, policy)
    max_attempts = _max_failed_executor_attempts_per_work_item(policy)
    for work_item_id in required:
        if work_item_id == current_work_item_id:
            continue
        if work_item_id in reviewed:
            continue
        if _failed_executor_attempt_count(completed_runs, policy, work_item_id) >= max_attempts:
            continue
        return work_item_id
    return None


def _first_available_recovery_role(
    available_roles: list[str],
    *,
    preferred_names: tuple[str, ...],
    excluded_roles: set[str],
) -> str | None:
    for preferred in preferred_names:
        if preferred in available_roles and preferred not in excluded_roles:
            return preferred
    for role in available_roles:
        if role not in excluded_roles:
            return role
    return None


def _generic_scientific_extraction_artifact_contracts() -> list[str]:
    return [
        "document_inventory.json",
        "candidate_source_files.json",
        "candidate_tables.json",
        "candidate_rows.json",
        "candidate_records.json",
        "validated_records.json",
        "final_records.jsonl",
    ]


def _scientific_handoff_bootstrap_enabled(policy_metadata: dict[str, Any]) -> bool:
    raw_value = policy_metadata.get("scientific_handoff_bootstrap_enabled")
    return raw_value is not False


def _tool_runtime_has_registered_tool(tool_runtime: ToolRuntime, name: str) -> bool:
    registry = getattr(tool_runtime, "_registry", None)
    get_spec = getattr(registry, "get_spec", None)
    return callable(get_spec) and get_spec(name) is not None


def _scientific_handoff_bootstrap_calls(
    *,
    role: str,
    role_instruction: str,
    task_goal: str,
    dispatch_metadata: dict[str, Any],
    lab_state_registry: FileLabStateRegistry | None,
    task_id: str,
    artifact_root: Path,
    runtime_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    role_name = role.casefold()
    runtime_metadata = runtime_metadata if isinstance(runtime_metadata, dict) else {}
    context = _scientific_work_item_context(
        role_instruction=role_instruction,
        task_goal=task_goal,
        dispatch_metadata=dispatch_metadata,
    )
    if role_name == "surveyagent":
        return _survey_bootstrap_calls(context, task_goal, artifact_root=artifact_root)
    if role_name in {"execagent", "recoveryextractionagent"}:
        return _exec_bootstrap_calls(context, task_goal, artifact_root=artifact_root, runtime_metadata=runtime_metadata)
    if role_name == "criticagent":
        candidate_path = _latest_scientific_artifact_uri(
            lab_state_registry,
            task_id=task_id,
            work_item_id=context.get("work_item_id"),
            artifact_kind="candidate_records",
            filename="candidate_records.json",
        )
        if candidate_path is None:
            return []
        return [
            {
                "name": "validate_candidate_records",
                "arguments": {
                    "candidate_records_path": candidate_path,
                    "work_item_id": context.get("work_item_id"),
                    "sequence_extraction_profile": _scientific_sequence_extraction_profile(runtime_metadata, task_goal),
                },
            }
        ]
    if role_name == "writeagent":
        current_work_item_id = context.get("work_item_id")
        current_records = _final_records_for_write_bootstrap(
            lab_state_registry,
            task_id=task_id,
            work_item_id=current_work_item_id if isinstance(current_work_item_id, str) else None,
        )
        if (
            not current_records
            and isinstance(current_work_item_id, str)
            and current_work_item_id
            and isinstance(context.get("article_package"), str)
            and context.get("article_package")
            and _dynamic_zero_output_recovery_enabled(runtime_metadata, dispatch_metadata)
        ):
            candidate_path = _bootstrap_artifact_path(current_work_item_id, "candidate_records.json", artifact_root)
            recovery_calls = [
                {
                    "name": "write_report",
                    "arguments": {
                        "artifact_name": _work_item_artifact_name(current_work_item_id, "recovery_reason.json"),
                        "format": "json",
                        "content": {
                            "work_item_id": current_work_item_id,
                            "reason": "zero final records before writer finalization",
                            "policy": "bounded generic scientific extraction recovery",
                        },
                    },
                },
                *_exec_bootstrap_calls(
                    context,
                    task_goal,
                    artifact_root=artifact_root,
                    runtime_metadata=runtime_metadata,
                ),
                {
                    "name": "serialize_final_records",
                    "arguments": {
                        "records_path": candidate_path,
                        "artifact_name": "biology_component_records.jsonl",
                        "also_write_final_records": True,
                        "work_item_id": current_work_item_id,
                    },
                },
                {
                    "name": "write_report",
                    "arguments": {
                        "artifact_name": _work_item_artifact_name(current_work_item_id, "biology_component_report.md"),
                        "format": "markdown",
                        "content": (
                            "# Scientific Extraction Recovery Finalization\n\n"
                            "Writer found zero current records and ran one bounded generic recovery pass "
                            "over the configured work-item sources. Ground truth was not used."
                        ),
                    },
                },
            ]
            return recovery_calls
        records = _final_records_for_write_bootstrap(lab_state_registry, task_id=task_id)
        return [
            {
                "name": "serialize_final_records",
                "arguments": {
                    "records": records,
                    "artifact_name": "biology_component_records.jsonl",
                    "also_write_final_records": True,
                },
            },
            {
                "name": "write_report",
                "arguments": {
                    "artifact_name": "biology_component_report.md",
                    "format": "markdown",
                    "content": (
                        "# Scientific Extraction Finalization\n\n"
                        f"Serialized {len(records)} record(s) from validated or candidate upstream artifacts. "
                        "This report was generated from generic handoff artifacts without using ground truth."
                    ),
                },
            },
        ]
    return []


def _survey_bootstrap_calls(context: dict[str, Any], task_goal: str, *, artifact_root: Path) -> list[dict[str, Any]]:
    root = context.get("article_package")
    if not isinstance(root, str) or not root:
        return []
    work_item_id = context.get("work_item_id")
    source_files = context.get("source_files") if isinstance(context.get("source_files"), list) else []
    return [
        {
            "name": "build_document_inventory",
            "arguments": {"root": root, "work_item_id": work_item_id, "source_files": source_files},
        },
        {
            "name": "discover_candidate_source_files",
            "arguments": {
                "document_inventory_path": _bootstrap_artifact_path(
                    work_item_id,
                    "document_inventory.json",
                    artifact_root,
                ),
                "work_item_id": work_item_id,
                "task_goal": task_goal,
            },
        },
        {
            "name": "discover_candidate_tables",
            "arguments": {
                "candidate_source_files_path": _bootstrap_artifact_path(
                    work_item_id,
                    "candidate_source_files.json",
                    artifact_root,
                ),
                "work_item_id": work_item_id,
            },
        },
    ]


def _exec_bootstrap_calls(
    context: dict[str, Any],
    task_goal: str,
    *,
    artifact_root: Path,
    runtime_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    runtime_metadata = runtime_metadata if isinstance(runtime_metadata, dict) else {}
    calls = _survey_bootstrap_calls(context, task_goal, artifact_root=artifact_root)
    work_item_id = context.get("work_item_id")
    if not calls:
        return []
    max_rows_per_table = _scientific_max_rows_per_table(runtime_metadata, task_goal)
    extraction_profile = _scientific_sequence_extraction_profile(runtime_metadata, task_goal)
    primary_component_tables_only = _scientific_primary_component_tables_only(runtime_metadata)
    calls.extend(
        [
            {
                "name": "extract_candidate_rows",
                "arguments": {
                    "candidate_tables_path": _bootstrap_artifact_path(
                        work_item_id,
                        "candidate_tables.json",
                        artifact_root,
                    ),
                    "work_item_id": work_item_id,
                    "max_rows_per_table": max_rows_per_table,
                    "sequence_extraction_profile": extraction_profile,
                    "primary_component_tables_only": primary_component_tables_only,
                },
            },
            {
                "name": "build_candidate_records",
                "arguments": {
                    "candidate_rows_path": _bootstrap_artifact_path(
                        work_item_id,
                        "candidate_rows.json",
                        artifact_root,
                    ),
                    "article_id": work_item_id,
                    "work_item_id": work_item_id,
                    "sequence_extraction_profile": extraction_profile,
                    "deduplicate_sequences": True,
                },
            },
        ]
    )
    return calls


def _scientific_max_rows_per_table(runtime_metadata: dict[str, Any], task_goal: str) -> int:
    for key in ("scientific_max_rows_per_table", "recovery_max_rows_per_table", "max_rows_per_table"):
        value = runtime_metadata.get(key)
        if isinstance(value, int) and value > 0:
            return value
    if "promoter" in task_goal.casefold() or "sequence" in task_goal.casefold():
        return 10_000
    return 500


def _scientific_sequence_extraction_profile(runtime_metadata: dict[str, Any], task_goal: str) -> str | None:
    for key in ("scientific_sequence_extraction_profile", "sequence_extraction_profile"):
        value = runtime_metadata.get(key)
        if isinstance(value, str) and value:
            return value
    lowered = task_goal.casefold()
    if "promoter" in lowered or "regulatory dna" in lowered:
        return "promoter"
    return None


def _scientific_primary_component_tables_only(runtime_metadata: dict[str, Any]) -> bool:
    return bool(runtime_metadata.get("scientific_primary_component_tables_only", False))


def _dynamic_zero_output_recovery_enabled(
    runtime_metadata: dict[str, Any],
    dispatch_metadata: dict[str, Any],
) -> bool:
    if dispatch_metadata.get("zero_output_recovery_enabled") is not None:
        return bool(dispatch_metadata.get("zero_output_recovery_enabled"))
    value = runtime_metadata.get("dynamic_zero_output_recovery_enabled")
    return value is not False


def _work_item_artifact_name(work_item_id: Any, filename: str) -> str:
    if isinstance(work_item_id, str) and work_item_id:
        return f"{_safe_filename(work_item_id)}/{filename}"
    return filename


def _bootstrap_artifact_path(work_item_id: Any, filename: str, artifact_root: Path | None = None) -> str:
    root = artifact_root or Path("artifacts") / "tools"
    if isinstance(work_item_id, str) and work_item_id:
        return str(root / _safe_filename(work_item_id) / filename)
    return str(root / filename)


def _scientific_work_item_context(
    *,
    role_instruction: str,
    task_goal: str,
    dispatch_metadata: dict[str, Any],
) -> dict[str, Any]:
    work_item_id = _work_item_id_from_any(dispatch_metadata.get("work_item_id")) or _work_item_id_from_text(
        role_instruction
    )
    metadata_article_package = _article_package_from_metadata(dispatch_metadata)
    metadata_source_files = _source_files_from_metadata(dispatch_metadata)
    instruction_article_package = _article_package_from_text(role_instruction)
    instruction_source_files = _source_files_from_text(role_instruction)
    block_article_package = None
    block_source_files: list[str] = []
    if work_item_id:
        block = _work_item_block(task_goal, work_item_id)
        if block:
            block_article_package = _article_package_from_text(block)
            block_source_files = _source_files_from_text(block)
    article_package = block_article_package or metadata_article_package or instruction_article_package
    source_files = block_source_files or metadata_source_files or instruction_source_files
    return {
        "work_item_id": work_item_id,
        "article_package": article_package,
        "source_files": source_files,
    }


def _work_item_preflight_issues(context: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    article_package = context.get("article_package")
    if isinstance(article_package, str) and article_package:
        root = Path(article_package).expanduser()
        try:
            root_exists = root.exists()
            root_is_dir = root.is_dir() if root_exists else False
        except OSError as exc:
            issues.append(f"lab_path could not be inspected: {root}: {exc}")
            root_exists = False
            root_is_dir = False
        if not root_exists:
            issues.append(f"lab_path is not an existing directory: {root}")
        elif not root_is_dir:
            issues.append(f"lab_path is not a directory: {root}")
    source_files = context.get("source_files")
    if isinstance(source_files, list):
        for source_file in source_files:
            if not isinstance(source_file, str) or not source_file:
                continue
            path = Path(source_file).expanduser()
            try:
                path_exists = path.exists()
                path_is_file = path.is_file() if path_exists else False
            except OSError as exc:
                issues.append(f"configured source file could not be inspected: {path}: {exc}")
                path_exists = False
                path_is_file = False
            if not path_exists:
                issues.append(f"configured source file does not exist: {path}")
            elif not path_is_file:
                issues.append(f"configured source path is not a file: {path}")
    return issues


def _work_item_id_from_any(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _work_item_id_from_text(text: str) -> str | None:
    match = re.search(r"work item ['\"](?P<id>[A-Za-z0-9_.-]+)['\"]", text)
    if match:
        return match.group("id")
    match = re.search(r"work_item_id:\s*(?P<id>[A-Za-z0-9_.-]+)", text)
    if match:
        return match.group("id")
    return None


def _article_package_from_text(text: str) -> str | None:
    patterns = [
        r"article_package:\s*(?P<path>/[^\n]+)",
        r"Article package path:\s*(?P<path>/[^\n]+)",
        r"article package:\s*(?P<path>/[^\n]+?)(?:\.\s+(?:Use|Read|Inspect|Identify|Cover|Produce|Report|Do not)\b|\n|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            path = match.group("path").strip()
            path = re.sub(r"\s+(Use|Exact) the exact source files:.*$", "", path, flags=re.IGNORECASE).strip()
            path = re.sub(r"\s+Exact source files:.*$", "", path, flags=re.IGNORECASE).strip()
            return _clean_article_package_path(path)
    return None


def _article_package_from_metadata(metadata: dict[str, Any]) -> str | None:
    for key in ("lab_path", "article_package", "article_path", "work_item_path", "root"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return _clean_article_package_path(value)
    return None


def _source_files_from_metadata(metadata: dict[str, Any]) -> list[str]:
    value = metadata.get("source_files") or metadata.get("exact_source_files")
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _clean_article_package_path(value: str) -> str:
    candidate = value.strip().rstrip(".")
    existing_prefix = _longest_existing_directory_prefix(candidate)
    if existing_prefix is not None:
        return existing_prefix
    candidate = re.sub(r"\s+(Use|Read|Inspect|Identify|Cover|Produce|Report|Do not)\b.*$", "", candidate, flags=re.IGNORECASE)
    return candidate.strip().rstrip(".")


def _longest_existing_directory_prefix(value: str) -> str | None:
    words = value.split()
    for end in range(len(words), 0, -1):
        candidate = " ".join(words[:end]).rstrip(".,;:")
        if not candidate.startswith(("/", "~")):
            continue
        path = Path(candidate).expanduser()
        try:
            is_dir = path.is_dir()
        except OSError:
            continue
        if is_dir:
            return str(path)
    path = Path(value.rstrip(".,;:")).expanduser()
    try:
        return str(path) if path.is_dir() else None
    except OSError:
        return None


def _source_files_from_text(text: str) -> list[str]:
    paths = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        value = stripped[2:].strip()
        if value.startswith("/"):
            paths.append(value)
    return paths


def _work_item_block(task_goal: str, work_item_id: str) -> str | None:
    pattern = re.compile(
        r"(?:^|\n)\s*\d+\.\s+work_item_id:\s*"
        + re.escape(work_item_id)
        + r"(?P<block>.*?)(?=\n\s*\d+\.\s+work_item_id:|\Z)",
        flags=re.DOTALL,
    )
    match = pattern.search(task_goal)
    if match:
        return match.group("block")
    pattern = re.compile(
        r"work_item_id:\s*" + re.escape(work_item_id) + r"(?P<block>.*?)(?=\n\s*work_item_id:|\Z)",
        flags=re.DOTALL,
    )
    match = pattern.search(task_goal)
    return match.group("block") if match else None


def _latest_scientific_artifact_uri(
    lab_state_registry: FileLabStateRegistry | None,
    *,
    task_id: str,
    work_item_id: str | None,
    artifact_kind: str,
    filename: str,
) -> str | None:
    if lab_state_registry is None:
        return None
    artifacts = lab_state_registry.list_artifacts(task_id)
    for artifact in reversed(artifacts):
        metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
        if work_item_id and metadata.get("work_item_id") != work_item_id:
            continue
        names = {str(metadata.get("artifact_kind") or ""), str(metadata.get("filename") or ""), Path(artifact.uri).name}
        if artifact_kind in names or filename in names:
            return artifact.uri
    return None


def _final_records_for_write_bootstrap(
    lab_state_registry: FileLabStateRegistry | None,
    *,
    task_id: str,
    work_item_id: str | None = None,
) -> list[dict[str, Any]]:
    if lab_state_registry is None:
        return []
    artifacts = lab_state_registry.list_artifacts(task_id)
    final_paths: dict[str, list[str]] = {}
    validated_paths: dict[str, list[str]] = {}
    candidate_paths: dict[str, list[str]] = {}
    for artifact in artifacts:
        metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
        artifact_work_item_id = metadata.get("work_item_id")
        if not isinstance(artifact_work_item_id, str) or not artifact_work_item_id:
            continue
        if work_item_id is not None and artifact_work_item_id != work_item_id:
            continue
        filename = str(metadata.get("filename") or Path(artifact.uri).name)
        kind = str(metadata.get("artifact_kind") or "")
        if kind == "final_records" or filename in {"final_records.jsonl", "biology_component_records.jsonl"}:
            final_paths.setdefault(artifact_work_item_id, []).append(artifact.uri)
        elif kind == "validated_records" or filename == "validated_records.json":
            validated_paths.setdefault(artifact_work_item_id, []).append(artifact.uri)
        elif kind == "candidate_records" or filename == "candidate_records.json":
            candidate_paths.setdefault(artifact_work_item_id, []).append(artifact.uri)
    work_item_ids = _dedupe(
        list(final_paths.keys()) + list(validated_paths.keys()) + list(candidate_paths.keys())
    )
    paths = [
        path
        for work_item in work_item_ids
        for path in [
            _preferred_scientific_artifact_path(
                final_paths.get(work_item, []),
                validated_paths.get(work_item, []),
                candidate_paths.get(work_item, []),
            )
        ]
        if path is not None
    ]
    records: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, Any]] = set()
    for path in paths:
        for record in _records_from_scientific_artifact(path):
            key = (record.get("article_id"), record.get("component_name"), record.get("sequence"))
            if key in seen:
                continue
            seen.add(key)
            records.append(record)
    return records


def _preferred_scientific_artifact_path(
    final_paths: list[str],
    validated_paths: list[str],
    candidate_paths: list[str],
) -> str | None:
    groups = [final_paths, validated_paths, candidate_paths]
    for group in groups:
        for path in reversed(group):
            if _scientific_artifact_record_count(path) > 0:
                return path
    for group in groups:
        if group:
            return group[-1]
    return None


def _scientific_artifact_record_count(path: str) -> int:
    return len(_records_from_scientific_artifact(path))


def _records_from_scientific_artifact(path: str) -> list[dict[str, Any]]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except Exception:
        return []
    if Path(path).suffix.casefold() == ".jsonl":
        records: list[dict[str, Any]] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except Exception:
                continue
            if isinstance(item, dict):
                records.append(item)
        return records
    try:
        payload = json.loads(text)
    except Exception:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    accepted = payload.get("accepted_records")
    if isinstance(accepted, list):
        return [item for item in accepted if isinstance(item, dict)]
    records = payload.get("records")
    if isinstance(records, list):
        return [item for item in records if isinstance(item, dict) and item.get("status") != "rejected"]
    return []


def _candidate_artifacts_for_prompt(artifact_refs: list[ArtifactRef]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for ref in artifact_refs:
        if not _artifact_ref_is_candidate_context(ref):
            continue
        candidates.append(
            {
                "uri": ref.uri,
                "type": ref.type,
                "semantic_kind": _artifact_semantic_kind(ref),
                "metadata": _json_compatible(ref.metadata),
            }
        )
    return candidates


def _workflow_output_node_contract(
    *,
    node: WorkflowNode,
    available_tool_names: list[str],
    artifact_refs: list[ArtifactRef],
    previous_summaries: list[dict[str, str]],
) -> dict[str, Any]:
    node_text = _workflow_node_text(node, available_tool_names)
    output_capable = _workflow_node_is_output_capable(node_text, available_tool_names)
    requires_concrete_candidates = output_capable and _workflow_node_requires_candidate_inputs(node_text)
    candidate_artifacts = _candidate_artifacts_for_prompt(artifact_refs)
    candidate_summary_refs = [
        {
            "node_id": summary.get("node_id"),
            "skill_id": summary.get("skill_id"),
            "name": summary.get("name"),
        }
        for summary in previous_summaries
        if _summary_mentions_candidate_context(summary)
    ]
    return {
        "schema_version": "v1",
        "output_capable": output_capable,
        "requires_concrete_candidates": requires_concrete_candidates,
        "candidate_artifacts_present": bool(candidate_artifacts),
        "candidate_artifacts": candidate_artifacts,
        "candidate_summary_refs": candidate_summary_refs,
        "accepted_intermediate_artifacts": _generic_scientific_extraction_artifact_contracts(),
        "required_handoff": (
            "Use explicit candidate artifacts, tables, rows, columns, and source references for record construction."
            if requires_concrete_candidates
            else "No record-construction handoff is required for this node."
        ),
        "missing_candidate_action": (
            "Request upstream Survey/Discovery to produce candidate source/table/row artifacts; do not call final "
            "record-writing tools from vague summaries alone."
            if requires_concrete_candidates and not candidate_artifacts
            else None
        ),
    }


def _workflow_node_text(node: WorkflowNode, available_tool_names: list[str]) -> str:
    return " ".join(
        str(item)
        for item in [
            node.name,
            node.purpose,
            *node.required_inputs,
            *node.expected_outputs,
            *node.required_tools,
            *available_tool_names,
        ]
        if isinstance(item, str)
    ).casefold()


def _workflow_node_is_output_capable(node_text: str, available_tool_names: list[str]) -> bool:
    output_tools = {"write_jsonl", "write_report"}
    if output_tools.intersection({name for name in available_tool_names if isinstance(name, str)}):
        return True
    return any(token in node_text for token in ("write", "serialize", "output", "artifact", "jsonl", "report"))


def _workflow_node_requires_candidate_inputs(node_text: str) -> bool:
    return any(token in node_text for token in ("record", "jsonl", "candidate", "row", "table", "source"))


def _summary_mentions_candidate_context(summary: dict[str, str]) -> bool:
    text = " ".join(str(value) for value in summary.values() if isinstance(value, str)).casefold()
    return any(token in text for token in ("candidate", "table", "row", "source", "inventory", "document"))


def _artifact_ref_is_candidate_context(ref: ArtifactRef) -> bool:
    metadata = ref.metadata if isinstance(ref.metadata, dict) else {}
    names = [
        ref.uri,
        metadata.get("filename"),
        metadata.get("name"),
        metadata.get("source_uri"),
    ]
    text = " ".join(str(item) for item in names if isinstance(item, str)).casefold()
    contract_names = {name.casefold() for name in _generic_scientific_extraction_artifact_contracts()}
    if any(name in text for name in contract_names):
        return True
    if any(token in text for token in ("candidate", "inventory", "table", "row", "source", "record", "jsonl")):
        return True
    return _artifact_semantic_kind(ref) in {"candidate", "final"}


def _work_item_id_from_metadata(metadata: dict[str, Any], field_name: str) -> str | None:
    value = metadata.get(field_name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _missing_required_final_artifacts(
    runtime_metadata: dict[str, Any],
    completed_runs: list[dict[str, Any]],
) -> list[str]:
    required = runtime_metadata.get("required_final_artifacts")
    if not isinstance(required, list):
        return []
    required_names = [str(item) for item in required if isinstance(item, str) and item.strip()]
    if not required_names:
        return []
    artifact_refs = _completed_artifact_refs(completed_runs)
    missing: list[str] = []
    for required_name in required_names:
        if not any(_artifact_ref_matches_required_name(ref, required_name) for ref in artifact_refs):
            missing.append(required_name)
    return missing


def _completed_artifact_refs(completed_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    artifact_refs: list[dict[str, Any]] = []
    for run in completed_runs:
        raw_refs = run.get("artifact_refs")
        if isinstance(raw_refs, list):
            artifact_refs.extend(ref for ref in raw_refs if isinstance(ref, dict))
    return artifact_refs


def _missing_required_final_artifact_groups(
    runtime_metadata: dict[str, Any],
    completed_runs: list[dict[str, Any]],
) -> list[str]:
    groups = runtime_metadata.get("required_final_artifact_groups")
    if not isinstance(groups, list):
        return []
    artifact_refs = _completed_artifact_refs(completed_runs)
    missing: list[str] = []
    for group in groups:
        names = _required_artifact_group_names(group)
        if not names:
            continue
        if not any(_artifact_ref_matches_required_name(ref, name) for ref in artifact_refs for name in names):
            missing.append("one of " + ", ".join(names))
    return missing


def _required_artifact_group_names(group: Any) -> list[str]:
    if isinstance(group, dict):
        raw_names = group.get("one_of")
    else:
        raw_names = group
    if not isinstance(raw_names, list):
        return []
    return [str(item) for item in raw_names if isinstance(item, str) and item.strip()]


def _incomplete_completed_runs(completed_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    incomplete: list[dict[str, Any]] = []
    for run in completed_runs:
        if _run_has_handoff_artifact(run, {"final_records"}):
            continue
        if run.get("status", "completed") != "completed":
            if _budget_exceeded_with_handoff_artifact(
                run,
                {"candidate_records", "validated_records", "final_records"},
            ):
                continue
            incomplete.append(
                {
                    "run_ref": run.get("run_ref"),
                    "role": run.get("role"),
                    "status": run.get("status"),
                    "failure_reason": run.get("failure_reason"),
                }
            )
            continue
        contract = run.get("completion_contract")
        if not isinstance(contract, dict):
            continue
        if contract.get("blocking_issues"):
            incomplete.append(
                {
                    "run_ref": run.get("run_ref"),
                    "role": run.get("role"),
                    "blocking_issues": contract.get("blocking_issues"),
                }
            )
    return incomplete


def _validate_no_progress_dispatch(
    *,
    decision: DispatchDecision,
    role_results: list[dict[str, Any]],
    runtime_metadata: dict[str, Any],
) -> None:
    target_role = decision.target_role
    if not target_role:
        return
    max_consecutive = runtime_metadata.get("max_consecutive_dispatches_per_role", 4)
    if not isinstance(max_consecutive, int) or max_consecutive < 1:
        max_consecutive = 4
    consecutive_count = 0
    for run in reversed(role_results):
        if run.get("role") != target_role:
            break
        consecutive_count += 1
    if consecutive_count >= max_consecutive:
        raise RuntimeError(
            "repeated same-role dispatch rejected: "
            + json.dumps(
                {
                    "target_role": target_role,
                    "consecutive_completed_dispatches": consecutive_count,
                    "max_consecutive_dispatches_per_role": max_consecutive,
                    "instruction": decision.instruction,
                },
                sort_keys=True,
            )
        )

    max_repeats = runtime_metadata.get("max_repeated_no_progress_dispatches", 3)
    if not isinstance(max_repeats, int) or max_repeats < 1:
        max_repeats = 3
    repeat_count = 0
    for run in reversed(role_results):
        if run.get("role") != target_role:
            break
        repeat_count += 1
        if _run_made_progress(run):
            break
    if repeat_count >= max_repeats:
        raise RuntimeError(
            "repeated no-progress dispatch rejected: "
            + json.dumps(
                {
                    "target_role": target_role,
                    "repeat_count": repeat_count,
                    "max_repeated_no_progress_dispatches": max_repeats,
                    "instruction": decision.instruction,
                },
                sort_keys=True,
            )
        )


def _run_made_progress(run: dict[str, Any]) -> bool:
    if run.get("status", "completed") != "completed":
        return False
    artifact_refs = run.get("artifact_refs")
    if isinstance(artifact_refs, list) and artifact_refs:
        return True
    contract = run.get("completion_contract")
    if isinstance(contract, dict) and contract.get("ready_for_task_end") is True:
        return True
    tool_count = run.get("tool_call_count")
    if isinstance(tool_count, int) and tool_count > 0:
        return True
    return False


def _artifact_ref_matches_required_name(ref: dict[str, Any], required_name: str) -> bool:
    uri = ref.get("uri")
    metadata = ref.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    candidates = [
        uri if isinstance(uri, str) else "",
        metadata.get("filename") if isinstance(metadata.get("filename"), str) else "",
        metadata.get("name") if isinstance(metadata.get("name"), str) else "",
    ]
    return any(Path(candidate).name == required_name or candidate.endswith(f"/{required_name}") for candidate in candidates)


def _invalid_required_final_artifacts(
    runtime_metadata: dict[str, Any],
    completed_runs: list[dict[str, Any]],
) -> list[str]:
    required = runtime_metadata.get("required_final_artifacts")
    if not isinstance(required, list):
        return []
    artifact_refs: list[dict[str, Any]] = []
    for run in completed_runs:
        raw_refs = run.get("artifact_refs")
        if isinstance(raw_refs, list):
            artifact_refs.extend(ref for ref in raw_refs if isinstance(ref, dict))
    invalid: list[str] = []
    for required_name in [item for item in required if isinstance(item, str)]:
        if not required_name.endswith(".jsonl"):
            continue
        for ref in artifact_refs:
            if not _artifact_ref_matches_required_name(ref, required_name):
                continue
            reason = _validate_required_jsonl_artifact(ref)
            if reason is not None:
                invalid.append(f"{required_name}: {reason}")
            break
    return invalid


def _validate_required_jsonl_artifact(ref: dict[str, Any]) -> str | None:
    uri = ref.get("uri")
    if not isinstance(uri, str) or not uri:
        return "missing artifact uri"
    path = _local_path_from_artifact_uri(uri)
    if path is None or not path.exists():
        return f"artifact file not found at {uri}"
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            return f"line {line_number} is not valid JSON: {exc}"
        if not isinstance(payload, dict):
            return f"line {line_number} is not a JSON object"
        records.append(payload)
    for index, record in enumerate(records, 1):
        reason = _placeholder_record_reason(record)
        if reason is not None:
            return f"record {index} {reason}"
    return None


def _placeholder_record_reason(record: dict[str, Any]) -> str | None:
    text_values = {
        key: value.strip().casefold()
        for key, value in record.items()
        if isinstance(value, str)
    }
    placeholder_values = {"unknown", "n/a", "na", "none", "tbd", "placeholder"}
    for key in ("article_id", "component_name", "component_type"):
        value = text_values.get(key)
        if value in placeholder_values:
            return f"has placeholder {key}={record.get(key)!r}"
    component_name = text_values.get("component_name", "")
    if "validated promoter record" in component_name or "record " == component_name[:7]:
        return f"has placeholder component_name={record.get('component_name')!r}"
    if records_require_evidence(record) and not _record_has_evidence(record):
        return "is missing evidence_text/evidence_source"
    return None


def records_require_evidence(record: dict[str, Any]) -> bool:
    return any(key in record for key in ("component_name", "component_type", "sequence", "article_id"))


def _record_has_evidence(record: dict[str, Any]) -> bool:
    evidence_text = record.get("evidence_text")
    evidence_source = record.get("evidence_source")
    return isinstance(evidence_text, str) and bool(evidence_text.strip()) and isinstance(evidence_source, str) and bool(evidence_source.strip())


def _pending_selected_workflow_nodes(
    decision: DispatchDecision,
    completed_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    workflow = _workflow_from_finish_decision_or_completed_runs(decision, completed_runs)
    if workflow is None:
        return []
    nodes = _workflow_nodes(workflow)
    if not nodes:
        return []
    completed_node_ids = {
        str(run.get("meta_workflow_node_id"))
        for run in completed_runs
        if run.get("status", "completed") == "completed" and run.get("meta_workflow_node_id")
    }
    completed_roles = {
        str(run.get("role"))
        for run in completed_runs
        if run.get("status", "completed") == "completed" and run.get("role")
    }
    pending: list[dict[str, Any]] = []
    for node in nodes:
        node_id = _workflow_node_id(node)
        role = _workflow_node_role(node)
        if node_id and node_id in completed_node_ids:
            continue
        if not node_id and role and role in completed_roles:
            continue
        pending.append(
            {
                "node_id": node_id,
                "generic_agent_type": role,
                "assigned_task": node.get("assigned_task"),
            }
        )
    return pending


def _workflow_from_finish_decision_or_completed_runs(
    decision: DispatchDecision,
    completed_runs: list[dict[str, Any]],
) -> Any | None:
    direct = _agent_workflow_dag(decision.metadata)
    if direct is not None:
        return direct
    for run in reversed(completed_runs):
        workflow = run.get("agent_level_workflow_dag")
        if workflow is not None:
            return workflow
        dispatch_metadata = run.get("dispatch_metadata")
        if isinstance(dispatch_metadata, dict):
            workflow = _agent_workflow_dag(dispatch_metadata)
            if workflow is not None:
                return workflow
    return None


def _workflow_nodes(workflow: Any) -> list[dict[str, Any]]:
    if isinstance(workflow, list):
        return [node for node in workflow if isinstance(node, dict)]
    if isinstance(workflow, dict):
        nodes = workflow.get("nodes")
        if isinstance(nodes, list):
            return [node for node in nodes if isinstance(node, dict)]
    return []


def _workflow_node_id(node: dict[str, Any]) -> str | None:
    for key in ("meta_workflow_node_id", "workflow_node_id", "node_id", "id"):
        value = node.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _workflow_node_role(node: dict[str, Any]) -> str | None:
    for key in ("generic_agent_type", "target_role", "role", "agent"):
        value = node.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _meta_dispatch_parse_retries(policy_metadata: dict[str, Any]) -> int:
    value = policy_metadata.get("max_meta_dispatch_parse_retries", 2)
    if isinstance(value, int) and value >= 0:
        return value
    return 2


def _meta_dispatch_expected_schema(allowed_roles: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "v1",
        "one_of": [
            {
                "action": "run_subagent",
                "target_role": allowed_roles,
                "instruction": "non-empty string",
                "retrieval_query": "optional string",
                "metadata": "optional object",
            },
            {
                "action": "finish_task",
                "metadata": {"final_answer": "optional string"},
            },
            {"action": "abort", "instruction": "reason string"},
        ],
        "json_only": True,
    }


def _meta_dispatch_repair_message(
    *,
    raw_output: str,
    error: Exception | None,
    expected_schema: dict[str, Any],
    retry_count: int,
) -> Message:
    return Message(
        role="user",
        content=json.dumps(
            {
                "repair_request": "Previous MetaAgent dispatch output was invalid. Return one valid JSON object only.",
                "retry_count": retry_count,
                "raw_invalid_output": raw_output,
                "parse_or_validation_error": str(error) if error is not None else None,
                "repair_guidance": _meta_dispatch_repair_guidance(error),
                "expected_schema": expected_schema,
                "required_response": "Return exactly one valid JSON object. No markdown. No prose.",
            },
            indent=2,
            sort_keys=True,
        ),
    )


def _meta_dispatch_repair_guidance(error: Exception | None) -> dict[str, Any]:
    if error is None:
        return {}
    message = str(error)
    marker = "work-item routing requires reviewer dispatch before continuing:"
    if marker in message:
        suffix = message.split(marker, 1)[1]
        work_item_ids = [item.strip() for item in suffix.split(",") if item.strip()]
        return {
            "required_next_action": "route_reviewer_for_pending_work_item",
            "work_item_ids": work_item_ids,
            "instruction": (
                "Route one configured reviewer role for the listed work_item_id before routing any executor, "
                "finalizer, or END."
            ),
        }
    marker = "finish_task rejected because work-item routing requires reviewer dispatch before finish:"
    if marker in message:
        suffix = message.split(marker, 1)[1]
        work_item_ids = [item.strip() for item in suffix.split(",") if item.strip()]
        return {
            "required_next_action": "route_reviewer_for_pending_work_item",
            "work_item_ids": work_item_ids,
            "instruction": "Route one configured reviewer role for the listed work_item_id before END.",
        }
    if "work-item routing requires finalizer dispatch before continuing:" in message:
        return {
            "required_next_action": "route_finalizer_for_resolved_work_items",
            "instruction": (
                "Route one configured finalizer role, such as WriteAgent, to serialize final records and reports "
                "from validated or candidate handoff artifacts before routing any executor, reviewer, or END."
            ),
        }
    if "finish_task rejected because work-item routing requires finalizer dispatch before finish:" in message:
        return {
            "required_next_action": "route_finalizer_for_resolved_work_items",
            "instruction": (
                "Route one configured finalizer role, such as WriteAgent, before END because final records are not written."
            ),
        }
    if "work-item routing requires END after final records are written" in message:
        return {
            "required_next_action": "route_end_after_final_records",
            "instruction": "Route END because final records have already been written.",
        }
    marker = "work-item routing reviewer rejected because no pending completed executor work item matches"
    if marker in message:
        work_item_id_match = re.search(r"work_item_id='([^']+)'", message)
        work_item_id = work_item_id_match.group(1) if work_item_id_match else None
        return {
            "required_next_action": "route_executor_for_uncompleted_work_item",
            "work_item_id": work_item_id,
            "instruction": (
                "Route one configured executor role for this work_item_id. Do not route a reviewer until that "
                "executor run completes successfully."
            ),
        }
    return {}


def _meta_dispatch_failure_message(
    *,
    raw_output: str,
    error: Exception,
    expected_schema: dict[str, Any],
    retry_count: int,
    task_id: str,
    step_index: int,
) -> str:
    return (
        f"MetaAgent dispatch parsing failed after {retry_count} retries: "
        + json.dumps(
            {
                "task_id": task_id,
                "step_index": step_index,
                "retry_count": retry_count,
                "raw_output": raw_output,
                "parse_or_validation_error": str(error),
                "expected_schema": expected_schema,
                "task_queue_state": "claimed",
            },
            sort_keys=True,
        )
    )


def _dispatch_from_selected_subagent_dag(
    payload: dict[str, Any],
    *,
    completed_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = [node for node in payload.get("selected_subagents", []) if isinstance(node, dict)]
    if not selected:
        return payload
    node = _next_selected_subagent_node(selected, completed_runs)
    if node is None:
        return {
            "schema_version": "v1",
            "action": "finish_task",
            "metadata": {
                "agent_level_workflow_dag": _json_compatible(_selected_subagent_dag(selected)),
                "selected_subagents": _json_compatible(selected),
                "dispatch_rationale": _json_compatible(payload.get("dispatch_rationale")),
                "stop_condition": _json_compatible(payload.get("stop_condition")),
                "source_decision_type": payload.get("decision_type"),
                "final_answer": "Selected generic workflow DAG completed.",
            },
        }
    node_id = str(node.get("node_id") or "selected-node-1")
    role = node.get("generic_agent_type")
    assigned_task = node.get("assigned_task")
    metadata = {
        "meta_workflow_node_id": node_id,
        "generic_agent_type": role,
        "assigned_task": assigned_task,
        "input_dependencies": _json_compatible(node.get("input_dependencies", [])),
        "expected_outputs": _json_compatible(node.get("expected_outputs", [])),
        "completion_criteria": _json_compatible(node.get("completion_criteria")),
        "recovery_policy": _json_compatible(node.get("recovery_policy")),
        "stage_index": _json_compatible(node.get("stage_index")),
        "workflow_dag": _json_compatible(node.get("workflow_dag", [])),
        "agent_level_workflow_dag": _json_compatible(_selected_subagent_dag(selected)),
        "selected_subagents": _json_compatible(selected),
        "dispatch_rationale": _json_compatible(payload.get("dispatch_rationale")),
        "stop_condition": _json_compatible(payload.get("stop_condition")),
        "source_decision_type": payload.get("decision_type"),
    }
    return {
        "schema_version": "v1",
        "action": "run_subagent",
        "target_role": role,
        "instruction": assigned_task,
        "retrieval_query": assigned_task,
        "metadata": metadata,
    }


def _dispatch_from_dispatch_list(
    payload: dict[str, Any],
    *,
    completed_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    dispatch_nodes = [node for node in payload.get("dispatch", []) if isinstance(node, dict)]
    if not dispatch_nodes:
        return payload
    normalized = []
    for node in dispatch_nodes:
        normalized.append(
            {
                "node_id": node.get("node_id") or node.get("meta_workflow_node_id"),
                "generic_agent_type": node.get("generic_agent_type"),
                "assigned_task": node.get("assigned_task"),
                "input_dependencies": node.get("input_dependencies", []),
                "expected_outputs": node.get("expected_outputs", []),
                "completion_criteria": node.get("completion_criteria"),
                "recovery_policy": node.get("recovery_policy"),
                "stage_index": node.get("stage_index"),
                "workflow_dag": node.get("workflow_dag"),
            }
        )
    converted = {
        "decision_type": payload.get("decision_type") or "run_subagent",
        "selected_subagents": normalized,
        "dispatch_rationale": payload.get("dispatch_rationale") or payload.get("rationale"),
        "stop_condition": payload.get("stop_condition") or payload.get("completion_policy"),
    }
    return _dispatch_from_selected_subagent_dag(converted, completed_runs=completed_runs)


def _dispatch_from_single_node_alias(payload: dict[str, Any]) -> dict[str, Any]:
    metadata_payload = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    merged = {**metadata_payload, **{key: value for key, value in payload.items() if key != "metadata"}}
    role = merged.get("target_role") or merged.get("generic_agent_type") or merged.get("selected_role")
    assigned_task = merged.get("instruction") or merged.get("assigned_task")
    node_id = merged.get("meta_workflow_node_id") or merged.get("node_id")
    return {
        "schema_version": "v1",
        "action": "run_subagent",
        "target_role": role,
        "instruction": assigned_task,
        "retrieval_query": merged.get("retrieval_query") or assigned_task,
        "metadata": {
            "meta_workflow_node_id": node_id,
            "generic_agent_type": role,
            "assigned_task": assigned_task,
            "input_dependencies": _json_compatible(merged.get("input_dependencies", [])),
            "expected_outputs": _json_compatible(merged.get("expected_outputs", [])),
            "completion_criteria": _json_compatible(merged.get("completion_criteria")),
            "recovery_policy": _json_compatible(merged.get("recovery_policy")),
            "stage_index": _json_compatible(merged.get("stage_index")),
            "workflow_dag": _json_compatible(merged.get("workflow_dag")),
            "agent_level_workflow_dag": _json_compatible(merged.get("workflow_dag")),
            "source_decision_type": merged.get("decision"),
        },
    }


def _dispatch_from_decision_wrapper(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("action") != "run_subagent":
        return payload
    role = payload.get("target_role") or payload.get("generic_agent_type")
    assigned_task = payload.get("instruction") or payload.get("assigned_task")
    node_id = payload.get("meta_workflow_node_id") or payload.get("node_id")
    return {
        "schema_version": "v1",
        "action": "run_subagent",
        "target_role": role,
        "instruction": assigned_task,
        "retrieval_query": payload.get("retrieval_query") or assigned_task,
        "metadata": {
            "meta_workflow_node_id": node_id,
            "generic_agent_type": role,
            "assigned_task": assigned_task,
            "input_dependencies": _json_compatible(payload.get("input_dependencies", [])),
            "expected_outputs": _json_compatible(payload.get("expected_outputs", [])),
            "completion_criteria": _json_compatible(payload.get("completion_criteria")),
            "recovery_policy": _json_compatible(payload.get("recovery_policy")),
            "stage_index": _json_compatible(payload.get("stage_index")),
            "workflow_dag": _json_compatible(payload.get("workflow_dag")),
            "agent_level_workflow_dag": _json_compatible(payload.get("workflow_dag")),
            "source_decision_type": "dispatch_decision",
        },
    }


def _dispatch_from_run_subagent_wrapper(payload: dict[str, Any]) -> dict[str, Any]:
    node = dict(payload.get("run_subagent") or {})
    node.setdefault("generic_agent_type", payload.get("selected_role"))
    node.setdefault("workflow_dag", payload.get("workflow_dag"))
    converted = _dispatch_from_single_node_alias(
        {
            "decision": "run_subagent",
            **node,
        }
    )
    converted["metadata"]["agent_level_workflow_dag"] = _json_compatible(payload.get("workflow_dag"))
    return converted


def _dispatch_from_finish_alias(payload: dict[str, Any]) -> dict[str, Any]:
    reason = payload.get("reason") or payload.get("final_answer") or "Selected workflow completed."
    return {
        "schema_version": "v1",
        "action": "finish_task",
        "metadata": {
            "final_answer": _json_compatible(reason),
            "agent_level_workflow_dag": _json_compatible(payload.get("selected_dag") or payload.get("workflow_dag")),
            "source_decision_type": "finish_alias",
            "dispatch": _json_compatible(payload.get("dispatch")),
        },
    }


def _dispatch_from_route_decision(payload: dict[str, Any]) -> dict[str, Any]:
    route = payload.get("route")
    instruction = payload.get("instruction")
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    if route == "END":
        return {
            "schema_version": "v1",
            "action": "finish_task",
            "instruction": instruction,
            "metadata": {
                **metadata,
                "final_answer": metadata.get("final_answer") or instruction or metadata.get("summary"),
                "source_decision_type": "route",
                "route": route,
            },
        }
    return {
        "schema_version": "v1",
        "action": "run_subagent",
        "target_role": route,
        "instruction": instruction,
        "retrieval_query": payload.get("retrieval_query") or instruction,
        "metadata": {
            **metadata,
            "meta_workflow_node_id": metadata.get("meta_workflow_node_id") or metadata.get("node_id"),
            "generic_agent_type": route,
            "assigned_task": instruction,
            "source_decision_type": "route",
            "route": route,
        },
    }


def _selected_subagent_dag(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dag = []
    for node in selected:
        dag.append(
            {
                "node_id": node.get("node_id"),
                "generic_agent_type": node.get("generic_agent_type"),
                "assigned_task": node.get("assigned_task"),
                "input_dependencies": node.get("input_dependencies", []),
                "expected_outputs": node.get("expected_outputs", []),
                "completion_criteria": node.get("completion_criteria"),
                "recovery_policy": node.get("recovery_policy"),
                "stage_index": node.get("stage_index"),
            }
        )
    return dag


def _next_selected_subagent_node(
    selected: list[dict[str, Any]],
    completed_runs: list[dict[str, Any]],
) -> dict[str, Any] | None:
    completed_node_ids = {
        str(run.get("meta_workflow_node_id"))
        for run in completed_runs
        if run.get("status", "completed") == "completed" and run.get("meta_workflow_node_id")
    }
    completed_roles = {
        str(run.get("role"))
        for run in completed_runs
        if run.get("status", "completed") == "completed" and run.get("role")
    }
    for node in selected:
        node_id = str(node.get("node_id") or "")
        role = str(node.get("generic_agent_type") or "")
        if node_id in completed_node_ids or (not node_id and role in completed_roles):
            continue
        dependencies = node.get("input_dependencies", [])
        if not isinstance(dependencies, list):
            dependencies = []
        if all(str(dep) in completed_node_ids for dep in dependencies):
            return node
    return None


def _final_answer_from_dispatch(
    decision: DispatchDecision,
    final_result: dict[str, Any],
) -> str:
    final_answer = decision.metadata.get("final_answer")
    if isinstance(final_answer, str) and final_answer:
        return final_answer
    if decision.instruction:
        return decision.instruction
    return str(final_result["final_answer"])


def _failed_subagent_result_message(prefix: str, result: dict[str, Any]) -> str:
    status = result.get("status") or "failed"
    run_ref = result.get("run_ref")
    reason = result.get("failure_reason") or "subagent did not complete successfully"
    if run_ref:
        return f"{prefix} non-completed subagent run {run_ref}: {status}; {reason}"
    return f"{prefix} non-completed subagent run: {status}; {reason}"


def _subagent_completion_contract(
    *,
    status: str,
    failure_reason: str | None,
    artifact_refs: list[ArtifactRef],
    node_records: list[NodeExecutionRecord],
    role: str,
    assigned_task: str = "",
    expected_outputs: list[dict[str, Any]] | None = None,
    tool_trace_records: list[ToolCallRecord] | None = None,
    final_answer: str = "",
    policy_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blocking_issues: list[dict[str, Any]] = []
    if status not in {"completed", "partial"}:
        blocking_issues.append(
            {
                "type": "subagent_status",
                "severity": "blocking",
                "message": failure_reason or f"subagent ended with status {status}",
            }
        )
    failed_nodes = [record for record in node_records if record.status == "failed"]
    for record in failed_nodes:
        blocking_issues.append(
            {
                "type": "internal_dag_node_failed",
                "severity": "blocking",
                "message": record.output_summary or "internal DAG node failed",
                "node_id": record.node_id,
                "skill_id": record.skill_id,
            }
        )
    tool_trace_records = tool_trace_records or []
    strict_output_contract = bool(expected_outputs) or bool(artifact_refs)
    allow_exploratory_tool_misses = any(
        isinstance(output.get("metadata"), dict) and output["metadata"].get("dynamic_output_artifact") is True
        for output in (expected_outputs or [])
    )
    error_tool_records = (
        _unresolved_tool_error_records(
            tool_trace_records,
            artifact_refs,
            allow_exploratory_tool_misses=allow_exploratory_tool_misses,
        )
        if strict_output_contract
        else []
    )
    for record in error_tool_records:
        blocking_issues.append(
            {
                "type": "tool_call_error",
                "severity": "blocking",
                "message": f"{record.tool_call.name} returned {record.result.status}: {record.result.content}",
                "tool_name": record.tool_call.name,
                "call_id": record.tool_call.call_id,
            }
        )
    guard_metadata = _completion_guard_metadata(policy_metadata or {}, role)
    for missing_tool in _missing_required_tool_calls_before_final(guard_metadata, tool_trace_records):
        blocking_issues.append(
            {
                "type": "missing_required_tool_call",
                "severity": "blocking",
                "message": (
                    "missing successful required tool call before completion: "
                    f"{missing_tool}. Continue by calling {missing_tool} successfully."
                ),
                "tool_name": missing_tool,
            }
        )
    skipped_nodes = [record for record in node_records if record.status == "skipped"]
    warnings = [
        {
            "type": "internal_dag_node_skipped",
            "message": record.output_summary or "internal DAG node skipped",
            "node_id": record.node_id,
            "skill_id": record.skill_id,
        }
        for record in skipped_nodes
    ]
    expected_outputs = expected_outputs or []
    missing_outputs = _missing_expected_outputs(expected_outputs, artifact_refs, final_answer)
    for missing in missing_outputs:
        blocking_issues.append(
            {
                "type": "missing_expected_output",
                "severity": "blocking",
                "message": f"missing expected output: {missing['name']}",
                "expected_output": missing,
            }
        )
    assigned_task_complete = status == "completed" and not failed_nodes and not blocking_issues
    if expected_outputs:
        produced_required_outputs = not missing_outputs
    else:
        produced_required_outputs = bool(artifact_refs) or _assigned_task_can_complete_without_artifact(assigned_task, final_answer)
    ready_for_task_end = assigned_task_complete and produced_required_outputs and not blocking_issues
    return {
        "schema_version": "v1",
        "assigned_task_complete": assigned_task_complete,
        "produced_required_outputs": produced_required_outputs,
        "ready_for_task_end": ready_for_task_end,
        "blocking_issues": blocking_issues,
        "recommended_next_route": "END" if ready_for_task_end else role,
        "warnings": warnings,
        "evidence": {
            "status": status,
            "artifact_count": len(artifact_refs),
            "expected_outputs": _json_compatible(expected_outputs),
            "failed_node_count": len(failed_nodes),
            "skipped_node_count": len(skipped_nodes),
            "tool_error_count": len(error_tool_records),
            "artifact_kinds": [_artifact_semantic_kind(ref) for ref in artifact_refs],
        },
    }


def _unresolved_tool_error_records(
    tool_trace_records: list[ToolCallRecord],
    artifact_refs: list[ArtifactRef],
    *,
    allow_exploratory_tool_misses: bool = False,
) -> list[ToolCallRecord]:
    successful_tool_targets = {
        _tool_call_resolution_key(record)
        for record in tool_trace_records
        if record.result.status == "ok"
    }
    has_artifact = bool(artifact_refs)
    unresolved: list[ToolCallRecord] = []
    for record in tool_trace_records:
        if record.result.status == "ok":
            continue
        if _tool_call_resolution_key(record) in successful_tool_targets:
            continue
        if has_artifact and _is_noncritical_tool_error(
            record,
            allow_exploratory_tool_misses=allow_exploratory_tool_misses,
        ):
            continue
        unresolved.append(record)
    return unresolved


def _tool_call_resolution_key(record: ToolCallRecord) -> tuple[str, str]:
    return record.tool_call.name, _tool_call_target_fingerprint(record.tool_call.arguments)


def _tool_call_target_fingerprint(arguments: dict[str, Any]) -> str:
    target: dict[str, Any] = {}
    for key in (
        "path",
        "artifact_name",
        "filename",
        "uri",
        "url",
        "query",
        "pattern",
        "sheet",
        "table",
        "source",
    ):
        value = arguments.get(key)
        if value not in (None, ""):
            target[key] = value
    if not target:
        target = arguments
    return json.dumps(_json_compatible(target), sort_keys=True, separators=(",", ":"))


def _is_noncritical_tool_error(
    record: ToolCallRecord,
    *,
    allow_exploratory_tool_misses: bool = False,
) -> bool:
    error_type = record.result.metadata.get("error_type")
    if error_type in {
        "unprepared_tool",
        "repeated_tool_call_suppressed",
        "completion_guard_budget_reserved",
        "completion_guard_required_outputs_due",
    }:
        return True
    if not allow_exploratory_tool_misses:
        return False
    if record.tool_call.name in {
        "list_files",
        "read_text",
        "search_text",
        "inspect_file_metadata",
        "extract_sections",
        "inspect_excel_workbook",
        "read_excel_sheet",
        "inspect_table",
        "read_table_slice",
        "detect_table_header",
        "normalize_table",
        "profile_table",
    }:
        return True
    if record.tool_call.name == "json_schema_validate":
        content = record.result.content.casefold()
        return "no such file or directory" in content or "schema or schema_path" in content
    return False


def _is_completion_guard_violation(record: ToolCallRecord) -> bool:
    return record.result.metadata.get("error_type") in {
        "completion_guard_budget_reserved",
        "completion_guard_required_outputs_due",
    }


def _is_repeated_tool_call_suppression(record: ToolCallRecord) -> bool:
    return record.result.metadata.get("error_type") == "repeated_tool_call_suppressed"


def _completion_guard_violation_limit(policy_metadata: dict[str, Any], role_name: str | None) -> int:
    guard_metadata = _completion_guard_metadata(policy_metadata, role_name)
    for source in (guard_metadata, policy_metadata):
        for key in ("completion_guard_max_violations", "max_completion_guard_violations"):
            value = _optional_positive_int(source.get(key))
            if value is not None:
                return value
    return 3


def _repeated_suppression_violation_limit(policy_metadata: dict[str, Any]) -> int:
    for key in ("repeated_tool_call_suppression_max_violations", "max_repeated_tool_call_suppression_violations"):
        value = _optional_positive_int(policy_metadata.get(key))
        if value is not None:
            return value
    return 5


def _completion_guard_violation_failure_reason(
    record: ToolCallRecord,
    *,
    violation_count: int,
    violation_limit: int,
) -> str:
    error_type = record.result.metadata.get("error_type") or "completion_guard_violation"
    return (
        f"{error_type}: stopped after {violation_count} repeated completion guard violation(s) "
        f"(limit {violation_limit}). Last violation: {record.result.content}"
    )


def _repeated_suppression_failure_reason(
    record: ToolCallRecord,
    *,
    violation_count: int,
    violation_limit: int,
) -> str:
    return (
        "repeated_tool_call_suppressed: stopped after "
        f"{violation_count} suppressed repeated tool call(s) (limit {violation_limit}). "
        f"Last suppression: {record.result.content}"
    )


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


def _tool_result_message(
    tool_name: str,
    result: ToolResult,
    *,
    policy_metadata: dict[str, Any] | None = None,
) -> Message:
    return Message(
        role="tool",
        content=_tool_result_content(result, policy_metadata=policy_metadata),
        name=tool_name,
        tool_call_id=result.call_id,
        metadata={"tool_result": result.model_dump(mode="json")},
    )


def _tool_result_content(result: ToolResult, *, policy_metadata: dict[str, Any] | None = None) -> str:
    if result.status != "ok" or (not result.metadata and not result.artifact_refs):
        return _truncate_tool_result_prompt(result.content, policy_metadata)
    payload: dict[str, Any] = {
        "status": result.status,
        "content": result.content,
    }
    if result.metadata:
        payload["metadata"] = _json_compatible(result.metadata)
    if result.artifact_refs:
        payload["artifact_refs"] = [artifact.model_dump(mode="json") for artifact in result.artifact_refs]
    content = f"{result.content}\n\nTool result payload:\n{json.dumps(payload, indent=2, sort_keys=True)}"
    return _truncate_tool_result_prompt(content, policy_metadata)


def _truncate_tool_result_prompt(content: str, policy_metadata: dict[str, Any] | None) -> str:
    max_chars = _tool_result_prompt_max_chars(policy_metadata)
    if max_chars is None:
        return content
    return _truncate_text(content, max_chars)


def _tool_result_prompt_max_chars(policy_metadata: dict[str, Any] | None) -> int | None:
    if not isinstance(policy_metadata, dict):
        return None
    raw_value = policy_metadata.get("tool_result_prompt_max_chars")
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        return None
    if isinstance(raw_value, int):
        return raw_value if raw_value > 0 else None
    if isinstance(raw_value, str) and raw_value.isdigit():
        parsed = int(raw_value)
        return parsed if parsed > 0 else None
    return None


def _completion_guard_budget_reserved_result(
    tool_call: ToolCall,
    tool_trace_records: list[ToolCallRecord],
    *,
    policy_metadata: dict[str, Any],
    role_name: str | None,
    remaining_tool_budget: int | None,
    available_tool_names: list[str] | None = None,
) -> ToolResult | None:
    guard_metadata = _completion_guard_metadata(policy_metadata, role_name)
    missing_tools = _missing_required_tool_calls_before_final(guard_metadata, tool_trace_records)
    minimum_jsonl_records = _optional_positive_int(guard_metadata.get("minimum_jsonl_records_before_final"))
    current_jsonl_records = _max_successful_jsonl_record_count(tool_trace_records)
    jsonl_records_missing = (
        minimum_jsonl_records is not None and current_jsonl_records < minimum_jsonl_records
    )
    due_tools = list(missing_tools)
    if jsonl_records_missing and "write_jsonl" not in due_tools:
        due_tools.append("write_jsonl")
    if not due_tools:
        return None
    if available_tool_names is not None:
        available = {name for name in available_tool_names if isinstance(name, str)}
        if due_tools[0] not in available:
            return None
        due_tools = [name for name in due_tools if name in available]
        if not due_tools:
            return None
    if tool_call.name in due_tools:
        return None

    max_non_required_tools = _optional_positive_int(
        guard_metadata.get("max_non_required_tool_calls_before_required_outputs")
    )
    if max_non_required_tools is not None:
        required_tools = _required_tool_calls_before_final(guard_metadata)
        non_required_count = _non_required_tool_call_count_before_required_outputs(
            tool_trace_records,
            required_tools=required_tools,
        )
        if non_required_count >= max_non_required_tools:
            missing = ", ".join(due_tools)
            earliest = due_tools[0]
            record_note = ""
            if jsonl_records_missing:
                record_note = (
                    f" Current JSONL record_count is {current_jsonl_records}; "
                    f"minimum required is {minimum_jsonl_records}."
                )
            return ToolResult(
                call_id=tool_call.call_id,
                status="error",
                content=(
                    "completion guard required outputs due: "
                    f"{non_required_count} non-required tool call(s) have already run while required tools are "
                    f"still due: {missing}.{record_note} Do not call {tool_call.name}; call {earliest} next."
                ),
                metadata={
                    "error_type": "completion_guard_required_outputs_due",
                    "missing_required_tools": due_tools,
                    "non_required_tool_call_count": non_required_count,
                    "max_non_required_tool_calls_before_required_outputs": max_non_required_tools,
                    "minimum_jsonl_records_before_final": minimum_jsonl_records,
                    "current_jsonl_record_count": current_jsonl_records,
                    "rejected_tool_name": tool_call.name,
                },
            )

    if remaining_tool_budget is None or remaining_tool_budget <= 0:
        return None
    if remaining_tool_budget > len(due_tools):
        return None
    missing = ", ".join(due_tools)
    earliest = due_tools[0]
    return ToolResult(
        call_id=tool_call.call_id,
        status="error",
        content=(
            "completion guard budget reserved: "
            f"only {remaining_tool_budget} executable tool call(s) remain, and required tools are still missing: "
            f"{missing}. Do not spend the remaining budget on {tool_call.name}; call {earliest} next."
        ),
        metadata={
            "error_type": "completion_guard_budget_reserved",
            "missing_required_tools": due_tools,
            "remaining_tool_budget": remaining_tool_budget,
            "minimum_jsonl_records_before_final": minimum_jsonl_records,
            "current_jsonl_record_count": current_jsonl_records,
            "rejected_tool_name": tool_call.name,
        },
    )


def _completion_guard_due_tools(
    guard_metadata: dict[str, Any],
    tool_trace_records: list[ToolCallRecord],
) -> list[str]:
    due_tools = list(_missing_required_tool_calls_before_final(guard_metadata, tool_trace_records))
    minimum_jsonl_records = _optional_positive_int(guard_metadata.get("minimum_jsonl_records_before_final"))
    current_jsonl_records = _max_successful_jsonl_record_count(tool_trace_records)
    if (
        minimum_jsonl_records is not None
        and current_jsonl_records < minimum_jsonl_records
        and "write_jsonl" not in due_tools
    ):
        due_tools.append("write_jsonl")
    return due_tools


def _repeated_tool_call_suppression_result(
    tool_call: ToolCall,
    tool_trace_records: list[ToolCallRecord],
    *,
    policy_metadata: dict[str, Any],
) -> ToolResult | None:
    repeat_limit = _max_repeated_tool_calls_per_run(policy_metadata)
    if repeat_limit is None:
        return None
    repeat_key = _tool_call_repeat_key(tool_call)
    prior_count = sum(1 for record in tool_trace_records if _tool_call_repeat_key(record.tool_call) == repeat_key)
    if prior_count < repeat_limit:
        return None
    return ToolResult(
        call_id=tool_call.call_id,
        status="error",
        content=(
            "repeated tool call suppressed: "
            f"{tool_call.name} has already been called {prior_count} time(s) for the same target in this run. "
            "Use the previous tool result, write the required artifact, or choose a materially different tool call."
        ),
        metadata={
            "error_type": "repeated_tool_call_suppressed",
            "repeat_key": list(repeat_key),
            "prior_count": prior_count,
            "max_repeated_tool_calls_per_run": repeat_limit,
        },
    )


def _max_repeated_tool_calls_per_run(policy_metadata: dict[str, Any]) -> int | None:
    raw_value = policy_metadata.get("max_repeated_tool_calls_per_run")
    if isinstance(raw_value, bool):
        return None
    if isinstance(raw_value, int) and raw_value > 0:
        return raw_value
    if isinstance(raw_value, str) and raw_value.isdigit():
        parsed = int(raw_value)
        return parsed if parsed > 0 else None
    return None


def _tool_call_repeat_key(tool_call: ToolCall) -> tuple[str, str, str]:
    arguments = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
    for target_key in ("path", "root", "artifact_name", "schema_path", "instance_path"):
        target_value = arguments.get(target_key)
        if isinstance(target_value, str) and target_value:
            return (tool_call.name, target_key, target_value)
    return (tool_call.name, "arguments", json.dumps(_json_compatible(arguments), sort_keys=True))


def _missing_required_tool_calls_before_final(
    policy_metadata: dict[str, Any],
    tool_trace_records: list[ToolCallRecord],
) -> list[str]:
    required = _required_tool_calls_before_final(policy_metadata)
    if not required:
        return []
    succeeded = {record.tool_call.name for record in tool_trace_records if record.result.status == "ok"}
    return [name for name in required if name not in succeeded]


def _required_tool_calls_before_final(policy_metadata: dict[str, Any]) -> list[str]:
    raw_required = policy_metadata.get("required_tool_calls_before_final", [])
    if not isinstance(raw_required, list):
        return []
    return [name for name in raw_required if isinstance(name, str) and name]


def _non_required_tool_call_count_before_required_outputs(
    tool_trace_records: list[ToolCallRecord],
    *,
    required_tools: list[str],
) -> int:
    ignored_error_types = {
        "completion_guard_budget_reserved",
        "completion_guard_required_outputs_due",
        "repeated_tool_call_suppressed",
    }
    required = set(required_tools)
    count = 0
    for record in tool_trace_records:
        if record.tool_call.name in required:
            continue
        error_type = record.result.metadata.get("error_type")
        if record.result.status == "error" and error_type in ignored_error_types:
            continue
        count += 1
    return count


def _final_answer_rejection_reason(
    policy_metadata: dict[str, Any],
    tool_trace_records: list[ToolCallRecord],
    role_name: str | None = None,
    *,
    expected_outputs: list[dict[str, Any]] | None = None,
    artifact_refs: list[ArtifactRef] | None = None,
    final_answer: str = "",
) -> str | None:
    guard_metadata = _completion_guard_metadata(policy_metadata, role_name)
    missing_tools = _missing_required_tool_calls_before_final(guard_metadata, tool_trace_records)
    if missing_tools:
        missing = ", ".join(missing_tools)
        return (
            "Missing successful required tools before final_answer: "
            f"{missing}. Continue by calling the earliest missing tool."
        )

    minimum_jsonl_records = _optional_positive_int(guard_metadata.get("minimum_jsonl_records_before_final"))
    if minimum_jsonl_records is not None:
        current_count = _max_successful_jsonl_record_count(tool_trace_records)
        if current_count < minimum_jsonl_records:
            return (
                "JSONL record_count is below the required minimum before final_answer: "
                f"current {current_count}, required at least {minimum_jsonl_records}. "
                "Continue by constructing all missing records and calling write_jsonl again."
            )
    missing_outputs = _missing_expected_outputs(expected_outputs or [], artifact_refs or [], final_answer)
    if missing_outputs:
        names = ", ".join(
            str(output.get("name") or output.get("description") or "unnamed output")
            for output in missing_outputs[:3]
        )
        return f"Missing expected outputs before final_answer: {names}. Continue by writing the required artifacts."
    return None


def _completion_policy_metadata_for_dispatch(
    policy_metadata: dict[str, Any],
    dispatch_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    if not _dispatch_is_dynamic(dispatch_metadata):
        return policy_metadata
    if dispatch_metadata.get("enable_static_completion_guards") is True:
        return policy_metadata
    metadata = dict(policy_metadata)
    for key in (
        "completion_guards_by_role",
        "required_tool_calls_before_final",
        "minimum_jsonl_records_before_final",
        "max_non_required_tool_calls_before_required_outputs",
    ):
        metadata.pop(key, None)
    metadata["static_completion_guards_disabled_for_dynamic_subagent"] = True
    return metadata


def _dispatch_is_dynamic(dispatch_metadata: dict[str, Any] | None) -> bool:
    if not isinstance(dispatch_metadata, dict):
        return False
    mode = dispatch_metadata.get("execution_mode") or dispatch_metadata.get("mode")
    return isinstance(mode, str) and mode.casefold() == "dynamic"


def _prepare_dynamic_subagent_tool_context(
    prepared_skills: Any,
    *,
    role: Any,
    tool_runtime: ToolRuntime | None,
    policy: Any,
    dispatch_metadata: dict[str, Any] | None,
) -> Any:
    if not isinstance(dispatch_metadata, dict):
        return prepared_skills
    mode = dispatch_metadata.get("execution_mode") or dispatch_metadata.get("mode")
    if not (isinstance(mode, str) and mode.casefold() == "dynamic"):
        return prepared_skills
    dynamic_tool_names = _dedupe([name for name in getattr(role, "allowed_tools", []) if isinstance(name, str) and name])
    if not dynamic_tool_names or tool_runtime is None:
        return prepared_skills
    tool_bundle = tool_runtime.prepare(
        required_tools=dynamic_tool_names,
        allowed_tools=dynamic_tool_names,
        policy=policy,
    )
    prepared_tool_names = [spec.name for spec in tool_bundle.tool_specs]
    skill_bundle = prepared_skills.skill_bundle.model_copy(
        update={
            "required_tools": _dedupe([*prepared_skills.skill_bundle.required_tools, *prepared_tool_names]),
            "metadata": {
                **prepared_skills.skill_bundle.metadata,
                "dynamic_allowed_tools_prepared": prepared_tool_names,
            },
        }
    )
    skill_context = {
        **prepared_skills.skill_context,
        "required_tools": skill_bundle.required_tools,
        "dynamic_allowed_tools_prepared": prepared_tool_names,
    }
    return prepared_skills.__class__(
        skill_bundle=skill_bundle,
        tool_bundle=tool_bundle,
        skill_context=skill_context,
    )


def _completion_guard_metadata(policy_metadata: dict[str, Any], role_name: str | None) -> dict[str, Any]:
    by_role = policy_metadata.get("completion_guards_by_role")
    if isinstance(by_role, dict):
        if isinstance(role_name, str):
            role_guards = by_role.get(role_name)
            if isinstance(role_guards, dict):
                return role_guards
        return {}
    return policy_metadata


def _max_successful_jsonl_record_count(tool_trace_records: list[ToolCallRecord]) -> int:
    counts = [
        record.result.metadata.get("record_count")
        for record in tool_trace_records
        if record.tool_call.name == "write_jsonl" and record.result.status == "ok"
    ]
    return max([count for count in counts if isinstance(count, int)], default=0)


def _optional_positive_int(value: Any) -> int | None:
    if isinstance(value, int) and value > 0:
        return value
    return None


def _optional_positive_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    return None


def _final_answer_rejection_message(reason: str) -> Message:
    return Message(
        role="user",
        content=(
            "The previous final_answer was rejected because completion guards have not passed. "
            f"{reason} Do not return final_answer until all completion guards pass."
        ),
    )


def _copy_messages(messages: list[Message]) -> list[Message]:
    return [message.model_copy(deep=True) for message in messages]


def _llm_output_messages(response: LLMRuntimeResponse) -> list[Message]:
    action = response.action
    if action.action == "tool_call":
        metadata: dict[str, Any] = {"action": action.action}
        tool_calls = _tool_calls_from_action(action)
        if tool_calls:
            metadata["tool_call"] = tool_calls[0].model_dump(mode="json")
            metadata["tool_calls"] = [tool_call.model_dump(mode="json") for tool_call in tool_calls]
        return [Message(role="assistant", content="", metadata=metadata)]
    return [
        Message(
            role="assistant",
            content=action.content or "",
            metadata={"action": action.action},
        )
    ]


def _tool_calls_from_action(action: SubAgentAction) -> list[ToolCall]:
    if action.tool_calls:
        return list(action.tool_calls)
    if action.tool_call is not None:
        return [action.tool_call]
    return []


def _llm_model_name(llm: Any, generation_config: LLMGenerationConfig) -> str:
    if generation_config.model:
        return generation_config.model
    model = getattr(llm, "model", None)
    return model if isinstance(model, str) else ""


def _meta_agent_memory_scope_id(meta_agent: MetaAgentSpec) -> str:
    return f"agent:{meta_agent.name}"


def _meta_agent_memory_query(
    *,
    request: TaskRequest,
    step_index: int,
    role_results: list[dict[str, Any]],
) -> str:
    payload = {
        "runtime_stage": "meta_agent_dispatch",
        "task_id": request.task_id,
        "goal": _truncate_text(request.goal, 2_000),
        "step_index": step_index,
        "recent_completed_runs": [
            {
                "run_ref": result.get("run_ref"),
                "role": result.get("role"),
                "status": result.get("status"),
                "failure_reason": _truncate_text(str(result.get("failure_reason") or ""), 500),
                "final_answer": _truncate_text(str(result.get("final_answer") or ""), 800),
            }
            for result in role_results[-5:]
        ],
    }
    return json.dumps(payload, sort_keys=True)


def _meta_memory_prompt_payload(memory_bundle: MemoryBundle | None) -> dict[str, Any] | None:
    if memory_bundle is None:
        return None
    return {
        "backend_id": memory_bundle.backend_id,
        "state_ref": memory_bundle.state_ref,
        "metadata": _json_compatible(memory_bundle.metadata),
        "items": _compact_memory_items_for_prompt(memory_bundle.items),
    }


def _meta_agent_memory_update_messages(
    *,
    request: TaskRequest,
    meta_agent: MetaAgentSpec,
    decision: DispatchDecision,
    step_index: int,
    role_results: list[dict[str, Any]],
    llm_call_ref: str | None,
) -> list[Message]:
    payload = {
        "runtime_stage": "meta_agent_dispatch_summary",
        "task_id": request.task_id,
        "meta_agent": meta_agent.name,
        "step_index": step_index,
        "llm_call_ref": llm_call_ref,
        "completed_run_count": len(role_results),
        "recent_completed_runs": [
            {
                "run_ref": result.get("run_ref"),
                "role": result.get("role"),
                "status": result.get("status"),
                "failure_reason": _truncate_text(str(result.get("failure_reason") or ""), 500),
                "final_answer": _truncate_text(str(result.get("final_answer") or ""), 800),
            }
            for result in role_results[-5:]
        ],
        "decision": decision.model_dump(mode="json"),
    }
    return [
        Message(
            role="assistant",
            content=json.dumps(payload, indent=2, sort_keys=True),
            metadata={
                "runtime_stage": "meta_agent_dispatch_summary",
                "memory_summary": True,
            },
        )
    ]


def _messages_for_memory_update(prompt_messages: list[Message], output_message: Message) -> list[Message]:
    return [_memory_compatible_message(message) for message in [*prompt_messages, output_message]]


def _flat_memory_update_messages(
    *,
    role: str,
    role_instruction: str,
    tool_trace_records: list[ToolCallRecord],
    artifact_refs: list[ArtifactRef],
    final_answer: str,
    status: str,
) -> list[Message]:
    payload = {
        "runtime_stage": "subagent_flat_summary",
        "role": role,
        "assigned_task": _truncate_text(role_instruction, 2_000),
        "status": status,
        "tool_call_count": len(tool_trace_records),
        "tool_summaries": [
            {
                "call_id": record.tool_call.call_id,
                "tool_name": record.tool_call.name,
                "status": record.result.status,
                "artifact_count": len(record.result.artifact_refs),
                "content_summary": _truncate_text(record.result.content or "", 500),
            }
            for record in tool_trace_records
        ],
        "artifact_refs": [ref.model_dump(mode="json") for ref in artifact_refs],
        "final_answer_summary": _truncate_text(final_answer, 2_000),
    }
    return [
        Message(
            role="assistant",
            content=json.dumps(payload, indent=2, sort_keys=True),
            metadata={
                "runtime_stage": "subagent_flat_summary",
                "memory_summary": True,
            },
        )
    ]


def _workflow_memory_update_messages(
    *,
    role: str,
    role_instruction: str,
    workflow_plan: WorkflowPlan,
    node_records: list[NodeExecutionRecord],
    artifact_refs: list[ArtifactRef],
    final_answer: str,
    status: str,
) -> list[Message]:
    payload = {
        "runtime_stage": "subagent_workflow_summary",
        "role": role,
        "assigned_task": _truncate_text(role_instruction, 2_000),
        "status": status,
        "workflow_plan_ref": workflow_plan.plan_id,
        "workflow_node_count": len(workflow_plan.nodes),
        "workflow_node_summaries": [
            {
                "node_id": record.node_id,
                "skill_id": record.skill_id,
                "status": record.status,
                "tool_call_count": len(record.tool_calls),
                "artifact_count": len(record.artifact_refs),
                "summary": _truncate_text(record.output_summary or "", 1_000),
            }
            for record in node_records
        ],
        "artifact_refs": [ref.model_dump(mode="json") for ref in artifact_refs],
        "final_answer_summary": _truncate_text(final_answer, 2_000),
    }
    return [
        Message(
            role="assistant",
            content=json.dumps(payload, indent=2, sort_keys=True),
            metadata={
                "runtime_stage": "subagent_workflow_summary",
                "compaction": "workflow_memory_update",
                "source_role": role,
            },
        )
    ]


def _compact_previous_node_summaries(previous_summaries: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            **summary,
            "summary": _truncate_text(summary.get("summary", ""), 2_000),
        }
        for summary in previous_summaries
    ]


def _compact_memory_items_for_prompt(items: list[MemoryItem]) -> list[dict[str, Any]]:
    return [
        {
            **item.model_dump(mode="json"),
            "content": _truncate_text(item.content, 2_000),
        }
        for item in items
    ]


def _lab_context_for_prompt(*, lab_root: Path | None, task_goal: str) -> dict[str, Any]:
    if lab_root is None:
        return {"lab_root": None, "configured_resource_paths": []}
    resource_paths = _resource_paths_from_text(task_goal)
    return {
        "lab_root": str(lab_root),
        "domain_packages_root": str(lab_root / "domain_packages"),
        "configured_resource_paths": [
            {
                "configured_path": path,
                "lab_path": str(lab_root / path),
                "exists": (lab_root / path).exists(),
            }
            for path in resource_paths
        ],
        "lab_domain_package_paths": [
            {
                "configured_path": package_path,
                "lab_path": str(lab_root / package_path),
                "exists": (lab_root / package_path).exists(),
            }
            for package_path in _domain_package_paths(resource_paths)
        ],
    }


def _resource_paths_from_text(text: str) -> list[str]:
    matches = re.findall(r"domain_packages/[^\s`'\"),]+", text)
    seen: set[str] = set()
    paths: list[str] = []
    for match in matches:
        path = match.rstrip(".,;:]")
        if path and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def _domain_package_paths(resource_paths: list[str]) -> list[str]:
    packages: list[str] = []
    seen: set[str] = set()
    for path in resource_paths:
        parts = Path(path).parts
        if len(parts) < 2 or parts[0] != "domain_packages":
            continue
        package = str(Path(parts[0]) / parts[1])
        if package not in seen:
            seen.add(package)
            packages.append(package)
    return packages


def _infer_lab_root(
    *,
    trajectory_registry: FileTrajectoryRegistry | None,
    lab_state_registry: FileLabStateRegistry | None,
    task_registry: FileTaskRegistry | None,
) -> Path | None:
    for root in (
        getattr(lab_state_registry, "root", None),
        getattr(trajectory_registry, "root", None),
        getattr(task_registry, "root", None),
    ):
        if isinstance(root, Path):
            inferred = _lab_root_from_registry_root(root)
            if inferred is not None:
                return inferred
    return None


def _lab_root_from_registry_root(root: Path) -> Path | None:
    parts = root.parts
    if "registries" in parts:
        index = len(parts) - 1 - list(reversed(parts)).index("registries")
        return Path(*parts[:index])
    return None


def _memory_compatible_message(message: Message) -> Message:
    if message.role != "tool":
        return message
    tool_name = message.name or "tool"
    return Message(
        role="assistant",
        content=f"Tool result ({tool_name}): {message.content}",
        metadata={
            "source_role": "tool",
            "source_tool_name": message.name,
            "source_tool_call_id": message.tool_call_id,
            "source_metadata": _json_compatible(message.metadata),
        },
    )


def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    omitted = len(value) - max_chars
    return f"{value[:max_chars]}...[truncated {omitted} chars]"


def _register_tool_result_artifacts(
    result: ToolResult,
    registrar: Callable[[ToolResult], None] | None,
) -> None:
    if registrar is not None and result.artifact_refs:
        registrar(result)


def _manage_tool_result_artifacts(
    *,
    result: ToolResult,
    request: TaskRequest,
    run_ref: str,
    artifact_root_factory: ToolArtifactRootFactory | None,
) -> ToolResult:
    if artifact_root_factory is None or not result.artifact_refs:
        return result
    artifact_root = Path(artifact_root_factory(request, run_ref))
    managed_refs = [
        _manage_artifact_ref(ref, artifact_root, result.call_id, index)
        for index, ref in enumerate(result.artifact_refs)
    ]
    return result.model_copy(update={"artifact_refs": managed_refs})


def _manage_artifact_ref(ref: ArtifactRef, artifact_root: Path, call_id: str, index: int) -> ArtifactRef:
    source_path = _local_path_from_artifact_uri(ref.uri)
    if source_path is None or not source_path.is_file():
        return ref
    destination = artifact_root / _safe_path_part(call_id) / f"{index}-{_safe_filename(source_path.name)}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() != destination.resolve():
        shutil.copy2(source_path, destination)
    metadata = {
        **ref.metadata,
        "filename": ref.metadata.get("filename") or source_path.name,
        "lab_managed": True,
        "source_uri": ref.uri,
    }
    return ref.model_copy(update={"uri": str(destination), "metadata": metadata})


def _local_path_from_artifact_uri(uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme == "":
        return Path(uri)
    if parsed.scheme == "file" and parsed.netloc in ("", "localhost"):
        return Path(unquote(parsed.path))
    return None


def _safe_path_part(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in value)
    return safe or "artifact"


def _safe_filename(value: str) -> str:
    safe = _safe_path_part(value)
    return safe if safe not in (".", "..") else "artifact"


def _parse_skill_update_result(value: Any) -> SkillUpdateResult | None:
    if isinstance(value, SkillUpdateResult):
        return value
    try:
        return SkillUpdateResult.model_validate(value)
    except Exception:
        return None


def _register_memory_state_update(
    *,
    registry: BackendStateRegistry | None,
    task_id: str,
    run_ref: str,
    role: str,
    memory_scope: str,
    memory_scope_id: str,
    memory_bundle: MemoryBundle,
    update_result: Any,
) -> None:
    if registry is None:
        return
    update_payload = _json_compatible(update_result)
    if not isinstance(update_payload, dict):
        return
    if update_payload.get("status") != "updated":
        return
    state_ref = update_payload.get("state_ref")
    if not isinstance(state_ref, str) or not state_ref:
        return
    artifact_refs, invalid_artifact_refs = _artifact_refs_from_update_payload(update_payload)
    metadata = {
        "memory_scope": memory_scope,
        "memory_scope_id": memory_scope_id,
        "role": role,
        "update_result": update_payload,
    }
    if invalid_artifact_refs:
        metadata["invalid_artifact_refs"] = invalid_artifact_refs
    registry.register_candidate(
        BackendStateRecord(
            state_ref=state_ref,
            backend_id=memory_bundle.backend_id,
            backend_type="memory",
            created_from_task_id=task_id,
            created_from_run_ref=run_ref,
            parent_state_refs=_memory_parent_state_refs(update_payload, memory_bundle),
            artifact_refs=artifact_refs,
            active=True,
            metadata=metadata,
        )
    )


def _memory_parent_state_refs(update_payload: dict[str, Any], memory_bundle: MemoryBundle) -> list[str]:
    previous_state_ref = update_payload.get("previous_state_ref")
    if isinstance(previous_state_ref, str) and previous_state_ref:
        return [previous_state_ref]
    if memory_bundle.state_ref:
        return [memory_bundle.state_ref]
    return []


def _artifact_refs_from_update_payload(update_payload: dict[str, Any]) -> tuple[list[ArtifactRef], list[Any]]:
    raw_refs = update_payload.get("artifact_refs", [])
    if not isinstance(raw_refs, list):
        return [], [raw_refs]
    refs: list[ArtifactRef] = []
    invalid_refs: list[Any] = []
    for raw_ref in raw_refs:
        if isinstance(raw_ref, ArtifactRef):
            refs.append(raw_ref)
        elif isinstance(raw_ref, dict):
            try:
                refs.append(ArtifactRef.model_validate(raw_ref))
            except ValueError:
                invalid_refs.append(raw_ref)
        else:
            invalid_refs.append(raw_ref)
    return refs, invalid_refs


def _register_skill_state_update(
    *,
    registry: BackendStateRegistry | None,
    request: TaskRequest,
    run_ref: str,
    skill_bundle: SkillBundle,
    update_result: SkillUpdateResult | None,
) -> None:
    if registry is None or update_result is None or update_result.skill_state_ref is None:
        return
    if registry.get_state(update_result.skill_state_ref) is not None:
        return
    parent_state_refs = []
    if skill_bundle.skill_state_ref and skill_bundle.skill_state_ref != update_result.skill_state_ref:
        parent_state_refs.append(skill_bundle.skill_state_ref)
    registry.register_candidate(
        BackendStateRecord(
            state_ref=update_result.skill_state_ref,
            backend_id=skill_bundle.backend_id,
            backend_type="skill",
            created_from_task_id=request.task_id,
            created_from_run_ref=run_ref,
            parent_state_refs=parent_state_refs,
            artifact_refs=update_result.artifact_refs,
            metadata={
                "graph_version_ref": update_result.graph_version_ref,
                "update_summary": update_result.update_summary,
                **update_result.metadata,
            },
        )
    )


def _safe_state_ref(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in value)
    if not safe or safe in {".", ".."}:
        raise ValueError(f"unsafe state ref: {value!r}")
    return safe


def _combined_memory_bundle(agent_memory_bundle: MemoryBundle, task_memory_bundle: MemoryBundle) -> MemoryBundle:
    return MemoryBundle(
        backend_id="combined",
        items=[
            *_scoped_memory_items("Agent Memory", "agent", agent_memory_bundle),
            *_scoped_memory_items("Task Memory", "task", task_memory_bundle),
        ],
        metadata={
            "agent_memory_backend_id": agent_memory_bundle.backend_id,
            "agent_memory_state_ref": agent_memory_bundle.state_ref,
            "task_memory_backend_id": task_memory_bundle.backend_id,
            "task_memory_state_ref": task_memory_bundle.state_ref,
        },
    )


def _scoped_memory_items(section_label: str, memory_scope: str, memory_bundle: MemoryBundle) -> list[MemoryItem]:
    return [
        MemoryItem(
            memory_id=item.memory_id,
            content=f"{section_label}:\n{item.content}",
            score=item.score,
            metadata={
                **item.metadata,
                "memory_scope": memory_scope,
                "source_backend_id": memory_bundle.backend_id,
                "source_state_ref": memory_bundle.state_ref,
            },
        )
        for item in memory_bundle.items
    ]
