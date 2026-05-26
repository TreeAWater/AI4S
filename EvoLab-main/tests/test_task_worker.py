import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evolab.backends.memory import MethodMemoryBackend
from evolab.backends.memory.methods.mem0 import Mem0MemoryMethod
from evolab.config.agents import render_agents_markdown
from evolab.config.task_config import BackendBinding, MetaAgentSpec, ReflectorSpec, RoleSpec, TaskConfig
from evolab.contracts.common import ArtifactRef, Message, RuntimePolicy
from evolab.contracts.dynamic_workflow import DynamicSubagentsConfig
from evolab.contracts.dispatch import DispatchAction, DispatchDecision
from evolab.contracts.embeddings import EmbeddingResponse
from evolab.contracts.lab_state import ArtifactIndexRecord, SubagentReportRecord
from evolab.contracts.llm import LLMGenerationConfig, LLMRuntimeResponse, SubAgentAction
from evolab.contracts.records import MetaAgentRunRecord, SubagentRunRecord, TrajectoryEventRecord
from evolab.contracts.state import BackendStateRecord
from evolab.contracts.retrieval import (
    MemoryBundle,
    MemoryItem,
    RetrievalRequest,
    SkillBundle,
    SkillRef,
    SkillUpdateResult,
)
from evolab.contracts.task import TaskOrigin, TaskPurpose, TaskRequest
from evolab.contracts.tools import ToolCall, ToolCallRecord, ToolResult, ToolSpec
from evolab.lab.layout import LabLayout
from evolab.lab.queue import FileWorkQueue
from evolab.registries.backend_state import FileBackendStateRegistry
from evolab.registries.lab_state import FileLabStateRegistry
from evolab.registries.trajectory import FileTrajectoryRegistry
from evolab.runtime.prompt_builder import PromptBuilder
from evolab.runtime.task_runtime import (
    TaskRuntime as _TaskRuntime,
    _apply_reflector_computed_metrics,
    _bootstrap_scientific_handoff_satisfies_expected_outputs,
    _bootstrap_scientific_handoff_satisfies_terminal_outputs,
    _compact_reflector_llm_payload,
    _final_records_for_write_bootstrap,
    _latest_scientific_artifact_uri,
    _reflector_result_payload,
    _reflector_sequence_evaluation,
    _subagent_completion_contract,
)
from evolab.runtime.task_worker import TaskWorker
from evolab.tools.runtime import ToolRegistry, ToolRuntime


STATIC_DEFAULT_CONFIG_REQUIRED_REMOVED = pytest.mark.skip(
    reason="removed static default worker path handled missing task_config before dynamic role-pool config is required"
)
STATIC_DEFAULT_EARLY_FAILURE_SUBAGENT_REMOVED = pytest.mark.skip(
    reason="removed static default worker path persisted an early synthetic subagent run before dynamic dispatch"
)
DEFAULT_META_DISPATCH_REMOVED = pytest.mark.skip(
    reason="removed default MetaAgent dispatch loop; dynamic role-pool planner is required for default execution"
)


class FakeRuntime:
    def __init__(self, result: dict[str, str] | None = None):
        self.requests: list[TaskRequest] = []
        self.result = result or {"status": "completed"}

    def run(self, request: TaskRequest) -> dict[str, str]:
        self.requests.append(request)
        return self.result


class RaisingRuntime:
    def run(self, request: TaskRequest) -> dict[str, str]:
        raise RuntimeError(f"boom {request.task_id}")


class InterruptingRuntime:
    def run(self, request: TaskRequest) -> dict[str, str]:
        raise KeyboardInterrupt()


class FakeBackend:
    def __init__(self, backend_id: str):
        self.backend_id = backend_id
        self.instantiated_state_refs: list[str | None] = []

    def instantiate(self, state_ref: str | None) -> object:
        self.instantiated_state_refs.append(state_ref)
        return {"backend_id": self.backend_id, "state_ref": state_ref}


class RaisingBackend:
    def __init__(self, backend_id: str):
        self.backend_id = backend_id
        self.instantiated_state_refs: list[str | None] = []

    def instantiate(self, state_ref: str | None) -> object:
        self.instantiated_state_refs.append(state_ref)
        raise RuntimeError(f"unused backend {self.backend_id} should not instantiate")


class DirectRuntime:
    pass


class RecordingMemoryRuntime:
    def __init__(
        self,
        *,
        backend_id: str = "memory-local",
        state_ref: str = "memory-state-before",
        memory_id: str = "memory-1",
        content: str = "Previous result: catalyst A improved yield.",
        metadata: dict | None = None,
        update_result: dict[str, str] | None = None,
    ):
        self.backend_id = backend_id
        self.state_ref = state_ref
        self.memory_id = memory_id
        self.content = content
        self.metadata = metadata or {}
        self.update_result = update_result or {"status": "updated", "state_ref": "memory-state-after"}
        self.search_requests: list[RetrievalRequest] = []
        self.add_calls: list[tuple[str, str, list[Message]]] = []

    def search(self, request: RetrievalRequest) -> MemoryBundle:
        self.search_requests.append(request)
        return MemoryBundle(
            backend_id=self.backend_id,
            state_ref=self.state_ref,
            items=[
                MemoryItem(
                    memory_id=self.memory_id,
                    content=self.content,
                    score=0.8,
                    metadata=self.metadata,
                )
            ],
        )

    def add(self, task_id: str, role: str, messages: list[Message]) -> dict[str, str]:
        self.add_calls.append((task_id, role, messages))
        return self.update_result


class RecordingSkillRuntime:
    def __init__(self, required_tools: list[str] | None = None):
        self.get_requests: list[RetrievalRequest] = []
        self.look_at_events: list[dict] = []
        self.required_tools = required_tools or []

    def get(self, request: RetrievalRequest) -> SkillBundle:
        self.get_requests.append(request)
        return SkillBundle(
            backend_id="skill-local",
            graph_version_ref="skill-graph-v1",
            required_tools=self.required_tools,
            skills=[
                SkillRef(
                    skill_id="skill-1",
                    name="Compare catalyst results",
                    content="Use prior experimental results to compare catalyst candidates.",
                    required_tools=self.required_tools,
                    metadata={
                        "retrieval": {
                            "matched_category_path": "Literature > Retrieval",
                            "retrieved_by": "direct",
                        }
                    },
                )
            ],
            metadata={
                "graph_context_summary": {
                    "graph_version": "skill-graph-v1",
                    "retrieval_paths": [
                        {
                            "category_ids": ["cap-literature", "task-retrieval"],
                            "endpoint_category_id": "task-retrieval",
                            "category_path": "Literature > Retrieval",
                        }
                    ],
                }
            },
        )

    def look_at(self, event: dict) -> dict[str, str]:
        self.look_at_events.append(event)
        return {"status": "recorded", "graph_version_after": "skill-graph-v1"}


class UpdatingSkillRuntime(RecordingSkillRuntime):
    def get(self, request: RetrievalRequest) -> SkillBundle:
        bundle = super().get(request)
        return bundle.model_copy(update={"skill_state_ref": "skill-state-v1"})

    def look_at(self, event: dict) -> SkillUpdateResult:
        self.look_at_events.append(event)
        return SkillUpdateResult(
            status="recorded",
            update_summary={"observed_run_ref": event["run_ref"]},
            graph_version_ref="skill-graph-v2",
            skill_state_ref="skill-state-v2",
        )


class RecordingLLMRuntime:
    def __init__(self):
        self.calls: list[tuple[list[Message], list[dict], LLMGenerationConfig]] = []

    def generate(
        self,
        messages: list[Message],
        tool_specs: list[dict],
        generation_config: LLMGenerationConfig,
    ) -> LLMRuntimeResponse:
        self.calls.append((messages, tool_specs, generation_config))
        return LLMRuntimeResponse(
            action=SubAgentAction(action="final_answer", content="Use catalyst A."),
            raw_response={"id": "response-1"},
        )


class ScriptedLLMRuntime:
    def __init__(self, responses: list[LLMRuntimeResponse]):
        self.responses = responses
        self.calls: list[tuple[list[Message], list[dict], LLMGenerationConfig]] = []

    def generate(
        self,
        messages: list[Message],
        tool_specs: list[dict],
        generation_config: LLMGenerationConfig,
    ) -> LLMRuntimeResponse:
        self.calls.append((messages, tool_specs, generation_config))
        return self.responses.pop(0)


class DynamicPlannerRuntime:
    def __init__(self, task_config: TaskConfig):
        self.task_config = task_config
        self.calls: list[tuple[list[Message], list[dict], LLMGenerationConfig]] = []

    def generate(
        self,
        messages: list[Message],
        tool_specs: list[dict],
        generation_config: LLMGenerationConfig,
    ) -> LLMRuntimeResponse:
        self.calls.append((messages, tool_specs, generation_config))
        return LLMRuntimeResponse(
            action=SubAgentAction(
                action="final_answer",
                content=json.dumps(_dynamic_workflow_payload_for_roles(self.task_config)),
            )
        )


class StaticMemoryExtractionLLM:
    def __init__(self, content: str = "Solver recorded first task finding."):
        self.content = content
        self.calls: list[tuple[list[Message], list[dict], LLMGenerationConfig]] = []

    def generate(
        self,
        messages: list[Message],
        tool_specs: list[dict],
        generation_config: LLMGenerationConfig,
    ) -> LLMRuntimeResponse:
        self.calls.append((messages, tool_specs, generation_config))
        return LLMRuntimeResponse(
            action=SubAgentAction(
                action="final_answer",
                content=json.dumps(
                    {
                        "memory": [
                            {
                                "id": "task-finding",
                                "text": self.content,
                                "attributed_to": "assistant",
                                "linked_memory_ids": [],
                            }
                        ]
                    }
                ),
            )
        )


class ConstantEmbeddingRuntime:
    def embed(self, texts: list[str], *, purpose: str) -> EmbeddingResponse:
        return EmbeddingResponse(
            backend_id="memory-embedding",
            model="constant",
            vectors=[[1.0, 0.0] for _ in texts],
            metadata={"purpose": purpose},
        )


def _native_mem0_backend(tmp_path: Path, backend_id: str) -> MethodMemoryBackend:
    backend = MethodMemoryBackend(
        backend_id=backend_id,
        method=Mem0MemoryMethod(
            store_path=tmp_path / f"{backend_id}.sqlite",
            llm_backend_id="memory-llm",
            embedding_backend_id="memory-embedding",
        ),
        default_search_threshold=0.0,
    )
    backend.bind_runtimes(
        llm_runtimes={"memory-llm": StaticMemoryExtractionLLM()},
        embedding_runtimes={"memory-embedding": ConstantEmbeddingRuntime()},
    )
    return backend


def _request(task_id: str = "task-1") -> TaskRequest:
    return TaskRequest(
        task_id=task_id,
        origin=TaskOrigin.HUMAN,
        purpose=TaskPurpose.SCIENCE,
        goal="Solve the task.",
    )


def _dynamic_config_for_roles(task_config: TaskConfig) -> DynamicSubagentsConfig:
    roles = list(task_config.roles.values())
    worker_backend_id = roles[0].llm_backend.backend_id if roles else "llm-local"
    allowed_tools = sorted({tool for role in roles for tool in role.allowed_tools})
    return DynamicSubagentsConfig(
        enabled=True,
        mode="dynamic",
        planner_backend={"backend_id": "planner-local"},
        default_worker_backend={"backend_id": worker_backend_id},
        allowed_worker_backend_ids=sorted({role.llm_backend.backend_id for role in roles}),
        allowed_tool_names=allowed_tools,
        max_planner_retries=0,
        require_output_schema=False,
        metadata={"enable_static_completion_guards": True},
    )


def _test_agents_ref(kwargs: dict) -> str:
    for key in ("trajectory_registry", "backend_state_registry", "lab_state_registry", "task_registry"):
        root = getattr(kwargs.get(key), "root", None)
        if isinstance(root, Path):
            return str(root.parent / "agents.md")
    return str(Path("/tmp/evolab-test-agents") / "task-worker-agents.md")


def _dynamic_task_config(task_config: TaskConfig, *, agents_ref: str) -> TaskConfig:
    if task_config.dynamic_subagents is not None:
        return task_config
    return task_config.model_copy(
        update={
            "agents_ref": task_config.agents_ref or agents_ref,
            "dynamic_subagents": _dynamic_config_for_roles(task_config),
        }
    )


def _dynamic_llm_runtimes(task_config: TaskConfig, runtimes: dict[str, object]) -> dict[str, object]:
    return {"planner-local": DynamicPlannerRuntime(task_config), **runtimes}


def _dynamic_workflow_payload_for_roles(task_config: TaskConfig) -> dict:
    agents = []
    nodes = []
    edges = []
    previous_node_id: str | None = None
    for index, role in enumerate(task_config.roles.values()):
        subagent_id = f"{role.name}-dynamic"
        node_id = f"node-{index}-{role.name}"
        agents.append(
            {
                "subagent_id": subagent_id,
                "role_name": role.name,
                "goal": task_config.goal,
                "system_prompt": role.system_prompt,
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
                "allowed_tools": list(role.allowed_tools),
                "required_skills": list(role.required_skills),
                "llm_backend_id": role.llm_backend.backend_id,
            }
        )
        nodes.append(
            {
                "node_id": node_id,
                "subagent_id": subagent_id,
                "dependencies": [previous_node_id] if previous_node_id is not None else [],
            }
        )
        if previous_node_id is not None:
            edges.append({"source_node_id": previous_node_id, "target_node_id": node_id})
        previous_node_id = node_id
    return {
        "workflow_id": "wf-test-role-pool",
        "task_summary": task_config.goal,
        "article_context_summary": "unit test",
        "dynamic_subagents": agents,
        "workflow_nodes": nodes,
        "workflow_edges": edges,
        "artifact_contracts": {},
        "validation_rules": [],
        "planner_rationale_summary": "Use one dynamic worker for each configured role.",
        "metadata": {"extraction_task": False},
    }


def TaskRuntime(**kwargs) -> _TaskRuntime:
    task_config = kwargs.get("task_config")
    if (
        isinstance(task_config, TaskConfig)
        and task_config.dynamic_subagents is None
        and task_config.meta_agent is None
    ):
        kwargs["task_config"] = _dynamic_task_config(task_config, agents_ref=_test_agents_ref(kwargs))
        kwargs["llm_runtimes"] = _dynamic_llm_runtimes(task_config, kwargs.get("llm_runtimes") or {})
        if kwargs.get("tool_runtime") is None:
            kwargs["tool_runtime"] = ToolRuntime(ToolRegistry())
    return _TaskRuntime(**kwargs)


def _enqueue_task(layout: LabLayout, request: TaskRequest, job_id: str = "job-1") -> None:
    request_path = layout.root / "requests" / f"{request.task_id}.json"
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(request.model_dump_json(), encoding="utf-8")
    FileWorkQueue(layout.tasks_queue_dir).enqueue(
        job_id,
        {"request_payload_uri": str(request_path)},
    )


class _ListArtifactRegistry:
    def __init__(self, artifacts: list[SimpleNamespace]):
        self.artifacts = artifacts

    def list_artifacts(self, task_id: str) -> list[SimpleNamespace]:
        return self.artifacts


def test_startup_initializes_lab_task_queue_registries_tool_runtime_and_prompt_builder(tmp_path: Path):
    layout = LabLayout(tmp_path / "lab")
    worker = TaskWorker(layout=layout, worker_id="worker-1")

    worker.startup()

    assert layout.tasks_queue_dir.exists()
    assert (layout.registries_dir / "task").exists()
    assert (layout.registries_dir / "backend_state").exists()
    assert worker.task_queue is not None
    assert worker.task_registry is not None
    assert worker.backend_state_registry is not None
    assert worker.trajectory_registry is not None
    assert isinstance(worker.tool_registry, ToolRegistry)
    assert isinstance(worker.tool_runtime, ToolRuntime)
    assert isinstance(worker.prompt_builder, PromptBuilder)


def test_run_once_loads_task_request_and_marks_done(tmp_path: Path):
    layout = LabLayout(tmp_path / "lab")
    request = _request()
    _enqueue_task(layout, request)
    runtime = FakeRuntime({"task_id": request.task_id, "status": "completed"})
    worker = TaskWorker(layout=layout, worker_id="worker-1", task_runtime=runtime)
    worker.startup()

    result = worker.run_once()

    assert result == {"task_id": "task-1", "status": "completed"}
    assert runtime.requests == [request]
    assert sorted((layout.tasks_queue_dir / "done").glob("*.json"))
    assert not list((layout.tasks_queue_dir / "failed").glob("*.json"))
    events = FileTrajectoryRegistry(layout.registries_dir / "trajectory").list_events()
    assert [event.event_type for event in events] == ["task_started", "task_completed"]
    assert events[0].metadata["worker_id"] == "worker-1"
    lab_state = FileLabStateRegistry(layout.registries_dir / "lab_state")
    ledger = lab_state.get_run_ledger(request.task_id)
    assert ledger is not None
    assert ledger.status == "completed"
    assert ledger.task_goal == request.goal
    assert ledger.final_answer is None


def test_run_once_prints_progress_updates(tmp_path: Path):
    layout = LabLayout(tmp_path / "lab")
    request = _request()
    _enqueue_task(layout, request)
    runtime = FakeRuntime(
        {
            "task_id": request.task_id,
            "status": "completed",
            "runs": [
                {
                    "run_ref": "subagent-1",
                    "role": "SurveyAgent",
                    "status": "completed",
                    "artifact_refs": [{"uri": "/tmp/artifact.txt", "type": "log", "metadata": {}}],
                }
            ],
            "final_answer": "done",
        }
    )
    messages: list[str] = []
    worker = TaskWorker(
        layout=layout,
        worker_id="worker-1",
        task_runtime=runtime,
        progress_callback=messages.append,
    )
    worker.startup()

    worker.run_once()

    assert any("task started" in message for message in messages)
    assert any("subagent SurveyAgent completed" in message for message in messages)
    assert any("task completed" in message for message in messages)


def test_run_once_writes_generic_final_artifact_index_when_task_completes(tmp_path: Path):
    layout = LabLayout(tmp_path / "lab")
    request = _request()
    _enqueue_task(layout, request)
    artifact_path = tmp_path / "artifact.jsonl"
    artifact_path.write_text('{"ok": true}\n', encoding="utf-8")
    artifact = ArtifactRef(
        uri=str(artifact_path),
        type="dataset",
        metadata={"status": "final", "validation_status": "not_checked"},
    )
    runtime = FakeRuntime(
        {
            "task_id": request.task_id,
            "status": "completed",
            "final_answer": "done",
            "run_refs": ["subagent-1"],
            "meta_run_refs": ["meta-1"],
            "runs": [
                {
                    "run_ref": "subagent-1",
                    "role": "WriteAgent",
                    "meta_workflow_node_id": "write-node",
                    "artifact_refs": [artifact.model_dump(mode="json")],
                }
            ],
        }
    )
    worker = TaskWorker(layout=layout, worker_id="worker-1", task_runtime=runtime)
    worker.startup()

    result = worker.run_once()

    assert result is not None
    lab_state = FileLabStateRegistry(layout.registries_dir / "lab_state")
    ledger = lab_state.get_run_ledger(request.task_id)
    assert ledger is not None
    assert ledger.status == "completed"
    assert ledger.final_artifact_refs == [artifact]
    index_path = layout.registries_dir / "lab_state" / "final_artifact_indexes" / f"{request.task_id}.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["warnings"] == []
    assert payload["artifacts"] == [
        {
            "artifact_path": str(artifact_path),
            "artifact_type": "dataset",
            "created_timestamp": None,
            "producing_agent": "WriteAgent",
            "producing_step": "write-node",
            "validation_status": "not_checked",
            "is_final": True,
            "metadata": {"status": "final", "validation_status": "not_checked"},
        }
    ]


def test_run_once_marks_failed_if_runtime_raises(tmp_path: Path):
    layout = LabLayout(tmp_path / "lab")
    _enqueue_task(layout, _request())
    worker = TaskWorker(layout=layout, worker_id="worker-1", task_runtime=RaisingRuntime())
    worker.startup()

    result = worker.run_once()

    assert result is None
    failed_paths = sorted((layout.tasks_queue_dir / "failed").glob("*.json"))
    assert len(failed_paths) == 1
    failed_payload = json.loads(failed_paths[0].read_text(encoding="utf-8"))
    assert failed_payload["error"] == "boom task-1"
    events = FileTrajectoryRegistry(layout.registries_dir / "trajectory").list_events()
    lab_state = FileLabStateRegistry(layout.registries_dir / "lab_state")
    ledger = lab_state.get_run_ledger("task-1")
    assert ledger is not None
    assert ledger.status == "failed"
    assert ledger.failure_reason == "boom task-1"
    assert [event.event_type for event in events] == ["task_started", "task_failed"]
    assert events[-1].metadata["error"] == "boom task-1"
    index_path = layout.registries_dir / "lab_state" / "final_artifact_indexes" / "task-1.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["warnings"] == ["no artifacts were recorded for this task run"]


def test_failed_run_ledger_includes_recorded_meta_and_subagent_refs(tmp_path: Path):
    layout = LabLayout(tmp_path / "lab")
    request = _request()
    _enqueue_task(layout, request)
    trajectory_registry = FileTrajectoryRegistry(layout.registries_dir / "trajectory")

    class RecordedFailureRuntime:
        def run(self, request: TaskRequest) -> dict[str, str]:
            trajectory_registry.save_meta_agent_run(
                MetaAgentRunRecord(
                    run_ref="meta-recorded",
                    task_id=request.task_id,
                    decision=DispatchDecision(
                        action=DispatchAction.RUN_SUBAGENT,
                        target_role="ExecAgent",
                        instruction="Run failed work.",
                    ),
                )
            )
            trajectory_registry.save_subagent_run(
                SubagentRunRecord(
                    run_ref="subagent-recorded",
                    task_id=request.task_id,
                    task_origin=request.origin,
                    task_purpose=request.purpose,
                    stage_index=0,
                    role="ExecAgent",
                    instruction="Run failed work.",
                    retrieval_request=RetrievalRequest(
                        task_id=request.task_id,
                        role="ExecAgent",
                        query="Run failed work.",
                    ),
                    memory_bundle=MemoryBundle(backend_id="memory-local"),
                    skill_bundle=SkillBundle(backend_id="skill-local"),
                    prompt_messages=[Message(role="user", content="Run failed work.")],
                    output_messages=[Message(role="assistant", content="failed")],
                    llm_backend_id="llm-local",
                    metadata={"status": "failed"},
                )
            )
            raise RuntimeError("boom after traces")

    worker = TaskWorker(
        layout=layout,
        worker_id="worker-1",
        task_runtime=RecordedFailureRuntime(),
        trajectory_registry=trajectory_registry,
    )
    worker.startup()

    worker.run_once()

    ledger = FileLabStateRegistry(layout.registries_dir / "lab_state").get_run_ledger(request.task_id)
    assert ledger is not None
    assert ledger.status == "failed"
    assert ledger.meta_run_refs == ["meta-recorded"]
    assert ledger.subagent_run_refs == ["subagent-recorded"]


def test_run_once_marks_interrupted_if_runtime_is_interrupted(tmp_path: Path):
    layout = LabLayout(tmp_path / "lab")
    _enqueue_task(layout, _request())
    worker = TaskWorker(layout=layout, worker_id="worker-1", task_runtime=InterruptingRuntime())
    worker.startup()

    result = worker.run_once()

    assert result is None
    assert not list((layout.tasks_queue_dir / "claimed").glob("*.json"))
    interrupted_paths = sorted((layout.tasks_queue_dir / "interrupted").glob("*.json"))
    assert len(interrupted_paths) == 1
    payload = json.loads(interrupted_paths[0].read_text(encoding="utf-8"))
    assert payload["interrupt_reason"] == "KeyboardInterrupt"
    events = FileTrajectoryRegistry(layout.registries_dir / "trajectory").list_events()
    assert [event.event_type for event in events] == ["task_started", "task_interrupted"]
    lab_state = FileLabStateRegistry(layout.registries_dir / "lab_state")
    ledger = lab_state.get_run_ledger("task-1")
    assert ledger is not None
    assert ledger.status == "interrupted"
    index_path = layout.registries_dir / "lab_state" / "final_artifact_indexes" / "task-1.json"
    assert json.loads(index_path.read_text(encoding="utf-8"))["status"] == "interrupted"


def test_run_once_marks_budget_exceeded_result_failed_and_writes_final_index(tmp_path: Path):
    layout = LabLayout(tmp_path / "lab")
    request = _request()
    _enqueue_task(layout, request)
    runtime = FakeRuntime(
        {
            "task_id": request.task_id,
            "status": "budget_exceeded",
            "failure_reason": "budget_exceeded: max_subagent_llm_calls exceeded",
            "run_refs": ["subagent-1"],
            "runs": [
                {
                    "run_ref": "subagent-1",
                    "role": "GenericAgent",
                    "status": "budget_exceeded",
                    "failure_reason": "budget_exceeded: max_subagent_llm_calls exceeded",
                    "artifact_refs": [],
                }
            ],
        }
    )
    worker = TaskWorker(layout=layout, worker_id="worker-1", task_runtime=runtime)
    worker.startup()

    result = worker.run_once()

    assert result is not None
    assert sorted((layout.tasks_queue_dir / "failed").glob("*.json"))
    assert not list((layout.tasks_queue_dir / "claimed").glob("*.json"))
    lab_state = FileLabStateRegistry(layout.registries_dir / "lab_state")
    ledger = lab_state.get_run_ledger(request.task_id)
    assert ledger is not None
    assert ledger.status == "failed"
    assert ledger.failure_reason == "budget_exceeded: max_subagent_llm_calls exceeded"
    index_path = layout.registries_dir / "lab_state" / "final_artifact_indexes" / f"{request.task_id}.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["failure_reason"] == "budget_exceeded: max_subagent_llm_calls exceeded"


def _write_work_item_lifecycle(layout: LabLayout, request: TaskRequest, *, status: str = "claimed") -> Path:
    path = layout.registries_dir / "lab_state" / "work_items" / request.task_id / "article-a.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "task_id": request.task_id,
                "work_item_id": "article-a",
                "status": status,
                "history": [{"event": "subagent_run", "status": status}],
            }
        ),
        encoding="utf-8",
    )
    return path


def _queue_counts(layout: LabLayout) -> dict[str, int]:
    return {
        name: len(list((layout.tasks_queue_dir / name).glob("*.json")))
        for name in ["queued", "claimed", "running", "done", "failed", "interrupted"]
        if (layout.tasks_queue_dir / name).exists()
    }


def test_completed_task_reconciles_claimed_work_item_lifecycle_to_completed(tmp_path: Path):
    layout = LabLayout(tmp_path / "lab")
    request = _request()
    _enqueue_task(layout, request)
    work_item_path = _write_work_item_lifecycle(layout, request, status="claimed")
    runtime = FakeRuntime({"task_id": request.task_id, "status": "completed"})
    worker = TaskWorker(layout=layout, worker_id="worker-1", task_runtime=runtime)
    worker.startup()

    worker.run_once()

    payload = json.loads(work_item_path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["history"][-1]["event"] == "task_terminal_reconcile"
    counts = _queue_counts(layout)
    assert counts.get("claimed", 0) == 0
    assert counts.get("running", 0) == 0


def test_failed_task_reconciles_claimed_work_item_lifecycle_to_failed(tmp_path: Path):
    layout = LabLayout(tmp_path / "lab")
    request = _request()
    _enqueue_task(layout, request)
    work_item_path = _write_work_item_lifecycle(layout, request, status="claimed")
    runtime = FakeRuntime({"task_id": request.task_id, "status": "failed", "failure_reason": "write failed"})
    worker = TaskWorker(layout=layout, worker_id="worker-1", task_runtime=runtime)
    worker.startup()

    worker.run_once()

    payload = json.loads(work_item_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["history"][-1]["event"] == "task_terminal_reconcile"
    counts = _queue_counts(layout)
    assert counts.get("claimed", 0) == 0
    assert counts.get("running", 0) == 0


def test_failed_task_reconciles_pending_work_item_lifecycle_to_failed(tmp_path: Path):
    layout = LabLayout(tmp_path / "lab")
    request = _request()
    _enqueue_task(layout, request)
    work_item_path = _write_work_item_lifecycle(layout, request, status="pending")
    runtime = FakeRuntime({"task_id": request.task_id, "status": "failed", "failure_reason": "api unavailable"})
    worker = TaskWorker(layout=layout, worker_id="worker-1", task_runtime=runtime)
    worker.startup()

    worker.run_once()

    payload = json.loads(work_item_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["history"][-1]["event"] == "task_terminal_reconcile"


def test_budget_exceeded_task_reconciliation_does_not_leave_work_item_claimed(tmp_path: Path):
    layout = LabLayout(tmp_path / "lab")
    request = _request()
    _enqueue_task(layout, request)
    work_item_path = _write_work_item_lifecycle(layout, request, status="claimed")
    runtime = FakeRuntime({"task_id": request.task_id, "status": "budget_exceeded", "failure_reason": "budget"})
    worker = TaskWorker(layout=layout, worker_id="worker-1", task_runtime=runtime)
    worker.startup()

    worker.run_once()

    payload = json.loads(work_item_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["history"][-1]["event"] == "task_terminal_reconcile"
    counts = _queue_counts(layout)
    assert counts.get("claimed", 0) == 0
    assert counts.get("running", 0) == 0


def test_interrupted_task_reconciles_claimed_work_item_lifecycle_to_interrupted(tmp_path: Path):
    layout = LabLayout(tmp_path / "lab")
    request = _request()
    _enqueue_task(layout, request)
    work_item_path = _write_work_item_lifecycle(layout, request, status="claimed")
    worker = TaskWorker(layout=layout, worker_id="worker-1", task_runtime=InterruptingRuntime())
    worker.startup()

    worker.run_once()

    payload = json.loads(work_item_path.read_text(encoding="utf-8"))
    assert payload["status"] == "interrupted"
    assert payload["history"][-1]["event"] == "task_terminal_reconcile"
    counts = _queue_counts(layout)
    assert counts.get("claimed", 0) == 0
    assert counts.get("running", 0) == 0


@STATIC_DEFAULT_CONFIG_REQUIRED_REMOVED
def test_default_task_runtime_marks_task_failed_without_task_config(tmp_path: Path):
    layout = LabLayout(tmp_path / "lab")
    _enqueue_task(layout, _request())
    worker = TaskWorker(layout=layout, worker_id="worker-1")
    worker.startup()

    result = worker.run_once()

    assert result is None
    assert not list((layout.tasks_queue_dir / "done").glob("*.json"))
    failed_paths = sorted((layout.tasks_queue_dir / "failed").glob("*.json"))
    assert len(failed_paths) == 1
    failed_payload = json.loads(failed_paths[0].read_text(encoding="utf-8"))
    assert "task_config" in failed_payload["error"]


@STATIC_DEFAULT_EARLY_FAILURE_SUBAGENT_REMOVED
def test_default_task_runtime_records_interrupted_subagent_for_early_runtime_failure(tmp_path: Path):
    layout = LabLayout(tmp_path / "lab")
    request = _request()
    _enqueue_task(layout, request)
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="You solve scientific tasks.",
                llm_backend=BackendBinding(backend_id="missing-llm"),
            )
        },
    )
    worker = TaskWorker(
        layout=layout,
        worker_id="worker-1",
        task_config=_dynamic_task_config(task_config),
        llm_runtimes=_dynamic_llm_runtimes(task_config, {}),
    )
    worker.startup()

    result = worker.run_once()

    assert result is None
    events = FileTrajectoryRegistry(layout.registries_dir / "trajectory").list_events()
    assert [event.event_type for event in events] == [
        "task_started",
        "subagent_started",
        "subagent_interrupted",
        "task_failed",
    ]
    assert events[1].run_ref == events[2].run_ref
    assert events[2].metadata["error"] == "missing LLM runtime for backend_id='missing-llm'"
    reports = FileLabStateRegistry(layout.registries_dir / "lab_state").list_subagent_reports(request.task_id)
    assert len(reports) == 1
    assert reports[0].status == "interrupted"
    assert reports[0].role == "solver"
    assert reports[0].failures == [{"reason": "missing LLM runtime for backend_id='missing-llm'"}]
    subagent_runs = FileTrajectoryRegistry(layout.registries_dir / "trajectory").list_subagent_runs()
    assert len(subagent_runs) == 1
    assert subagent_runs[0].role == "solver"
    assert subagent_runs[0].metadata["status"] == "interrupted"
    assert subagent_runs[0].metadata["failure_reason"] == "missing LLM runtime for backend_id='missing-llm'"


def test_interruption_records_complete_open_subagent_run_and_cleans_claimed_queue(tmp_path: Path):
    layout = LabLayout(tmp_path / "lab")
    request = _request()
    _enqueue_task(layout, request)

    class StartedThenInterruptedRuntime:
        def __init__(self, trajectory_registry: FileTrajectoryRegistry):
            self.trajectory_registry = trajectory_registry

        def run(self, request: TaskRequest) -> dict[str, str]:
            self.trajectory_registry.save_event(
                TrajectoryEventRecord(
                event_ref="event-open-subagent",
                event_type="subagent_started",
                subject_type="subagent",
                subject_ref="subagent-open",
                task_id=request.task_id,
                run_ref="subagent-open",
                metadata={
                    "role": "GenericAgent",
                    "generic_agent_type": "GenericAgent",
                    "assigned_task": "Do generic work.",
                    "stage_index": 0,
                    "llm_backend_id": "llm-local",
                    "meta_workflow_node_id": "dispatch-001-GenericAgent",
                },
                )
            )
            raise KeyboardInterrupt()

    trajectory_registry = FileTrajectoryRegistry(layout.registries_dir / "trajectory")
    worker = TaskWorker(
        layout=layout,
        worker_id="worker-1",
        task_runtime=StartedThenInterruptedRuntime(trajectory_registry),
        trajectory_registry=trajectory_registry,
    )
    worker.startup()

    result = worker.run_once()

    assert result is None
    assert not list((layout.tasks_queue_dir / "claimed").glob("*.json"))
    assert sorted((layout.tasks_queue_dir / "interrupted").glob("*.json"))
    subagent_runs = FileTrajectoryRegistry(layout.registries_dir / "trajectory").list_subagent_runs()
    assert len(subagent_runs) == 1
    run = subagent_runs[0]
    assert run.run_ref == "subagent-open"
    assert run.role == "GenericAgent"
    assert run.instruction == "Do generic work."
    assert run.metadata["status"] == "interrupted"
    assert run.metadata["partial"] is True
    assert run.metadata["failure_reason"] == "KeyboardInterrupt"
    assert (layout.registries_dir / "lab_state" / "final_artifact_indexes" / f"{request.task_id}.json").exists()


def test_interruption_recording_handles_missing_request_file(tmp_path: Path):
    layout = LabLayout(tmp_path / "lab")
    trajectory_registry = FileTrajectoryRegistry(layout.registries_dir / "trajectory")
    trajectory_registry.save_event(
        TrajectoryEventRecord(
            event_ref="event-open-subagent",
            event_type="subagent_started",
            subject_type="subagent",
            subject_ref="subagent-open",
            task_id="task-without-request-file",
            run_ref="subagent-open",
            metadata={
                "role": "GenericAgent",
                "generic_agent_type": "GenericAgent",
                "assigned_task": "Do generic work.",
                "stage_index": 0,
                "llm_backend_id": "llm-local",
                "meta_workflow_node_id": "dispatch-001-GenericAgent",
            },
        )
    )
    worker = TaskWorker(
        layout=layout,
        worker_id="worker-1",
        trajectory_registry=trajectory_registry,
    )
    worker.startup()

    worker._record_open_subagents_interrupted("task-without-request-file", "KeyboardInterrupt")

    run = trajectory_registry.get_subagent_run("subagent-open")
    assert run is not None
    assert run.task_origin == TaskOrigin.HUMAN
    assert run.task_purpose == TaskPurpose.SCIENCE
    assert run.metadata["status"] == "interrupted"


def test_default_task_runtime_runs_memory_skill_llm_and_records_updates(tmp_path: Path):
    request = _request()
    memory = RecordingMemoryRuntime()
    class RoleAwareSkillRuntime(RecordingSkillRuntime):
        def get(self, request: RetrievalRequest) -> SkillBundle:
            self.required_tools = ["write_jsonl"] if request.role == "WriteAgent" else []
            return super().get(request)

    skill = RoleAwareSkillRuntime()
    llm = RecordingLLMRuntime()
    trajectory_registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="You solve scientific tasks.",
                llm_backend=BackendBinding(
                    backend_id="llm-local",
                    config_ref="configs/llm.yaml",
                    state_ref="llm-state-1",
                ),
            )
        },
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        trajectory_registry=trajectory_registry,
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": skill},
    )

    result = runtime.run(request)

    assert result["task_id"] == request.task_id
    assert result["role"] == "solver"
    assert result["final_answer"] == "Use catalyst A."
    assert len(memory.search_requests) == 1
    task_request = memory.search_requests[0]
    assert task_request.query == request.goal
    assert task_request.task_origin == TaskOrigin.HUMAN
    assert task_request.role == "task"
    assert task_request.filters["memory_scope"] == "task"
    assert task_request.filters["memory_scope_id"] == "task:task-1"
    worker_skill_requests = [
        request
        for request in skill.get_requests
        if request.role == "solver" and request.filters.get("memory_scope") == "agent"
    ]
    assert len(worker_skill_requests) == 1
    assert len(llm.calls) == 1
    prompt_text = "\n".join(message.content for message in llm.calls[0][0])
    assert "Compare catalyst results" in prompt_text
    assert len(memory.add_calls) == 1
    add_task_id, add_role, add_messages = memory.add_calls[0]
    assert (add_task_id, add_role) == (request.task_id, "task")
    assert "Use catalyst A." in add_messages[-1].content
    assert len(skill.look_at_events) == 1
    assert skill.look_at_events[0]["task_id"] == request.task_id
    assert skill.look_at_events[0]["run_ref"] == result["run_ref"]

    saved = trajectory_registry.get_subagent_run(result["run_ref"])
    assert saved is not None
    assert saved.task_id == request.task_id
    assert saved.role == "solver"
    assert saved.skill_bundle.graph_version_ref == "skill-graph-v1"
    assert saved.llm_backend_id == "llm-local"
    assert saved.llm_backend_config_ref == "configs/llm.yaml"
    assert saved.llm_backend_state_ref == "llm-state-1"
    assert saved.output_messages == [Message(role="assistant", content="Use catalyst A.")]
    assert saved.metadata["memory_update_result"] == {
        "status": "updated",
        "state_ref": "memory-state-after",
    }
    assert saved.metadata["task_memory_update_result"] == {
        "status": "updated",
        "state_ref": "memory-state-after",
    }
    assert saved.metadata["skill_update_result"] == {
        "status": "recorded",
        "graph_version_after": "skill-graph-v1",
    }
    events = trajectory_registry.list_events()
    assert [event.event_type for event in events] == ["subagent_started", "subagent_completed"]
    assert events[0].run_ref == result["run_ref"]


def test_default_task_runtime_prepares_skill_required_tools_for_llm_prompt_and_record(tmp_path: Path):
    request = _request()
    memory = RecordingMemoryRuntime()
    skill = RecordingSkillRuntime(required_tools=["lookup"])
    llm = RecordingLLMRuntime()
    trajectory_registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    tool_registry = ToolRegistry()
    tool_registry.register(
        ToolSpec(
            name="lookup",
            description="lookup tool",
            parameters_schema={"type": "object"},
        ),
        lambda arguments: "ok",
    )
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="You solve scientific tasks.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["lookup"],
            )
        },
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(tool_registry),
        trajectory_registry=trajectory_registry,
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": skill},
    )

    result = runtime.run(request)

    assert [spec["name"] for spec in llm.calls[0][1]] == ["lookup"]
    prompt_text = "\n".join(message.content for message in llm.calls[0][0])
    assert "Skill Context:" in prompt_text
    assert "Literature > Retrieval" in prompt_text
    saved = trajectory_registry.get_subagent_run(result["run_ref"])
    assert saved is not None
    assert saved.metadata["tool_bundle"]["tool_specs"][0]["name"] == "lookup"
    assert saved.metadata["skill_context"]["required_tools"] == ["lookup"]


def test_default_task_runtime_executes_single_tool_call_and_records_trace(tmp_path: Path):
    request = _request()
    memory = RecordingMemoryRuntime()
    skill = RecordingSkillRuntime(required_tools=["lookup"])
    tool_registry = ToolRegistry()
    tool_calls: list[dict] = []
    tool_registry.register(
        ToolSpec(name="lookup", description="lookup tool", parameters_schema={"type": "object"}),
        lambda arguments: tool_calls.append(arguments) or f"found {arguments['query']}",
    )
    llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(
                        call_id="call-1",
                        name="lookup",
                        arguments={"query": "catalyst"},
                    ),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(action="final_answer", content="Use the lookup result."),
            ),
        ]
    )
    trajectory_registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="You solve scientific tasks.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["lookup"],
            )
        },
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(tool_registry),
        trajectory_registry=trajectory_registry,
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": skill},
    )

    result = runtime.run(request)

    assert result["final_answer"] == "Use the lookup result."
    assert tool_calls == [{"query": "catalyst"}]
    for _, _, add_messages in memory.add_calls:
        assert all(message.role != "tool" for message in add_messages)
        assert len(add_messages) == 1
        summary = json.loads(add_messages[0].content)
        assert summary["runtime_stage"] == "subagent_flat_summary"
        assert summary["tool_summaries"][0]["tool_name"] == "lookup"
        assert summary["tool_summaries"][0]["content_summary"] == "found catalyst"
    assert len(llm.calls) == 2
    second_round_messages = llm.calls[1][0]
    assert second_round_messages[-1] == Message(
        role="tool",
        content="found catalyst",
        name="lookup",
        tool_call_id="call-1",
        metadata={
            "tool_result": {
                "schema_version": "v1",
                "call_id": "call-1",
                "status": "ok",
                "content": "found catalyst",
                "artifact_refs": [],
                "metadata": {},
            }
        },
    )
    saved = trajectory_registry.get_subagent_run(result["run_ref"])
    assert saved is not None
    llm_calls = trajectory_registry.list_llm_calls()
    worker_llm_calls = [call for call in llm_calls if call.run_ref == result["run_ref"]]
    assert saved.llm_call_refs == [call.call_ref for call in worker_llm_calls]
    assert len(worker_llm_calls) == 2
    assert worker_llm_calls[0].backend_id == "llm-local"
    assert worker_llm_calls[0].output_messages == [
        Message(
            role="assistant",
            content="",
            metadata={
                "action": "tool_call",
                "tool_call": {
                    "schema_version": "v1",
                    "call_id": "call-1",
                    "name": "lookup",
                    "arguments": {"query": "catalyst"},
                },
                "tool_calls": [
                    {
                        "schema_version": "v1",
                        "call_id": "call-1",
                        "name": "lookup",
                        "arguments": {"query": "catalyst"},
                    }
                ],
            },
        )
    ]
    assert worker_llm_calls[1].input_messages[-1].role == "tool"
    assert worker_llm_calls[1].input_messages[-1].content == "found catalyst"
    assert worker_llm_calls[1].output_messages[0].content == "Use the lookup result."
    assert saved.metadata["tool_trace"]["calls"] == [
        {
            "schema_version": "v1",
            "tool_call": {
                "schema_version": "v1",
                "call_id": "call-1",
                "name": "lookup",
                "arguments": {"query": "catalyst"},
            },
            "result": {
                "schema_version": "v1",
                "call_id": "call-1",
                "status": "ok",
                "content": "found catalyst",
                "artifact_refs": [],
                "metadata": {},
            },
        }
    ]
    tool_records = trajectory_registry.list_tool_call_records()
    assert len(tool_records) == 1
    assert tool_records[0].run_ref == result["run_ref"]
    assert tool_records[0].task_id == request.task_id
    assert tool_records[0].tool_call_id == "call-1"
    assert tool_records[0].tool_name == "lookup"
    assert tool_records[0].runtime_stage == "subagent_flat"
    assert tool_records[0].step_index == 0


def test_default_task_runtime_rejects_final_answer_until_required_tools_run(tmp_path: Path):
    request = _request()
    memory = RecordingMemoryRuntime()
    skill = RecordingSkillRuntime(required_tools=["lookup", "write_report"])
    tool_registry = ToolRegistry()
    tool_calls: list[str] = []
    tool_registry.register(
        ToolSpec(name="lookup", description="lookup tool", parameters_schema={"type": "object"}),
        lambda arguments: tool_calls.append("lookup") or "found catalyst",
    )
    tool_registry.register(
        ToolSpec(name="write_report", description="write report", parameters_schema={"type": "object"}),
        lambda arguments: tool_calls.append("write_report") or "wrote report",
    )
    llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="Too early.")),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="call-1", name="lookup", arguments={}),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="call-2", name="write_report", arguments={}),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="Done.")),
        ]
    )
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="You solve scientific tasks.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["lookup", "write_report"],
            )
        },
        runtime_policy=RuntimePolicy(
            metadata={"required_tool_calls_before_final": ["lookup", "write_report"]}
        ),
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(tool_registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": skill},
    )

    result = runtime.run(request)

    assert result["final_answer"] == "role completion guards satisfied after successful required tool calls"
    assert tool_calls == ["lookup", "write_report"]
    assert len(llm.calls) == 3
    rejection_messages = [message.content for message in llm.calls[1][0] if "rejected" in message.content]
    assert rejection_messages
    assert "lookup, write_report" in rejection_messages[0]


def test_default_task_runtime_rejects_final_answer_until_required_tools_succeed(tmp_path: Path):
    request = _request()
    memory = RecordingMemoryRuntime()
    skill = RecordingSkillRuntime(required_tools=["write_report"])
    tool_registry = ToolRegistry()
    attempts = 0

    def write_report(arguments: dict) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("missing content")
        return "wrote report"

    tool_registry.register(
        ToolSpec(name="write_report", description="write report", parameters_schema={"type": "object"}),
        write_report,
    )
    llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="call-1", name="write_report", arguments={}),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="Too early.")),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="call-2", name="write_report", arguments={"content": "ok"}),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="Done.")),
        ]
    )
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="You solve scientific tasks.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["write_report"],
            )
        },
        runtime_policy=RuntimePolicy(metadata={"required_tool_calls_before_final": ["write_report"]}),
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(tool_registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": skill},
    )

    result = runtime.run(request)

    assert result["final_answer"] == "role completion guards satisfied after successful required tool calls"
    assert attempts == 2
    assert len(llm.calls) == 3


def test_default_task_runtime_rejects_final_answer_until_min_jsonl_records_written(tmp_path: Path):
    request = _request()
    memory = RecordingMemoryRuntime()

    class FinalArtifactSkillRuntime(RecordingSkillRuntime):
        def get(self, request: RetrievalRequest) -> SkillBundle:
            self.required_tools = ["write_jsonl"] if request.role == "WriteAgent" else []
            return super().get(request)

    skill = FinalArtifactSkillRuntime()
    tool_registry = ToolRegistry()
    record_counts: list[int] = []
    tool_registry.register(
        ToolSpec(name="write_jsonl", description="write jsonl", parameters_schema={"type": "object"}),
        lambda arguments: ToolResult(
            call_id="write_jsonl",
            status="ok",
            content="wrote records",
            metadata={"record_count": len(arguments["records"])},
        ),
    )
    llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(
                        call_id="call-1",
                        name="write_jsonl",
                        arguments={"records": [{"name": "one"}]},
                    ),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="Too few.")),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(
                        call_id="call-2",
                        name="write_jsonl",
                        arguments={"records": [{"name": "one"}, {"name": "two"}]},
                    ),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="Done.")),
        ]
    )
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="You solve scientific tasks.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["write_jsonl"],
            )
        },
        runtime_policy=RuntimePolicy(
            metadata={
                "required_tool_calls_before_final": ["write_jsonl"],
                "minimum_jsonl_records_before_final": 2,
            }
        ),
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(tool_registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": skill},
    )

    result = runtime.run(request)

    assert result["final_answer"] == "role completion guards satisfied after successful required tool calls"
    assert len(llm.calls) == 3
    rejection_messages = [message.content for message in llm.calls[2][0] if "rejected" in message.content]
    assert rejection_messages
    assert "at least 2" in rejection_messages[0]


def test_default_task_runtime_saves_partial_subagent_before_raising_guard_failure(tmp_path: Path):
    request = _request()
    memory = RecordingMemoryRuntime()
    skill = RecordingSkillRuntime(required_tools=["write_jsonl"])
    tool_registry = ToolRegistry()
    tool_registry.register(
        ToolSpec(name="write_jsonl", description="write jsonl", parameters_schema={"type": "object"}),
        lambda arguments: ToolResult(
            call_id="write_jsonl",
            status="ok",
            content="wrote records",
            metadata={"record_count": len(arguments["records"])},
        ),
    )
    llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(
                        call_id="call-1",
                        name="write_jsonl",
                        arguments={"records": []},
                    ),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="Too few.")),
        ]
    )
    trajectory_registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    backend_state_registry = FileBackendStateRegistry(tmp_path / "backend_state")
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="You solve scientific tasks.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["write_jsonl"],
            )
        },
        runtime_policy=RuntimePolicy(
            max_tool_steps=1,
            metadata={
                "required_tool_calls_before_final": ["write_jsonl"],
                "minimum_jsonl_records_before_final": 2,
            },
        ),
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(tool_registry),
        trajectory_registry=trajectory_registry,
        backend_state_registry=backend_state_registry,
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": skill},
    )

    result = runtime.run(request)

    saved_runs = trajectory_registry.list_subagent_runs()
    assert len(saved_runs) == 1
    assert result["status"] == "failed"
    assert saved_runs[0].metadata["status"] == "guard_failed"
    assert "required at least 2" in saved_runs[0].metadata["failure_reason"]
    assert [event.event_type for event in trajectory_registry.list_events()] == [
        "subagent_started",
        "subagent_failed",
    ]
    assert len(memory.add_calls) == 1
    assert len(skill.look_at_events) == 1
    assert backend_state_registry.list_states()


@DEFAULT_META_DISPATCH_REMOVED
def test_meta_agent_receives_guard_failed_subagent_result_for_recovery_decision(tmp_path: Path):
    request = _request()
    memory = RecordingMemoryRuntime()
    skill = RecordingSkillRuntime(required_tools=["write_jsonl"])
    tool_registry = ToolRegistry()
    tool_registry.register(
        ToolSpec(name="write_jsonl", description="write jsonl", parameters_schema={"type": "object"}),
        lambda arguments: ToolResult(
            call_id="write_jsonl",
            status="ok",
            content="wrote records",
            metadata={"record_count": len(arguments["records"])},
        ),
    )

    class InspectingMetaLLMRuntime:
        def __init__(self):
            self.payloads: list[dict] = []

        def generate(
            self,
            messages: list[Message],
            tool_specs: list[dict],
            generation_config: LLMGenerationConfig,
        ) -> LLMRuntimeResponse:
            payload = json.loads(messages[-1].content)
            self.payloads.append(payload)
            if len(self.payloads) == 1:
                content = json.dumps(
                    {
                        "action": "run_subagent",
                        "target_role": "solver",
                        "instruction": "Write at least two records.",
                    }
                )
            else:
                assert payload["completed_runs"][0]["status"] == "guard_failed"
                assert "required at least 2" in payload["completed_runs"][0]["failure_reason"]
                content = json.dumps(
                    {
                        "action": "abort",
                        "instruction": "guard failure observed by meta-agent",
                    }
                )
            return LLMRuntimeResponse(
                action=SubAgentAction(action="final_answer", content=content),
            )

    meta_llm = InspectingMetaLLMRuntime()
    solver_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(
                        call_id="call-1",
                        name="write_jsonl",
                        arguments={"records": []},
                    ),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="Too few.")),
        ]
    )
    trajectory_registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    backend_state_registry = FileBackendStateRegistry(tmp_path / "backend_state")
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        max_dispatch_steps=2,
        meta_agent=MetaAgentSpec(
            name="meta",
            system_prompt="Dispatch solver and inspect recovery state.",
            llm_backend=BackendBinding(backend_id="meta-llm"),
        ),
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="You solve scientific tasks.",
                llm_backend=BackendBinding(backend_id="solver-llm"),
                allowed_tools=["write_jsonl"],
            )
        },
        runtime_policy=RuntimePolicy(
            max_tool_steps=1,
            metadata={
                "required_tool_calls_before_final": ["write_jsonl"],
                "minimum_jsonl_records_before_final": 2,
            },
        ),
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(tool_registry),
        trajectory_registry=trajectory_registry,
        backend_state_registry=backend_state_registry,
        llm_runtimes={"meta-llm": meta_llm, "solver-llm": solver_llm},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": skill},
    )

    with pytest.raises(RuntimeError, match="guard failure observed by meta-agent"):
        runtime.run(request)

    assert len(meta_llm.payloads) == 2
    assert len(trajectory_registry.list_meta_agent_runs()) == 2
    saved_runs = trajectory_registry.list_subagent_runs()
    assert len(saved_runs) == 1
    assert saved_runs[0].metadata["status"] == "guard_failed"
    assert [event.event_type for event in trajectory_registry.list_events()] == [
        "subagent_started",
        "subagent_failed",
    ]
    assert len(memory.add_calls) == 2
    assert len(skill.look_at_events) == 1
    assert backend_state_registry.list_states()


@DEFAULT_META_DISPATCH_REMOVED
def test_meta_agent_receives_current_lab_state_summary(tmp_path: Path):
    request = _request()

    class InspectingMetaLLMRuntime:
        def __init__(self):
            self.payloads: list[dict] = []

        def generate(
            self,
            messages: list[Message],
            tool_specs: list[dict],
            generation_config: LLMGenerationConfig,
        ) -> LLMRuntimeResponse:
            payload = json.loads(messages[-1].content)
            self.payloads.append(payload)
            return LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "route": "END",
                            "instruction": "Already complete.",
                            "metadata": {"final_answer": "Done."},
                        }
                    ),
                ),
            )

    meta_llm = InspectingMetaLLMRuntime()
    trajectory_registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    backend_state_registry = FileBackendStateRegistry(tmp_path / "backend_state")
    lab_state_registry = FileLabStateRegistry(tmp_path / "lab_state")
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        max_dispatch_steps=1,
        meta_agent=MetaAgentSpec(
            name="meta",
            system_prompt="Route to a subagent or END.",
            llm_backend=BackendBinding(backend_id="meta-llm"),
        ),
        roles={
            "SurveyAgent": RoleSpec(
                name="SurveyAgent",
                system_prompt="Survey lab state.",
                llm_backend=BackendBinding(backend_id="meta-llm"),
                allowed_tools=[],
            )
        },
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        trajectory_registry=trajectory_registry,
        backend_state_registry=backend_state_registry,
        lab_state_registry=lab_state_registry,
        llm_runtimes={"meta-llm": meta_llm},
        memory_runtimes={"memory-local": RecordingMemoryRuntime()},
        skill_runtimes={"skill-local": RecordingSkillRuntime()},
    )

    with pytest.raises(RuntimeError, match="finished without running a subagent"):
        runtime.run(request)

    assert len(meta_llm.payloads) == 1
    lab_state = meta_llm.payloads[0]["lab_state"]
    assert set(lab_state) == {"index", "digest", "requested_details"}
    assert lab_state["index"]["task"]["task_id"] == request.task_id
    assert lab_state["index"]["trajectory"]["meta_agent_run_count"] == 0
    assert lab_state["index"]["trajectory"]["subagent_run_count"] == 0
    assert lab_state["index"]["backend_states"] == []
    assert lab_state["digest"]["task_id"] == request.task_id
    assert "0 subagent report" in lab_state["digest"]["summary"]
    assert lab_state["requested_details"] == {}
    assert lab_state_registry.latest_index(request.task_id) is not None
    assert lab_state_registry.latest_digest(request.task_id) is not None


@DEFAULT_META_DISPATCH_REMOVED
def test_meta_agent_can_request_lab_state_detail_refs_for_next_dispatch(tmp_path: Path):
    request = _request()
    lab_state_registry = FileLabStateRegistry(tmp_path / "lab_state")
    lab_state_registry.save_subagent_report(
        SubagentReportRecord(
            report_ref="report-survey",
            task_id=request.task_id,
            run_ref="subagent-survey",
            role="SurveyAgent",
            status="completed",
            assigned_task="Survey inputs.",
            summary="Detailed survey report.",
            coverage={"processed_article_count": 2},
        )
    )

    class InspectingMetaLLMRuntime:
        def __init__(self):
            self.payloads: list[dict] = []

        def generate(
            self,
            messages: list[Message],
            tool_specs: list[dict],
            generation_config: LLMGenerationConfig,
        ) -> LLMRuntimeResponse:
            payload = json.loads(messages[-1].content)
            self.payloads.append(payload)
            if len(self.payloads) == 1:
                return LLMRuntimeResponse(
                    action=SubAgentAction(
                        action="final_answer",
                        content=json.dumps(
                            {
                                "action": "run_subagent",
                                "target_role": "SurveyAgent",
                                "instruction": "Inspect report details.",
                                "metadata": {
                                    "lab_state_detail_requests": {
                                        "subagent_reports": ["report-survey"],
                                    }
                                },
                            }
                        ),
                    ),
                )
            return LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "route": "END",
                            "instruction": "Done.",
                            "metadata": {"final_answer": "Done."},
                        }
                    ),
                ),
            )

    meta_llm = InspectingMetaLLMRuntime()
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        max_dispatch_steps=2,
        meta_agent=MetaAgentSpec(
            name="meta",
            system_prompt="Route to a subagent or END.",
            llm_backend=BackendBinding(backend_id="meta-llm"),
        ),
        roles={
            "SurveyAgent": RoleSpec(
                name="SurveyAgent",
                system_prompt="Survey lab state.",
                llm_backend=BackendBinding(backend_id="survey-llm"),
                allowed_tools=[],
            )
        },
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        lab_state_registry=lab_state_registry,
        llm_runtimes={
            "meta-llm": meta_llm,
            "survey-llm": RecordingLLMRuntime(),
        },
        memory_runtimes={"memory-local": RecordingMemoryRuntime()},
        skill_runtimes={"skill-local": RecordingSkillRuntime()},
    )

    runtime.run(request)

    assert meta_llm.payloads[0]["lab_state"]["requested_details"] == {}
    details = meta_llm.payloads[1]["lab_state"]["requested_details"]
    assert details["subagent_reports"][0]["report_ref"] == "report-survey"
    assert details["subagent_reports"][0]["summary"] == "Detailed survey report."


def test_default_task_runtime_applies_completion_guards_by_role(tmp_path: Path):
    request = _request()
    memory = RecordingMemoryRuntime()
    skill = RecordingSkillRuntime(required_tools=["write_jsonl"])
    tool_registry = ToolRegistry()
    tool_registry.register(
        ToolSpec(name="write_jsonl", description="write jsonl", parameters_schema={"type": "object"}),
        lambda arguments: ToolResult(
            call_id="write_jsonl",
            status="ok",
            content="wrote records",
            metadata={"record_count": len(arguments["records"])},
        ),
    )
    intake_llm = ScriptedLLMRuntime(
        [LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="Intake complete."))]
    )
    writer_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(
                        call_id="call-1",
                        name="write_jsonl",
                        arguments={"records": [{"name": "only-one"}]},
                    ),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="Writer complete.")),
        ]
    )
    trajectory_registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        roles={
            "intake": RoleSpec(
                name="intake",
                system_prompt="Read and summarize sources.",
                llm_backend=BackendBinding(backend_id="intake-llm"),
                allowed_tools=["write_jsonl"],
            ),
            "writer": RoleSpec(
                name="writer",
                system_prompt="Write final records.",
                llm_backend=BackendBinding(backend_id="writer-llm"),
                allowed_tools=["write_jsonl"],
            ),
        },
        runtime_policy=RuntimePolicy(
            max_tool_steps=1,
            metadata={
                "completion_guards_by_role": {
                    "writer": {
                        "required_tool_calls_before_final": ["write_jsonl"],
                        "minimum_jsonl_records_before_final": 2,
                    }
                }
            },
        ),
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(tool_registry),
        trajectory_registry=trajectory_registry,
        llm_runtimes={"intake-llm": intake_llm, "writer-llm": writer_llm},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": skill},
    )

    result = runtime.run(request)

    saved_runs = trajectory_registry.list_subagent_runs()
    assert result["status"] == "failed"
    assert [run.role for run in saved_runs] == ["intake", "writer"]
    assert saved_runs[0].metadata["status"] == "completed"
    assert saved_runs[1].metadata["status"] == "guard_failed"


def test_default_task_runtime_exposes_completion_guard_tools_to_role_llm(tmp_path: Path):
    request = _request()
    memory = RecordingMemoryRuntime()
    skill = RecordingSkillRuntime(required_tools=["read_text"])
    tool_registry = ToolRegistry()
    tool_registry.register(
        ToolSpec(name="read_text", description="read text", parameters_schema={"type": "object"}),
        lambda arguments: ToolResult(call_id="read_text", status="ok", content="read text"),
    )
    tool_registry.register(
        ToolSpec(name="list_files", description="list files", parameters_schema={"type": "object"}),
        lambda arguments: ToolResult(call_id="list_files", status="ok", content="listed files"),
    )

    class GuardToolAwareLLMRuntime:
        def __init__(self):
            self.calls = 0

        def generate(
            self,
            messages: list[Message],
            tool_specs: list[dict],
            generation_config: LLMGenerationConfig,
        ) -> LLMRuntimeResponse:
            self.calls += 1
            assert "list_files" in {spec["name"] for spec in tool_specs}
            if self.calls == 1:
                return LLMRuntimeResponse(
                    action=SubAgentAction(
                        action="tool_call",
                        tool_call=ToolCall(call_id="call-1", name="list_files", arguments={}),
                    )
                )
            return LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="Intake complete."))

    llm = GuardToolAwareLLMRuntime()
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        roles={
            "intake": RoleSpec(
                name="intake",
                system_prompt="Read and summarize sources.",
                llm_backend=BackendBinding(backend_id="intake-llm"),
                allowed_tools=["read_text", "list_files"],
            )
        },
        runtime_policy=RuntimePolicy(
            max_tool_steps=2,
            metadata={
                "completion_guards_by_role": {
                    "intake": {
                        "required_tool_calls_before_final": ["list_files"],
                    }
                }
            },
        ),
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(tool_registry),
        llm_runtimes={"intake-llm": llm},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": skill},
    )

    result = runtime.run(request)

    assert result["role"] == "intake"
    assert result["final_answer"] == "role completion guards satisfied after successful required tool calls"


def test_default_task_runtime_exposes_tool_result_metadata_to_next_llm_context(tmp_path: Path):
    request = _request()
    memory = RecordingMemoryRuntime()
    skill = RecordingSkillRuntime(required_tools=["read_text"])
    tool_registry = ToolRegistry()
    tool_registry.register(
        ToolSpec(name="read_text", description="read text", parameters_schema={"type": "object"}),
        lambda arguments: ToolResult(
            call_id="handler-call",
            status="ok",
            content="read 11 characters",
            metadata={"path": arguments["path"], "text": "Alpha beta\n"},
        ),
    )
    llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(
                        call_id="call-1",
                        name="read_text",
                        arguments={"path": "inputs/paper.md"},
                    ),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="Read the text.")),
        ]
    )
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="You solve scientific tasks.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["read_text"],
            )
        },
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(tool_registry),
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": skill},
    )

    runtime.run(request)

    tool_message = llm.calls[1][0][-1]
    assert tool_message.role == "tool"
    assert "read 11 characters" in tool_message.content
    assert '"text": "Alpha beta\\n"' in tool_message.content
    assert '"path": "inputs/paper.md"' in tool_message.content


def test_default_task_runtime_registers_tool_artifacts_and_records_them_in_trace(tmp_path: Path):
    request = _request()
    memory = RecordingMemoryRuntime()
    skill = RecordingSkillRuntime(required_tools=["lookup"])
    artifact = ArtifactRef(
        uri=str(tmp_path / "artifacts" / "lookup.json"),
        type="dataset",
        metadata={"role": "lookup_result"},
    )
    tool_registry = ToolRegistry()
    tool_registry.register(
        ToolSpec(name="lookup", description="lookup tool", parameters_schema={"type": "object"}),
        lambda arguments: ToolResult(
            call_id="local-handler-call",
            status="ok",
            content=f"wrote artifact for {arguments['query']}",
            artifact_refs=[artifact],
            metadata={"rows": 2},
        ),
    )
    llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(
                        call_id="call-1",
                        name="lookup",
                        arguments={"query": "catalyst"},
                    ),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(action="final_answer", content="Use the artifact."),
            ),
        ]
    )
    registered_results: list[ToolResult] = []
    trajectory_registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="You solve scientific tasks.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["lookup"],
            )
        },
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(tool_registry),
        tool_artifact_registrar=registered_results.append,
        trajectory_registry=trajectory_registry,
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": skill},
    )

    result = runtime.run(request)

    assert [registered.call_id for registered in registered_results] == ["call-1"]
    assert registered_results[0].artifact_refs == [artifact]
    saved = trajectory_registry.get_subagent_run(result["run_ref"])
    assert saved is not None
    trace_result = saved.metadata["tool_trace"]["calls"][0]["result"]
    assert trace_result["call_id"] == "call-1"
    assert trace_result["artifact_refs"] == [artifact.model_dump(mode="json")]
    assert trace_result["metadata"] == {"rows": 2}


def test_default_task_runtime_writes_subagent_report_artifact_and_training_indexes(tmp_path: Path):
    request = _request()
    memory = RecordingMemoryRuntime()
    skill = RecordingSkillRuntime(required_tools=["lookup"])
    artifact = ArtifactRef(
        uri=str(tmp_path / "artifacts" / "lookup.json"),
        type="dataset",
        metadata={"role": "lookup_result"},
    )
    tool_registry = ToolRegistry()
    tool_registry.register(
        ToolSpec(name="lookup", description="lookup tool", parameters_schema={"type": "object"}),
        lambda arguments: ToolResult(
            call_id="local-handler-call",
            status="ok",
            content="wrote artifact",
            artifact_refs=[artifact],
            metadata={"rows": 2},
        ),
    )
    llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="call-1", name="lookup", arguments={"query": "catalyst"}),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="Use the artifact.")),
        ]
    )
    lab_state = FileLabStateRegistry(tmp_path / "lab_state")
    trajectory_registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="You solve scientific tasks.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["lookup"],
            )
        },
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(tool_registry),
        trajectory_registry=trajectory_registry,
        lab_state_registry=lab_state,
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": skill},
    )

    result = runtime.run(request)

    reports = lab_state.list_subagent_reports(request.task_id)
    assert len(reports) == 1
    assert reports[0].run_ref == result["run_ref"]
    assert reports[0].role == "solver"
    assert reports[0].summary == "Use the artifact."
    assert reports[0].artifact_refs == [artifact]
    artifacts = lab_state.list_artifacts(request.task_id)
    assert len(artifacts) == 1
    assert artifacts[0].producer_run_ref == result["run_ref"]
    assert artifacts[0].uri == artifact.uri
    assert artifacts[0].artifact_type == "dataset"
    assert artifacts[0].role == "lookup_result"
    samples = lab_state.list_training_samples(request.task_id)
    assert len(samples) == 1
    assert samples[0].source_run_ref == result["run_ref"]
    assert samples[0].source_llm_call_refs == result["runs"][0]["llm_call_refs"]
    assert samples[0].sample_kind == "subagent_trace"


def test_default_task_runtime_subagent_report_includes_internal_dag_skill_and_coverage_summaries(tmp_path: Path):
    request = _request()
    memory = RecordingMemoryRuntime()
    skill = RecordingSkillRuntime(required_tools=["lookup"])
    tool_registry = ToolRegistry()
    tool_registry.register(
        ToolSpec(name="lookup", description="lookup tool", parameters_schema={"type": "object"}),
        lambda arguments: ToolResult(
            call_id="local-handler-call",
            status="ok",
            content="found rows",
            metadata={"rows": 2},
        ),
    )
    llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="call-1", name="lookup", arguments={"query": "catalyst"}),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "processed_article_count": 2,
                            "processed_table_count": 1,
                            "candidate_count": 1,
                            "failure_count": 0,
                            "records": [{"component_name": "pA"}],
                        }
                    ),
                )
            ),
        ]
    )
    lab_state = FileLabStateRegistry(tmp_path / "lab_state")
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        roles={
            "ExecAgent": RoleSpec(
                name="ExecAgent",
                system_prompt="Execute.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["lookup"],
            )
        },
        runtime_policy=RuntimePolicy(enable_workflow_planning=True),
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(tool_registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        lab_state_registry=lab_state,
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": skill},
    )

    runtime.run(request)

    report = lab_state.list_subagent_reports(request.task_id)[0]
    assert report.coverage == {
        "processed_article_count": 2,
        "processed_table_count": 1,
        "candidate_count": 1,
        "failure_count": 0,
    }
    assert report.metadata["internal_dag"] is None
    assert report.metadata["retrieved_skills"][0]["skill_id"] == "skill-1"
    assert report.metadata["tool_calls"][0]["tool_name"] == "lookup"
    assert report.metadata["artifacts"] == []


def test_default_task_runtime_writes_unregistered_tool_error_to_next_context(tmp_path: Path):
    request = _request()
    memory = RecordingMemoryRuntime()
    skill = RecordingSkillRuntime()
    llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="call-1", name="missing", arguments={}),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="Handled missing tool.")),
        ]
    )
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="You solve scientific tasks.",
                llm_backend=BackendBinding(backend_id="llm-local"),
            )
        },
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(ToolRegistry()),
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": skill},
    )

    runtime.run(request)

    tool_message = llm.calls[1][0][-1]
    assert tool_message.role == "tool"
    assert tool_message.name == "missing"
    assert tool_message.metadata["tool_result"]["status"] == "error"
    assert "not prepared" in tool_message.content
    assert tool_message.metadata["tool_result"]["metadata"]["error_type"] == "unprepared_tool"


def test_default_task_runtime_writes_tool_handler_exception_to_next_context(tmp_path: Path):
    request = _request()
    memory = RecordingMemoryRuntime()
    skill = RecordingSkillRuntime(required_tools=["lookup"])
    tool_registry = ToolRegistry()

    def raise_handler(arguments: dict) -> str:
        raise RuntimeError("lookup failed")

    tool_registry.register(
        ToolSpec(name="lookup", description="lookup tool", parameters_schema={"type": "object"}),
        raise_handler,
    )
    llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="call-1", name="lookup", arguments={}),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="Handled tool error.")),
        ]
    )
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="You solve scientific tasks.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["lookup"],
            )
        },
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(tool_registry),
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": skill},
    )

    runtime.run(request)

    tool_message = llm.calls[1][0][-1]
    assert tool_message.role == "tool"
    assert tool_message.metadata["tool_result"]["status"] == "error"
    assert tool_message.content == "lookup failed"


def test_default_task_runtime_records_valid_and_invalid_tool_traces_and_lab_artifacts(tmp_path: Path):
    request = _request()
    memory = RecordingMemoryRuntime()
    skill = RecordingSkillRuntime(required_tools=["write_file"])
    source_file = tmp_path / "tool-output.txt"
    tool_registry = ToolRegistry()

    def write_file(arguments: dict) -> ToolResult:
        source_file.write_text(arguments["content"], encoding="utf-8")
        return ToolResult(
            call_id="handler-call",
            status="ok",
            content="wrote file",
            artifact_refs=[ArtifactRef(uri=str(source_file), type="text")],
        )

    tool_registry.register(
        ToolSpec(name="write_file", description="write file", parameters_schema={"type": "object"}),
        write_file,
    )
    llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(
                        call_id="call-1",
                        name="write_file",
                        arguments={"content": "managed artifact"},
                    ),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="call-2", name="unprepared", arguments={}),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="done")),
        ]
    )
    trajectory_registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="You solve scientific tasks.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["write_file"],
            )
        },
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(tool_registry),
        trajectory_registry=trajectory_registry,
        tool_artifact_root_factory=lambda task_request, run_ref: tmp_path
        / "lab"
        / "tasks"
        / task_request.task_id
        / "artifacts"
        / run_ref,
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": skill},
    )

    result = runtime.run(request)

    saved = trajectory_registry.get_subagent_run(result["run_ref"])
    assert saved is not None
    assert [call.tool_call.name for call in saved.tool_calls] == ["write_file", "unprepared"]
    assert saved.tool_calls[0].result.status == "ok"
    assert saved.tool_calls[1].result.status == "error"
    assert saved.tool_calls[1].result.metadata["error_type"] == "unprepared_tool"
    managed_ref = saved.artifact_refs[0]
    assert managed_ref.metadata["lab_managed"] is True
    assert managed_ref.metadata["source_uri"] == str(source_file)
    assert Path(managed_ref.uri).is_file()
    assert str(Path(managed_ref.uri)).startswith(str(tmp_path / "lab" / "tasks" / request.task_id))
    assert Path(managed_ref.uri).read_text(encoding="utf-8") == "managed artifact"
    assert saved.metadata["tool_trace"]["calls"][0]["result"]["artifact_refs"][0]["uri"] == managed_ref.uri
    assert saved.metadata["tool_trace"]["calls"][1]["result"]["metadata"]["error_type"] == "unprepared_tool"


def test_default_task_runtime_saves_skill_update_result_and_backend_state_lineage(tmp_path: Path):
    request = _request()
    memory = RecordingMemoryRuntime()
    skill = UpdatingSkillRuntime()
    llm = RecordingLLMRuntime()
    trajectory_registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    backend_state_registry = FileBackendStateRegistry(tmp_path / "backend-state")
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="You solve scientific tasks.",
                llm_backend=BackendBinding(backend_id="llm-local"),
            )
        },
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        trajectory_registry=trajectory_registry,
        backend_state_registry=backend_state_registry,
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": skill},
    )

    result = runtime.run(request)

    saved = trajectory_registry.get_subagent_run(result["run_ref"])
    assert saved is not None
    assert saved.skill_bundle.skill_state_ref == "skill-state-v1"
    assert skill.look_at_events[0]["graph_version_ref"] == "skill-graph-v1"
    assert skill.look_at_events[0]["skill_state_ref"] == "skill-state-v1"
    assert saved.metadata["skill_observation_request"]["skill_state_ref"] == "skill-state-v1"
    assert saved.metadata["skill_update_result"] == {
        "schema_version": "v1",
        "status": "recorded",
        "update_summary": {"observed_run_ref": result["run_ref"]},
        "graph_version_ref": "skill-graph-v2",
        "skill_state_ref": "skill-state-v2",
        "artifact_refs": [],
        "metadata": {},
    }
    state_record = backend_state_registry.get_state("skill-state-v2")
    assert state_record is not None
    assert state_record.backend_id == "skill-local"
    assert state_record.backend_type == "skill"
    assert state_record.created_from_task_id == request.task_id
    assert state_record.created_from_run_ref == result["run_ref"]
    assert state_record.parent_state_refs == ["skill-state-v1"]
    assert state_record.metadata["graph_version_ref"] == "skill-graph-v2"
    assert state_record.metadata["update_summary"] == {"observed_run_ref": result["run_ref"]}

def test_default_task_runtime_reads_and_writes_agent_and_task_memory_scopes(tmp_path: Path):
    request = _request()
    agent_memory = RecordingMemoryRuntime(
        backend_id="agent-memory",
        state_ref="agent-state-before",
        memory_id="agent-memory-1",
        content="Agent Memory: solver prefers catalyst A.",
        metadata={"origin": "agent-note"},
        update_result={"status": "updated", "state_ref": "agent-state-after"},
    )
    task_memory = RecordingMemoryRuntime(
        backend_id="task-memory",
        state_ref="task-state-before",
        memory_id="task-memory-1",
        content="Task Memory: surveyor ruled out catalyst B.",
        metadata={"origin": "task-note"},
        update_result={"status": "updated", "state_ref": "task-state-after"},
    )
    skill = RecordingSkillRuntime()
    llm = RecordingLLMRuntime()
    trajectory_registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        task_memory_backend=BackendBinding(backend_id="task-memory"),
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="You solve scientific tasks.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                agent_memory_backend=BackendBinding(backend_id="agent-memory"),
            )
        },
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        trajectory_registry=trajectory_registry,
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"agent-memory": agent_memory, "task-memory": task_memory},
        skill_runtimes={"skill-local": skill},
    )

    result = runtime.run(request)

    assert result["task_id"] == request.task_id
    assert agent_memory.search_requests == []
    assert agent_memory.add_calls == []
    assert len(task_memory.search_requests) == 1
    task_request = task_memory.search_requests[0]
    assert task_request.task_id == request.task_id
    assert task_request.role == "task"
    assert task_request.query == request.goal
    assert task_request.filters["memory_scope"] == "task"
    assert task_request.filters["memory_scope_id"] == "task:task-1"

    assert len(llm.calls) == 1
    prompt_text = "\n".join(message.content for message in llm.calls[0][0])
    assert "Task Memory" in prompt_text
    assert "surveyor ruled out catalyst B" in prompt_text

    assert len(task_memory.add_calls) == 1
    add_task_id, add_role, add_messages = task_memory.add_calls[0]
    assert (add_task_id, add_role) == (request.task_id, "task")
    assert "Use catalyst A." in add_messages[-1].content

    saved = trajectory_registry.get_subagent_run(result["run_ref"])
    assert saved is not None
    saved_memory_by_id = {item.memory_id: item for item in saved.memory_bundle.items}
    assert set(saved_memory_by_id) == {"task-memory-1"}
    assert saved_memory_by_id["task-memory-1"].score == 0.8
    assert saved_memory_by_id["task-memory-1"].metadata["origin"] == "task-note"
    assert saved.metadata["task_memory_bundle"]["state_ref"] == "task-state-before"
    assert saved.metadata["task_memory_update_result"] == {
        "status": "updated",
        "state_ref": "task-state-after",
    }


def test_default_task_runtime_records_memory_state_lineage(tmp_path: Path):
    request = _request()
    agent_memory = RecordingMemoryRuntime(
        backend_id="agent-memory",
        state_ref="agent-state-before",
        update_result={"status": "updated", "state_ref": "agent-state-after"},
    )
    task_memory = RecordingMemoryRuntime(
        backend_id="task-memory",
        state_ref="task-state-before",
        update_result={"status": "updated", "state_ref": "task-state-after"},
    )
    backend_state_registry = FileBackendStateRegistry(tmp_path / "backend-state")
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        task_memory_backend=BackendBinding(backend_id="task-memory"),
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="You solve scientific tasks.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                agent_memory_backend=BackendBinding(backend_id="agent-memory"),
            )
        },
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        backend_state_registry=backend_state_registry,
        llm_runtimes={"llm-local": RecordingLLMRuntime()},
        memory_runtimes={"agent-memory": agent_memory, "task-memory": task_memory},
        skill_runtimes={"skill-local": RecordingSkillRuntime()},
    )

    result = runtime.run(request)

    records = {
        (record.backend_id, record.state_ref): record
        for record in backend_state_registry.list_states()
    }
    assert ("agent-memory", "agent-state-after") not in records
    task_record = records[("task-memory", "task-state-after")]
    assert task_record.backend_type == "memory"
    assert task_record.created_from_task_id == request.task_id
    assert task_record.created_from_run_ref == result["run_ref"]
    assert task_record.parent_state_refs == ["task-state-before"]
    assert task_record.active is True
    assert task_record.metadata["memory_scope"] == "task"
    assert task_record.metadata["memory_scope_id"] == "task:task-1"
    assert task_record.metadata["role"] == "solver"
    assert task_record.metadata["update_result"] == {
        "status": "updated",
        "state_ref": "task-state-after",
    }


def test_default_task_runtime_records_mem0_agent_and_task_memory_lineage(tmp_path: Path):
    request = _request("mem0-task")
    agent_memory = _native_mem0_backend(tmp_path, "mem0-agent")
    task_memory = _native_mem0_backend(tmp_path, "mem0-task")
    solver_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(action="final_answer", content="Solver recorded first task finding."),
            )
        ]
    )
    reviewer_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(action="final_answer", content="Reviewer used shared task finding."),
            )
        ]
    )
    trajectory_registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    backend_state_registry = FileBackendStateRegistry(tmp_path / "backend-state")
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        task_memory_backend=BackendBinding(backend_id="mem0-task"),
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="You solve scientific tasks.",
                llm_backend=BackendBinding(backend_id="solver-llm"),
                agent_memory_backend=BackendBinding(backend_id="mem0-agent"),
            ),
            "reviewer": RoleSpec(
                name="reviewer",
                system_prompt="You review scientific answers.",
                llm_backend=BackendBinding(backend_id="reviewer-llm"),
                agent_memory_backend=BackendBinding(backend_id="mem0-agent"),
            ),
        },
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        trajectory_registry=trajectory_registry,
        backend_state_registry=backend_state_registry,
        llm_runtimes={"solver-llm": solver_llm, "reviewer-llm": reviewer_llm},
        memory_runtimes={"mem0-agent": agent_memory, "mem0-task": task_memory},
        skill_runtimes={"skill-local": RecordingSkillRuntime()},
    )

    result = runtime.run(request)

    assert [run["role"] for run in result["runs"]] == ["solver", "reviewer"]
    saved_runs = trajectory_registry.list_subagent_runs()
    assert "agent_memory_bundle" not in saved_runs[0].metadata
    assert saved_runs[0].metadata["task_memory_bundle"]["state_ref"] == (
        "method://mem0/9:mem0-task/4:task/14:task:mem0-task/v0"
    )
    assert saved_runs[0].metadata["task_memory_update_result"]["state_ref"] == (
        "method://mem0/9:mem0-task/4:task/14:task:mem0-task/v2"
    )
    assert "agent_memory_bundle" not in saved_runs[1].metadata
    assert saved_runs[1].metadata["task_memory_bundle"]["state_ref"] == (
        "method://mem0/9:mem0-task/4:task/14:task:mem0-task/v2"
    )
    assert saved_runs[1].metadata["task_memory_update_result"]["state_ref"] == (
        "method://mem0/9:mem0-task/4:task/14:task:mem0-task/v3"
    )
    assert "Solver recorded first task finding." in "\n".join(message.content for message in saved_runs[1].prompt_messages)

    memory_records = [record for record in backend_state_registry.list_states() if record.backend_type == "memory"]
    assert [(record.backend_id, record.metadata["memory_scope"], record.metadata["memory_scope_id"]) for record in memory_records] == [
        ("mem0-task", "task", "task:mem0-task"),
        ("mem0-task", "task", "task:mem0-task"),
    ]
    assert all(record.backend_id != "mem0-agent" for record in memory_records)
    assert memory_records[0].parent_state_refs == ["method://mem0/9:mem0-task/4:task/14:task:mem0-task/v0"]
    assert memory_records[1].parent_state_refs == ["method://mem0/9:mem0-task/4:task/14:task:mem0-task/v2"]

def test_default_task_runtime_runs_each_configured_role_in_order(tmp_path: Path):
    request = _request()
    memory = RecordingMemoryRuntime()
    skill = RecordingSkillRuntime()
    solver_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(action="final_answer", content="Solver answer."),
            )
        ]
    )
    reviewer_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(action="final_answer", content="Reviewer answer."),
            )
        ]
    )
    trajectory_registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="You solve scientific tasks.",
                llm_backend=BackendBinding(backend_id="solver-llm"),
            ),
            "reviewer": RoleSpec(
                name="reviewer",
                system_prompt="You review scientific answers.",
                llm_backend=BackendBinding(backend_id="reviewer-llm"),
            ),
        },
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        trajectory_registry=trajectory_registry,
        llm_runtimes={"solver-llm": solver_llm, "reviewer-llm": reviewer_llm},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": skill},
    )

    result = runtime.run(request)

    assert result["run_refs"] == [run["run_ref"] for run in result["runs"]]
    assert [run["role"] for run in result["runs"]] == ["solver", "reviewer"]
    assert result["run_ref"] == result["runs"][-1]["run_ref"]
    assert result["role"] == "reviewer"
    assert result["final_answer"] == "Reviewer answer."
    saved = trajectory_registry.list_subagent_runs()
    assert [record.role for record in saved] == ["solver", "reviewer"]
    assert [record.stage_index for record in saved] == [0, 1]
    assert [record.output_messages[0].content for record in saved] == [
        "Solver answer.",
        "Reviewer answer.",
    ]
    assert [call[1] for call in memory.add_calls] == ["task", "task"]


@DEFAULT_META_DISPATCH_REMOVED
def test_default_task_runtime_uses_meta_agent_dispatch_and_records_decisions(tmp_path: Path):
    request = _request()
    memory = RecordingMemoryRuntime()
    skill = RecordingSkillRuntime()
    meta_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "action": "run_subagent",
                            "target_role": "solver",
                            "instruction": "Read the evidence file before answering.",
                            "retrieval_query": "catalyst evidence",
                        }
                    ),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "action": "run_subagent",
                            "target_role": "reviewer",
                            "instruction": "Review the solver recommendation.",
                        }
                    ),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "action": "finish_task",
                            "metadata": {"final_answer": "Reviewed catalyst A recommendation."},
                        }
                    ),
                )
            ),
        ]
    )
    solver_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(action="final_answer", content="Solver recommends catalyst A."),
            )
        ]
    )
    reviewer_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(action="final_answer", content="Reviewer accepts catalyst A."),
            )
        ]
    )
    trajectory_registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        max_dispatch_steps=3,
        meta_agent=MetaAgentSpec(
            name="meta",
            system_prompt="Dispatch solver then reviewer and finish.",
            llm_backend=BackendBinding(backend_id="meta-llm"),
        ),
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="You solve scientific tasks.",
                llm_backend=BackendBinding(backend_id="solver-llm"),
            ),
            "reviewer": RoleSpec(
                name="reviewer",
                system_prompt="You review scientific answers.",
                llm_backend=BackendBinding(backend_id="reviewer-llm"),
            ),
        },
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        trajectory_registry=trajectory_registry,
        llm_runtimes={
            "meta-llm": meta_llm,
            "solver-llm": solver_llm,
            "reviewer-llm": reviewer_llm,
        },
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": skill},
    )

    result = runtime.run(request)

    assert result["role"] == "reviewer"
    assert result["final_answer"] == "Reviewed catalyst A recommendation."
    assert [run["role"] for run in result["runs"]] == ["solver", "reviewer"]
    assert len(result["meta_run_refs"]) == 3
    meta_records = trajectory_registry.list_meta_agent_runs()
    assert [record.decision.action.value for record in meta_records] == [
        "run_subagent",
        "run_subagent",
        "finish_task",
    ]
    llm_calls = trajectory_registry.list_llm_calls()
    assert len(llm_calls) == 5
    assert [record.metadata["llm_call_refs"][0] for record in meta_records] == [
        llm_calls[0].call_ref,
        llm_calls[2].call_ref,
        llm_calls[4].call_ref,
    ]
    assert [call.backend_id for call in llm_calls] == [
        "meta-llm",
        "solver-llm",
        "meta-llm",
        "reviewer-llm",
        "meta-llm",
    ]
    assert [record.decision.target_role for record in meta_records[:2]] == ["solver", "reviewer"]
    saved = trajectory_registry.list_subagent_runs()
    assert [record.llm_call_refs for record in saved] == [[llm_calls[1].call_ref], [llm_calls[3].call_ref]]
    assert [record.instruction for record in saved] == [
        "Read the evidence file before answering.",
        "Review the solver recommendation.",
    ]
    assert saved[0].retrieval_request.query == "catalyst evidence"


@DEFAULT_META_DISPATCH_REMOVED
def test_meta_agent_updates_agents_md_before_subagent_dispatch(tmp_path: Path):
    request = _request()
    agents_path = tmp_path / "agents.md"
    agents_path.write_text(
        render_agents_markdown(
            {
                "solver": RoleSpec(
                    name="solver",
                    system_prompt="Old solver prompt.",
                    llm_backend=BackendBinding(backend_id="solver-llm"),
                    allowed_tools=["read_text"],
                )
            }
        ),
        encoding="utf-8",
    )
    meta_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "action": "run_subagent",
                            "target_role": "solver",
                            "instruction": "Solve with the updated prompt.",
                            "metadata": {
                                "agent_config_update": {
                                    "reason": "Previous feedback requires stricter evidence use.",
                                    "roles": {
                                        "solver": {
                                            "system_prompt": "Evolved solver prompt.",
                                            "system_prompt_append": "Append reflector note.",
                                            "allowed_tools": ["read_text", "write_report"],
                                            "required_skills": ["skill.evidence.v1"],
                                            "memory_policy": {"prefer_recent_failures": True},
                                            "metadata": {"evolved_by": "meta"},
                                        }
                                    },
                                }
                            },
                        }
                    ),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps({"action": "finish_task", "metadata": {"final_answer": "done"}}),
                )
            ),
        ]
    )
    solver_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(action="final_answer", content="Solved with updated prompt."),
            )
        ]
    )
    trajectory_registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        agents_ref=str(agents_path),
        max_dispatch_steps=2,
        meta_agent=MetaAgentSpec(
            name="meta",
            system_prompt="Dispatch solver and evolve agents.md when feedback requires it.",
            llm_backend=BackendBinding(backend_id="meta-llm"),
        ),
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        trajectory_registry=trajectory_registry,
        llm_runtimes={"meta-llm": meta_llm, "solver-llm": solver_llm},
        memory_runtimes={"memory-local": RecordingMemoryRuntime()},
        skill_runtimes={"skill-local": RecordingSkillRuntime()},
    )

    result = runtime.run(request)

    assert result["final_answer"] == "done"
    assert solver_llm.calls[0][0][0].content == "Evolved solver prompt.\n\nAppend reflector note."
    refreshed_prompt = json.loads(meta_llm.calls[1][0][-1].content)
    assert "Evolved solver prompt." in refreshed_prompt["agents_config"]["agents_md"]
    updated_roles = json.loads(agents_path.read_text(encoding="utf-8").split("```json", 1)[1].split("```", 1)[0])
    updated_solver = updated_roles["agents"][0]
    assert updated_solver["system_prompt"] == "Evolved solver prompt.\n\nAppend reflector note."
    assert updated_solver["allowed_tools"] == ["read_text", "write_report"]
    assert updated_solver["required_skills"] == ["skill.evidence.v1"]
    meta_record = trajectory_registry.list_meta_agent_runs()[0]
    update_result = meta_record.metadata["dispatch_metadata"]["agent_config_update_result"]
    assert update_result["status"] == "updated"
    assert update_result["updated_roles"] == ["solver"]
    assert (tmp_path / "agents.md.updates.jsonl").is_file()


@DEFAULT_META_DISPATCH_REMOVED
def test_subagent_prompt_includes_active_role_specific_prompt_overlay(tmp_path: Path):
    request = _request()
    meta_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "action": "run_subagent",
                            "target_role": "solver",
                            "instruction": "Solve with evolved prompt overlay.",
                        }
                    ),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps({"action": "finish_task", "metadata": {"final_answer": "done"}}),
                )
            ),
        ]
    )
    solver_llm = ScriptedLLMRuntime(
        [LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="Solved with overlay."))]
    )
    backend_state_registry = FileBackendStateRegistry(tmp_path / "backend_state")
    state = BackendStateRecord(
        state_ref="prompt-overlay://solver-llm/solver/abc123",
        backend_id="solver-llm",
        backend_type="llm",
        metadata={
            "state_kind": "prompt_overlay",
            "prompt_overlay": {
                "role": "solver",
                "system_prompt_append": "Reflector-guided prompt evolution notes:\n- Inspect all supplementary tables.",
            },
        },
    )
    backend_state_registry.register_candidate(state)
    backend_state_registry.promote("solver-llm", state.state_ref, "evo-1")
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        max_dispatch_steps=2,
        meta_agent=MetaAgentSpec(
            name="meta",
            system_prompt="Dispatch solver then finish.",
            llm_backend=BackendBinding(backend_id="meta-llm"),
        ),
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="Base solver prompt.",
                llm_backend=BackendBinding(backend_id="solver-llm"),
            )
        },
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        backend_state_registry=backend_state_registry,
        llm_runtimes={"meta-llm": meta_llm, "solver-llm": solver_llm},
        memory_runtimes={"memory-local": RecordingMemoryRuntime()},
        skill_runtimes={"skill-local": RecordingSkillRuntime()},
    )

    result = runtime.run(request)

    assert result["final_answer"] == "done"
    system_prompt = solver_llm.calls[0][0][0].content
    assert "Base solver prompt." in system_prompt
    assert "Reflector-guided prompt evolution notes" in system_prompt
    assert "Inspect all supplementary tables." in system_prompt


@DEFAULT_META_DISPATCH_REMOVED
def test_meta_agent_prompt_ref_is_hot_read_and_evolvable(tmp_path: Path):
    request = _request()
    prompt_path = tmp_path / "meta_prompt.md"
    prompt_path.write_text("Initial MetaAgent prompt.", encoding="utf-8")
    meta_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "action": "run_subagent",
                            "target_role": "solver",
                            "instruction": "Solve once.",
                            "metadata": {
                                "meta_agent_prompt_update": {
                                    "reason": "Route feedback says use a stricter router prompt.",
                                    "prompt": "Evolved MetaAgent prompt.",
                                }
                            },
                        }
                    ),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps({"action": "finish_task", "metadata": {"final_answer": "done"}}),
                )
            ),
        ]
    )
    solver_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(action="final_answer", content="Solved."),
            )
        ]
    )
    trajectory_registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        max_dispatch_steps=2,
        meta_agent=MetaAgentSpec(
            name="meta",
            prompt_ref=str(prompt_path),
            llm_backend=BackendBinding(backend_id="meta-llm"),
        ),
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="Solve tasks.",
                llm_backend=BackendBinding(backend_id="solver-llm"),
            ),
        },
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        trajectory_registry=trajectory_registry,
        llm_runtimes={"meta-llm": meta_llm, "solver-llm": solver_llm},
        memory_runtimes={"memory-local": RecordingMemoryRuntime()},
        skill_runtimes={"skill-local": RecordingSkillRuntime()},
    )

    result = runtime.run(request)

    assert result["final_answer"] == "done"
    assert meta_llm.calls[0][0][0].content == "Initial MetaAgent prompt."
    assert meta_llm.calls[1][0][0].content == "Evolved MetaAgent prompt."
    assert prompt_path.read_text(encoding="utf-8") == "Evolved MetaAgent prompt."
    meta_record = trajectory_registry.list_meta_agent_runs()[0]
    update_result = meta_record.metadata["dispatch_metadata"]["meta_agent_prompt_update_result"]
    assert update_result["status"] == "updated"
    assert (tmp_path / "meta_prompt.md.updates.jsonl").is_file()


@DEFAULT_META_DISPATCH_REMOVED
def test_reflector_evaluates_task_with_ground_truth_and_guides_next_meta_prompt(tmp_path: Path):
    request = _request()
    meta_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "action": "run_subagent",
                            "target_role": "solver",
                            "instruction": "Solve with evidence.",
                        }
                    ),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps({"action": "finish_task", "metadata": {"final_answer": "done"}}),
                )
            ),
        ]
    )
    solver_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(action="final_answer", content="Catalyst A is best."),
            )
        ]
    )
    reflector_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "score": 1.0,
                            "passed": True,
                            "summary": "Matches the ground truth.",
                            "errors": [],
                            "credit_assignment": {"solver": "correct"},
                            "evolution_recommendations": {
                                "meta_agent": "Keep routing to solver first.",
                                "subagents": {"solver": "No change needed."},
                            },
                        }
                    ),
                )
            )
        ]
    )
    trajectory_registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        max_dispatch_steps=2,
        meta_agent=MetaAgentSpec(
            name="meta",
            system_prompt="Return dispatch JSON.",
            llm_backend=BackendBinding(backend_id="meta-llm"),
        ),
        reflector=ReflectorSpec(
            name="reflector",
            system_prompt="Evaluate the completed task against ground truth.",
            llm_backend=BackendBinding(backend_id="reflector-llm"),
            ground_truth={"answer": "Catalyst A is best."},
            rubric="Score exact answer correctness.",
        ),
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="Solve tasks.",
                llm_backend=BackendBinding(backend_id="solver-llm"),
            ),
        },
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        trajectory_registry=trajectory_registry,
        llm_runtimes={
            "meta-llm": meta_llm,
            "solver-llm": solver_llm,
            "reflector-llm": reflector_llm,
        },
        memory_runtimes={"memory-local": RecordingMemoryRuntime()},
        skill_runtimes={"skill-local": RecordingSkillRuntime()},
    )

    result = runtime.run(request)

    assert result["reflector_evaluation_status"] == "completed"
    assert result["reflector_evaluation"]["score"] == 1.0
    reflector_prompt = json.loads(reflector_llm.calls[0][0][-1].content)
    assert reflector_prompt["ground_truth"]["content"] == {"answer": "Catalyst A is best."}
    assert reflector_prompt["task_result"]["runs"][0]["role"] == "solver"
    events = trajectory_registry.list_events()
    assert [event.event_type for event in events if event.event_type == "reflector_evaluation"] == [
        "reflector_evaluation"
    ]

    next_meta_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "action": "run_subagent",
                            "target_role": "solver",
                            "instruction": "Solve after feedback.",
                        }
                    ),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps({"action": "finish_task", "metadata": {"final_answer": "done"}}),
                )
            ),
        ]
    )
    next_solver_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(action="final_answer", content="Catalyst A remains best."),
            )
        ]
    )
    next_runtime = TaskRuntime(
        task_config=task_config.model_copy(update={"reflector": None}),
        prompt_builder=PromptBuilder(),
        trajectory_registry=trajectory_registry,
        llm_runtimes={"meta-llm": next_meta_llm, "solver-llm": next_solver_llm},
        memory_runtimes={"memory-local": RecordingMemoryRuntime()},
        skill_runtimes={"skill-local": RecordingSkillRuntime()},
    )

    next_runtime.run(_request("task-2"))

    next_meta_prompt = json.loads(next_meta_llm.calls[0][0][-1].content)
    assert next_meta_prompt["recent_reflector_feedback"][0]["evaluation"]["summary"] == "Matches the ground truth."


def test_reflector_payload_includes_bounded_final_artifact_preview(tmp_path: Path):
    final_records = tmp_path / "final_records.jsonl"
    validated_records = tmp_path / "validated_records.json"
    records = [
        {"article_id": "article-a", "sequence": "AACCGGTT", "component_name": "pA"},
        {"article_id": "article-a", "sequence": "TTGGCCAA", "component_name": "pB"},
    ]
    extra_validated = [*records, {"article_id": "article-a", "sequence": "CCCCAAAA", "component_name": "pC"}]
    final_records.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    validated_records.write_text(
        json.dumps({"accepted_records": extra_validated, "accepted_count": 3}, sort_keys=True),
        encoding="utf-8",
    )

    payload = _reflector_result_payload(
        {
            "task_id": "task-1",
            "status": "completed",
            "runs": [
                {
                    "run_ref": "writer-run",
                    "role": "SchemaWriterAgent",
                    "status": "completed",
                    "final_answer": "The final artifact is empty.",
                    "artifact_refs": [
                        ArtifactRef(
                            uri=str(final_records),
                            type="dataset",
                            metadata={
                                "filename": "final_records.jsonl",
                                "artifact_kind": "final_records",
                                "record_count": 2,
                            },
                        ).model_dump(mode="json"),
                        ArtifactRef(
                            uri=str(validated_records),
                            type="dataset",
                            metadata={
                                "filename": "validated_records.json",
                                "artifact_kind": "validated_records",
                                "record_count": 3,
                            },
                        ).model_dump(mode="json")
                    ],
                }
            ],
        }
    )

    preview = payload["runs"][0]["artifact_previews"][0]
    assert preview["filename"] == "final_records.jsonl"
    assert preview["metadata_record_count"] == 2
    assert preview["record_count"] == 2
    assert preview["records"] == records
    assert payload["final_predictions"]["record_count"] == 2
    assert payload["final_predictions"]["records"] == records


def test_reflector_overrides_llm_sequence_metrics_with_runtime_computed_metrics():
    evaluation = {
        "score": 1.0,
        "passed": True,
        "summary": "LLM claims all records matched.",
        "metrics": {
            "gt_count": 2,
            "predicted_count": 2,
            "true_positive": 2,
            "false_positive": 0,
            "false_negative": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        },
    }

    updated = _apply_reflector_computed_metrics(
        evaluation,
        task_result={
            "final_predictions": {
                "record_count": 1,
                "records": [{"article_id": "article-a", "sequence": "AACCGGTT"}],
                "truncated": False,
            }
        },
        ground_truth={
            "content": {
                "ground_truth_records": [
                    {"article_id": "article-a", "sequence": "AACCGGTT"},
                    {"article_id": "article-a", "sequence": "CCCCAAAA"},
                ]
            }
        },
    )

    assert updated["metric_source"] == "runtime_sequence_evaluator"
    assert updated["llm_reported_metrics"] == evaluation["metrics"]
    assert updated["llm_reported_summary"] == "LLM claims all records matched."
    assert "precision=1.0, recall=0.5, f1=0.666667" in updated["summary"]
    assert updated["score"] == 0.666667
    assert updated["passed"] is False
    assert updated["metrics"] == {
        "gt_count": 2,
        "predicted_count": 1,
        "true_positive": 1,
        "false_positive": 0,
        "false_negative": 1,
        "precision": 1.0,
        "recall": 0.5,
        "f1": 0.666667,
    }
    assert updated["sequence_error_analysis"]["matched_examples"][0]["match_type"] == "exact"
    assert updated["sequence_error_analysis"]["false_negative_examples"] == [
        {
            "sequence": "CCCCAAAA",
            "ground_truth_record": {"article_id": "article-a"},
        }
    ]
    assert updated["specific_evolution_instructions"][0]["stage"] == "source_discovery_and_candidate_extraction"
    assert "ground-truth sequence(s) missing" in updated["errors"][-1]


def test_reflector_runtime_metrics_read_full_final_artifact_when_preview_is_truncated(tmp_path: Path):
    final_records = tmp_path / "final_records.jsonl"
    alphabet = "ACGT"
    records = [
        {
            "article_id": "article-a",
            "work_item_id": "article-a",
            "sequence": "AACCGGTT" + "".join(alphabet[(i >> shift) & 3] for shift in (0, 2, 4)),
        }
        for i in range(30)
    ]
    final_records.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    payload = _reflector_result_payload(
        {
            "task_id": "task-1",
            "status": "completed",
            "runs": [
                {
                    "run_ref": "writer-run",
                    "role": "SchemaWriterAgent",
                    "status": "completed",
                    "artifact_refs": [
                        ArtifactRef(
                            uri=str(final_records),
                            type="dataset",
                            metadata={
                                "filename": "final_records.jsonl",
                                "artifact_kind": "final_records",
                                "record_count": 30,
                            },
                        ).model_dump(mode="json")
                    ],
                }
            ],
        }
    )
    assert payload["final_predictions"]["truncated"] is True
    assert payload["final_predictions"]["preview_record_count"] == 25

    updated = _apply_reflector_computed_metrics(
        {"summary": "LLM could not score.", "metrics": None},
        task_result=payload,
        ground_truth={"content": {"ground_truth_records": records}},
    )

    assert updated["metric_source"] == "runtime_sequence_evaluator"
    assert updated["metrics"]["predicted_count"] == 30
    assert updated["metrics"]["true_positive"] == 30
    assert updated["metrics"]["recall"] == 1.0


def test_reflector_runtime_sequence_evaluation_is_article_aligned_and_deduped():
    evaluation = _reflector_sequence_evaluation(
        {
            "final_predictions": {
                "records": [
                    {"article_id": "article-a", "sequence": "AACCGGTT"},
                    {"article_id": "article-a", "sequence": "AACCGGTT"},
                    {"article_id": "article-b", "sequence": "AACCGGTT"},
                    {"article_id": "article-b", "sequence": "CCCCAAAA"},
                ]
            }
        },
        {
            "content": {
                "ground_truth_records": [
                    {"article_id": "article-a", "sequence": "AACCGGTT"},
                    {"article_id": "article-b", "sequence": "GGGGAAAA"},
                ]
            }
        },
    )

    assert evaluation is not None
    assert evaluation["metrics"] == {
        "gt_count": 2,
        "predicted_count": 3,
        "true_positive": 1,
        "false_positive": 2,
        "false_negative": 1,
        "precision": 0.333333,
        "recall": 0.5,
        "f1": 0.4,
    }
    assert evaluation["sequence_error_analysis"]["false_positive_examples"][0]["prediction_record"][
        "article_id"
    ] == "article-b"
    assert evaluation["sequence_error_analysis"]["false_negative_examples"][0]["ground_truth_record"][
        "article_id"
    ] == "article-b"


def test_reflector_runtime_sequence_evaluation_handles_large_indexed_matching():
    def sequence_for(index: int) -> str:
        alphabet = "ACGT"
        suffix = "".join(alphabet[(index >> shift) & 3] for shift in range(0, 18, 2))
        return f"AAAACCCC{suffix}"

    gt_records = [
        {"article_id": "article-a", "sequence": sequence_for(i)}
        for i in range(500)
    ]
    predictions = [
        {"article_id": "article-a", "sequence": record["sequence"]}
        for record in gt_records
    ] + [{"article_id": "article-a", "sequence": "CCCCGGGGTTTTAAAA"}]

    evaluation = _reflector_sequence_evaluation(
        {"final_predictions": {"records": predictions}},
        {"content": {"ground_truth_records": gt_records}},
    )

    assert evaluation is not None
    assert evaluation["metrics"]["gt_count"] == 500
    assert evaluation["metrics"]["predicted_count"] == 501
    assert evaluation["metrics"]["true_positive"] == 500
    assert evaluation["metrics"]["false_positive"] == 1
    assert evaluation["metrics"]["false_negative"] == 0


def test_reflector_llm_payload_compacts_large_sequence_ground_truth_and_artifacts(tmp_path: Path):
    final_records = tmp_path / "final_records.jsonl"
    records = [
        {
            "article_id": "article-a",
            "work_item_id": "article-a",
            "sequence": f"AACCGGTT{i:03d}",
            "component_name": f"p{i}",
        }
        for i in range(30)
    ]
    final_records.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    task_result = _reflector_result_payload(
        {
            "task_id": "task-1",
            "status": "completed",
            "runs": [
                {
                    "run_ref": "writer-run",
                    "role": "SchemaWriterAgent",
                    "status": "completed",
                    "assigned_task": "x" * 100_000,
                    "artifact_refs": [
                        ArtifactRef(
                            uri=str(final_records),
                            type="dataset",
                            metadata={
                                "filename": "final_records.jsonl",
                                "artifact_kind": "final_records",
                                "record_count": 30,
                            },
                        ).model_dump(mode="json")
                    ],
                }
            ],
        }
    )
    ground_truth = {
        "source_ref": "gt.json",
        "path": str(tmp_path / "gt.json"),
        "content": {
            "article_count": 1,
            "articles": [{"work_item_id": "article-a"}],
            "ground_truth_records": records,
        },
    }
    runtime_eval = {
        "metrics": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
        "sequence_error_analysis": {
            "matched_examples": [{"prediction_sequence": record["sequence"]} for record in records],
            "false_positive_examples": [{"sequence": "TTTT" + str(i)} for i in range(40)],
            "false_negative_examples": [{"sequence": "CCCC" + str(i)} for i in range(40)],
            "matched_example_count": 30,
            "false_positive_example_count": 40,
            "false_negative_example_count": 40,
        },
        "specific_evolution_instructions": [{"instruction": f"fix {i}"} for i in range(20)],
        "deterministic_errors": [],
    }

    payload = _compact_reflector_llm_payload(
        task_id="task-1",
        goal="Evaluate.",
        ground_truth=ground_truth,
        rubric={"source_ref": None, "path": None, "content": "Score."},
        task_result=task_result,
        runtime_sequence_evaluation=runtime_eval,
        recent_reflector_feedback=[],
    )

    assert "ground_truth_records" not in json.dumps(payload["ground_truth"])
    assert payload["ground_truth"]["content_summary"]["record_count"] == 30
    assert "assigned_task" not in payload["task_result"]["runs"][0]
    assert "records" not in payload["task_result"]["runs"][0]["artifact_previews"][0]
    assert payload["task_result"]["final_predictions"]["source_uris"] == [str(final_records)]
    assert len(payload["runtime_sequence_evaluation"]["sequence_error_analysis"]["false_positive_examples"]) == 20
    assert len(payload["runtime_sequence_evaluation"]["sequence_error_analysis"]["false_negative_examples"]) == 20
    assert len(payload["runtime_sequence_evaluation"]["specific_evolution_instructions"]) == 8
    assert len(json.dumps(payload)) < len(json.dumps(ground_truth)) + len(json.dumps(task_result))


def test_latest_scientific_artifact_uses_matching_unscoped_richer_candidate_records(tmp_path: Path):
    scoped_candidate = tmp_path / "work-item-a" / "candidate_records.json"
    scoped_candidate.parent.mkdir()
    scoped_candidate.write_text(
        json.dumps(
            {
                "records": [
                    {"work_item_id": "work-item-a", "article_id": "work-item-a", "sequence": "AAAACCCC"}
                ]
            }
        ),
        encoding="utf-8",
    )
    global_candidate = tmp_path / "candidate_records.json"
    global_candidate.write_text(
        json.dumps(
            {
                "records": [
                    {"work_item_id": "work-item-a", "article_id": "work-item-a", "sequence": "AAAACCCC"},
                    {"work_item_id": "work-item-a", "article_id": "work-item-a", "sequence": "GGGGTTTT"},
                ]
            }
        ),
        encoding="utf-8",
    )
    other_candidate = tmp_path / "other_candidate_records.json"
    other_candidate.write_text(
        json.dumps(
            {
                "records": [
                    {"work_item_id": "work-item-b", "article_id": "work-item-b", "sequence": "CCCCGGGG"},
                    {"work_item_id": "work-item-b", "article_id": "work-item-b", "sequence": "TTTTAAAA"},
                    {"work_item_id": "work-item-b", "article_id": "work-item-b", "sequence": "AACCGGTT"},
                ]
            }
        ),
        encoding="utf-8",
    )
    registry = _ListArtifactRegistry(
        [
            SimpleNamespace(
                uri=str(scoped_candidate),
                metadata={
                    "filename": "candidate_records.json",
                    "artifact_kind": "candidate_records",
                    "work_item_id": "work-item-a",
                },
            ),
            SimpleNamespace(
                uri=str(other_candidate),
                metadata={"filename": "candidate_records.json", "artifact_kind": "candidate_records"},
            ),
            SimpleNamespace(
                uri=str(global_candidate),
                metadata={"filename": "candidate_records.json", "artifact_kind": "candidate_records"},
            ),
        ]
    )

    selected = _latest_scientific_artifact_uri(
        registry,
        task_id="task-1",
        work_item_id="work-item-a",
        artifact_kind="candidate_records",
        filename="candidate_records.json",
    )

    assert selected == str(global_candidate)


def test_final_records_bootstrap_merges_scoped_final_and_unscoped_validated_records(tmp_path: Path):
    scoped_final = tmp_path / "work-item-a" / "final_records.jsonl"
    scoped_final.parent.mkdir()
    scoped_final.write_text(
        "\n".join(
            [
                json.dumps({"work_item_id": "work-item-a", "article_id": "work-item-a", "sequence": "AAAACCCC"}),
                json.dumps({"work_item_id": "work-item-a", "article_id": "work-item-a", "sequence": "GGGGTTTT"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    global_validated = tmp_path / "validated_records.json"
    global_validated.write_text(
        json.dumps(
            {
                "accepted_records": [
                    {"work_item_id": "work-item-a", "article_id": "work-item-a", "sequence": "AAAACCCC"},
                    {"work_item_id": "work-item-a", "article_id": "work-item-a", "sequence": "GGGGTTTT"},
                    {"work_item_id": "work-item-a", "article_id": "work-item-a", "sequence": "CCCCAAAA"},
                ]
            }
        ),
        encoding="utf-8",
    )
    rejected_global = tmp_path / "rejected_validated_records.json"
    rejected_global.write_text(
        json.dumps(
            {
                "records": [
                    {"work_item_id": "work-item-a", "article_id": "work-item-a", "sequence": "TTTTGGGG", "status": "rejected"}
                ]
            }
        ),
        encoding="utf-8",
    )
    other_validated = tmp_path / "other_validated_records.json"
    other_validated.write_text(
        json.dumps(
            {
                "accepted_records": [
                    {"work_item_id": "work-item-b", "article_id": "work-item-b", "sequence": "CCCCGGGG"}
                ]
            }
        ),
        encoding="utf-8",
    )
    registry = _ListArtifactRegistry(
        [
            SimpleNamespace(
                uri=str(scoped_final),
                metadata={
                    "filename": "final_records.jsonl",
                    "artifact_kind": "final_records",
                    "work_item_id": "work-item-a",
                },
            ),
            SimpleNamespace(
                uri=str(global_validated),
                metadata={"filename": "validated_records.json", "artifact_kind": "validated_records"},
            ),
            SimpleNamespace(
                uri=str(rejected_global),
                metadata={"filename": "validated_records.json", "artifact_kind": "validated_records"},
            ),
            SimpleNamespace(
                uri=str(other_validated),
                metadata={"filename": "validated_records.json", "artifact_kind": "validated_records"},
            ),
        ]
    )

    records = _final_records_for_write_bootstrap(registry, task_id="task-1", work_item_id="work-item-a")

    assert [record["sequence"] for record in records] == ["AAAACCCC", "GGGGTTTT", "CCCCAAAA"]


def test_bootstrap_terminal_validated_artifact_satisfies_dynamic_expected_output():
    artifact_ref = ArtifactRef(
        uri="/tmp/work-item-a/validated_records.json",
        type="dataset",
        metadata={
            "filename": "validated_records.json",
            "artifact_kind": "validated_records",
            "work_item_id": "work-item-a",
        },
    )
    record = ToolCallRecord(
        tool_call=ToolCall(
            call_id="bootstrap-1-validate_candidate_records",
            name="validate_candidate_records",
            arguments={},
        ),
        result=ToolResult(
            call_id="bootstrap-1-validate_candidate_records",
            status="ok",
            content="validated",
            artifact_refs=[artifact_ref],
        ),
    )
    expected_outputs = [
        {
            "name": "validated_records.json",
            "required": True,
            "metadata": {"requires_artifact": True},
        }
    ]

    assert _bootstrap_scientific_handoff_satisfies_terminal_outputs(
        bootstrap_records=[record],
        artifact_refs=[artifact_ref],
        expected_outputs=expected_outputs,
        dispatch_metadata={"execution_mode": "dynamic"},
    )


def test_bootstrap_completion_does_not_short_circuit_nonterminal_candidate_output():
    artifact_ref = ArtifactRef(
        uri="/tmp/work-item-a/candidate_records.json",
        type="dataset",
        metadata={
            "filename": "candidate_records.json",
            "artifact_kind": "candidate_records",
            "work_item_id": "work-item-a",
        },
    )
    record = ToolCallRecord(
        tool_call=ToolCall(call_id="bootstrap-1-extract", name="extract_schema_records", arguments={}),
        result=ToolResult(call_id="bootstrap-1-extract", status="ok", content="candidates", artifact_refs=[artifact_ref]),
    )

    assert not _bootstrap_scientific_handoff_satisfies_terminal_outputs(
        bootstrap_records=[record],
        artifact_refs=[artifact_ref],
        expected_outputs=[
            {
                "name": "candidate_records.json",
                "required": True,
                "metadata": {"requires_artifact": True},
            }
        ],
        dispatch_metadata={"execution_mode": "dynamic"},
    )


def test_bootstrap_completion_satisfies_dynamic_candidate_output_contract():
    artifact_ref = ArtifactRef(
        uri="/tmp/work-item-a/candidate_records.json",
        type="dataset",
        metadata={
            "filename": "candidate_records.json",
            "artifact_kind": "candidate_records",
            "work_item_id": "work-item-a",
        },
    )
    record = ToolCallRecord(
        tool_call=ToolCall(call_id="bootstrap-1-build", name="build_candidate_records", arguments={}),
        result=ToolResult(call_id="bootstrap-1-build", status="ok", content="candidates", artifact_refs=[artifact_ref]),
    )

    assert _bootstrap_scientific_handoff_satisfies_expected_outputs(
        bootstrap_records=[record],
        artifact_refs=[artifact_ref],
        expected_outputs=[
            {
                "name": "candidate_records.json",
                "required": True,
                "metadata": {"requires_artifact": True},
            }
        ],
        dispatch_metadata={"execution_mode": "dynamic"},
    )


def test_bootstrap_completion_requires_dynamic_mode_and_successful_artifact():
    artifact_ref = ArtifactRef(
        uri="/tmp/work-item-a/final_records.jsonl",
        type="dataset",
        metadata={
            "filename": "final_records.jsonl",
            "artifact_kind": "final_records",
            "work_item_id": "work-item-a",
        },
    )
    record = ToolCallRecord(
        tool_call=ToolCall(call_id="bootstrap-1-serialize", name="serialize_final_records", arguments={}),
        result=ToolResult(call_id="bootstrap-1-serialize", status="ok", content="serialized", artifact_refs=[artifact_ref]),
    )
    expected_outputs = [
        {
            "name": "final_records.jsonl",
            "required": True,
            "metadata": {"requires_artifact": True},
        }
    ]

    assert not _bootstrap_scientific_handoff_satisfies_terminal_outputs(
        bootstrap_records=[record],
        artifact_refs=[artifact_ref],
        expected_outputs=expected_outputs,
        dispatch_metadata={"execution_mode": "static"},
    )
    assert not _bootstrap_scientific_handoff_satisfies_terminal_outputs(
        bootstrap_records=[],
        artifact_refs=[artifact_ref],
        expected_outputs=expected_outputs,
        dispatch_metadata={"execution_mode": "dynamic"},
    )


@DEFAULT_META_DISPATCH_REMOVED
def test_meta_agent_uses_configured_memory_backend(tmp_path: Path):
    request = _request()
    meta_memory = RecordingMemoryRuntime(
        backend_id="meta-memory",
        state_ref="meta-memory-state-before",
        content="Previous MetaAgent routing memory: dispatch solver before finishing.",
        update_result={
            "status": "updated",
            "state_ref": "meta-memory-state-after",
            "previous_state_ref": "meta-memory-state-before",
        },
    )
    agent_memory = RecordingMemoryRuntime(backend_id="agent-memory")
    task_memory = RecordingMemoryRuntime(backend_id="task-memory")
    meta_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "action": "run_subagent",
                            "target_role": "solver",
                            "instruction": "Use the routed evidence.",
                        }
                    ),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "action": "finish_task",
                            "metadata": {"final_answer": "Solver completed after meta memory."},
                        }
                    ),
                )
            ),
        ]
    )
    solver_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(action="final_answer", content="Solver used routed evidence."),
            )
        ]
    )
    trajectory_registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    backend_state_registry = FileBackendStateRegistry(tmp_path / "backend_state")
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        max_dispatch_steps=2,
        task_memory_backend=BackendBinding(backend_id="task-memory"),
        meta_agent=MetaAgentSpec(
            name="meta",
            system_prompt="Dispatch solver and finish.",
            llm_backend=BackendBinding(backend_id="meta-llm"),
            memory_backend=BackendBinding(backend_id="meta-memory"),
        ),
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="You solve scientific tasks.",
                llm_backend=BackendBinding(backend_id="solver-llm"),
                agent_memory_backend=BackendBinding(backend_id="agent-memory"),
            )
        },
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        trajectory_registry=trajectory_registry,
        backend_state_registry=backend_state_registry,
        llm_runtimes={"meta-llm": meta_llm, "solver-llm": solver_llm},
        memory_runtimes={
            "meta-memory": meta_memory,
            "agent-memory": agent_memory,
            "task-memory": task_memory,
        },
        skill_runtimes={"skill-local": RecordingSkillRuntime()},
    )

    result = runtime.run(request)

    assert result["final_answer"] == "Solver completed after meta memory."
    assert len(meta_memory.search_requests) == 2
    assert all(search_request.role == "meta" for search_request in meta_memory.search_requests)
    assert all(search_request.filters["memory_scope"] == "agent" for search_request in meta_memory.search_requests)
    assert all(
        search_request.filters["memory_scope_id"] == "agent:meta"
        for search_request in meta_memory.search_requests
    )
    first_meta_payload = json.loads(meta_llm.calls[0][0][-1].content)
    assert first_meta_payload["meta_memory"]["backend_id"] == "meta-memory"
    assert first_meta_payload["meta_memory"]["state_ref"] == "meta-memory-state-before"
    assert first_meta_payload["meta_memory"]["items"][0]["content"].startswith("Previous MetaAgent routing memory")
    assert len(meta_memory.add_calls) == 2
    assert all(call[0] == request.task_id and call[1] == "meta" for call in meta_memory.add_calls)
    assert "meta_agent_dispatch_summary" in meta_memory.add_calls[0][2][0].content
    meta_records = trajectory_registry.list_meta_agent_runs()
    assert len(meta_records) == 2
    assert all(record.metadata["meta_memory_bundle"]["backend_id"] == "meta-memory" for record in meta_records)
    assert all(record.metadata["meta_memory_update_result"]["status"] == "updated" for record in meta_records)
    meta_states = backend_state_registry.list_states("meta-memory")
    assert len(meta_states) == 2
    assert all(state.metadata["memory_scope"] == "agent" for state in meta_states)
    assert all(state.metadata["memory_scope_id"] == "agent:meta" for state in meta_states)


@DEFAULT_META_DISPATCH_REMOVED
def test_meta_agent_malformed_json_triggers_repair_retry(tmp_path: Path):
    request = _request()
    memory = RecordingMemoryRuntime()
    skill = RecordingSkillRuntime()
    meta_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(action="final_answer", content='{"action":"run_subagent",'),
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "action": "run_subagent",
                            "target_role": "solver",
                            "instruction": "Solve after retry.",
                        }
                    ),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps({"action": "finish_task", "metadata": {"final_answer": "done"}}),
                )
            ),
        ]
    )
    solver_llm = ScriptedLLMRuntime(
        [LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="Solved."))]
    )
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        max_dispatch_steps=2,
        meta_agent=MetaAgentSpec(
            name="meta",
            system_prompt="Return dispatch JSON.",
            llm_backend=BackendBinding(backend_id="meta-llm"),
        ),
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="Solve.",
                llm_backend=BackendBinding(backend_id="solver-llm"),
            )
        },
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"meta-llm": meta_llm, "solver-llm": solver_llm},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": skill},
    )

    result = runtime.run(request)

    assert result["final_answer"] == "done"
    assert len(meta_llm.calls) == 3
    retry_messages = meta_llm.calls[1][0]
    assert "Previous MetaAgent dispatch output was invalid" in retry_messages[-1].content
    repair_payload = json.loads(retry_messages[-1].content)
    assert repair_payload["raw_invalid_output"] == '{"action":"run_subagent",'
    assert "expected_schema" in repair_payload


@DEFAULT_META_DISPATCH_REMOVED
def test_meta_agent_invalid_schema_fails_clearly_after_retry_limit(tmp_path: Path):
    request = _request()
    bad_response = LLMRuntimeResponse(
        action=SubAgentAction(
            action="final_answer",
            content=json.dumps({"action": "run_subagent", "instruction": "missing target role"}),
        )
    )
    meta_llm = ScriptedLLMRuntime([bad_response, bad_response, bad_response])
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        max_dispatch_steps=1,
        meta_agent=MetaAgentSpec(
            name="meta",
            system_prompt="Return dispatch JSON.",
            llm_backend=BackendBinding(backend_id="meta-llm"),
        ),
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="Solve.",
                llm_backend=BackendBinding(backend_id="solver-llm"),
            )
        },
        runtime_policy=RuntimePolicy(metadata={"max_meta_dispatch_parse_retries": 1}),
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"meta-llm": meta_llm, "solver-llm": RecordingLLMRuntime()},
        memory_runtimes={"memory-local": RecordingMemoryRuntime()},
        skill_runtimes={"skill-local": RecordingSkillRuntime()},
    )

    with pytest.raises(RuntimeError) as exc_info:
        runtime.run(request)

    message = str(exc_info.value)
    assert "MetaAgent dispatch parsing failed after 1 retries" in message
    assert "raw_output" in message
    assert "expected_schema" in message
    assert "retry_count" in message


@DEFAULT_META_DISPATCH_REMOVED
def test_meta_agent_invalid_target_role_is_rejected(tmp_path: Path):
    request = _request()
    meta_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {"action": "run_subagent", "target_role": "OtherAgent", "instruction": "bad role"}
                    ),
                )
            )
        ]
    )
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        max_dispatch_steps=1,
        meta_agent=MetaAgentSpec(
            name="meta",
            system_prompt="Return dispatch JSON.",
            llm_backend=BackendBinding(backend_id="meta-llm"),
        ),
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="Solve.",
                llm_backend=BackendBinding(backend_id="solver-llm"),
            )
        },
        runtime_policy=RuntimePolicy(metadata={"max_meta_dispatch_parse_retries": 0}),
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"meta-llm": meta_llm, "solver-llm": RecordingLLMRuntime()},
        memory_runtimes={"memory-local": RecordingMemoryRuntime()},
        skill_runtimes={"skill-local": RecordingSkillRuntime()},
    )

    with pytest.raises(RuntimeError, match="invalid target_role"):
        runtime.run(request)


@DEFAULT_META_DISPATCH_REMOVED
def test_meta_agent_finish_with_pending_selected_workflow_node_triggers_retry(tmp_path: Path):
    request = _request()
    memory = RecordingMemoryRuntime()
    skill = RecordingSkillRuntime()
    selected_workflow = {
        "nodes": [
            {
                "node_id": "survey_1",
                "generic_agent_type": "SurveyAgent",
                "assigned_task": "Survey inputs.",
                "input_dependencies": [],
                "expected_outputs": ["inventory"],
                "completion_criteria": "inventory complete",
            },
            {
                "node_id": "write_1",
                "generic_agent_type": "WriteAgent",
                "assigned_task": "Write outputs.",
                "input_dependencies": ["survey_1"],
                "expected_outputs": ["records"],
                "completion_criteria": "records written",
            },
        ]
    }
    meta_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "action": "run_subagent",
                            "target_role": "SurveyAgent",
                            "instruction": "Survey inputs.",
                            "metadata": {
                                "meta_workflow_node_id": "survey_1",
                                "generic_agent_type": "SurveyAgent",
                                "workflow_dag": selected_workflow,
                            },
                        }
                    ),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "action": "finish_task",
                            "metadata": {"final_answer": "done too early"},
                        }
                    ),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "action": "run_subagent",
                            "target_role": "WriteAgent",
                            "instruction": "Write outputs.",
                            "metadata": {
                                "meta_workflow_node_id": "write_1",
                                "generic_agent_type": "WriteAgent",
                                "workflow_dag": selected_workflow,
                            },
                        }
                    ),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps({"action": "finish_task", "metadata": {"final_answer": "done"}}),
                )
            ),
        ]
    )
    survey_llm = ScriptedLLMRuntime(
        [LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="Surveyed."))]
    )
    write_llm = ScriptedLLMRuntime(
        [LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="Wrote."))]
    )
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        max_dispatch_steps=3,
        meta_agent=MetaAgentSpec(
            name="meta",
            system_prompt="Return dispatch JSON.",
            llm_backend=BackendBinding(backend_id="meta-llm"),
        ),
        roles={
            "SurveyAgent": RoleSpec(
                name="SurveyAgent",
                system_prompt="Survey.",
                llm_backend=BackendBinding(backend_id="survey-llm"),
            ),
            "WriteAgent": RoleSpec(
                name="WriteAgent",
                system_prompt="Write.",
                llm_backend=BackendBinding(backend_id="write-llm"),
            ),
        },
        runtime_policy=RuntimePolicy(metadata={"max_meta_dispatch_parse_retries": 1}),
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"meta-llm": meta_llm, "survey-llm": survey_llm, "write-llm": write_llm},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": skill},
    )

    result = runtime.run(request)

    assert result["final_answer"] == "done"
    assert [run["role"] for run in result["runs"]] == ["SurveyAgent", "WriteAgent"]
    assert len(meta_llm.calls) == 4
    retry_messages = meta_llm.calls[2][0]
    assert "Previous MetaAgent dispatch output was invalid" in retry_messages[-1].content
    assert "pending selected workflow nodes" in retry_messages[-1].content


@DEFAULT_META_DISPATCH_REMOVED
def test_meta_agent_finish_without_required_final_artifact_triggers_retry(tmp_path: Path):
    request = _request()
    request = request.model_copy(
        update={
            "goal": (
                "Extract records. Final outputs should be written as lab artifacts: "
                "biology_component_records.jsonl and biology_component_report.md."
            )
        }
    )
    memory = RecordingMemoryRuntime()

    class FinalArtifactSkillRuntime(RecordingSkillRuntime):
        def get(self, request: RetrievalRequest) -> SkillBundle:
            self.required_tools = ["write_jsonl"] if request.role == "WriteAgent" else []
            return super().get(request)

    skill = FinalArtifactSkillRuntime()
    meta_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {"action": "run_subagent", "target_role": "ExecAgent", "instruction": "Extract records."}
                    ),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "action": "finish_task",
                            "metadata": {"final_answer": "empty audit only"},
                        }
                    ),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps({"action": "run_subagent", "target_role": "WriteAgent", "instruction": "Write final artifacts."}),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "action": "finish_task",
                            "metadata": {"final_answer": "final artifacts written"},
                        }
                    ),
                )
            ),
        ]
    )
    exec_llm = ScriptedLLMRuntime(
        [LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="Extracted zero records."))]
    )
    write_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(
                        call_id="write-records",
                        name="write_jsonl",
                        arguments={"path": "biology_component_records.jsonl", "records": []},
                    ),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="Wrote records.")),
        ]
    )

    registry = ToolRegistry()

    def write_empty_jsonl(arguments: dict) -> ToolResult:
        output_path = tmp_path / "biology_component_records.jsonl"
        output_path.write_text("", encoding="utf-8")
        return ToolResult(
            call_id="write-records",
            status="ok",
            content="wrote jsonl",
            artifact_refs=[
                ArtifactRef(
                    uri=str(output_path),
                    type="dataset",
                    metadata={"filename": "biology_component_records.jsonl", "status": "final"},
                )
            ],
        )

    registry.register(
        ToolSpec(name="write_jsonl", description="write", parameters_schema={"type": "object"}),
        write_empty_jsonl,
    )
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        max_dispatch_steps=4,
        meta_agent=MetaAgentSpec(
            name="meta",
            system_prompt="Return dispatch JSON.",
            llm_backend=BackendBinding(backend_id="meta-llm"),
        ),
        roles={
            "ExecAgent": RoleSpec(
                name="ExecAgent",
                system_prompt="Extract.",
                llm_backend=BackendBinding(backend_id="exec-llm"),
            ),
            "WriteAgent": RoleSpec(
                name="WriteAgent",
                system_prompt="Write.",
                llm_backend=BackendBinding(backend_id="write-llm"),
                allowed_tools=["write_jsonl"],
            ),
        },
        runtime_policy=RuntimePolicy(
            enable_workflow_planning=False,
            metadata={
                "max_meta_dispatch_parse_retries": 1,
                "required_final_artifacts": ["biology_component_records.jsonl"],
            }
        ),
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"meta-llm": meta_llm, "exec-llm": exec_llm, "write-llm": write_llm},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": skill},
    )

    result = runtime.run(request)

    assert result["final_answer"] == "final artifacts written"
    assert [run["role"] for run in result["runs"]] == ["ExecAgent", "WriteAgent"]
    retry_messages = meta_llm.calls[2][0]
    assert "Previous MetaAgent dispatch output was invalid" in retry_messages[-1].content
    assert "missing required final artifact" in retry_messages[-1].content


@DEFAULT_META_DISPATCH_REMOVED
def test_meta_agent_finish_accepts_managed_artifact_by_original_filename(tmp_path: Path):
    request = _request()
    source_path = tmp_path / "source" / "biology_component_records.jsonl"
    source_path.parent.mkdir()
    source_path.write_text("", encoding="utf-8")
    meta_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "action": "run_subagent",
                            "target_role": "WriteAgent",
                            "instruction": "Write final records.",
                        }
                    ),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps({"action": "finish_task", "metadata": {"final_answer": "done"}}),
                )
            ),
        ]
    )
    write_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="write-records", name="write_jsonl", arguments={}),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="Wrote records.")),
        ]
    )
    registry = ToolRegistry()
    registry.register(
        ToolSpec(name="write_jsonl", description="write", parameters_schema={"type": "object"}),
        lambda arguments: ToolResult(
            call_id="write-records",
            status="ok",
            content="wrote records",
            artifact_refs=[ArtifactRef(uri=str(source_path), type="dataset", metadata={"format": "jsonl"})],
        ),
    )
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        max_dispatch_steps=2,
        meta_agent=MetaAgentSpec(
            name="meta",
            system_prompt="Return dispatch JSON.",
            llm_backend=BackendBinding(backend_id="meta-llm"),
        ),
        roles={
            "WriteAgent": RoleSpec(
                name="WriteAgent",
                system_prompt="Write.",
                llm_backend=BackendBinding(backend_id="write-llm"),
                allowed_tools=["write_jsonl"],
            )
        },
        runtime_policy=RuntimePolicy(
            enable_workflow_planning=False,
            metadata={"required_final_artifacts": ["biology_component_records.jsonl"]},
        ),
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        tool_artifact_root_factory=lambda req, run_ref: tmp_path / "lab-artifacts" / run_ref,
        llm_runtimes={"meta-llm": meta_llm, "write-llm": write_llm},
        memory_runtimes={"memory-local": RecordingMemoryRuntime()},
        skill_runtimes={"skill-local": RecordingSkillRuntime(required_tools=["write_jsonl"])},
    )

    result = runtime.run(request)

    assert result["final_answer"] == "done"
    artifact_ref = result["runs"][0]["artifact_refs"][0]
    assert Path(artifact_ref["uri"]).name == "0-biology_component_records.jsonl"
    assert artifact_ref["metadata"]["filename"] == "biology_component_records.jsonl"


@DEFAULT_META_DISPATCH_REMOVED
def test_meta_agent_context_includes_subagent_completion_contract(tmp_path: Path):
    request = _request()
    memory = RecordingMemoryRuntime()
    skill = RecordingSkillRuntime()
    meta_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {"action": "run_subagent", "target_role": "WriteAgent", "instruction": "Write final artifacts."}
                    ),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps({"action": "finish_task", "metadata": {"final_answer": "done"}}),
                )
            ),
        ]
    )
    writer_llm = ScriptedLLMRuntime(
        [LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="No artifacts were written."))]
    )
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        max_dispatch_steps=2,
        meta_agent=MetaAgentSpec(
            name="meta",
            system_prompt="Return dispatch JSON.",
            llm_backend=BackendBinding(backend_id="meta-llm"),
        ),
        roles={
            "WriteAgent": RoleSpec(
                name="WriteAgent",
                system_prompt="Write.",
                llm_backend=BackendBinding(backend_id="write-llm"),
            )
        },
    )
    lab_state_registry = FileLabStateRegistry(tmp_path / "lab-state")
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        lab_state_registry=lab_state_registry,
        llm_runtimes={"meta-llm": meta_llm, "write-llm": writer_llm},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": skill},
    )

    runtime.run(request)

    completed_run = meta_llm.calls[1][0][-1]
    payload = json.loads(completed_run.content)
    contract = payload["completed_runs"][0]["completion_contract"]
    assert contract["assigned_task_complete"] is True
    assert contract["ready_for_task_end"] is False
    assert contract["recommended_next_route"] == "WriteAgent"
    reports = lab_state_registry.list_subagent_reports(request.task_id)
    assert reports[0].metadata["completion_contract"] == contract


@DEFAULT_META_DISPATCH_REMOVED
def test_downstream_subagent_prompt_includes_generic_upstream_outputs(tmp_path: Path):
    request = _request()
    artifact_path = tmp_path / "alpha-output.json"
    alpha_memory = RecordingMemoryRuntime()

    class RoleAwareSkillRuntime(RecordingSkillRuntime):
        def get(self, request: RetrievalRequest) -> SkillBundle:
            self.required_tools = ["write_artifact"] if request.role == "AlphaAgent" else []
            return super().get(request)

    skill = RoleAwareSkillRuntime()
    tool_registry = ToolRegistry()
    tool_registry.register(
        ToolSpec(name="write_artifact", description="write artifact", parameters_schema={"type": "object"}),
        lambda arguments: ToolResult(
            call_id="write_artifact",
            status="ok",
            content="wrote alpha artifact",
            artifact_refs=[
                ArtifactRef(
                    uri=str(artifact_path),
                    type="dataset",
                    metadata={"name": "alpha-output", "status": "intermediate"},
                )
            ],
        ),
    )
    meta_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {"action": "run_subagent", "target_role": "AlphaAgent", "instruction": "Produce upstream output."}
                    ),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {"action": "run_subagent", "target_role": "BetaAgent", "instruction": "Consume upstream output."}
                    ),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps({"action": "finish_task", "metadata": {"final_answer": "done"}}),
                )
            ),
        ]
    )
    alpha_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="alpha-call", name="write_artifact", arguments={}),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="Alpha summary for downstream.")),
        ]
    )

    class InspectingBetaLLMRuntime:
        def __init__(self):
            self.payloads: list[dict] = []

        def generate(
            self,
            messages: list[Message],
            tool_specs: list[dict],
            generation_config: LLMGenerationConfig,
        ) -> LLMRuntimeResponse:
            content = messages[-1].content
            self.payloads.append({"raw": content})
            return LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="Beta saw upstream output."))

    beta_llm = InspectingBetaLLMRuntime()
    lab_state_registry = FileLabStateRegistry(tmp_path / "lab_state")
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        max_dispatch_steps=3,
        meta_agent=MetaAgentSpec(
            name="meta",
            system_prompt="Route generic agents.",
            llm_backend=BackendBinding(backend_id="meta-llm"),
        ),
        roles={
            "AlphaAgent": RoleSpec(
                name="AlphaAgent",
                system_prompt="Produce.",
                llm_backend=BackendBinding(backend_id="alpha-llm"),
                allowed_tools=["write_artifact"],
            ),
            "BetaAgent": RoleSpec(
                name="BetaAgent",
                system_prompt="Consume.",
                llm_backend=BackendBinding(backend_id="beta-llm"),
            ),
        },
        runtime_policy=RuntimePolicy(enable_workflow_planning=False, max_tool_steps=2),
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(tool_registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        lab_state_registry=lab_state_registry,
        llm_runtimes={"meta-llm": meta_llm, "alpha-llm": alpha_llm, "beta-llm": beta_llm},
        memory_runtimes={"memory-local": alpha_memory},
        skill_runtimes={"skill-local": skill},
    )

    runtime.run(request)

    assert beta_llm.payloads
    beta_prompt = beta_llm.payloads[0]["raw"]
    assert '"upstream_outputs"' in beta_prompt
    assert '"role": "AlphaAgent"' in beta_prompt
    assert "Alpha summary for downstream." in beta_prompt
    assert '"assigned_task_complete": true' in beta_prompt
    assert str(artifact_path) in beta_prompt
    assert '"producer_role": "AlphaAgent"' in beta_prompt


def test_completion_contract_treats_exploratory_misses_as_nonblocking_after_artifact(tmp_path: Path):
    artifact = ArtifactRef(
        uri=str(tmp_path / "context_summary.json"),
        type="dataset",
        metadata={"filename": "context_summary.json"},
    )
    exploratory_error = ToolCallRecord(
        tool_call=ToolCall(call_id="table-miss", name="inspect_table", arguments={"table_caption": "Table S1"}),
        result=ToolResult(call_id="table-miss", status="error", content="'Table S1'"),
    )

    contract = _subagent_completion_contract(
        status="completed",
        failure_reason=None,
        artifact_refs=[artifact],
        node_records=[],
        role="ContextAgent",
        assigned_task="Inspect source context.",
        expected_outputs=[
            {
                "name": "context_summary.json",
                "required": True,
                "metadata": {"requires_artifact": True, "dynamic_output_artifact": True},
            }
        ],
        tool_trace_records=[exploratory_error],
        final_answer="wrote context",
        policy_metadata={},
    )

    assert contract["ready_for_task_end"] is True
    assert contract["blocking_issues"] == []


@DEFAULT_META_DISPATCH_REMOVED
def test_completion_contract_rejects_audit_only_artifact_for_record_expected_output(tmp_path: Path):
    request = _request()
    audit_path = tmp_path / "audit.md"
    audit_path.write_text("candidate_count: 0\n", encoding="utf-8")
    memory = RecordingMemoryRuntime()

    class RecordSkillRuntime(RecordingSkillRuntime):
        def get(self, request: RetrievalRequest) -> SkillBundle:
            self.required_tools = ["write_report"]
            return super().get(request)

    skill = RecordSkillRuntime()
    registry = ToolRegistry()
    registry.register(
        ToolSpec(name="write_report", description="write report", parameters_schema={"type": "object"}),
        lambda arguments: ToolResult(
            call_id="write_report",
            status="ok",
            content="wrote audit",
            artifact_refs=[
                ArtifactRef(
                    uri=str(audit_path),
                    type="log",
                    metadata={"filename": "audit.md", "status": "audit", "artifact_kind": "audit"},
                )
            ],
        ),
    )
    meta_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "action": "run_subagent",
                            "target_role": "ExecAgent",
                            "instruction": "Produce candidate records.",
                            "expected_outputs": [
                                {"name": "candidate records JSONL", "description": "candidate record artifact"}
                            ],
                            "metadata": {
                                "expected_outputs": ["candidate records JSONL"],
                                "completion_criteria": "candidate records artifact exists or a justified no-candidate audit with coverage exists",
                            },
                        }
                    ),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps({"action": "finish_task", "metadata": {"final_answer": "done"}}),
                )
            ),
        ]
    )
    exec_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="audit", name="write_report", arguments={}),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content="No candidate record was serialized in this node.",
                )
            ),
        ]
    )
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        max_dispatch_steps=2,
        meta_agent=MetaAgentSpec(
            name="meta",
            system_prompt="Return dispatch JSON.",
            llm_backend=BackendBinding(backend_id="meta-llm"),
        ),
        roles={
            "ExecAgent": RoleSpec(
                name="ExecAgent",
                system_prompt="Execute.",
                llm_backend=BackendBinding(backend_id="exec-llm"),
                allowed_tools=["write_report"],
            )
        },
        runtime_policy=RuntimePolicy(enable_workflow_planning=False, metadata={"max_meta_dispatch_parse_retries": 0}),
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"meta-llm": meta_llm, "exec-llm": exec_llm},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": skill},
    )

    with pytest.raises(RuntimeError, match="incomplete completed subagent outputs"):
        runtime.run(request)

    reports = FileTrajectoryRegistry(tmp_path / "trajectory").list_subagent_runs()
    contract = reports[0].metadata["completion_contract"]
    assert contract["assigned_task_complete"] is False
    assert contract["produced_required_outputs"] is False
    assert "candidate records JSONL" in json.dumps(contract)


@DEFAULT_META_DISPATCH_REMOVED
def test_completion_contract_marks_tool_errors_as_blocking(tmp_path: Path):
    request = _request()
    memory = RecordingMemoryRuntime()
    skill = RecordingSkillRuntime(required_tools=["write_jsonl"])
    registry = ToolRegistry()
    registry.register(
        ToolSpec(name="write_jsonl", description="write jsonl", parameters_schema={"type": "object"}),
        lambda arguments: ToolResult(call_id="write_jsonl", status="error", content="records must be a list of objects"),
    )
    meta_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "action": "run_subagent",
                            "target_role": "WriteAgent",
                            "instruction": "Write records.",
                            "expected_outputs": [{"name": "records JSONL", "description": "final records artifact"}],
                            "metadata": {"expected_outputs": ["records JSONL"]},
                        }
                    ),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps({"action": "finish_task", "metadata": {"final_answer": "done"}}),
                )
            ),
        ]
    )
    write_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="bad-jsonl", name="write_jsonl", arguments={"path": "records.jsonl"}),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="Done.")),
        ]
    )
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        max_dispatch_steps=2,
        meta_agent=MetaAgentSpec(
            name="meta",
            system_prompt="Return dispatch JSON.",
            llm_backend=BackendBinding(backend_id="meta-llm"),
        ),
        roles={
            "WriteAgent": RoleSpec(
                name="WriteAgent",
                system_prompt="Write.",
                llm_backend=BackendBinding(backend_id="write-llm"),
                allowed_tools=["write_jsonl"],
            )
        },
        runtime_policy=RuntimePolicy(enable_workflow_planning=False, metadata={"max_meta_dispatch_parse_retries": 0}),
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"meta-llm": meta_llm, "write-llm": write_llm},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": skill},
    )

    with pytest.raises(RuntimeError, match="incomplete completed subagent outputs"):
        runtime.run(request)

    run = FileTrajectoryRegistry(tmp_path / "trajectory").list_subagent_runs()[0]
    contract = run.metadata["completion_contract"]
    assert contract["assigned_task_complete"] is False
    assert contract["blocking_issues"]
    assert "write_jsonl" in json.dumps(contract["blocking_issues"])


@DEFAULT_META_DISPATCH_REMOVED
def test_meta_agent_repeated_no_progress_route_is_aborted(tmp_path: Path):
    request = _request()
    memory = RecordingMemoryRuntime()
    skill = RecordingSkillRuntime()

    class RepeatingMetaLLMRuntime:
        def __init__(self):
            self.calls = []

        def generate(
            self,
            messages: list[Message],
            tool_specs: list[dict],
            generation_config: LLMGenerationConfig,
        ) -> LLMRuntimeResponse:
            self.calls.append((messages, tool_specs, generation_config))
            return LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "action": "run_subagent",
                            "target_role": "WriteAgent",
                            "instruction": "Retry writing outputs.",
                        }
                    ),
                )
            )

    meta_llm = RepeatingMetaLLMRuntime()
    writer_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="No new artifacts.")),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="No new artifacts.")),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="No new artifacts.")),
        ]
    )
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        max_dispatch_steps=5,
        meta_agent=MetaAgentSpec(
            name="meta",
            system_prompt="Return dispatch JSON.",
            llm_backend=BackendBinding(backend_id="meta-llm"),
        ),
        roles={
            "WriteAgent": RoleSpec(
                name="WriteAgent",
                system_prompt="Write.",
                llm_backend=BackendBinding(backend_id="write-llm"),
            )
        },
        runtime_policy=RuntimePolicy(
            metadata={
                "max_repeated_no_progress_dispatches": 2,
            }
        ),
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"meta-llm": meta_llm, "write-llm": writer_llm},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": skill},
    )

    with pytest.raises(RuntimeError, match="repeated no-progress dispatch"):
        runtime.run(request)

    assert len(meta_llm.calls) == 3


@DEFAULT_META_DISPATCH_REMOVED
def test_meta_agent_repeated_guard_failed_tool_only_route_is_aborted(tmp_path: Path):
    request = _request()
    memory = RecordingMemoryRuntime()
    skill = RecordingSkillRuntime(required_tools=["write_report"])

    class RepeatingMetaLLMRuntime:
        def __init__(self):
            self.calls = []

        def generate(
            self,
            messages: list[Message],
            tool_specs: list[dict],
            generation_config: LLMGenerationConfig,
        ) -> LLMRuntimeResponse:
            self.calls.append((messages, tool_specs, generation_config))
            return LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "action": "run_subagent",
                            "target_role": "SurveyAgent",
                            "instruction": "Survey and write a report.",
                        }
                    ),
                )
            )

    survey_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="call-1", name="write_report", arguments={}),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="Survey did not produce a valid artifact.")),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="call-2", name="write_report", arguments={}),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="Survey still did not produce a valid artifact.")),
        ]
    )
    registry = ToolRegistry()
    registry.register(
        ToolSpec(name="write_report", description="write report", parameters_schema={"type": "object"}),
        lambda arguments: ToolResult(call_id="write_report", status="error", content="missing content"),
    )
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        max_dispatch_steps=5,
        meta_agent=MetaAgentSpec(
            name="meta",
            system_prompt="Return dispatch JSON.",
            llm_backend=BackendBinding(backend_id="meta-llm"),
        ),
        roles={
            "SurveyAgent": RoleSpec(
                name="SurveyAgent",
                system_prompt="Survey.",
                llm_backend=BackendBinding(backend_id="survey-llm"),
                allowed_tools=["write_report"],
            )
        },
        runtime_policy=RuntimePolicy(
            max_tool_steps=1,
            metadata={
                "max_repeated_no_progress_dispatches": 2,
                "completion_guards_by_role": {"SurveyAgent": {"required_tool_calls_before_final": ["write_report"]}},
            },
        ),
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"meta-llm": RepeatingMetaLLMRuntime(), "survey-llm": survey_llm},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": skill},
    )

    with pytest.raises(RuntimeError, match="repeated no-progress dispatch"):
        runtime.run(request)


@DEFAULT_META_DISPATCH_REMOVED
def test_meta_agent_guard_failed_same_work_item_retries_with_survey_strategy(tmp_path: Path):
    request = _request()
    memory = RecordingMemoryRuntime()
    skill = RecordingSkillRuntime()

    class RepeatingExecMetaLLMRuntime:
        def __init__(self):
            self.calls = []

        def generate(
            self,
            messages: list[Message],
            tool_specs: list[dict],
            generation_config: LLMGenerationConfig,
        ) -> LLMRuntimeResponse:
            self.calls.append((messages, tool_specs, generation_config))
            return LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "action": "run_subagent",
                            "target_role": "ExecAgent",
                            "instruction": "Extract work item a.",
                            "metadata": {"work_item_id": "a"},
                        }
                    ),
                )
            )

    registry = ToolRegistry()
    registry.register(
        ToolSpec(name="write_jsonl", description="write jsonl", parameters_schema={"type": "object"}),
        lambda arguments: ToolResult(
            call_id="write_jsonl",
            status="ok",
            content="wrote records",
            metadata={"record_count": len(arguments.get("records") or [])},
        ),
    )
    meta_llm = RepeatingExecMetaLLMRuntime()
    exec_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="jsonl", name="write_jsonl", arguments={"records": []}),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="no records")),
        ]
    )
    survey_llm = ScriptedLLMRuntime([LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="surveyed sources"))])
    trajectory_registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    lab_state_registry = FileLabStateRegistry(tmp_path / "lab_state")
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        max_dispatch_steps=2,
        meta_agent=MetaAgentSpec(
            name="meta",
            system_prompt="Return dispatch JSON.",
            llm_backend=BackendBinding(backend_id="meta-llm"),
        ),
        roles={
            "ExecAgent": RoleSpec(
                name="ExecAgent",
                system_prompt="Execute extraction.",
                llm_backend=BackendBinding(backend_id="exec-llm"),
                allowed_tools=["write_jsonl"],
            ),
            "SurveyAgent": RoleSpec(
                name="SurveyAgent",
                system_prompt="Survey sources.",
                llm_backend=BackendBinding(backend_id="survey-llm"),
            ),
        },
        runtime_policy=RuntimePolicy(
            max_tool_steps=1,
            metadata={
                "work_item_routing": {
                    "enabled": True,
                    "executor_roles": ["ExecAgent"],
                    "reviewer_roles": [],
                    "finalizer_roles": [],
                    "required_work_item_ids": [],
                    "work_item_id_field": "work_item_id",
                },
                "completion_guards_by_role": {
                    "ExecAgent": {
                        "required_tool_calls_before_final": ["write_jsonl"],
                        "minimum_jsonl_records_before_final": 1,
                    }
                },
            },
        ),
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=trajectory_registry,
        lab_state_registry=lab_state_registry,
        llm_runtimes={"meta-llm": meta_llm, "exec-llm": exec_llm, "survey-llm": survey_llm},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": skill},
    )

    with pytest.raises(RuntimeError, match="meta-agent exceeded max_dispatch_steps"):
        runtime.run(request)

    saved_runs = trajectory_registry.list_subagent_runs()
    assert [run.role for run in saved_runs] == ["ExecAgent", "SurveyAgent"]
    assert saved_runs[1].metadata["dispatch_metadata"]["recovery_strategy"] == "survey_before_retry"
    assert saved_runs[1].metadata["dispatch_metadata"]["route"] == "SurveyAgent"
    assert saved_runs[1].metadata["dispatch_metadata"]["generic_agent_type"] == "SurveyAgent"
    work_item_path = tmp_path / "lab_state" / "work_items" / request.task_id / "a.json"
    payload = json.loads(work_item_path.read_text(encoding="utf-8"))
    assert payload["status"] == "claimed"
    assert payload["history"][0]["status"] == "failed"
    assert payload["history"][1]["role"] == "SurveyAgent"


@DEFAULT_META_DISPATCH_REMOVED
def test_failed_work_item_retry_budget_moves_scheduler_to_next_work_item(tmp_path: Path):
    request = _request()
    memory = RecordingMemoryRuntime()
    skill = RecordingSkillRuntime()

    class RepeatingFirstWorkItemMetaLLMRuntime:
        def __init__(self):
            self.calls = []

        def generate(
            self,
            messages: list[Message],
            tool_specs: list[dict],
            generation_config: LLMGenerationConfig,
        ) -> LLMRuntimeResponse:
            self.calls.append((messages, tool_specs, generation_config))
            return LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "action": "run_subagent",
                            "target_role": "ExecAgent",
                            "instruction": "Extract work item a.",
                            "metadata": {"work_item_id": "a"},
                        }
                    ),
                )
            )

    registry = ToolRegistry()
    registry.register(
        ToolSpec(name="write_jsonl", description="write jsonl", parameters_schema={"type": "object"}),
        lambda arguments: ToolResult(
            call_id="write_jsonl",
            status="ok",
            content="wrote records",
            metadata={"record_count": len(arguments.get("records") or [])},
        ),
    )
    exec_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="jsonl-a", name="write_jsonl", arguments={"records": []}),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="a failed")),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="jsonl-b", name="write_jsonl", arguments={"records": []}),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="b failed")),
        ]
    )
    trajectory_registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    lab_state_registry = FileLabStateRegistry(tmp_path / "lab_state")
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        max_dispatch_steps=2,
        meta_agent=MetaAgentSpec(
            name="meta",
            system_prompt="Return dispatch JSON.",
            llm_backend=BackendBinding(backend_id="meta-llm"),
        ),
        roles={
            "ExecAgent": RoleSpec(
                name="ExecAgent",
                system_prompt="Execute extraction.",
                llm_backend=BackendBinding(backend_id="exec-llm"),
                allowed_tools=["write_jsonl"],
            ),
        },
        runtime_policy=RuntimePolicy(
            max_tool_steps=1,
            metadata={
                "work_item_routing": {
                    "enabled": True,
                    "executor_roles": ["ExecAgent"],
                    "reviewer_roles": [],
                    "finalizer_roles": [],
                    "required_work_item_ids": ["a", "b"],
                    "work_item_id_field": "work_item_id",
                    "max_failed_executor_attempts_per_work_item": 1,
                },
                "completion_guards_by_role": {
                    "ExecAgent": {
                        "required_tool_calls_before_final": ["write_jsonl"],
                        "minimum_jsonl_records_before_final": 1,
                    }
                },
            },
        ),
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=trajectory_registry,
        lab_state_registry=lab_state_registry,
        llm_runtimes={"meta-llm": RepeatingFirstWorkItemMetaLLMRuntime(), "exec-llm": exec_llm},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": skill},
    )

    with pytest.raises(RuntimeError, match="meta-agent exceeded max_dispatch_steps"):
        runtime.run(request)

    saved_runs = trajectory_registry.list_subagent_runs()
    assert [run.metadata["dispatch_metadata"]["work_item_id"] for run in saved_runs] == ["a", "b"]
    assert saved_runs[1].metadata["dispatch_metadata"]["skipped_failed_work_item_id"] == "a"
    assert saved_runs[1].metadata["dispatch_metadata"]["route"] == "ExecAgent"
    assert saved_runs[1].metadata["dispatch_metadata"]["generic_agent_type"] == "ExecAgent"
    work_item_root = tmp_path / "lab_state" / "work_items" / request.task_id
    assert json.loads((work_item_root / "a.json").read_text(encoding="utf-8"))["status"] == "failed"
    assert json.loads((work_item_root / "b.json").read_text(encoding="utf-8"))["status"] == "failed"


@DEFAULT_META_DISPATCH_REMOVED
def test_work_item_retry_budget_exhaustion_terminates_when_no_unresolved_items_remain(tmp_path: Path):
    request = _request()
    memory = RecordingMemoryRuntime()
    skill = RecordingSkillRuntime()

    class RepeatingMetaLLMRuntime:
        def generate(
            self,
            messages: list[Message],
            tool_specs: list[dict],
            generation_config: LLMGenerationConfig,
        ) -> LLMRuntimeResponse:
            return LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "action": "run_subagent",
                            "target_role": "ExecAgent",
                            "instruction": "Extract work item a.",
                            "metadata": {"work_item_id": "a"},
                        }
                    ),
                )
            )

    registry = ToolRegistry()
    registry.register(
        ToolSpec(name="write_jsonl", description="write jsonl", parameters_schema={"type": "object"}),
        lambda arguments: ToolResult(
            call_id="write_jsonl",
            status="ok",
            content="wrote records",
            metadata={"record_count": len(arguments.get("records") or [])},
        ),
    )
    exec_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="jsonl-a", name="write_jsonl", arguments={"records": []}),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="a failed")),
        ]
    )
    trajectory_registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    lab_state_registry = FileLabStateRegistry(tmp_path / "lab_state")
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        max_dispatch_steps=3,
        meta_agent=MetaAgentSpec(
            name="meta",
            system_prompt="Return dispatch JSON.",
            llm_backend=BackendBinding(backend_id="meta-llm"),
        ),
        roles={
            "ExecAgent": RoleSpec(
                name="ExecAgent",
                system_prompt="Execute extraction.",
                llm_backend=BackendBinding(backend_id="exec-llm"),
                allowed_tools=["write_jsonl"],
            ),
        },
        runtime_policy=RuntimePolicy(
            max_tool_steps=1,
            metadata={
                "work_item_routing": {
                    "enabled": True,
                    "executor_roles": ["ExecAgent"],
                    "reviewer_roles": [],
                    "finalizer_roles": [],
                    "required_work_item_ids": ["a"],
                    "work_item_id_field": "work_item_id",
                    "max_failed_executor_attempts_per_work_item": 1,
                },
                "completion_guards_by_role": {
                    "ExecAgent": {
                        "required_tool_calls_before_final": ["write_jsonl"],
                        "minimum_jsonl_records_before_final": 1,
                    }
                },
            },
        ),
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=trajectory_registry,
        lab_state_registry=lab_state_registry,
        llm_runtimes={"meta-llm": RepeatingMetaLLMRuntime(), "exec-llm": exec_llm},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": skill},
    )

    with pytest.raises(RuntimeError, match="no unresolved configured work items remain"):
        runtime.run(request)

    assert len(trajectory_registry.list_subagent_runs()) == 1
    payload = json.loads((tmp_path / "lab_state" / "work_items" / request.task_id / "a.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["history"][-1]["event"] == "retry_budget_exhausted"


@DEFAULT_META_DISPATCH_REMOVED
def test_meta_agent_repeated_same_role_audit_artifact_route_is_aborted(tmp_path: Path):
    request = _request()
    memory = RecordingMemoryRuntime()

    class ReportSkillRuntime(RecordingSkillRuntime):
        def get(self, request: RetrievalRequest) -> SkillBundle:
            self.required_tools = ["write_report"]
            return super().get(request)

    class RepeatingMetaLLMRuntime:
        def __init__(self):
            self.calls = []

        def generate(
            self,
            messages: list[Message],
            tool_specs: list[dict],
            generation_config: LLMGenerationConfig,
        ) -> LLMRuntimeResponse:
            self.calls.append((messages, tool_specs, generation_config))
            return LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "action": "run_subagent",
                            "target_role": "WriteAgent",
                            "instruction": "Retry writing the final report.",
                            "expected_outputs": [
                                {"name": "final report", "description": "final report", "required": True}
                            ],
                        }
                    ),
                )
            )

    class RepeatingWriterLLMRuntime:
        def __init__(self):
            self.calls = []

        def generate(
            self,
            messages: list[Message],
            tool_specs: list[dict],
            generation_config: LLMGenerationConfig,
        ) -> LLMRuntimeResponse:
            self.calls.append((messages, tool_specs, generation_config))
            if len(self.calls) % 2 == 1:
                return LLMRuntimeResponse(
                    action=SubAgentAction(
                        action="tool_call",
                        tool_call=ToolCall(call_id=f"report-{len(self.calls)}", name="write_report", arguments={}),
                    )
                )
            return LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="Wrote audit report."))

    artifact_path = tmp_path / "audit.md"
    registry = ToolRegistry()
    registry.register(
        ToolSpec(name="write_report", description="write report", parameters_schema={"type": "object"}),
        lambda arguments: ToolResult(
            call_id="write_report",
            status="ok",
            content="wrote audit",
            artifact_refs=[
                ArtifactRef(
                    uri=str(artifact_path),
                    type="log",
                    metadata={"filename": "audit.md", "status": "audit", "artifact_kind": "audit"},
                )
            ],
        ),
    )
    meta_llm = RepeatingMetaLLMRuntime()
    writer_llm = RepeatingWriterLLMRuntime()
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        max_dispatch_steps=5,
        meta_agent=MetaAgentSpec(
            name="meta",
            system_prompt="Return dispatch JSON.",
            llm_backend=BackendBinding(backend_id="meta-llm"),
        ),
        roles={
            "WriteAgent": RoleSpec(
                name="WriteAgent",
                system_prompt="Write.",
                llm_backend=BackendBinding(backend_id="write-llm"),
                allowed_tools=["write_report"],
            )
        },
        runtime_policy=RuntimePolicy(
            metadata={
                "max_consecutive_dispatches_per_role": 2,
            }
        ),
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"meta-llm": meta_llm, "write-llm": writer_llm},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": ReportSkillRuntime()},
    )

    with pytest.raises(RuntimeError, match="repeated same-role dispatch"):
        runtime.run(request)

    assert len(meta_llm.calls) == 3
    assert len(writer_llm.calls) == 4


def test_meta_agent_finish_rejects_placeholder_required_jsonl_artifact(tmp_path: Path):
    request = _request()
    request = request.model_copy(update={"goal": "Write biology_component_records.jsonl."})
    placeholder_path = tmp_path / "biology_component_records.jsonl"
    placeholder_path.write_text(
        '{"article_id":"unknown","component_name":"validated promoter record 1","component_type":"promoter"}\n',
        encoding="utf-8",
    )
    meta_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps({"action": "finish_task", "metadata": {"final_answer": "done"}}),
                )
            )
        ]
    )
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        max_dispatch_steps=1,
        meta_agent=MetaAgentSpec(
            name="meta",
            system_prompt="Return dispatch JSON.",
            llm_backend=BackendBinding(backend_id="meta-llm"),
        ),
        roles={
            "WriteAgent": RoleSpec(
                name="WriteAgent",
                system_prompt="Write.",
                llm_backend=BackendBinding(backend_id="write-llm"),
            )
        },
        runtime_policy=RuntimePolicy(
            metadata={
                "max_meta_dispatch_parse_retries": 0,
                "required_final_artifacts": ["biology_component_records.jsonl"],
            }
        ),
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"meta-llm": meta_llm, "write-llm": RecordingLLMRuntime()},
        memory_runtimes={"memory-local": RecordingMemoryRuntime()},
        skill_runtimes={"skill-local": RecordingSkillRuntime()},
    )

    with pytest.raises(RuntimeError, match="invalid required final artifact"):
        runtime._next_dispatch_decision(
            request=request,
            meta_agent=task_config.meta_agent,
            meta_llm=meta_llm,
            run_ref="meta-test",
            step_index=0,
            role_results=[
                {
                    "status": "completed",
                    "role": "WriteAgent",
                    "run_ref": "write-run",
                    "artifact_refs": [
                        ArtifactRef(
                            uri=str(placeholder_path),
                            type="dataset",
                            metadata={"record_count": 1, "format": "jsonl"},
                        ).model_dump(mode="json")
                    ],
                }
            ],
        )


def test_meta_agent_finish_rejects_json_array_required_jsonl_artifact(tmp_path: Path):
    request = _request()
    request = request.model_copy(update={"goal": "Write biology_component_records.jsonl."})
    array_path = tmp_path / "biology_component_records.jsonl"
    array_path.write_text("[]", encoding="utf-8")
    meta_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps({"action": "finish_task", "metadata": {"final_answer": "done"}}),
                )
            )
        ]
    )
    task_config = TaskConfig(
        task_id=request.task_id,
        goal="Write biology_component_records.jsonl.",
        max_dispatch_steps=1,
        meta_agent=MetaAgentSpec(
            name="meta",
            system_prompt="Return dispatch JSON.",
            llm_backend=BackendBinding(backend_id="meta-llm"),
        ),
        roles={
            "WriteAgent": RoleSpec(
                name="WriteAgent",
                system_prompt="Write.",
                llm_backend=BackendBinding(backend_id="write-llm"),
            )
        },
        runtime_policy=RuntimePolicy(
            metadata={
                "max_meta_dispatch_parse_retries": 0,
                "required_final_artifacts": ["biology_component_records.jsonl"],
            }
        ),
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"meta-llm": meta_llm, "write-llm": RecordingLLMRuntime()},
        memory_runtimes={"memory-local": RecordingMemoryRuntime()},
        skill_runtimes={"skill-local": RecordingSkillRuntime()},
    )

    with pytest.raises(RuntimeError, match="line 1 is not a JSON object"):
        runtime._next_dispatch_decision(
            request=request,
            meta_agent=task_config.meta_agent,
            meta_llm=meta_llm,
            run_ref="meta-test",
            step_index=0,
            role_results=[
                {
                    "status": "completed",
                    "role": "WriteAgent",
                    "run_ref": "write-run",
                    "artifact_refs": [
                        ArtifactRef(
                            uri=str(array_path),
                            type="dataset",
                            metadata={"record_count": 0, "format": "jsonl"},
                        ).model_dump(mode="json")
                    ],
                }
            ],
        )


def test_meta_agent_finish_does_not_count_report_source_uri_as_required_jsonl(tmp_path: Path):
    request = _request()
    report_path = tmp_path / "0-biology_component_report.json"
    report_path.write_text('{"records_path":"biology_component_records.jsonl"}', encoding="utf-8")
    meta_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps({"action": "finish_task", "metadata": {"final_answer": "done"}}),
                )
            )
        ]
    )
    task_config = TaskConfig(
        task_id=request.task_id,
        goal="Write biology_component_records.jsonl.",
        max_dispatch_steps=1,
        meta_agent=MetaAgentSpec(
            name="meta",
            system_prompt="Return dispatch JSON.",
            llm_backend=BackendBinding(backend_id="meta-llm"),
        ),
        roles={
            "WriteAgent": RoleSpec(
                name="WriteAgent",
                system_prompt="Write.",
                llm_backend=BackendBinding(backend_id="write-llm"),
            )
        },
        runtime_policy=RuntimePolicy(
            metadata={
                "max_meta_dispatch_parse_retries": 0,
                "required_final_artifacts": ["biology_component_records.jsonl"],
            }
        ),
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"meta-llm": meta_llm, "write-llm": RecordingLLMRuntime()},
        memory_runtimes={"memory-local": RecordingMemoryRuntime()},
        skill_runtimes={"skill-local": RecordingSkillRuntime()},
    )

    with pytest.raises(RuntimeError, match="missing required final artifact"):
        runtime._next_dispatch_decision(
            request=request,
            meta_agent=task_config.meta_agent,
            meta_llm=meta_llm,
            run_ref="meta-test",
            step_index=0,
            role_results=[
                {
                    "status": "completed",
                    "role": "WriteAgent",
                    "run_ref": "write-run",
                    "artifact_refs": [
                        ArtifactRef(
                            uri=str(report_path),
                            type="log",
                            metadata={
                                "format": "json",
                                "source_uri": str(tmp_path / "biology_component_records.jsonl"),
                            },
                        ).model_dump(mode="json")
                    ],
                }
            ],
        )


def test_meta_agent_finish_rejects_missing_required_artifact_group(tmp_path: Path):
    request = _request()
    records_path = tmp_path / "biology_component_records.jsonl"
    records_path.write_text(
        (
            '{"article_id":"paper-1","component_name":"pTet","component_type":"promoter",'
            '"evidence_text":"pTet promoter was used.","evidence_source":"main_text.md"}\n'
        ),
        encoding="utf-8",
    )
    meta_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps({"action": "finish_task", "metadata": {"final_answer": "done"}}),
                )
            )
        ]
    )
    task_config = TaskConfig(
        task_id=request.task_id,
        goal="Write biology_component_records.jsonl and a final report.",
        max_dispatch_steps=1,
        meta_agent=MetaAgentSpec(
            name="meta",
            system_prompt="Return dispatch JSON.",
            llm_backend=BackendBinding(backend_id="meta-llm"),
        ),
        roles={
            "WriteAgent": RoleSpec(
                name="WriteAgent",
                system_prompt="Write.",
                llm_backend=BackendBinding(backend_id="write-llm"),
            )
        },
        runtime_policy=RuntimePolicy(
            metadata={
                "max_meta_dispatch_parse_retries": 0,
                "required_final_artifacts": ["biology_component_records.jsonl"],
                "required_final_artifact_groups": [
                    {"one_of": ["biology_component_report.md", "biology_component_report.json"]}
                ],
            }
        ),
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"meta-llm": meta_llm, "write-llm": RecordingLLMRuntime()},
        memory_runtimes={"memory-local": RecordingMemoryRuntime()},
        skill_runtimes={"skill-local": RecordingSkillRuntime()},
    )

    with pytest.raises(RuntimeError, match="missing required final artifact group"):
        runtime._next_dispatch_decision(
            request=request,
            meta_agent=task_config.meta_agent,
            meta_llm=meta_llm,
            run_ref="meta-test",
            step_index=0,
            role_results=[
                {
                    "status": "completed",
                    "role": "WriteAgent",
                    "run_ref": "write-run",
                    "artifact_refs": [
                        ArtifactRef(
                            uri=str(records_path),
                            type="dataset",
                            metadata={"record_count": 1, "format": "jsonl"},
                        ).model_dump(mode="json")
                    ],
                }
            ],
        )


def test_meta_agent_finish_accepts_one_required_artifact_group_member(tmp_path: Path):
    request = _request()
    records_path = tmp_path / "biology_component_records.jsonl"
    records_path.write_text(
        (
            '{"article_id":"paper-1","component_name":"pTet","component_type":"promoter",'
            '"evidence_text":"pTet promoter was used.","evidence_source":"main_text.md"}\n'
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "biology_component_report.md"
    report_path.write_text("# Report\n", encoding="utf-8")
    meta_llm = ScriptedLLMRuntime(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps({"action": "finish_task", "metadata": {"final_answer": "done"}}),
                )
            )
        ]
    )
    task_config = TaskConfig(
        task_id=request.task_id,
        goal="Write biology_component_records.jsonl and a final report.",
        max_dispatch_steps=1,
        meta_agent=MetaAgentSpec(
            name="meta",
            system_prompt="Return dispatch JSON.",
            llm_backend=BackendBinding(backend_id="meta-llm"),
        ),
        roles={
            "WriteAgent": RoleSpec(
                name="WriteAgent",
                system_prompt="Write.",
                llm_backend=BackendBinding(backend_id="write-llm"),
            )
        },
        runtime_policy=RuntimePolicy(
            metadata={
                "max_meta_dispatch_parse_retries": 0,
                "required_final_artifacts": ["biology_component_records.jsonl"],
                "required_final_artifact_groups": [
                    {"one_of": ["biology_component_report.md", "biology_component_report.json"]}
                ],
            }
        ),
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"meta-llm": meta_llm, "write-llm": RecordingLLMRuntime()},
        memory_runtimes={"memory-local": RecordingMemoryRuntime()},
        skill_runtimes={"skill-local": RecordingSkillRuntime()},
    )

    decision, *_ = runtime._next_dispatch_decision(
        request=request,
        meta_agent=task_config.meta_agent,
        meta_llm=meta_llm,
        run_ref="meta-test",
        step_index=0,
        role_results=[
            {
                "status": "completed",
                "role": "WriteAgent",
                "run_ref": "write-run",
                "artifact_refs": [
                    ArtifactRef(
                        uri=str(records_path),
                        type="dataset",
                        metadata={"record_count": 1, "format": "jsonl"},
                    ).model_dump(mode="json"),
                    ArtifactRef(
                        uri=str(report_path),
                        type="log",
                        metadata={"filename": "biology_component_report.md"},
                    ).model_dump(mode="json"),
                ],
            }
        ],
    )

    assert decision.action == "finish_task"


def test_default_task_runtime_skips_non_updated_memory_state_lineage(tmp_path: Path):
    request = _request()
    agent_memory = RecordingMemoryRuntime(
        backend_id="agent-memory",
        state_ref="agent-state-before",
        update_result={"status": "failed", "state_ref": "agent-state-after"},
    )
    task_memory = RecordingMemoryRuntime(
        backend_id="task-memory",
        state_ref="task-state-before",
        update_result={"status": "degraded", "state_ref": "task-state-after"},
    )
    backend_state_registry = FileBackendStateRegistry(tmp_path / "backend-state")
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        task_memory_backend=BackendBinding(backend_id="task-memory"),
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="You solve scientific tasks.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                agent_memory_backend=BackendBinding(backend_id="agent-memory"),
            )
        },
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        backend_state_registry=backend_state_registry,
        llm_runtimes={"llm-local": RecordingLLMRuntime()},
        memory_runtimes={"agent-memory": agent_memory, "task-memory": task_memory},
        skill_runtimes={"skill-local": RecordingSkillRuntime()},
    )

    runtime.run(request)

    assert backend_state_registry.list_states() == []

def test_default_task_runtime_records_valid_memory_artifacts_without_failing_on_invalid_refs(
    tmp_path: Path,
):
    request = _request()
    valid_artifact = ArtifactRef(
        uri=str(tmp_path / "memory-update.json"),
        type="log",
        metadata={"scope": "task"},
    )
    agent_memory = RecordingMemoryRuntime(
        backend_id="agent-memory",
        state_ref="agent-state-before",
        update_result={"status": "updated", "state_ref": "agent-state-after"},
    )
    task_memory = RecordingMemoryRuntime(
        backend_id="task-memory",
        state_ref="task-state-before",
        update_result={
            "status": "updated",
            "state_ref": "task-state-after",
            "artifact_refs": [
                valid_artifact.model_dump(mode="json"),
                {"uri": "file:///missing-type"},
            ],
        },
    )
    backend_state_registry = FileBackendStateRegistry(tmp_path / "backend-state")
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        task_memory_backend=BackendBinding(backend_id="task-memory"),
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="You solve scientific tasks.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                agent_memory_backend=BackendBinding(backend_id="agent-memory"),
            )
        },
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        backend_state_registry=backend_state_registry,
        llm_runtimes={"llm-local": RecordingLLMRuntime()},
        memory_runtimes={"agent-memory": agent_memory, "task-memory": task_memory},
        skill_runtimes={"skill-local": RecordingSkillRuntime()},
    )

    runtime.run(request)

    assert backend_state_registry.get_state("agent-state-after") is None
    task_record = backend_state_registry.get_state("task-state-after")
    assert task_record is not None
    assert task_record.artifact_refs == [valid_artifact]
    assert task_record.metadata["invalid_artifact_refs"] == [{"uri": "file:///missing-type"}]


@pytest.mark.parametrize(
    ("role_agent_memory_backend", "task_memory_backend", "missing_field"),
    [
        (None, BackendBinding(backend_id="task-memory"), "agent_memory_backend"),
        (BackendBinding(backend_id="agent-memory"), None, "task_memory_backend"),
    ],
)
def test_default_task_runtime_tolerates_partial_memory_scope_bindings_with_task_level_memory(
    role_agent_memory_backend: BackendBinding | None,
    task_memory_backend: BackendBinding | None,
    missing_field: str,
):
    request = _request()
    task_config = TaskConfig(
        task_id=request.task_id,
        goal=request.goal,
        task_memory_backend=task_memory_backend,
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="You solve scientific tasks.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                agent_memory_backend=role_agent_memory_backend,
            )
        },
    )
    memory_runtimes = {
        "memory-local": RecordingMemoryRuntime(),
        "agent-memory": RecordingMemoryRuntime(),
        "task-memory": RecordingMemoryRuntime(),
    }
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        llm_runtimes={"llm-local": RecordingLLMRuntime()},
        memory_runtimes=memory_runtimes,
        skill_runtimes={"skill-local": RecordingSkillRuntime()},
    )

    result = runtime.run(request)

    assert result["status"] == "completed"
    assert memory_runtimes["agent-memory"].search_requests == []
    assert memory_runtimes["agent-memory"].add_calls == []
    if task_memory_backend is None:
        assert memory_runtimes["memory-local"].search_requests
    else:
        assert memory_runtimes["task-memory"].search_requests

def test_supplied_backend_categories_are_initialized_at_startup(tmp_path: Path):
    layout = LabLayout(tmp_path / "lab")
    llm_backend = FakeBackend("llm-local")
    memory_backend = FakeBackend("memory-local")
    skill_runtime = DirectRuntime()
    worker = TaskWorker(
        layout=layout,
        worker_id="worker-1",
        llm_backends={"llm-local": llm_backend},
        memory_backends={"memory-local": memory_backend},
        skill_backends={"skill-local": skill_runtime},
        llm_backend_bindings=[BackendBinding(backend_id="llm-local", state_ref="state-1")],
        memory_backend_bindings=[BackendBinding(backend_id="memory-local", state_ref="state-2")],
        skill_backend_bindings=[BackendBinding(backend_id="skill-local", state_ref="state-3")],
    )

    worker.startup()

    assert llm_backend.instantiated_state_refs == ["state-1"]
    assert memory_backend.instantiated_state_refs == ["state-2"]
    assert worker.llm_runtimes["llm-local"] == {
        "backend_id": "llm-local",
        "state_ref": "state-1",
    }
    assert worker.memory_runtimes["memory-local"] == {
        "backend_id": "memory-local",
        "state_ref": "state-2",
    }
    assert worker.skill_runtimes["skill-local"] is skill_runtime


def test_task_worker_instantiates_llm_backend_with_active_state_when_binding_has_none(tmp_path: Path):
    layout = LabLayout(tmp_path / "lab")
    state_registry = FileBackendStateRegistry(layout.registries_dir / "backend_state")
    state_registry.register_candidate(
        BackendStateRecord(
            state_ref="local-trainable://llm-local/state/promoted",
            backend_id="llm-local",
            backend_type="llm",
        )
    )
    state_registry.promote(
        "llm-local",
        "local-trainable://llm-local/state/promoted",
        "evo-1",
    )
    llm_backend = FakeBackend("llm-local")
    worker = TaskWorker(
        layout=layout,
        worker_id="worker-1",
        backend_state_registry=state_registry,
        llm_backends={"llm-local": llm_backend},
        llm_backend_bindings=[BackendBinding(backend_id="llm-local")],
    )

    worker.startup()

    assert llm_backend.instantiated_state_refs == ["local-trainable://llm-local/state/promoted"]
    assert worker.llm_runtimes["llm-local"] == {
        "backend_id": "llm-local",
        "state_ref": "local-trainable://llm-local/state/promoted",
    }


def test_task_worker_initializes_only_bound_and_memory_dependency_llm_backends(tmp_path: Path):
    layout = LabLayout(tmp_path / "lab")
    solver_llm = FakeBackend("solver-llm")
    memory_llm = FakeBackend("memory-llm")
    unused_llm = RaisingBackend("unused-llm")
    embedding_backend = FakeBackend("memory-embedding")
    memory_backend = MethodMemoryBackend(
        backend_id="agent-memory",
        method=Mem0MemoryMethod(
            store_path=tmp_path / "agent-memory.sqlite",
            llm_backend_id="memory-llm",
            embedding_backend_id="memory-embedding",
        ),
    )
    task_config = TaskConfig(
        task_id="task-1",
        goal="Solve the task.",
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="Solve.",
                llm_backend=BackendBinding(backend_id="solver-llm"),
                agent_memory_backend=BackendBinding(backend_id="agent-memory"),
            )
        },
    )
    worker = TaskWorker(
        layout=layout,
        worker_id="worker-1",
        task_config=task_config,
        llm_backends={
            "solver-llm": solver_llm,
            "memory-llm": memory_llm,
            "unused-llm": unused_llm,
        },
        embedding_backends={"memory-embedding": embedding_backend},
        memory_backends={"agent-memory": memory_backend},
        llm_backend_bindings=[BackendBinding(backend_id="solver-llm")],
        memory_backend_bindings=[BackendBinding(backend_id="agent-memory")],
    )

    worker.startup()

    assert solver_llm.instantiated_state_refs == [None]
    assert memory_llm.instantiated_state_refs == [None]
    assert unused_llm.instantiated_state_refs == []
    assert memory_backend.method.llm_runtime == {
        "backend_id": "memory-llm",
        "state_ref": None,
    }


def test_task_worker_does_not_consume_or_schedule_evolve_jobs(tmp_path: Path):
    layout = LabLayout(tmp_path / "lab")
    _enqueue_task(layout, _request())
    FileWorkQueue(layout.evolve_queue_dir).enqueue(
        "evolve-1",
        {"request_payload_uri": str(layout.root / "evolve-request.json")},
    )
    worker = TaskWorker(layout=layout, worker_id="worker-1", task_runtime=FakeRuntime())
    worker.startup()

    worker.run_once()

    assert sorted((layout.evolve_queue_dir / "queued").glob("*.json"))
    assert not list((layout.evolve_queue_dir / "claimed").glob("*.json"))
    assert not list((layout.evolve_queue_dir / "done").glob("*.json"))
    assert not list((layout.evolve_queue_dir / "failed").glob("*.json"))
