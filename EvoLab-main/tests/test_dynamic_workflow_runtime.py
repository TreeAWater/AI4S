from __future__ import annotations

import json
from pathlib import Path

import pytest

from evolab.backends.llm.fake import FakeLLMRuntime
from evolab.backends.memory import FakeMemoryBackend, NullMemoryBackend
from evolab.backends.skills import FakeSkillBackend
from evolab.config.agents import render_agents_markdown
from evolab.config.task_config import BackendBinding, MetaAgentSpec, RoleSpec, TaskConfig
from evolab.contracts.common import ArtifactRef, RuntimePolicy
from evolab.contracts.dynamic_workflow import DynamicSubagentsConfig, DynamicWorkflowSpec, DynamicWorkflowValidationReport
from evolab.contracts.generated_tools import TaskEffectiveToolCatalog
from evolab.contracts.lab_state import ArtifactIndexRecord
from evolab.contracts.llm import LLMGenerationConfig, LLMRuntimeRequest, LLMRuntimeResponse, SubAgentAction
from evolab.contracts.retrieval import SkillItem
from evolab.contracts.state import BackendStateRecord
from evolab.contracts.task import TaskOrigin, TaskPurpose, TaskRequest
from evolab.contracts.tools import ToolCall, ToolResult, ToolSpec
from evolab.lab.layout import LabLayout
from evolab.lab.resolver import LabResolver
from evolab.registries.backend_state import FileBackendStateRegistry
from evolab.registries.lab_state import FileLabStateRegistry
from evolab.registries.task import FileTaskRegistry
from evolab.registries.trajectory import FileTrajectoryRegistry
from evolab.runtime.prompt_builder import PromptBuilder
from evolab.runtime.dynamic_workflow import persist_dynamic_workflow_artifacts, validate_dynamic_workflow_spec
from evolab.runtime.task_runtime import (
    TaskRuntime,
    _enforce_dynamic_node_output_contract,
    _recover_dynamic_final_records_outputs,
)
from evolab.tools.runtime import ToolRegistry, ToolRuntime


def _layout(lab_root: Path) -> LabLayout:
    return LabLayout(lab_root)


def _trajectory_registry(lab_root: Path) -> FileTrajectoryRegistry:
    return FileTrajectoryRegistry(_layout(lab_root).registries_dir / "trajectory")


def _task_registry(lab_root: Path) -> FileTaskRegistry:
    return FileTaskRegistry(_layout(lab_root).registries_dir / "task")


def _lab_state_registry(lab_root: Path) -> FileLabStateRegistry:
    return FileLabStateRegistry(_layout(lab_root).registries_dir / "lab_state")


def _backend_state_registry(lab_root: Path) -> FileBackendStateRegistry:
    return FileBackendStateRegistry(_layout(lab_root).registries_dir / "backend_state")


def _work_item_path(lab_root: Path, task_id: str, work_item_id: str) -> Path:
    return _layout(lab_root).registries_dir / "lab_state" / "work_items" / task_id / f"{work_item_id}.json"


def _agents_updates_path(lab_root: Path) -> Path:
    return _layout(lab_root).agents_path.with_name(_layout(lab_root).agents_path.name + ".updates.jsonl")


def test_dynamic_runtime_executes_two_node_workflow_and_records_traces(tmp_path: Path):
    lab_root = tmp_path / "lab"
    runtime = _runtime(
        lab_root,
        planner_payload=_workflow_payload(),
        worker_responses=[
            _tool_response("write_report", {"artifact_name": "context.json", "content": {"summary": "context"}}),
            "context done",
            _tool_response("write_report", {"artifact_name": "final.json", "content": {"summary": "final"}}),
            "writer done",
        ],
        dynamic_config=_dynamic_config(),
    )
    request = _request()

    result = runtime.run(request)

    assert result["execution_mode"] == "dynamic"
    assert result["status"] == "completed"
    assert [run["role"] for run in result["runs"]] == ["TextContextAgent", "EvidenceWriterAgent"]
    trajectory = _trajectory_registry(lab_root)
    assert [record.role for record in trajectory.list_subagent_runs()] == ["TextContextAgent", "EvidenceWriterAgent"]
    dynamic_root = _layout(lab_root).state_root / "dynamic_workflows" / request.task_id / "wf-runtime"
    assert (dynamic_root / "dynamic_workflow_spec.json").exists()
    assert (dynamic_root / "dynamic_subagents.json").exists()
    assert (dynamic_root / "dynamic_workflow_trace.json").exists()
    assert (dynamic_root / "dynamic_subagent_records.jsonl").exists()
    trace = json.loads((dynamic_root / "dynamic_workflow_trace.json").read_text(encoding="utf-8"))
    assert trace["status"] == "completed"
    assert len(trace["run_refs"]) == 2


def test_dynamic_worker_uses_task_memory_only(tmp_path: Path):
    result, resolver, worker_memory = _run_minimal_dynamic_task_with_memory(tmp_path)

    assert result["status"] == "completed"
    assert worker_memory.search_requests == []
    assert worker_memory.add_requests == []
    subagent_run = resolver.trajectory_registry().list_subagent_runs()[-1]
    metadata = subagent_run.metadata
    assert metadata["memory_mode"] == "task_only"
    assert "agent_memory_update_result" not in metadata
    assert metadata["task_memory_update_result"]["metadata"]["memory_scope"] == "task"
    state_records = resolver.backend_state_registry().list_states()
    assert all(
        record.metadata.get("memory_scope") != "agent"
        for record in state_records
        if record.metadata.get("role") != "meta"
    )


def test_dynamic_worker_memory_mode_metadata_cannot_enable_agent_memory():
    from evolab.runtime.task_runtime import _worker_memory_mode

    assert _worker_memory_mode({"execution_mode": "dynamic", "worker_memory_mode": "agent_and_task"}) == "task_only"


def test_dynamic_workflow_validation_accepts_generated_tool_from_effective_catalog():
    registry = ToolRegistry()
    runtime = ToolRuntime(registry)
    generated_spec = ToolSpec(
        name="gt_task_extract",
        description="Generated extract.",
        parameters_schema={"type": "object"},
        metadata={"generated_tool": True},
    )
    runtime.activate_generated_tool_scope("task-dynamic")
    runtime.register_task_generated_tool(generated_spec, lambda args: "ok", provenance={"code_hash": "abc"})
    catalog = TaskEffectiveToolCatalog(
        task_id="task-dynamic",
        builtin_allowed_tool_names=[],
        generated_tool_names=["gt_task_extract"],
        tool_specs_by_name={"gt_task_extract": generated_spec},
        provenance_by_name={"gt_task_extract": {"code_hash": "abc"}},
    )
    payload = _single_node_payload()
    payload["dynamic_subagents"][0]["allowed_tools"] = ["gt_task_extract"]
    spec = DynamicWorkflowSpec.model_validate(payload)

    _prepared, report = validate_dynamic_workflow_spec(
        spec,
        config=_dynamic_config(allowed_tool_names=[]),
        available_llm_backend_ids={"planner", "worker"},
        tool_runtime=runtime,
        skill_backend=None,
        task_id="task-dynamic",
        effective_tool_catalog=catalog,
    )

    assert report.valid is True
    assert report.allowed_tool_names == ["gt_task_extract"]
    assert report.metadata["configured_builtin_allowed_tool_names"] == []


def test_dynamic_runtime_runs_tool_code_preplanning_before_planner(tmp_path: Path):
    lab_root = tmp_path / "lab"
    generated_package = {
        "schema_version": "v1",
        "tool_name": "extract_fact",
        "reason": "The task benefits from a task-local extraction helper.",
        "manifest": {"parameters_schema": {"type": "object"}},
        "primary_module": "tool.py",
        "files": [
            {
                "schema_version": "v1",
                "path": "tool.py",
                "content": (
                    "TOOL_SPEC = {\n"
                    "    'name': 'extract_fact',\n"
                    "    'description': 'Extract a synthetic fact.',\n"
                    "    'parameters_schema': {'type': 'object'},\n"
                    "}\n\n"
                    "def run(arguments, context):\n"
                    "    return {'status': 'ok', 'content': 'fact: ok', 'metadata': {'used': True}}\n"
                ),
            }
        ],
        "smoke_tests": [{"schema_version": "v1", "name": "loads", "arguments": {}}],
    }
    meta = FakeLLMRuntime(
        responses=[
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "route": "END",
                            "instruction": "Register a task-local generated tool before planning.",
                            "metadata": {"generated_tool_package": generated_package},
                        }
                    ),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "route": "END",
                            "instruction": "No reusable role-pool update needed.",
                            "metadata": {"no_role_pool_update_reason": "existing roles are enough"},
                        }
                    ),
                )
            ),
        ]
    )
    planner = _GeneratedToolAwarePlanner()
    task_config = TaskConfig(
        task_id="task-dynamic",
        goal="Process synthetic item.",
        meta_agent=MetaAgentSpec(
            system_prompt="Preplan generated tools before dynamic planning.",
            llm_backend=BackendBinding(backend_id="meta"),
        ),
        dynamic_subagents=_dynamic_config(),
        runtime_policy=RuntimePolicy(max_tool_steps=1, metadata={"max_meta_dispatch_parse_retries": 0}),
    )
    runtime = _base_runtime(
        lab_root,
        task_config=task_config,
        llm_runtimes={
            "planner": planner,
            "meta": meta,
            "worker": FakeLLMRuntime(
                responses=[LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="done"))]
            ),
        },
    )

    result = runtime.run(_request())

    assert result["status"] == "completed"
    assert len(meta.requests) == 2
    tool_preplanning_payload = json.loads(meta.requests[0].messages[-1].content)
    assert tool_preplanning_payload["preplanning_context"]["stage"] == "tool_code_evolution"
    planner_payload = json.loads(planner.requests[0].messages[-1].content)
    generated_names = [
        name for name in planner_payload["effective_allowed_tool_names"] if name.startswith("gt_task_dynamic_")
    ]
    assert len(generated_names) == 1
    generated_name = generated_names[0]
    assert planner.selected_tool_name == generated_name
    assert generated_name in planner_payload["effective_tool_catalog"]
    assert planner_payload["effective_tool_catalog"][generated_name]["description"] == "Extract a synthetic fact."
    provenance = result["runs"][0]["dispatch_metadata"]["dynamic_subagent_provenance"]
    assert generated_name in provenance["prepared_tool_names"]
    events = _trajectory_registry(lab_root).list_events()
    assert any(event.event_type == "generated_tool_registered" for event in events)
    meta_runs = _trajectory_registry(lab_root).list_meta_agent_runs()
    tool_code_run = next(run for run in meta_runs if run.metadata["step_index"] == -2)
    registration_result = tool_code_run.metadata["dispatch_metadata"]["generated_tool_registration_result"]
    assert registration_result["status"] == "registered"
    assert registration_result["registration"]["registered_tool_name"] == generated_name


def test_dynamic_runtime_records_generated_tool_provenance_in_skill_observation(tmp_path: Path):
    lab_root = tmp_path / "lab"
    generated_package = {
        "schema_version": "v1",
        "tool_name": "extract_fact",
        "reason": "The task benefits from a task-local extraction helper.",
        "manifest": {"parameters_schema": {"type": "object"}},
        "primary_module": "tool.py",
        "files": [
            {
                "schema_version": "v1",
                "path": "tool.py",
                "content": (
                    "TOOL_SPEC = {\n"
                    "    'name': 'extract_fact',\n"
                    "    'description': 'Extract a synthetic fact.',\n"
                    "    'parameters_schema': {'type': 'object'},\n"
                    "}\n\n"
                    "def run(arguments, context):\n"
                    "    return {\n"
                    "        'status': 'ok',\n"
                    "        'content': 'fact: ok',\n"
                    "        'metadata': {'used': True, 'row_count': 1},\n"
                    "    }\n"
                ),
            }
        ],
        "smoke_tests": [{"schema_version": "v1", "name": "loads", "arguments": {}}],
    }
    meta = FakeLLMRuntime(responses=[_generated_tool_package_response(generated_package), _role_pool_noop_response()])
    planner = _GeneratedToolAwarePlanner()
    worker = _GeneratedToolUser()
    task_config = TaskConfig(
        task_id="task-dynamic",
        goal="Process synthetic item.",
        meta_agent=MetaAgentSpec(
            system_prompt="Preplan generated tools before dynamic planning.",
            llm_backend=BackendBinding(backend_id="meta"),
        ),
        dynamic_subagents=_dynamic_config(),
        runtime_policy=RuntimePolicy(max_tool_steps=1, metadata={"max_meta_dispatch_parse_retries": 0}),
    )
    runtime = _base_runtime(
        lab_root,
        task_config=task_config,
        llm_runtimes={"planner": planner, "meta": meta, "worker": worker},
    )
    skill = runtime.skill_runtimes["skill"]

    result = runtime.run(_request())

    assert result["status"] == "completed"
    generated_name = planner.selected_tool_name
    assert generated_name is not None
    assert worker.selected_tool_name == generated_name
    assert skill.look_at_events
    observation_metadata = skill.look_at_events[-1]["metadata"]
    generated_tools = observation_metadata["generated_tools"]
    assert len(generated_tools) == 1
    assert generated_tools[0]["name"] == generated_name
    assert generated_tools[0]["requested_tool_name"] == "extract_fact"
    assert generated_tools[0]["registered_tool_name"] == generated_name
    assert isinstance(generated_tools[0]["code_hash"], str)
    assert len(generated_tools[0]["code_hash"]) == 64
    assert generated_tools[0]["validation"]["valid"] is True
    assert generated_tools[0]["module_path"].endswith(f"{generated_name}/tool.py")
    subagent_run = _trajectory_registry(lab_root).list_subagent_runs()[-1]
    tool_call = subagent_run.tool_calls[0]
    assert tool_call.tool_call.name == generated_name
    assert tool_call.result.metadata["used"] is True
    assert tool_call.result.metadata["row_count"] == 1
    assert tool_call.result.metadata["generated_tool"]["registered_tool_name"] == generated_name


def test_dynamic_runtime_builds_incomplete_tool_package_with_builder(tmp_path: Path):
    lab_root = tmp_path / "lab"
    complete_package = {
        "schema_version": "v1",
        "tool_name": "extract_fact",
        "reason": "The builder completed the task-local extraction helper.",
        "manifest": {"parameters_schema": {"type": "object"}},
        "primary_module": "tool.py",
        "files": [
            {
                "schema_version": "v1",
                "path": "tool.py",
                "content": (
                    "TOOL_SPEC = {\n"
                    "    'name': 'extract_fact',\n"
                    "    'description': 'Extract a built synthetic fact.',\n"
                    "    'parameters_schema': {'type': 'object'},\n"
                    "}\n\n"
                    "def run(arguments, context):\n"
                    "    return {'status': 'ok', 'content': 'built fact', 'metadata': {}}\n"
                ),
            }
        ],
        "smoke_tests": [{"schema_version": "v1", "name": "loads", "arguments": {}}],
    }
    meta = FakeLLMRuntime(
        responses=[
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "route": "END",
                            "instruction": "Build a task-local generated tool before planning.",
                            "metadata": {
                                "generated_tool_package": {
                                    "schema_version": "v1",
                                    "tool_name": "extract_fact",
                                    "reason": "Need a generated helper, but source is delegated to builder.",
                                    "files": [],
                                }
                            },
                        }
                    ),
                )
            ),
            _role_pool_noop_response(),
        ]
    )
    builder = FakeLLMRuntime(
        responses=[
            LLMRuntimeResponse(
                action=SubAgentAction(action="final_answer", content=json.dumps(complete_package))
            )
        ]
    )
    planner = _GeneratedToolAwarePlanner()
    task_config = TaskConfig(
        task_id="task-dynamic",
        goal="Process synthetic item.",
        meta_agent=MetaAgentSpec(
            system_prompt="Preplan generated tools before dynamic planning.",
            llm_backend=BackendBinding(backend_id="meta"),
        ),
        dynamic_subagents=_dynamic_config(),
        runtime_policy=RuntimePolicy(
            max_tool_steps=1,
            metadata={
                "max_meta_dispatch_parse_retries": 0,
                "generated_tool_builder_backend_id": "builder",
            },
        ),
    )
    runtime = _base_runtime(
        lab_root,
        task_config=task_config,
        llm_runtimes={
            "planner": planner,
            "meta": meta,
            "builder": builder,
            "worker": FakeLLMRuntime(default_content="done"),
        },
    )

    result = runtime.run(_request())

    assert result["status"] == "completed"
    assert builder.requests
    assert builder.requests[0].generation_config.metadata["runtime_stage"] == "generated_tool_builder"
    builder_payload = json.loads(builder.requests[0].messages[-1].content)
    assert builder_payload["requested_tool_name"] == "extract_fact"
    assert builder_payload["capability_grant"]["allow_network"] is False
    assert {role["name"] for role in builder_payload["role_pool_templates"]} >= {
        "TextContextAgent",
        "EvidenceWriterAgent",
    }
    assert planner.selected_tool_name is not None
    assert planner.selected_tool_name.startswith("gt_task_dynamic_")
    meta_runs = _trajectory_registry(lab_root).list_meta_agent_runs()
    tool_code_run = next(run for run in meta_runs if run.metadata["step_index"] == -2)
    registration_result = tool_code_run.metadata["dispatch_metadata"]["generated_tool_registration_result"]
    assert registration_result["status"] == "registered"


def test_dynamic_runtime_records_rejected_generated_tool_when_builder_fails(tmp_path: Path):
    lab_root = tmp_path / "lab"
    meta = FakeLLMRuntime(
        responses=[
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "route": "END",
                            "instruction": "Build a task-local generated tool before planning.",
                            "metadata": {
                                "generated_tool_package": {
                                    "schema_version": "v1",
                                    "tool_name": "extract_fact",
                                    "reason": "Need a generated helper, but source is delegated to builder.",
                                    "files": [],
                                }
                            },
                        }
                    ),
                )
            ),
            _role_pool_noop_response(),
        ]
    )
    builder = FakeLLMRuntime(
        responses=[
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="{not valid json"))
        ]
    )
    planner = FakeLLMRuntime(default_content=json.dumps(_single_node_payload()))
    task_config = TaskConfig(
        task_id="task-dynamic",
        goal="Process synthetic item.",
        meta_agent=MetaAgentSpec(
            system_prompt="Preplan generated tools before dynamic planning.",
            llm_backend=BackendBinding(backend_id="meta"),
        ),
        dynamic_subagents=_dynamic_config(),
        runtime_policy=RuntimePolicy(
            max_tool_steps=1,
            metadata={
                "max_meta_dispatch_parse_retries": 0,
                "generated_tool_builder_backend_id": "builder",
            },
        ),
    )
    runtime = _base_runtime(
        lab_root,
        task_config=task_config,
        llm_runtimes={
            "planner": planner,
            "meta": meta,
            "builder": builder,
            "worker": FakeLLMRuntime(default_content="done"),
        },
    )

    result = runtime.run(_request())

    assert result["status"] == "completed"
    planner_payload = json.loads(planner.requests[0].messages[-1].content)
    assert not any(name.startswith("gt_task_dynamic_") for name in planner_payload["effective_allowed_tool_names"])
    events = _trajectory_registry(lab_root).list_events()
    rejected = next(event for event in events if event.event_type == "generated_tool_rejected")
    assert rejected.metadata["status"] == "rejected"
    assert "invalid JSON" in rejected.metadata["failure_reason"]
    meta_runs = _trajectory_registry(lab_root).list_meta_agent_runs()
    tool_code_run = next(run for run in meta_runs if run.metadata["step_index"] == -2)
    registration_result = tool_code_run.metadata["dispatch_metadata"]["generated_tool_registration_result"]
    assert registration_result["status"] == "rejected"


def test_dynamic_runtime_repair_builder_prefers_resolved_role_backend(tmp_path: Path):
    lab_root = tmp_path / "lab"
    generated_package = {
        "schema_version": "v1",
        "tool_name": "recover_missing_rows",
        "reason": "The role requested a missing task-local recovery helper.",
        "manifest": {"parameters_schema": {"type": "object"}},
        "primary_module": "tool.py",
        "files": [
            {
                "schema_version": "v1",
                "path": "tool.py",
                "content": (
                    "TOOL_SPEC = {\n"
                    "    'name': 'recover_missing_rows',\n"
                    "    'description': 'Recover rows for this task.',\n"
                    "    'parameters_schema': {'type': 'object'},\n"
                    "}\n\n"
                    "def run(arguments, context):\n"
                    "    return {'status': 'ok', 'content': 'recovered', 'metadata': {'source': 'role_builder'}}\n"
                ),
            }
        ],
        "smoke_tests": [{"schema_version": "v1", "name": "loads", "arguments": {}}],
    }
    workflow_payload = _single_node_payload()
    workflow_payload["dynamic_subagents"][0]["role_name"] = "RoleBackendAgent"
    workflow_payload["dynamic_subagents"][0]["allowed_tools"] = ["read_text"]
    workflow_payload["dynamic_subagents"][0]["llm_backend_id"] = "role-builder"
    role_builder = FakeLLMRuntime(
        responses=[
            _tool_response("recover_missing_rows", {"path": "/tmp/table.md"}),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content=json.dumps(generated_package))),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="done")),
        ]
    )
    worker = FakeLLMRuntime(default_content="worker should not build generated tools")
    task_config = TaskConfig(
        task_id="task-dynamic",
        goal="Process synthetic item.",
        meta_agent=MetaAgentSpec(
            system_prompt="Preplan generated tools before dynamic planning.",
            llm_backend=BackendBinding(backend_id="meta"),
        ),
        dynamic_subagents=_dynamic_config(),
        runtime_policy=RuntimePolicy(
            enable_runtime_capability_repair=True,
            max_tool_steps=2,
            metadata={"max_meta_dispatch_parse_retries": 0},
        ),
    )
    runtime = _base_runtime(
        lab_root,
        task_config=task_config,
        llm_runtimes={
            "planner": FakeLLMRuntime(default_content=json.dumps(workflow_payload)),
            "meta": FakeLLMRuntime(responses=[_no_generated_tool_response(), _role_pool_noop_response()]),
            "role-builder": role_builder,
            "worker": worker,
        },
    )

    result = runtime.run(_request())

    assert result["status"] == "completed"
    builder_requests = [
        request
        for request in role_builder.requests
        if request.generation_config.metadata.get("runtime_stage") == "generated_tool_builder"
    ]
    assert len(builder_requests) == 1
    repair_builder_payload = json.loads(builder_requests[0].messages[-1].content)
    assert {role["name"] for role in repair_builder_payload["role_pool_templates"]} >= {
        "TextContextAgent",
        "EvidenceWriterAgent",
    }
    assert not any(
        request.generation_config.metadata.get("runtime_stage") == "generated_tool_builder"
        for request in worker.requests
    )
    events = _trajectory_registry(lab_root).list_events()
    assert any(event.event_type == "generated_tool_registered" for event in events)


def test_dynamic_runtime_can_disable_tool_code_preplanning_extra_meta_call(tmp_path: Path):
    lab_root = tmp_path / "lab"
    meta = FakeLLMRuntime(responses=[_role_pool_noop_response()])
    planner = FakeLLMRuntime(default_content=json.dumps(_single_node_payload()))
    task_config = TaskConfig(
        task_id="task-dynamic",
        goal="Process synthetic item.",
        meta_agent=MetaAgentSpec(
            system_prompt="Manage role-pool changes.",
            llm_backend=BackendBinding(backend_id="meta"),
        ),
        dynamic_subagents=_dynamic_config(),
        runtime_policy=RuntimePolicy(
            max_tool_steps=1,
            allow_runtime_tool_creation=False,
            metadata={"max_meta_dispatch_parse_retries": 0},
        ),
    )
    runtime = _base_runtime(
        lab_root,
        task_config=task_config,
        llm_runtimes={
            "planner": planner,
            "meta": meta,
            "worker": FakeLLMRuntime(default_content="done"),
        },
    )

    result = runtime.run(_request())

    assert result["status"] == "completed"
    assert len(meta.requests) == 1
    meta_payload = json.loads(meta.requests[0].messages[-1].content)
    assert meta_payload["preplanning_context"]["stage"] == "role_pool_evolution"
    planner_payload = json.loads(planner.requests[0].messages[-1].content)
    assert planner_payload["effective_allowed_tool_names"] == ["read_text", "write_report"]
    events = _trajectory_registry(lab_root).list_events()
    no_op_event = next(event for event in events if event.event_type == "generated_tool_no_op")
    assert no_op_event.metadata["no_generated_tool_reason"] == "runtime tool creation is disabled by policy"


def test_dynamic_runtime_resets_generated_tools_between_tasks(tmp_path: Path):
    lab_root = tmp_path / "lab"
    tool_runtime = _base_runtime(
        lab_root,
        task_config=TaskConfig(
            task_id="task-dynamic",
            goal="Process synthetic item.",
            dynamic_subagents=_dynamic_config(),
            runtime_policy=RuntimePolicy(max_tool_steps=1),
        ),
        llm_runtimes={
            "planner": FakeLLMRuntime(default_content=json.dumps(_single_node_payload())),
            "worker": FakeLLMRuntime(default_content="done"),
        },
    ).tool_runtime
    assert tool_runtime is not None
    tool_runtime.activate_generated_tool_scope("previous-task")
    tool_runtime.register_task_generated_tool(
        ToolSpec(
            name="gt_previous_task_tool",
            description="Previous generated tool.",
            parameters_schema={"type": "object"},
            metadata={"generated_tool": True},
        ),
        lambda args: "old",
        provenance={"task_id": "previous-task"},
    )
    runtime = _base_runtime(
        lab_root,
        task_config=TaskConfig(
            task_id="task-dynamic",
            goal="Process synthetic item.",
            dynamic_subagents=_dynamic_config(),
            runtime_policy=RuntimePolicy(max_tool_steps=1),
        ),
        llm_runtimes={
            "planner": FakeLLMRuntime(default_content=json.dumps(_single_node_payload())),
            "worker": FakeLLMRuntime(default_content="done"),
        },
    )
    runtime.tool_runtime = tool_runtime

    result = runtime.run(_request())

    assert result["status"] == "completed"
    assert tool_runtime.generated_tool_names() == []


def test_dynamic_per_work_item_run_records_lifecycle_without_routing_policy(tmp_path: Path):
    lab_root = tmp_path / "lab"
    payload = _single_node_payload()
    runtime = _runtime(
        lab_root,
        planner_payload=payload,
        worker_responses=["item done"],
        dynamic_config=_dynamic_config(scope="per_work_item"),
    )

    result = runtime.run(_request_with_work_items(["item-a"]))

    assert result["status"] == "completed"
    work_item_record = _work_item_path(lab_root, "task-dynamic", "item-a")
    payload = json.loads(work_item_record.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["history"][-1]["event"] == "subagent_run"
    assert payload["history"][-1]["role"] == "TextContextAgent"


def test_dynamic_per_work_item_runtime_overrides_planner_work_item_id(tmp_path: Path):
    lab_root = tmp_path / "lab"
    payload = _single_node_payload()
    payload["work_item_id"] = "wrong-item"
    runtime = _runtime(
        lab_root,
        planner_payload=payload,
        worker_responses=["item done"],
        dynamic_config=_dynamic_config(scope="per_work_item"),
    )

    result = runtime.run(_request_with_work_items(["item-a"]))

    assert result["status"] == "completed"
    assert result["dynamic_workflows"][0]["work_item_id"] == "item-a"
    dynamic_root = _layout(lab_root).state_root / "dynamic_workflows" / "task-dynamic" / "item-a__wf-runtime"
    trace = json.loads((dynamic_root / "dynamic_workflow_trace.json").read_text(encoding="utf-8"))
    assert trace["work_item_id"] == "item-a"
    assert (_work_item_path(lab_root, "task-dynamic", "item-a")).exists()
    assert not (_work_item_path(lab_root, "task-dynamic", "wrong-item")).exists()


def test_dynamic_final_records_complete_lifecycle_even_when_policy_roles_differ(tmp_path: Path):
    lab_root = tmp_path / "lab"
    payload = _workflow_payload()
    payload["dynamic_subagents"] = [
        {
            "subagent_id": "writer",
            "role_name": "SchemaWriterAgent",
            "goal": "Write final records.",
            "system_prompt": "Write final records.",
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"},
            "allowed_tools": ["write_jsonl"],
            "artifact_outputs": ["final_records.jsonl"],
        }
    ]
    payload["workflow_nodes"] = [
        {"node_id": "node-writer", "subagent_id": "writer", "output_artifacts": ["final_records.jsonl"]}
    ]
    payload["workflow_edges"] = []
    payload["artifact_contracts"] = {"final_records.jsonl": {"type": "jsonl"}}
    runtime = _base_runtime(
        lab_root,
        task_config=TaskConfig(
            task_id="task-dynamic",
            goal="Process synthetic work item.",
            dynamic_subagents=_dynamic_config(scope="per_work_item", allowed_tool_names=["write_jsonl", "read_text"]),
            runtime_policy=RuntimePolicy(
                max_tool_steps=4,
                metadata={
                    "work_item_routing": {
                        "enabled": True,
                        "executor_roles": ["ExecAgent"],
                        "reviewer_roles": [],
                        "finalizer_roles": ["WriteAgent"],
                        "required_work_item_ids": ["item-a"],
                        "work_item_id_field": "work_item_id",
                    }
                },
            ),
        ),
        llm_runtimes={
            "planner": FakeLLMRuntime(default_content=json.dumps(payload)),
            "worker": FakeLLMRuntime(
                responses=[
                    _tool_response(
                        "write_jsonl",
                        {"artifact_name": "final_records.jsonl", "records": [{"id": "alpha"}]},
                    ),
                    LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="done")),
                ]
            ),
        },
    )

    result = runtime.run(_request_with_work_items(["item-a"]))

    assert result["status"] == "completed"
    work_item_record = _work_item_path(lab_root, "task-dynamic", "item-a")
    lifecycle = json.loads(work_item_record.read_text(encoding="utf-8"))
    assert lifecycle["status"] == "completed"
    assert lifecycle["history"][-1]["role"] == "SchemaWriterAgent"
    assert lifecycle["history"][-1]["status"] == "completed"


def test_dynamic_context_run_does_not_complete_policy_work_item_without_final_records(tmp_path: Path):
    lab_root = tmp_path / "lab"
    runtime = _runtime(
        lab_root,
        planner_payload=_single_node_payload(),
        worker_responses=["context done"],
        dynamic_config=_dynamic_config(scope="per_work_item"),
    )
    runtime.task_config = runtime.task_config.model_copy(
        update={
            "runtime_policy": RuntimePolicy(
                max_tool_steps=4,
                metadata={
                    "work_item_routing": {
                        "enabled": True,
                        "executor_roles": ["ExecAgent"],
                        "reviewer_roles": [],
                        "finalizer_roles": ["WriteAgent"],
                        "required_work_item_ids": ["item-a"],
                        "work_item_id_field": "work_item_id",
                    }
                },
            )
        }
    )

    result = runtime.run(_request_with_work_items(["item-a"]))

    assert result["status"] == "completed"
    work_item_record = _work_item_path(lab_root, "task-dynamic", "item-a")
    lifecycle = json.loads(work_item_record.read_text(encoding="utf-8"))
    assert lifecycle["status"] == "claimed"
    assert lifecycle["history"][-1]["role"] == "TextContextAgent"
    assert lifecycle["history"][-1]["status"] == "claimed"


def test_dynamic_runtime_fails_missing_declared_output_artifact_and_blocks_downstream(tmp_path: Path):
    lab_root = tmp_path / "lab"
    runtime = _runtime(
        lab_root,
        planner_payload=_workflow_payload(),
        worker_responses=["context summary without artifact", "writer should not run"],
        dynamic_config=_dynamic_config(),
    )

    result = runtime.run(_request())

    assert result["status"] == "failed"
    runs = result["runs"]
    assert runs[0]["status"] == "guard_failed"
    assert "Missing expected outputs before final_answer: context.json" in runs[0]["failure_reason"]
    assert runs[1]["status"] == "failed"
    assert "failed dependencies: node-text" in runs[1]["failure_reason"]


def test_dynamic_runtime_recovers_declared_artifact_from_final_answer_json(tmp_path: Path):
    lab_root = tmp_path / "lab"
    runtime = _runtime(
        lab_root,
        planner_payload=_workflow_payload(),
        worker_responses=[
            json.dumps({"context.json": {"summary": "context from final answer"}}),
            _tool_response("write_report", {"artifact_name": "final.json", "content": {"summary": "final"}}),
            "writer done",
        ],
        dynamic_config=_dynamic_config(),
    )

    result = runtime.run(_request())

    assert result["status"] == "completed"
    assert result["runs"][0]["status"] == "completed"
    recovered = lab_root / "artifacts" / "tools" / "dynamic-work-item" / "context.json"
    assert recovered.exists()
    assert json.loads(recovered.read_text(encoding="utf-8")) == {"summary": "context from final answer"}
    assert any(
        ref["metadata"].get("recovered_from_final_answer") is True
        for ref in result["runs"][0]["artifact_refs"]
    )


def test_dynamic_runtime_recovers_single_declared_artifact_from_unwrapped_json(tmp_path: Path):
    lab_root = tmp_path / "lab"
    runtime = _runtime(
        lab_root,
        planner_payload=_workflow_payload(),
        worker_responses=[
            json.dumps({"summary": "context from unwrapped final answer", "status": "success"}),
            _tool_response("write_report", {"artifact_name": "final.json", "content": {"summary": "final"}}),
            "writer done",
        ],
        dynamic_config=_dynamic_config(),
    )

    result = runtime.run(_request())

    assert result["status"] == "completed"
    recovered = lab_root / "artifacts" / "tools" / "dynamic-work-item" / "context.json"
    assert recovered.exists()
    recovered_payload = json.loads(recovered.read_text(encoding="utf-8"))
    assert recovered_payload["summary"] == "context from unwrapped final answer"


def test_dynamic_runtime_recovers_context_summary_from_report_handoff(tmp_path: Path):
    payload = _workflow_payload()
    payload["dynamic_subagents"][0]["artifact_outputs"] = ["context_summary.json"]
    payload["dynamic_subagents"][1]["artifact_inputs"] = ["context_summary.json"]
    payload["workflow_nodes"][0]["output_artifacts"] = ["context_summary.json"]
    payload["workflow_nodes"][1]["input_artifacts"] = ["context_summary.json"]
    payload["artifact_contracts"] = {
        "context_summary.json": {"type": "object"},
        "final.json": {"type": "object"},
    }
    lab_root = tmp_path / "lab"
    runtime = _runtime(
        lab_root,
        planner_payload=payload,
        worker_responses=[
            _tool_response("write_report", {"artifact_name": "report.md", "content": "context report"}),
            "context report final answer",
            _tool_response("write_report", {"artifact_name": "final.json", "content": {"summary": "final"}}),
            "writer done",
        ],
        dynamic_config=_dynamic_config(),
    )

    result = runtime.run(_request())

    assert len(result["runs"]) == 2
    assert result["runs"][0]["status"] == "completed"
    assert result["runs"][1]["role"] == "EvidenceWriterAgent"
    assert "failed dependencies" not in str(result["runs"][1].get("failure_reason"))
    recovered = lab_root / "artifacts" / "tools" / "dynamic-work-item" / "context_summary.json"
    assert recovered.exists()
    payload = json.loads(recovered.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "dynamic_context_summary"
    first_run_refs = result["runs"][0]["artifact_refs"]
    assert any(ref["metadata"]["filename"] == "context_summary.json" for ref in first_run_refs)


def test_dynamic_runtime_adds_artifact_dependency_when_planner_omits_edge(tmp_path: Path):
    payload = _workflow_payload()
    payload["workflow_nodes"][1].pop("dependencies")
    payload["workflow_edges"] = []
    lab_root = tmp_path / "lab"
    runtime = _runtime(
        lab_root,
        planner_payload=payload,
        worker_responses=[
            "context summary without artifact",
            "writer should not run because context artifact is missing",
        ],
        dynamic_config=_dynamic_config(),
    )

    result = runtime.run(_request())

    assert result["status"] == "failed"
    assert result["runs"][1]["status"] == "failed"
    assert "failed dependencies: node-text" in result["runs"][1]["failure_reason"]
    spec_path = _layout(lab_root).state_root / "dynamic_workflows" / "task-dynamic" / "wf-runtime" / "dynamic_workflow_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    writer_node = next(node for node in spec["workflow_nodes"] if node["node_id"] == "node-writer")
    assert writer_node["dependencies"] == ["node-text"]


def test_dynamic_runtime_ignores_static_role_completion_guards_for_dynamic_subagent(tmp_path: Path):
    lab_root = tmp_path / "lab"
    task_config = TaskConfig(
        task_id="task-dynamic",
        goal="Process synthetic item.",
        dynamic_subagents=_dynamic_config(allowed_tool_names=["read_text", "write_report"]),
        runtime_policy=RuntimePolicy(
            max_tool_steps=4,
            metadata={
                "completion_guards_by_role": {
                    "ExecAgent": {
                        "required_tool_calls_before_final": ["write_jsonl", "write_report"],
                        "max_non_required_tool_calls_before_required_outputs": 0,
                    }
                }
            },
        ),
    )
    payload = _workflow_payload()
    payload["dynamic_subagents"][0]["role_name"] = "ExecAgent"
    payload["dynamic_subagents"] = [payload["dynamic_subagents"][0]]
    payload["workflow_nodes"] = [payload["workflow_nodes"][0]]
    payload["workflow_edges"] = []
    runtime = _base_runtime(
        lab_root,
        task_config=task_config,
        llm_runtimes={
            "planner": FakeLLMRuntime(default_content=json.dumps(payload)),
            "worker": FakeLLMRuntime(
                responses=[
                    _tool_response("write_report", {"artifact_name": "context.json", "content": "context"}),
                    LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="done")),
                ]
            ),
        },
    )

    result = runtime.run(_request())

    assert result["status"] == "completed"
    assert result["runs"][0]["status"] == "completed"
    assert result["runs"][0]["failure_reason"] is None


def test_dynamic_runtime_prompts_output_artifact_policy_for_declared_outputs(tmp_path: Path):
    lab_root = tmp_path / "lab"
    planner = FakeLLMRuntime(default_content=json.dumps(_workflow_payload()))
    worker = FakeLLMRuntime(
        responses=[
            _tool_response("write_report", {"artifact_name": "context.json", "content": "context"}),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="done")),
        ]
    )
    runtime = _base_runtime(
        lab_root,
        task_config=TaskConfig(
            task_id="task-dynamic",
            goal="Process synthetic item.",
            dynamic_subagents=_dynamic_config(),
            runtime_policy=RuntimePolicy(max_tool_steps=2, metadata={}),
        ),
        llm_runtimes={"planner": planner, "worker": worker},
    )

    runtime.run(_request())

    first_worker_prompt = worker.requests[0].messages[1].content
    assert "output_artifact_policy" in first_worker_prompt
    assert "Write every expected_output_artifact before final_answer" in first_worker_prompt
    assert "do not retry the same tool target indefinitely" in first_worker_prompt


def test_dynamic_runtime_passes_concrete_input_artifact_refs_downstream(tmp_path: Path):
    lab_root = tmp_path / "lab"
    planner = FakeLLMRuntime(default_content=json.dumps(_workflow_payload()))
    worker = FakeLLMRuntime(
        responses=[
            _tool_response("write_report", {"artifact_name": "context.json", "content": "context"}),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="context done")),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="writer done")),
        ]
    )
    runtime = _base_runtime(
        lab_root,
        task_config=TaskConfig(
            task_id="task-dynamic",
            goal="Process synthetic item.",
            dynamic_subagents=_dynamic_config(),
            runtime_policy=RuntimePolicy(max_tool_steps=3, metadata={}),
        ),
        llm_runtimes={"planner": planner, "worker": worker},
    )

    runtime.run(_request())

    writer_instruction = worker.requests[2].messages[1].content
    assert "available_input_artifact_refs" in writer_instruction
    assert "context.json" in writer_instruction
    assert str(lab_root / "artifacts" / "context.json") in writer_instruction


def test_dynamic_runtime_captures_node_failure(tmp_path: Path):
    lab_root = tmp_path / "lab"
    runtime = _runtime(
        lab_root,
        planner_payload=_workflow_payload(),
        worker_responses=[
            _tool_response("write_report", {"artifact_name": "context.json", "content": "context"}),
            "context done",
        ],
        dynamic_config=_dynamic_config(),
        extra_worker_response=LLMRuntimeResponse(action=SubAgentAction(action="ask_human", content="unsupported")),
    )

    result = runtime.run(_request())

    assert result["status"] == "failed"
    assert "EvidenceWriterAgent" in result["failure_reason"]


def test_dynamic_planning_failure_records_failed_dynamic_result(tmp_path: Path):
    lab_root = tmp_path / "lab"
    task_config = TaskConfig(
        task_id="task-dynamic",
        goal="Process synthetic item.",
        roles={
            "UnusedStaticAgent": RoleSpec(
                name="UnusedStaticAgent",
                system_prompt="This role must not run when dynamic planning fails.",
                llm_backend=BackendBinding(backend_id="worker"),
                allowed_tools=["read_text"],
            )
        },
        max_dispatch_steps=2,
        dynamic_subagents=_dynamic_config(max_planner_retries=0),
        runtime_policy=RuntimePolicy(max_tool_steps=2, metadata={}),
    )
    runtime = _base_runtime(
        lab_root,
        task_config=task_config,
        llm_runtimes={
            "planner": FakeLLMRuntime(default_content="not json"),
            "worker": FakeLLMRuntime(default_content="static worker done"),
        },
    )

    result = runtime.run(_request())

    assert result["execution_mode"] == "dynamic"
    assert result["status"] == "failed"
    assert result["runs"][0]["role"] == "DynamicWorkflowPlanner"
    assert result["runs"][0]["metadata"]["planning_failed"] is True
    assert "dynamic planning failed validation" in result["failure_reason"]
    fallback_path = _layout(lab_root).state_root / "dynamic_workflows" / "task-dynamic" / "planning_failed" / "fallback_reason.json"
    assert fallback_path.exists()


def test_dynamic_mode_runs_meta_preplanning_and_updates_agents_md_before_planner(tmp_path: Path):
    lab_root = tmp_path / "lab"
    meta_response = {
        "route": "END",
        "instruction": "Updated reusable subagent library before dynamic planning.",
        "metadata": {
            "role_pool_update": {
                "reason": "Reflector feedback showed the static extraction role needs stronger evidence handling.",
                "roles": {
                    "StaticAgent": {
                        "system_prompt": "Evolved static prompt. Require explicit evidence before writing records.",
                        "allowed_tools": ["read_text", "write_report"],
                        "metadata": {"evolved_from": "reflector-feedback"},
                    },
                    "EvidenceReviewAgent": {
                        "system_prompt": "Review extracted evidence before records are finalized.",
                        "llm_backend": {"backend_id": "worker"},
                        "allowed_tools": ["read_text", "write_report"],
                        "metadata": {"created_from": "reflector-feedback"},
                    }
                },
            }
        },
    }
    workflow_payload = _workflow_payload()
    workflow_payload["dynamic_subagents"][0]["role_name"] = "StaticAgent"
    workflow_payload["dynamic_subagents"][0]["system_prompt"] = "Planner context prompt."
    workflow_payload["dynamic_subagents"][1]["role_name"] = "EvidenceReviewAgent"
    planner = FakeLLMRuntime(default_content=json.dumps(workflow_payload))
    worker = FakeLLMRuntime(
        responses=[
            _tool_response("write_report", {"artifact_name": "context.json", "content": {"summary": "context"}}),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="context done")),
            _tool_response("write_report", {"artifact_name": "final.json", "content": {"summary": "final"}}),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="writer done")),
        ]
    )
    task_config = TaskConfig(
        task_id="task-dynamic",
        goal="Process synthetic item.",
        meta_agent=MetaAgentSpec(
            system_prompt="Use feedback to update agents.md before dynamic workflow planning.",
            llm_backend=BackendBinding(backend_id="meta"),
        ),
        roles={
            "StaticAgent": RoleSpec(
                name="StaticAgent",
                system_prompt="Old static prompt.",
                llm_backend=BackendBinding(backend_id="worker"),
                allowed_tools=["read_text"],
            )
        },
        dynamic_subagents=_dynamic_config(metadata={"meta_agent_preplanning_enabled": True}),
        runtime_policy=RuntimePolicy(max_tool_steps=2, metadata={}),
    )
    runtime = _base_runtime(
        lab_root,
        task_config=task_config,
        llm_runtimes={
            "planner": planner,
            "meta": FakeLLMRuntime(default_content=json.dumps(meta_response)),
            "worker": worker,
        },
    )

    result = runtime.run(_request())

    assert result["status"] == "completed"
    agents_path = _layout(lab_root).agents_path
    agents_text = agents_path.read_text(encoding="utf-8")
    assert "Evolved static prompt" in agents_text
    assert "EvidenceReviewAgent" in agents_text
    assert (_agents_updates_path(lab_root)).exists()
    planner_payload = json.loads(planner.requests[0].messages[1].content)
    role_pool_templates = {role["name"]: role for role in planner_payload["role_pool_templates"]}
    assert "static_fallback_subagents" not in planner_payload
    assert "system_prompt" not in role_pool_templates["StaticAgent"]
    assert "Evolved static prompt. Require explicit evidence" in role_pool_templates["StaticAgent"]["system_prompt_summary"]
    assert "Review extracted evidence before records are finalized" in role_pool_templates["EvidenceReviewAgent"]["system_prompt_summary"]
    assert role_pool_templates["EvidenceReviewAgent"]["llm_backend_id"] == "worker"
    assert worker.requests[0].messages[0].content.startswith(
        "Evolved static prompt. Require explicit evidence before writing records."
    )
    assert "Dynamic workflow assignment context from planner" in worker.requests[0].messages[0].content
    assert "Planner context prompt." in worker.requests[0].messages[0].content
    meta_runs = _trajectory_registry(lab_root).list_meta_agent_runs()
    assert len(meta_runs) == 1
    assert meta_runs[0].metadata["step_index"] == -1
    assert meta_runs[0].metadata["dispatch_metadata"]["role_pool_update_result"]["status"] == "updated"


def test_dynamic_runtime_updates_agents_md_before_planner_reads_role_pool(tmp_path: Path):
    lab_root = tmp_path / "lab"
    agents_path = _layout(lab_root).agents_path
    agents_path.parent.mkdir(parents=True, exist_ok=True)
    agents_path.write_text(
        render_agents_markdown(
            {
                "GeneralistAgent": RoleSpec(
                    name="GeneralistAgent",
                    system_prompt="Generalist prompt.",
                    llm_backend=BackendBinding(backend_id="worker-llm"),
                    agent_memory_backend=BackendBinding(backend_id="worker-memory"),
                    allowed_tools=["write_report"],
                )
            }
        ),
        encoding="utf-8",
    )
    meta_response = {
        "route": "END",
        "instruction": "Role pool evolved before dynamic planning.",
        "metadata": {
            "role_pool_update": {
                "reason": "Planner needs a reusable writer role.",
                "roles": {
                    "WriterAgent": {
                        "system_prompt": "Write the final report.",
                        "llm_backend": {"backend_id": "worker-llm"},
                        "agent_memory_backend": {"backend_id": "worker-memory"},
                        "allowed_tools": ["write_report"],
                    }
                },
            }
        },
    }
    workflow_payload = {
        "workflow_id": "wf-role-pool",
        "task_summary": "Write a synthetic report.",
        "article_context_summary": "none",
        "dynamic_subagents": [
            {
                "subagent_id": "writer",
                "role_name": "WriterAgent",
                "goal": "Write the report.",
                "system_prompt": "Use the evolved writer role.",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
                "allowed_tools": ["write_report"],
            }
        ],
        "workflow_nodes": [{"node_id": "node-writer", "subagent_id": "writer"}],
        "workflow_edges": [],
        "artifact_contracts": {},
        "validation_rules": [],
        "planner_rationale_summary": "The evolved writer role is enough.",
    }
    planner = FakeLLMRuntime(default_content=json.dumps(workflow_payload))
    meta = FakeLLMRuntime(
        default_content="not json",
        responses=[
            _no_generated_tool_response(),
            LLMRuntimeResponse(
                action=SubAgentAction(action="final_answer", content=json.dumps(meta_response)),
            )
        ],
    )
    worker = FakeLLMRuntime(default_content="done")
    task_config = TaskConfig(
        task_id="task-dynamic",
        goal="Process synthetic item.",
        task_memory_backend=BackendBinding(backend_id="task-memory"),
        meta_agent=MetaAgentSpec(
            system_prompt="Evolve agents.md before dynamic workflow planning.",
            llm_backend=BackendBinding(backend_id="meta-llm"),
            memory_backend=BackendBinding(backend_id="meta-memory"),
        ),
        agents_ref=str(agents_path),
        dynamic_subagents=_dynamic_config(
            default_worker_backend={"backend_id": "worker-llm"},
            allowed_tool_names=["read_text", "write_report"],
            metadata={"meta_agent_preplanning_enabled": True},
        ),
        runtime_policy=RuntimePolicy(max_tool_steps=1, metadata={"max_meta_dispatch_parse_retries": 0}),
    )
    runtime = _base_runtime(
        lab_root,
        task_config=task_config,
        llm_runtimes={
            "planner": planner,
            "meta-llm": meta,
            "worker-llm": worker,
        },
    )

    result = runtime.run(_request())

    assert result["status"] == "completed", result
    assert result["final_answer"] == "done"
    assert len(meta.requests) == 2
    assert "WriterAgent" in agents_path.read_text(encoding="utf-8")
    planner_payload = json.loads(planner.requests[0].messages[1].content)
    role_pool_templates = {role["name"]: role for role in planner_payload["role_pool_templates"]}
    assert "system_prompt" not in role_pool_templates["WriterAgent"]
    assert role_pool_templates["WriterAgent"]["system_prompt_summary"] == "Write the final report."
    assert worker.requests[0].messages[0].content.startswith("Write the final report.")


def test_dynamic_runtime_records_role_pool_no_op_event(tmp_path: Path):
    lab_root = tmp_path / "lab"
    agents_path = _layout(lab_root).agents_path
    agents_path.parent.mkdir(parents=True, exist_ok=True)
    agents_path.write_text(
        render_agents_markdown(
            {
                "GeneralistAgent": RoleSpec(
                    name="GeneralistAgent",
                    system_prompt="Generalist prompt.",
                    llm_backend=BackendBinding(backend_id="worker"),
                    allowed_tools=["write_report"],
                )
            }
        ),
        encoding="utf-8",
    )
    workflow_payload = {
        "workflow_id": "wf-role-pool-no-op",
        "task_summary": "Write a synthetic report.",
        "article_context_summary": "none",
        "dynamic_subagents": [
            {
                "subagent_id": "writer",
                "role_name": "GeneralistAgent",
                "goal": "Write the report.",
                "system_prompt": "Use the existing generalist role.",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
                "allowed_tools": ["write_report"],
            }
        ],
        "workflow_nodes": [{"node_id": "node-writer", "subagent_id": "writer"}],
        "workflow_edges": [],
        "artifact_contracts": {},
        "validation_rules": [],
        "planner_rationale_summary": "The existing role is enough.",
    }
    task_config = TaskConfig(
        task_id="task-dynamic",
        goal="Process synthetic item.",
        meta_agent=MetaAgentSpec(
            system_prompt="Inspect agents.md before dynamic workflow planning.",
            llm_backend=BackendBinding(backend_id="meta"),
        ),
        agents_ref=str(agents_path),
        dynamic_subagents=_dynamic_config(),
        runtime_policy=RuntimePolicy(max_tool_steps=1, metadata={}),
    )
    runtime = _base_runtime(
        lab_root,
        task_config=task_config,
        llm_runtimes={
            "planner": FakeLLMRuntime(default_content=json.dumps(workflow_payload)),
            "meta": FakeLLMRuntime(
                default_content=json.dumps(
                    {
                        "route": "END",
                        "instruction": "Existing role pool is already sufficient.",
                        "metadata": {"role_pool_update": {"reason": "No reusable role changes needed."}},
                    }
                )
            ),
            "worker": FakeLLMRuntime(default_content="done"),
        },
    )

    result = runtime.run(_request())

    assert result["status"] == "completed", result
    event_types = [
        event.event_type
        for event in _trajectory_registry(lab_root).list_events()
    ]
    assert "role_pool_update_no_op" in event_types
    assert "role_pool_update_rejected" not in event_types


def test_dynamic_runtime_retries_empty_role_pool_preplanning_then_records_no_op_reason(tmp_path: Path):
    lab_root = tmp_path / "lab"
    agents_path = _layout(lab_root).agents_path
    agents_path.parent.mkdir(parents=True, exist_ok=True)
    agents_path.write_text(
        render_agents_markdown(
            {
                "GeneralistAgent": RoleSpec(
                    name="GeneralistAgent",
                    system_prompt="Generalist prompt.",
                    llm_backend=BackendBinding(backend_id="worker"),
                    allowed_tools=["write_report"],
                )
            }
        ),
        encoding="utf-8",
    )
    workflow_payload = {
        "workflow_id": "wf-role-pool-explicit-no-op",
        "task_summary": "Write a synthetic report.",
        "article_context_summary": "none",
        "dynamic_subagents": [
            {
                "subagent_id": "writer",
                "role_name": "GeneralistAgent",
                "goal": "Write the report.",
                "system_prompt": "Use the existing generalist role.",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
                "allowed_tools": ["write_report"],
            }
        ],
        "workflow_nodes": [{"node_id": "node-writer", "subagent_id": "writer"}],
        "workflow_edges": [],
        "artifact_contracts": {},
        "validation_rules": [],
        "planner_rationale_summary": "The existing role is enough.",
    }
    meta = FakeLLMRuntime(
        responses=[
            _no_generated_tool_response(),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps({"route": "END", "instruction": "No change.", "metadata": {}}),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "route": "END",
                            "instruction": "No reusable role-pool update is needed.",
                            "metadata": {"no_role_pool_update_reason": "Existing role pool is sufficient."},
                        }
                    ),
                )
            ),
        ]
    )
    task_config = TaskConfig(
        task_id="task-dynamic",
        goal="Process synthetic item.",
        meta_agent=MetaAgentSpec(
            system_prompt="Inspect agents.md before dynamic workflow planning.",
            llm_backend=BackendBinding(backend_id="meta"),
        ),
        agents_ref=str(agents_path),
        dynamic_subagents=_dynamic_config(),
        runtime_policy=RuntimePolicy(max_tool_steps=1, metadata={}),
    )
    runtime = _base_runtime(
        lab_root,
        task_config=task_config,
        llm_runtimes={
            "planner": FakeLLMRuntime(default_content=json.dumps(workflow_payload)),
            "meta": meta,
            "worker": FakeLLMRuntime(default_content="done"),
        },
    )

    result = runtime.run(_request())

    assert result["status"] == "completed", result
    assert len(meta.requests) == 3
    events = _trajectory_registry(lab_root).list_events()
    event_types = [event.event_type for event in events]
    assert "role_pool_update_no_op" in event_types
    assert "role_pool_update_rejected" not in event_types
    role_pool_events = [event for event in events if event.event_type.startswith("role_pool_update_")]
    assert role_pool_events[0].metadata["status"] == "no_op"
    meta_runs = _trajectory_registry(lab_root).list_meta_agent_runs()
    assert meta_runs[1].metadata["dispatch_metadata"]["no_role_pool_update_reason"] == (
        "Existing role pool is sufficient."
    )


def test_dynamic_runtime_accepts_legacy_role_pool_preplanning_no_op_reason(tmp_path: Path):
    lab_root = tmp_path / "lab"
    agents_path = _layout(lab_root).agents_path
    agents_path.parent.mkdir(parents=True, exist_ok=True)
    agents_path.write_text(
        render_agents_markdown(
            {
                "GeneralistAgent": RoleSpec(
                    name="GeneralistAgent",
                    system_prompt="Generalist prompt.",
                    llm_backend=BackendBinding(backend_id="worker"),
                    allowed_tools=["write_report"],
                )
            }
        ),
        encoding="utf-8",
    )
    workflow_payload = {
        "workflow_id": "wf-role-pool-legacy-no-op",
        "task_summary": "Write a synthetic report.",
        "article_context_summary": "none",
        "dynamic_subagents": [
            {
                "subagent_id": "writer",
                "role_name": "GeneralistAgent",
                "goal": "Write the report.",
                "system_prompt": "Use the existing generalist role.",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
                "allowed_tools": ["write_report"],
            }
        ],
        "workflow_nodes": [{"node_id": "node-writer", "subagent_id": "writer"}],
        "workflow_edges": [],
        "artifact_contracts": {},
        "validation_rules": [],
        "planner_rationale_summary": "The existing role is enough.",
    }
    task_config = TaskConfig(
        task_id="task-dynamic",
        goal="Process synthetic item.",
        meta_agent=MetaAgentSpec(
            system_prompt="Inspect agents.md before dynamic workflow planning.",
            llm_backend=BackendBinding(backend_id="meta"),
        ),
        agents_ref=str(agents_path),
        dynamic_subagents=_dynamic_config(),
        runtime_policy=RuntimePolicy(max_tool_steps=1, metadata={}),
    )
    runtime = _base_runtime(
        lab_root,
        task_config=task_config,
        llm_runtimes={
            "planner": FakeLLMRuntime(default_content=json.dumps(workflow_payload)),
            "meta": FakeLLMRuntime(
                default_content=json.dumps(
                    {
                        "route": "END",
                        "instruction": "Legacy no-op reason.",
                        "metadata": {"no_agent_config_update_reason": "inspection only"},
                    }
                )
            ),
            "worker": FakeLLMRuntime(default_content="done"),
        },
    )

    result = runtime.run(_request())

    assert result["status"] == "completed", result
    events = _trajectory_registry(lab_root).list_events()
    event_types = [event.event_type for event in events]
    assert "role_pool_update_no_op" in event_types
    assert "role_pool_update_rejected" not in event_types
    role_pool_event = next(event for event in events if event.event_type == "role_pool_update_no_op")
    assert role_pool_event.metadata["no_role_pool_update_reason"] == "inspection only"


def test_dynamic_runtime_records_rejected_role_pool_event_when_preplanning_fails(tmp_path: Path):
    lab_root = tmp_path / "lab"
    agents_path = _layout(lab_root).agents_path
    agents_path.parent.mkdir(parents=True, exist_ok=True)
    agents_path.write_text(
        render_agents_markdown(
            {
                "GeneralistAgent": RoleSpec(
                    name="GeneralistAgent",
                    system_prompt="Generalist prompt.",
                    llm_backend=BackendBinding(backend_id="worker"),
                    allowed_tools=["write_report"],
                )
            }
        ),
        encoding="utf-8",
    )
    meta = FakeLLMRuntime(
        responses=[
            _no_generated_tool_response(),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="not json")),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="still not json")),
        ]
    )
    task_config = TaskConfig(
        task_id="task-dynamic",
        goal="Process synthetic item.",
        meta_agent=MetaAgentSpec(
            system_prompt="Inspect agents.md before dynamic workflow planning.",
            llm_backend=BackendBinding(backend_id="meta"),
        ),
        agents_ref=str(agents_path),
        dynamic_subagents=_dynamic_config(),
        runtime_policy=RuntimePolicy(max_tool_steps=1, metadata={"max_meta_dispatch_parse_retries": 1}),
    )
    runtime = _base_runtime(
        lab_root,
        task_config=task_config,
        llm_runtimes={
            "planner": FakeLLMRuntime(default_content=json.dumps(_workflow_payload())),
            "meta": meta,
            "worker": FakeLLMRuntime(default_content="done"),
        },
    )

    with pytest.raises(RuntimeError, match="MetaAgent dispatch parsing failed") as exc_info:
        runtime.run(_request())

    failure_message = str(exc_info.value)
    assert "run_subagent" not in failure_message
    assert "abort" not in failure_message
    assert "role_pool_update" in failure_message
    assert "no_role_pool_update_reason" in failure_message
    assert len(meta.requests) == 3
    events = _trajectory_registry(lab_root).list_events()
    rejected_events = [event for event in events if event.event_type == "role_pool_update_rejected"]
    assert len(rejected_events) == 1
    assert rejected_events[0].subject_ref == str(agents_path)
    assert rejected_events[0].metadata["status"] == "rejected"
    assert rejected_events[0].metadata["stage"] == "role_pool_evolution"
    assert rejected_events[0].metadata["errors"]


def test_dynamic_runtime_retries_abort_role_pool_preplanning_then_accepts_end_no_op(tmp_path: Path):
    lab_root = tmp_path / "lab"
    agents_path = _layout(lab_root).agents_path
    agents_path.parent.mkdir(parents=True, exist_ok=True)
    agents_path.write_text(
        render_agents_markdown(
            {
                "GeneralistAgent": RoleSpec(
                    name="GeneralistAgent",
                    system_prompt="Generalist prompt.",
                    llm_backend=BackendBinding(backend_id="worker"),
                    allowed_tools=["write_report"],
                )
            }
        ),
        encoding="utf-8",
    )
    workflow_payload = {
        "workflow_id": "wf-role-pool-abort-retry",
        "task_summary": "Write a synthetic report.",
        "article_context_summary": "none",
        "dynamic_subagents": [
            {
                "subagent_id": "writer",
                "role_name": "GeneralistAgent",
                "goal": "Write the report.",
                "system_prompt": "Use the existing generalist role.",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
                "allowed_tools": ["write_report"],
            }
        ],
        "workflow_nodes": [{"node_id": "node-writer", "subagent_id": "writer"}],
        "workflow_edges": [],
        "artifact_contracts": {},
        "validation_rules": [],
        "planner_rationale_summary": "The existing role is enough.",
    }
    meta = FakeLLMRuntime(
        responses=[
            _no_generated_tool_response(),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "action": "abort",
                            "instruction": "No role-pool update is needed.",
                            "metadata": {"no_role_pool_update_reason": "Existing role pool is sufficient."},
                        }
                    ),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "route": "END",
                            "instruction": "No reusable role-pool update is needed.",
                            "metadata": {"no_role_pool_update_reason": "Existing role pool is sufficient."},
                        }
                    ),
                )
            ),
        ]
    )
    task_config = TaskConfig(
        task_id="task-dynamic",
        goal="Process synthetic item.",
        meta_agent=MetaAgentSpec(
            system_prompt="Inspect agents.md before dynamic workflow planning.",
            llm_backend=BackendBinding(backend_id="meta"),
        ),
        agents_ref=str(agents_path),
        dynamic_subagents=_dynamic_config(),
        runtime_policy=RuntimePolicy(max_tool_steps=1, metadata={}),
    )
    runtime = _base_runtime(
        lab_root,
        task_config=task_config,
        llm_runtimes={
            "planner": FakeLLMRuntime(default_content=json.dumps(workflow_payload)),
            "meta": meta,
            "worker": FakeLLMRuntime(default_content="done"),
        },
    )

    result = runtime.run(_request())

    assert result["status"] == "completed", result
    assert len(meta.requests) == 3
    repair_payload = json.loads(meta.requests[2].messages[-1].content)
    repair_schema = json.dumps(repair_payload["expected_schema"], sort_keys=True)
    assert "run_subagent" not in repair_schema
    assert "abort" not in repair_schema
    assert "role_pool_update" in repair_schema
    assert "no_role_pool_update_reason" in repair_schema
    events = _trajectory_registry(lab_root).list_events()
    role_pool_events = [event for event in events if event.event_type.startswith("role_pool_update_")]
    assert [event.event_type for event in role_pool_events] == ["role_pool_update_no_op"]
    assert role_pool_events[0].metadata["no_role_pool_update_reason"] == "Existing role pool is sufficient."


def test_dynamic_meta_preplanning_sees_evolved_feedback_for_prior_dynamic_roles(tmp_path: Path):
    lab_root = tmp_path / "lab"
    state_registry = _backend_state_registry(lab_root)
    for role_name in ("StaticAgent", "ContextAgent"):
        state = BackendStateRecord(
            state_ref=f"prompt-overlay://worker/{role_name.casefold()}",
            backend_id="worker",
            backend_type="llm",
            metadata={
                "state_kind": "prompt_overlay",
                "prompt_overlay": {
                    "role": role_name,
                    "backend_id": "worker",
                    "system_prompt_append": f"Reflector note for {role_name}.",
                    "metrics": {"f1": 0.5},
                },
            },
        )
        state_registry.register_candidate(state)
        state_registry.promote("worker", state.state_ref, f"evo-{role_name}")
    planner = FakeLLMRuntime(default_content=json.dumps(_workflow_payload()))
    meta = FakeLLMRuntime(
        responses=[
            _no_generated_tool_response(),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "route": "END",
                            "instruction": "No library update for this test.",
                            "metadata": {"no_role_pool_update_reason": "inspection only"},
                        }
                    ),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(
                        {
                            "route": "END",
                            "instruction": "No library update for this test.",
                            "metadata": {"no_agent_config_update_reason": "inspection only"},
                        }
                    ),
                )
            ),
        ]
    )
    task_config = TaskConfig(
        task_id="task-dynamic",
        goal="Process synthetic item.",
        meta_agent=MetaAgentSpec(
            system_prompt="Inspect active evolved feedback before dynamic planning.",
            llm_backend=BackendBinding(backend_id="meta"),
        ),
        roles={
            "StaticAgent": RoleSpec(
                name="StaticAgent",
                system_prompt="Old static prompt.",
                llm_backend=BackendBinding(backend_id="worker"),
                allowed_tools=["read_text"],
            )
        },
        dynamic_subagents=_dynamic_config(metadata={"meta_agent_preplanning_enabled": True}),
        runtime_policy=RuntimePolicy(max_tool_steps=2, metadata={}),
    )
    runtime = _base_runtime(
        lab_root,
        task_config=task_config,
        llm_runtimes={
            "planner": planner,
            "meta": meta,
            "worker": FakeLLMRuntime(
                responses=[
                    _tool_response("write_report", {"artifact_name": "context.json", "content": {"summary": "context"}}),
                    LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="context done")),
                    _tool_response("write_report", {"artifact_name": "final.json", "content": {"summary": "final"}}),
                    LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="writer done")),
                ]
            ),
        },
    )

    result = runtime.run(_request())

    assert result["status"] == "completed"
    assert len(meta.requests) == 2
    meta_payload = json.loads(meta.requests[1].messages[-1].content)
    feedback_by_role = {
        item["role"]: item
        for item in meta_payload["active_evolved_role_feedback"]
    }
    assert "StaticAgent" in feedback_by_role
    assert "ContextAgent" in feedback_by_role
    assert feedback_by_role["ContextAgent"]["active_state_ref"] == "prompt-overlay://worker/contextagent"
    assert "ContextAgent" not in {
        role["name"]
        for role in meta_payload["available_roles"]
    }


def test_dynamic_subagents_consume_active_prompt_overlay_state(tmp_path: Path):
    lab_root = tmp_path / "lab"
    state_registry = _backend_state_registry(lab_root)
    overlays = {
        "ContextAgent": "EVOLVED CONTEXT OVERLAY: inspect source inventory before summarizing.",
        "ExecAgent": "EVOLVED EXEC OVERLAY: preserve evidence-backed candidate rows.",
    }
    for role_name, append_text in overlays.items():
        state = BackendStateRecord(
            state_ref=f"prompt-overlay://worker/{role_name.casefold()}/test",
            backend_id="worker",
            backend_type="llm",
            metadata={
                "state_kind": "prompt_overlay",
                "prompt_overlay": {
                    "role": role_name,
                    "backend_id": "worker",
                    "system_prompt_append": append_text,
                    "metrics": {"f1": 0.42},
                },
            },
        )
        state_registry.register_candidate(state)
        state_registry.promote("worker", state.state_ref, f"evo-{role_name}")

    workflow_payload = _workflow_payload()
    workflow_payload["dynamic_subagents"][0]["role_name"] = "ContextAgent"
    workflow_payload["dynamic_subagents"][1]["role_name"] = "ExecAgent"
    worker = FakeLLMRuntime(
        responses=[
            _tool_response("write_report", {"artifact_name": "context.json", "content": {"summary": "context"}}),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="context done")),
            _tool_response("write_report", {"artifact_name": "final.json", "content": {"summary": "final"}}),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="writer done")),
        ]
    )
    runtime = _base_runtime(
        lab_root,
        task_config=TaskConfig(
            task_id="task-dynamic",
            goal="Process synthetic item.",
            dynamic_subagents=_dynamic_config(),
            runtime_policy=RuntimePolicy(max_tool_steps=2, metadata={}),
        ),
        llm_runtimes={
            "planner": FakeLLMRuntime(default_content=json.dumps(workflow_payload)),
            "worker": worker,
        },
    )

    result = runtime.run(_request())

    assert result["status"] == "completed"
    system_prompts = [request.messages[0].content for request in worker.requests]
    assert any(overlays["ContextAgent"] in prompt for prompt in system_prompts)
    assert any(overlays["ExecAgent"] in prompt for prompt in system_prompts)
    trajectory = _trajectory_registry(lab_root)
    subagent_runs = trajectory.list_subagent_runs()
    assert [record.llm_backend_state_ref for record in subagent_runs] == [
        "prompt-overlay://worker/contextagent/test",
        "prompt-overlay://worker/execagent/test",
    ]
    events_text = (_layout(lab_root).registries_dir / "trajectory" / "events.jsonl").read_text(encoding="utf-8")
    assert "active_prompt_overlay_state_ref" in events_text
    assert "prompt-overlay://worker/contextagent/test" in events_text
    assert "prompt-overlay://worker/execagent/test" in events_text


def test_dynamic_meta_preplanning_failure_is_recorded_and_does_not_block_planner(tmp_path: Path):
    lab_root = tmp_path / "lab"
    state_registry = _backend_state_registry(lab_root)
    state = BackendStateRecord(
        state_ref="prompt-overlay://worker/contextagent",
        backend_id="worker",
        backend_type="llm",
        metadata={
            "state_kind": "prompt_overlay",
            "prompt_overlay": {
                "role": "ContextAgent",
                "backend_id": "worker",
                "system_prompt_append": "Reflector note for ContextAgent.",
            },
        },
    )
    state_registry.register_candidate(state)
    state_registry.promote("worker", state.state_ref, "evo-context")
    planner = FakeLLMRuntime(default_content=json.dumps(_workflow_payload()))
    task_config = TaskConfig(
        task_id="task-dynamic",
        goal="Process synthetic item.",
        meta_agent=MetaAgentSpec(
            system_prompt="Inspect active evolved feedback before dynamic planning.",
            llm_backend=BackendBinding(backend_id="meta"),
        ),
        roles={
            "StaticAgent": RoleSpec(
                name="StaticAgent",
                system_prompt="Old static prompt.",
                llm_backend=BackendBinding(backend_id="worker"),
                allowed_tools=["read_text"],
            )
        },
        dynamic_subagents=_dynamic_config(metadata={"meta_agent_preplanning_enabled": True}),
        runtime_policy=RuntimePolicy(max_tool_steps=2, metadata={"max_meta_dispatch_parse_retries": 0}),
    )
    runtime = _base_runtime(
        lab_root,
        task_config=task_config,
        llm_runtimes={
            "planner": planner,
            "meta": FakeLLMRuntime(
                responses=[
                    _no_generated_tool_response(),
                    LLMRuntimeResponse(
                        action=SubAgentAction(
                            action="final_answer",
                            content=json.dumps(
                                {
                                    "route": "END",
                                    "instruction": "No reusable role update needed.",
                                    "metadata": {"no_role_pool_update_reason": "inspection only"},
                                }
                            ),
                        )
                    ),
                    LLMRuntimeResponse(
                        action=SubAgentAction(
                            action="final_answer",
                            content=json.dumps(
                                {
                                    "route": "StaticAgent",
                                    "instruction": "This executable route is not valid preplanning.",
                                    "metadata": {},
                                }
                            ),
                        )
                    ),
                ]
            ),
            "worker": FakeLLMRuntime(
                responses=[
                    _tool_response("write_report", {"artifact_name": "context.json", "content": {"summary": "context"}}),
                    LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="context done")),
                    _tool_response("write_report", {"artifact_name": "final.json", "content": {"summary": "final"}}),
                    LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="writer done")),
                ]
            ),
        },
    )

    result = runtime.run(_request())

    assert result["status"] == "completed"
    assert planner.requests
    assert len(runtime.llm_runtimes["meta"].requests) == 2
    events = _trajectory_registry(lab_root).list_events()
    assert any(event.event_type == "role_pool_update_no_op" for event in events)
    assert not any(event.event_type == "dynamic_meta_preplanning_skipped" for event in events)


def test_per_work_item_dynamic_planning_failure_after_partial_run_records_dynamic_failure(tmp_path: Path):
    lab_root = tmp_path / "lab"
    first_payload = _workflow_payload_for_work_item("item-a")
    third_payload = _workflow_payload_for_work_item("item-c")
    task_config = TaskConfig(
        task_id="task-dynamic",
        goal="Process synthetic work items.",
        roles={
            "UnusedStaticAgent": RoleSpec(
                name="UnusedStaticAgent",
                system_prompt="This role must not run during dynamic planning failure.",
                llm_backend=BackendBinding(backend_id="worker"),
                allowed_tools=["read_text"],
            )
        },
        dynamic_subagents=_dynamic_config(scope="per_work_item", max_planner_retries=0),
        runtime_policy=RuntimePolicy(max_tool_steps=2, metadata={}),
    )
    runtime = _base_runtime(
        lab_root,
        task_config=task_config,
        llm_runtimes={
            "planner": FakeLLMRuntime(
                responses=[
                    LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content=json.dumps(first_payload))),
                    LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="{}")),
                    LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content=json.dumps(third_payload))),
                ]
            ),
            "worker": FakeLLMRuntime(
                responses=[
                    _tool_response("write_report", {"artifact_name": "item-a-context.json", "content": "context"}),
                    LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="item-a context done")),
                    _tool_response("write_report", {"artifact_name": "item-a-final.json", "content": "final"}),
                    LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="item-a final done")),
                    _tool_response("write_report", {"artifact_name": "item-c-context.json", "content": "context"}),
                    LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="item-c context done")),
                    _tool_response("write_report", {"artifact_name": "item-c-final.json", "content": "final"}),
                    LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="item-c final done")),
                ]
            ),
        },
    )
    request = _request_with_work_items(["item-a", "item-b", "item-c"])

    result = runtime.run(request)

    assert result["execution_mode"] == "dynamic"
    assert result["status"] == "failed"
    assert "dynamic planning failed validation" in result["failure_reason"]
    assert [run["role"] for run in result["runs"]] == [
        "TextContextAgent",
        "EvidenceWriterAgent",
        "DynamicWorkflowPlanner",
        "TextContextAgent",
        "EvidenceWriterAgent",
    ]
    assert result["runs"][2]["metadata"]["work_item_id"] == "item-b"
    assert result["runs"][2]["metadata"]["planning_failed"] is True
    assert not any(run["role"] == "UnusedStaticAgent" for run in result["runs"])
    assert (
        _layout(lab_root).state_root / "dynamic_workflows" / request.task_id / "item-b__planning_failed" / "fallback_reason.json"
    ).exists()
    assert (_layout(lab_root).state_root / "dynamic_workflows" / request.task_id / "wf-item-a" / "dynamic_workflow_trace.json").exists()
    assert (_layout(lab_root).state_root / "dynamic_workflows" / request.task_id / "wf-item-c" / "dynamic_workflow_trace.json").exists()


def test_per_work_item_dynamic_planning_failure_without_static_fallback_records_failed_item_and_continues(tmp_path: Path):
    lab_root = tmp_path / "lab"
    first_payload = _workflow_payload_for_work_item("item-a")
    third_payload = _workflow_payload_for_work_item("item-c")
    task_config = TaskConfig(
        task_id="task-dynamic",
        goal="Process synthetic work items.",
        dynamic_subagents=_dynamic_config(scope="per_work_item", fallback_to_static=False, max_planner_retries=0),
        runtime_policy=RuntimePolicy(max_tool_steps=2, metadata={}),
    )
    runtime = _base_runtime(
        lab_root,
        task_config=task_config,
        llm_runtimes={
            "planner": FakeLLMRuntime(
                responses=[
                    LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content=json.dumps(first_payload))),
                    LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="{")),
                    LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content=json.dumps(third_payload))),
                ]
            ),
            "worker": FakeLLMRuntime(
                responses=[
                    _tool_response("write_report", {"artifact_name": "item-a-context.json", "content": "context"}),
                    LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="item-a context done")),
                    _tool_response("write_report", {"artifact_name": "item-a-final.json", "content": "final"}),
                    LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="item-a final done")),
                    _tool_response("write_report", {"artifact_name": "item-c-context.json", "content": "context"}),
                    LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="item-c context done")),
                    _tool_response("write_report", {"artifact_name": "item-c-final.json", "content": "final"}),
                    LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="item-c final done")),
                ]
            ),
        },
    )
    request = _request_with_work_items(["item-a", "item-b", "item-c"])

    result = runtime.run(request)

    assert result["execution_mode"] == "dynamic"
    assert result["status"] == "failed"
    assert [run["role"] for run in result["runs"]] == [
        "TextContextAgent",
        "EvidenceWriterAgent",
        "DynamicWorkflowPlanner",
        "TextContextAgent",
        "EvidenceWriterAgent",
    ]
    assert result["runs"][2]["metadata"]["work_item_id"] == "item-b"
    assert result["runs"][2]["metadata"]["fallback_to_static"] is False
    assert "dynamic planning failed validation" in result["runs"][2]["failure_reason"]
    assert (
        _layout(lab_root).state_root
        / "dynamic_workflows"
        / request.task_id
        / "item-b__planning_failed"
        / "fallback_reason.json"
    ).exists()
    assert (
        _layout(lab_root).state_root
        / "dynamic_workflows"
        / request.task_id
        / "wf-item-a"
        / "dynamic_workflow_trace.json"
    ).exists()
    assert (
        _layout(lab_root).state_root
        / "dynamic_workflows"
        / request.task_id
        / "wf-item-c"
        / "dynamic_workflow_trace.json"
    ).exists()
    events = _trajectory_registry(lab_root).list_events()
    assert any(event.event_type == "dynamic_workflow_planning_failed" for event in events)


def test_dynamic_workflow_artifact_paths_include_work_item_when_workflow_id_repeats(tmp_path: Path):
    lab_root = tmp_path / "lab"
    payload_a = _workflow_payload()
    payload_a["work_item_id"] = "article-a"
    payload_b = _workflow_payload()
    payload_b["work_item_id"] = "article-b"
    spec_a = DynamicWorkflowSpec.model_validate(payload_a)
    spec_b = DynamicWorkflowSpec.model_validate(payload_b)
    report = DynamicWorkflowValidationReport(valid=True)

    paths_a = persist_dynamic_workflow_artifacts(
        lab_root=lab_root,
        task_id="task-dynamic",
        spec=spec_a,
        validation_report=report,
    )
    paths_b = persist_dynamic_workflow_artifacts(
        lab_root=lab_root,
        task_id="task-dynamic",
        spec=spec_b,
        validation_report=report,
    )

    assert paths_a["dynamic_workflow_spec"] != paths_b["dynamic_workflow_spec"]
    assert "article-a__wf-runtime" in paths_a["dynamic_workflow_spec"]
    assert "article-b__wf-runtime" in paths_b["dynamic_workflow_spec"]
    assert Path(paths_a["dynamic_workflow_spec"]).exists()
    assert Path(paths_b["dynamic_workflow_spec"]).exists()


def test_dynamic_aggregate_final_records_preserves_per_work_item_outputs(tmp_path: Path):
    lab_root = tmp_path / "lab"
    runtime = _base_runtime(
        lab_root,
        task_config=TaskConfig(
            task_id="task-dynamic",
            goal="Extract structured records.",
            dynamic_subagents=_dynamic_config(),
            runtime_policy=RuntimePolicy(max_tool_steps=2, metadata={}),
        ),
        llm_runtimes={"planner": FakeLLMRuntime(default_content="{}"), "worker": FakeLLMRuntime(default_content="done")},
    )
    registry = _lab_state_registry(lab_root)
    runtime.lab_state_registry = registry
    first_path = tmp_path / "article-a-candidate.json"
    second_path = tmp_path / "article-b-candidate.json"
    first_path.write_text(
        json.dumps({"records": [{"article_id": "a", "work_item_id": "a", "component_name": "A", "sequence": "AAAA"}]}),
        encoding="utf-8",
    )
    second_path.write_text(
        json.dumps({"records": [{"article_id": "b", "work_item_id": "b", "component_name": "B", "sequence": "CCCC"}]}),
        encoding="utf-8",
    )
    registry.save_artifact_index_record(
        ArtifactIndexRecord(
            artifact_ref="a",
            task_id="task-dynamic",
            uri=str(first_path),
            artifact_type="dataset",
            metadata={"work_item_id": "a", "filename": "candidate_records.json", "artifact_kind": "candidate_records"},
        )
    )
    registry.save_artifact_index_record(
        ArtifactIndexRecord(
            artifact_ref="b",
            task_id="task-dynamic",
            uri=str(second_path),
            artifact_type="dataset",
            metadata={"work_item_id": "b", "filename": "candidate_records.json", "artifact_kind": "candidate_records"},
        )
    )

    runtime._write_dynamic_aggregate_final_records(_request())

    final_path = lab_root / "artifacts" / "tools" / "biology_component_records.jsonl"
    records = [json.loads(line) for line in final_path.read_text(encoding="utf-8").splitlines()]
    assert [record["work_item_id"] for record in records] == ["a", "b"]
    assert (lab_root / "artifacts" / "tools" / "final_records.jsonl").exists()


def test_dynamic_aggregate_prefers_non_empty_validated_over_empty_final(tmp_path: Path):
    lab_root = tmp_path / "lab"
    runtime = _base_runtime(
        lab_root,
        task_config=TaskConfig(
            task_id="task-dynamic",
            goal="Extract structured records.",
            dynamic_subagents=_dynamic_config(),
            runtime_policy=RuntimePolicy(max_tool_steps=2, metadata={}),
        ),
        llm_runtimes={"planner": FakeLLMRuntime(default_content="{}"), "worker": FakeLLMRuntime(default_content="done")},
    )
    registry = _lab_state_registry(lab_root)
    runtime.lab_state_registry = registry
    empty_final = tmp_path / "empty-final.jsonl"
    empty_final.write_text("", encoding="utf-8")
    validated = tmp_path / "validated.json"
    validated.write_text(
        json.dumps(
            {
                "accepted_records": [
                    {"article_id": "a", "work_item_id": "a", "component_name": "A", "sequence": "AAAA"}
                ]
            }
        ),
        encoding="utf-8",
    )
    registry.save_artifact_index_record(
        ArtifactIndexRecord(
            artifact_ref="validated",
            task_id="task-dynamic",
            uri=str(validated),
            artifact_type="dataset",
            metadata={"work_item_id": "a", "filename": "validated_records.json", "artifact_kind": "validated_records"},
        )
    )
    registry.save_artifact_index_record(
        ArtifactIndexRecord(
            artifact_ref="final-empty",
            task_id="task-dynamic",
            uri=str(empty_final),
            artifact_type="dataset",
            metadata={"work_item_id": "a", "filename": "final_records.jsonl", "artifact_kind": "final_records"},
        )
    )

    runtime._write_dynamic_aggregate_final_records(_request())

    records = [
        json.loads(line)
        for line in (lab_root / "artifacts" / "tools" / "biology_component_records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert records == [{"article_id": "a", "component_name": "A", "sequence": "AAAA", "work_item_id": "a"}]


def test_dynamic_output_contract_allows_downstream_when_required_artifact_was_written():
    result = {
        "status": "failed",
        "failure_reason": "repeated_tool_call_suppressed after artifact output",
        "artifact_refs": [
            {
                "uri": "/tmp/candidate_records.json",
                "type": "dataset",
                "metadata": {"filename": "candidate_records.json"},
            }
        ],
        "completion_contract": {
            "produced_required_outputs": True,
            "blocking_issues": [{"type": "subagent_status", "message": "later failure"}],
            "warnings": [],
        },
    }

    updated = _enforce_dynamic_node_output_contract(
        result,
        expected_output_artifacts=["candidate_records.json"],
    )

    assert updated["status"] == "completed"
    assert updated["failure_reason"] is None
    assert updated["completion_contract"]["blocking_issues"] == []
    assert updated["dynamic_output_contract_override"]["original_status"] == "failed"


def test_dynamic_writer_recovers_final_records_from_validated_handoff(tmp_path: Path):
    lab_root = tmp_path / "lab"
    registry = _lab_state_registry(lab_root)
    validated = tmp_path / "validated_records.json"
    validated.write_text(
        json.dumps(
            {
                "accepted_records": [
                    {
                        "article_id": "article-a",
                        "work_item_id": "article-a",
                        "component_name": "promoter-a",
                        "sequence": "AACCGGTT",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    registry.save_artifact_index_record(
        ArtifactIndexRecord(
            artifact_ref="validated-a",
            task_id="task-dynamic",
            uri=str(validated),
            artifact_type="dataset",
            metadata={
                "work_item_id": "article-a",
                "filename": "validated_records.json",
                "artifact_kind": "validated_records",
            },
        )
    )
    failed_writer = {
        "run_ref": "writer-run",
        "role": "SchemaWriterAgent",
        "status": "failed",
        "failure_reason": "writer inspected sources instead of serializing final records",
        "artifact_refs": [],
        "completion_contract": {"blocking_issues": [{"type": "missing_final_records"}]},
    }

    recovered = _recover_dynamic_final_records_outputs(
        failed_writer,
        expected_output_artifacts=["final_records.jsonl"],
        lab_root=lab_root,
        lab_state_registry=registry,
        task_id="task-dynamic",
        producer_run_ref="writer-run",
        producer_role="SchemaWriterAgent",
        work_item_id="article-a",
    )
    updated = _enforce_dynamic_node_output_contract(
        recovered,
        expected_output_artifacts=["final_records.jsonl"],
    )

    final_path = lab_root / "artifacts" / "tools" / "article-a" / "final_records.jsonl"
    biology_path = lab_root / "artifacts" / "tools" / "article-a" / "biology_component_records.jsonl"
    assert final_path.exists()
    assert biology_path.exists()
    assert json.loads(final_path.read_text(encoding="utf-8").strip())["sequence"] == "AACCGGTT"
    assert updated["status"] == "completed"
    assert updated["failure_reason"] is None
    indexed = registry.list_artifacts("task-dynamic")
    recovered_indexes = [
        item
        for item in indexed
        if item.metadata.get("source") == "dynamic_final_records_recovery"
        and item.metadata.get("work_item_id") == "article-a"
    ]
    assert {Path(item.uri).name for item in recovered_indexes} == {
        "biology_component_records.jsonl",
        "final_records.jsonl",
    }


def test_dynamic_writer_ignores_global_empty_final_when_scoped_validated_records_exist(tmp_path: Path):
    lab_root = tmp_path / "lab"
    registry = _lab_state_registry(lab_root)
    validated = tmp_path / "validated_records.json"
    validated.write_text(
        json.dumps(
            {
                "accepted_records": [
                    {
                        "article_id": "article-a",
                        "work_item_id": "article-a",
                        "component_name": "promoter-a",
                        "sequence": "AACCGGTT",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    registry.save_artifact_index_record(
        ArtifactIndexRecord(
            artifact_ref="validated-a",
            task_id="task-dynamic",
            uri=str(validated),
            artifact_type="dataset",
            metadata={
                "work_item_id": "article-a",
                "filename": "validated_records.json",
                "artifact_kind": "validated_records",
            },
        )
    )
    global_empty = lab_root / "artifacts" / "tools" / "final_records.jsonl"
    global_empty.parent.mkdir(parents=True, exist_ok=True)
    global_empty.write_text("", encoding="utf-8")
    writer_with_empty_global_final = {
        "run_ref": "writer-run",
        "role": "SchemaWriterAgent",
        "status": "completed",
        "failure_reason": None,
        "artifact_refs": [
            {
                "schema_version": "v1",
                "uri": str(global_empty),
                "type": "dataset",
                "metadata": {
                    "filename": "final_records.jsonl",
                    "artifact_kind": "final_records",
                    "record_count": 0,
                },
            }
        ],
        "completion_contract": {"warnings": []},
    }

    recovered = _recover_dynamic_final_records_outputs(
        writer_with_empty_global_final,
        expected_output_artifacts=["final_records.jsonl"],
        lab_root=lab_root,
        lab_state_registry=registry,
        task_id="task-dynamic",
        producer_run_ref="writer-run",
        producer_role="SchemaWriterAgent",
        work_item_id="article-a",
    )
    updated = _enforce_dynamic_node_output_contract(
        recovered,
        expected_output_artifacts=["final_records.jsonl"],
    )

    scoped_final = lab_root / "artifacts" / "tools" / "article-a" / "final_records.jsonl"
    assert scoped_final.exists()
    assert json.loads(scoped_final.read_text(encoding="utf-8").strip())["sequence"] == "AACCGGTT"
    assert updated["status"] == "completed"
    assert any(
        ref.get("metadata", {}).get("work_item_id") == "article-a"
        and ref.get("metadata", {}).get("filename") == "final_records.jsonl"
        for ref in updated["artifact_refs"]
        if isinstance(ref, dict)
    )


def test_dynamic_writer_stops_after_repeated_finalization_suppression_with_existing_artifact(tmp_path: Path):
    lab_root = tmp_path / "lab"
    payload = _workflow_payload()
    payload["dynamic_subagents"] = [
        {
            "subagent_id": "writer",
            "role_name": "SchemaWriterAgent",
            "goal": "Write final records.",
            "system_prompt": "Write final records.",
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"},
            "allowed_tools": ["write_jsonl"],
            "artifact_outputs": ["final_records.jsonl"],
        }
    ]
    payload["workflow_nodes"] = [
        {"node_id": "node-writer", "subagent_id": "writer", "output_artifacts": ["final_records.jsonl"]}
    ]
    payload["workflow_edges"] = []
    payload["artifact_contracts"] = {"final_records.jsonl": {"type": "jsonl"}}
    repeated_write = _tool_response(
        "write_jsonl",
        {"artifact_name": "final_records.jsonl", "records": [{"id": "alpha"}]},
    )
    worker = FakeLLMRuntime(responses=[repeated_write, repeated_write, repeated_write, repeated_write])
    runtime = _base_runtime(
        lab_root,
        task_config=TaskConfig(
            task_id="task-dynamic",
            goal="Process synthetic item.",
            dynamic_subagents=_dynamic_config(allowed_tool_names=["write_jsonl", "read_text"]),
            runtime_policy=RuntimePolicy(
                max_tool_steps=8,
                metadata={
                    "max_repeated_tool_calls_per_run": 1,
                    "repeated_tool_call_suppression_max_violations": 5,
                },
            ),
        ),
        llm_runtimes={
            "planner": FakeLLMRuntime(default_content=json.dumps(payload)),
            "worker": worker,
        },
    )

    result = runtime.run(_request())

    assert result["status"] == "completed"
    assert result["runs"][0]["status"] == "completed"
    assert "repeated finalization tool call suppressed" in result["runs"][0]["final_answer"]
    assert len(worker.requests) == 2
    final_path = lab_root / "artifacts" / "final_records.jsonl"
    assert final_path.exists()
    assert final_path.read_text(encoding="utf-8") == '{"id": "alpha"}\n'
    tool_records = _trajectory_registry(lab_root).list_tool_call_records()
    assert [record.record.result.status for record in tool_records] == ["ok", "error"]


def _runtime(
    lab_root: Path,
    *,
    planner_payload: dict,
    worker_responses: list[str | LLMRuntimeResponse],
    dynamic_config: DynamicSubagentsConfig,
    extra_worker_response: LLMRuntimeResponse | None = None,
) -> TaskRuntime:
    responses = [
        item
        if isinstance(item, LLMRuntimeResponse)
        else LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content=item or "node failed"))
        for item in worker_responses
    ]
    if extra_worker_response is not None:
        responses.append(extra_worker_response)
    return _base_runtime(
        lab_root,
        task_config=TaskConfig(
            task_id="task-dynamic",
            goal="Process synthetic item.",
            dynamic_subagents=dynamic_config,
            runtime_policy=RuntimePolicy(max_tool_steps=2, metadata={}),
        ),
        llm_runtimes={
            "planner": FakeLLMRuntime(default_content=json.dumps(planner_payload)),
            "worker": FakeLLMRuntime(responses=responses),
        },
    )


def _run_minimal_dynamic_task_with_memory(tmp_path: Path):
    lab_root = tmp_path / "lab"
    agents_path = _layout(lab_root).agents_path
    agents_path.parent.mkdir(parents=True, exist_ok=True)
    agents_path.write_text(
        render_agents_markdown(
            {
                "MemoryWorkerAgent": RoleSpec(
                    name="MemoryWorkerAgent",
                    system_prompt="Use memory to answer.",
                    llm_backend=BackendBinding(backend_id="worker"),
                    agent_memory_backend=BackendBinding(backend_id="worker-memory"),
                    allowed_tools=["read_text"],
                )
            }
        ),
        encoding="utf-8",
    )
    workflow_payload = {
        "workflow_id": "wf-memory",
        "task_summary": "Answer with task memory only.",
        "article_context_summary": "none",
        "dynamic_subagents": [
            {
                "subagent_id": "memory-worker",
                "role_name": "MemoryWorkerAgent",
                "goal": "Answer from task context.",
                "system_prompt": "Answer from task context.",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
                "allowed_tools": ["read_text"],
            }
        ],
        "workflow_nodes": [{"node_id": "node-memory-worker", "subagent_id": "memory-worker"}],
        "workflow_edges": [],
        "artifact_contracts": {},
        "validation_rules": [],
        "planner_rationale_summary": "A single worker is enough.",
    }
    task_config = TaskConfig(
        task_id="task-dynamic",
        goal="Process synthetic item.",
        agents_ref=str(agents_path),
        task_memory_backend=BackendBinding(backend_id="task-memory"),
        dynamic_subagents=_dynamic_config(allowed_tool_names=["read_text"]),
        runtime_policy=RuntimePolicy(max_tool_steps=1, metadata={}),
    )
    resolver = LabResolver(lab_root)
    worker_memory = _RecordingMemoryBackend(backend_id="worker-memory")
    runtime = _base_runtime(
        lab_root,
        task_config=task_config,
        llm_runtimes={
            "planner": FakeLLMRuntime(default_content=json.dumps(workflow_payload)),
            "worker": FakeLLMRuntime(default_content="done"),
        },
        memory_runtimes={
            "memory": FakeMemoryBackend(),
            "task-memory": FakeMemoryBackend(backend_id="task-memory"),
            "worker-memory": worker_memory,
        },
    )

    return runtime.run(_request()), resolver, worker_memory


class _RecordingMemoryBackend(FakeMemoryBackend):
    def __init__(self, backend_id: str = "fake-memory"):
        super().__init__(backend_id=backend_id)
        self.search_requests = []
        self.add_requests = []

    def search(self, request):
        self.search_requests.append(request)
        return super().search(request)

    def add(self, task_id, role, messages):
        self.add_requests.append((task_id, role, messages))
        return super().add(task_id, role, messages)


class _GeneratedToolAwarePlanner:
    def __init__(self) -> None:
        self.requests: list[LLMRuntimeRequest] = []
        self.selected_tool_name: str | None = None

    def generate(
        self,
        messages: list,
        tool_specs: list[dict],
        generation_config: LLMGenerationConfig,
    ) -> LLMRuntimeResponse:
        self.requests.append(
            LLMRuntimeRequest(
                messages=messages,
                tool_specs=tool_specs,
                generation_config=generation_config,
            )
        )
        prompt_payload = json.loads(messages[-1].content)
        generated_names = [
            name for name in prompt_payload["effective_allowed_tool_names"] if name.startswith("gt_task_dynamic_")
        ]
        self.selected_tool_name = generated_names[0] if generated_names else "__missing_generated_tool__"
        payload = _single_node_payload()
        payload["dynamic_subagents"][0]["allowed_tools"] = [self.selected_tool_name]
        payload["dynamic_subagents"][0]["role_name"] = "GeneratedToolAgent"
        payload["dynamic_subagents"][0]["goal"] = "Use the generated helper."
        payload["dynamic_subagents"][0]["system_prompt"] = "Use the generated helper."
        return LLMRuntimeResponse(
            action=SubAgentAction(action="final_answer", content=json.dumps(payload)),
            raw_response={"backend": "generated-tool-aware-planner"},
        )


class _GeneratedToolUser:
    def __init__(self) -> None:
        self.requests: list[LLMRuntimeRequest] = []
        self.selected_tool_name: str | None = None
        self._called_tool = False

    def generate(
        self,
        messages: list,
        tool_specs: list[dict],
        generation_config: LLMGenerationConfig,
    ) -> LLMRuntimeResponse:
        self.requests.append(
            LLMRuntimeRequest(
                messages=messages,
                tool_specs=tool_specs,
                generation_config=generation_config,
            )
        )
        generated_names = [spec["name"] for spec in tool_specs if spec["name"].startswith("gt_task_dynamic_")]
        self.selected_tool_name = generated_names[0] if generated_names else "__missing_generated_tool__"
        if not self._called_tool:
            self._called_tool = True
            return _tool_response(self.selected_tool_name, {"source": "synthetic"})
        return LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="done"))


def _base_runtime(
    lab_root: Path,
    *,
    task_config: TaskConfig,
    llm_runtimes: dict[str, FakeLLMRuntime],
    memory_runtimes: dict[str, object] | None = None,
) -> TaskRuntime:
    if task_config.dynamic_subagents is not None and task_config.agents_ref is None:
        role_pool = task_config.roles if task_config.roles else _default_role_pool_roles()
        task_config = task_config.model_copy(update={"agents_ref": _write_agents_ref(lab_root, role_pool)})
    registry = ToolRegistry()
    registry.register(ToolSpec(name="read_text", description="read", parameters_schema={}), lambda args: "text")
    registry.register(
        ToolSpec(name="write_jsonl", description="write jsonl", parameters_schema={}),
        lambda args: _write_jsonl_tool_result(lab_root, args),
    )
    registry.register(
        ToolSpec(name="write_report", description="write report", parameters_schema={}),
        lambda args: ToolResult(
            call_id="write_report",
            status="ok",
            content=f"wrote {args.get('artifact_name') or 'report.md'}",
            artifact_refs=[
                ArtifactRef(
                    uri=str(lab_root / "artifacts" / str(args.get("artifact_name") or "report.md")),
                    type="log",
                    metadata={"filename": str(args.get("artifact_name") or "report.md")},
                )
            ],
        ),
    )
    skill = FakeSkillBackend(
        skills=[SkillItem(skill_id="skill.generic", name="Generic", content="Generic skill.", required_tools=["read_text"])]
    )
    return TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        task_registry=_task_registry(lab_root),
        trajectory_registry=_trajectory_registry(lab_root),
        lab_state_registry=_lab_state_registry(lab_root),
        backend_state_registry=_backend_state_registry(lab_root),
        llm_runtimes=llm_runtimes,
        memory_runtimes=memory_runtimes or {
            "memory": NullMemoryBackend(),
            "meta-memory": NullMemoryBackend(backend_id="meta-memory"),
            "task-memory": NullMemoryBackend(backend_id="task-memory"),
            "worker-memory": NullMemoryBackend(backend_id="worker-memory"),
        },
        skill_runtimes={"skill": skill},
        lab_root=lab_root,
        state_root=_layout(lab_root).state_root,
    )


def _write_jsonl_tool_result(lab_root: Path, args: dict) -> ToolResult:
    records = args.get("records") or []
    path = lab_root / "artifacts" / str(args.get("artifact_name") or "records.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n", encoding="utf-8")
    return ToolResult(
        call_id="write_jsonl",
        status="ok",
        content=f"wrote {len(records)} records to {path}",
        artifact_refs=[
            ArtifactRef(
                uri=str(path),
                type="dataset",
                metadata={
                    "filename": path.name,
                    "artifact_kind": "final_records",
                    "record_count": len(records),
                    "format": "jsonl",
                },
            )
        ],
        metadata={"path": str(path), "record_count": len(records)},
    )


def _dynamic_config(**updates):
    payload = {
        "enabled": True,
        "mode": "dynamic",
        "scope": "per_task",
        "planner_backend": {"backend_id": "planner"},
        "default_worker_backend": {"backend_id": "worker"},
        "allowed_tool_names": ["read_text", "write_report"],
        "max_planner_retries": 0,
    }
    payload.update(updates)
    return DynamicSubagentsConfig.model_validate(payload)


def _workflow_payload():
    return {
        "workflow_id": "wf-runtime",
        "task_summary": "Process a synthetic document.",
        "article_context_summary": "text only",
        "dynamic_subagents": [
            {
                "subagent_id": "text",
                "role_name": "TextContextAgent",
                "goal": "Read text context.",
                "system_prompt": "Read text context.",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
                "allowed_tools": ["read_text", "write_report"],
                "artifact_outputs": ["context.json"],
            },
            {
                "subagent_id": "writer",
                "role_name": "EvidenceWriterAgent",
                "goal": "Write evidence output.",
                "system_prompt": "Write evidence output.",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
                "allowed_tools": ["read_text", "write_report"],
                "artifact_outputs": ["final.json"],
            },
        ],
        "workflow_nodes": [
            {"node_id": "node-text", "subagent_id": "text", "output_artifacts": ["context.json"]},
            {
                "node_id": "node-writer",
                "subagent_id": "writer",
                "input_artifacts": ["context.json"],
                "dependencies": ["node-text"],
                "output_artifacts": ["final.json"],
            },
        ],
        "workflow_edges": [{"source_node_id": "node-text", "target_node_id": "node-writer", "relation": "depends_on"}],
        "artifact_contracts": {"context.json": {"type": "object"}, "final.json": {"type": "object"}},
        "validation_rules": ["outputs must be evidence grounded"],
        "planner_rationale_summary": "Text item needs context then writer.",
    }


def _single_node_payload() -> dict:
    payload = _workflow_payload()
    payload["dynamic_subagents"] = [payload["dynamic_subagents"][0]]
    payload["workflow_nodes"] = [payload["workflow_nodes"][0]]
    payload["workflow_edges"] = []
    payload["artifact_contracts"] = {}
    payload["dynamic_subagents"][0]["artifact_outputs"] = []
    payload["workflow_nodes"][0]["output_artifacts"] = []
    return payload


def _workflow_payload_for_work_item(work_item_id: str) -> dict:
    payload = _workflow_payload()
    payload["workflow_id"] = f"wf-{work_item_id}"
    payload["work_item_id"] = work_item_id
    payload["dynamic_subagents"][0]["artifact_outputs"] = [f"{work_item_id}-context.json"]
    payload["dynamic_subagents"][1]["artifact_inputs"] = [f"{work_item_id}-context.json"]
    payload["dynamic_subagents"][1]["artifact_outputs"] = [f"{work_item_id}-final.json"]
    payload["workflow_nodes"][0]["output_artifacts"] = [f"{work_item_id}-context.json"]
    payload["workflow_nodes"][1]["input_artifacts"] = [f"{work_item_id}-context.json"]
    payload["workflow_nodes"][1]["output_artifacts"] = [f"{work_item_id}-final.json"]
    payload["artifact_contracts"] = {
        f"{work_item_id}-context.json": {"type": "object"},
        f"{work_item_id}-final.json": {"type": "object"},
    }
    return payload


def _default_role_pool_roles() -> dict[str, RoleSpec]:
    return {
        "TextContextAgent": RoleSpec(
            name="TextContextAgent",
            system_prompt="Read text context.",
            llm_backend=BackendBinding(backend_id="worker"),
            allowed_tools=["read_text", "write_report"],
        ),
        "EvidenceWriterAgent": RoleSpec(
            name="EvidenceWriterAgent",
            system_prompt="Write evidence output.",
            llm_backend=BackendBinding(backend_id="worker"),
            allowed_tools=["read_text", "write_report"],
        ),
    }


def _write_agents_ref(lab_root: Path, roles: dict[str, RoleSpec] | None = None) -> str:
    agents_path = _layout(lab_root).agents_path
    agents_path.parent.mkdir(parents=True, exist_ok=True)
    agents_path.write_text(render_agents_markdown(roles or _default_role_pool_roles()), encoding="utf-8")
    return str(agents_path)


def _request() -> TaskRequest:
    return TaskRequest(
        task_id="task-dynamic",
        origin=TaskOrigin.HUMAN,
        purpose=TaskPurpose.SCIENCE,
        goal="Process synthetic item.",
    )


def _request_with_work_items(work_item_ids: list[str]) -> TaskRequest:
    return TaskRequest(
        task_id="task-dynamic",
        origin=TaskOrigin.HUMAN,
        purpose=TaskPurpose.SCIENCE,
        goal="Process synthetic work items.",
        metadata={"work_items": [{"work_item_id": work_item_id} for work_item_id in work_item_ids]},
    )


def _tool_response(name: str, arguments: dict) -> LLMRuntimeResponse:
    return LLMRuntimeResponse(
        action=SubAgentAction(
            action="tool_call",
            tool_call=ToolCall(call_id=name, name=name, arguments=arguments),
        )
    )


def _generated_tool_package_response(generated_package: dict) -> LLMRuntimeResponse:
    return LLMRuntimeResponse(
        action=SubAgentAction(
            action="final_answer",
            content=json.dumps(
                {
                    "route": "END",
                    "instruction": "Register a task-local generated tool before planning.",
                    "metadata": {"generated_tool_package": generated_package},
                }
            ),
        )
    )


def _no_generated_tool_response(reason: str = "No generated runtime tool is needed.") -> LLMRuntimeResponse:
    return LLMRuntimeResponse(
        action=SubAgentAction(
            action="final_answer",
            content=json.dumps(
                {
                    "route": "END",
                    "instruction": reason,
                    "metadata": {"no_generated_tool_reason": reason},
                }
            ),
        )
    )


def _role_pool_noop_response(reason: str = "No reusable role-pool update is needed.") -> LLMRuntimeResponse:
    return LLMRuntimeResponse(
        action=SubAgentAction(
            action="final_answer",
            content=json.dumps(
                {
                    "route": "END",
                    "instruction": reason,
                    "metadata": {"no_role_pool_update_reason": reason},
                }
            ),
        )
    )
