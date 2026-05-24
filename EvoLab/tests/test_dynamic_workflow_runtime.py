from __future__ import annotations

import json
from pathlib import Path

from evolab.backends.llm.fake import FakeLLMRuntime
from evolab.backends.memory import NullMemoryBackend
from evolab.backends.skills import FakeSkillBackend
from evolab.config.task_config import BackendBinding, MetaAgentSpec, RoleSpec, TaskConfig
from evolab.contracts.common import ArtifactRef, RuntimePolicy
from evolab.contracts.dynamic_workflow import DynamicSubagentsConfig
from evolab.contracts.lab_state import ArtifactIndexRecord
from evolab.contracts.llm import LLMRuntimeResponse, SubAgentAction
from evolab.contracts.retrieval import SkillItem
from evolab.contracts.task import TaskOrigin, TaskPurpose, TaskRequest
from evolab.contracts.tools import ToolCall, ToolResult, ToolSpec
from evolab.registries.lab_state import FileLabStateRegistry
from evolab.registries.task import FileTaskRegistry
from evolab.registries.trajectory import FileTrajectoryRegistry
from evolab.runtime.prompt_builder import PromptBuilder
from evolab.runtime.task_runtime import TaskRuntime, _enforce_dynamic_node_output_contract
from evolab.tools.runtime import ToolRegistry, ToolRuntime


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
    trajectory = FileTrajectoryRegistry(lab_root / "registries" / "trajectory")
    assert [record.role for record in trajectory.list_subagent_runs()] == ["TextContextAgent", "EvidenceWriterAgent"]
    dynamic_root = lab_root / "dynamic_workflows" / request.task_id / "wf-runtime"
    assert (dynamic_root / "dynamic_workflow_spec.json").exists()
    assert (dynamic_root / "dynamic_subagents.json").exists()
    assert (dynamic_root / "dynamic_workflow_trace.json").exists()
    assert (dynamic_root / "dynamic_subagent_records.jsonl").exists()
    trace = json.loads((dynamic_root / "dynamic_workflow_trace.json").read_text(encoding="utf-8"))
    assert trace["status"] == "completed"
    assert len(trace["run_refs"]) == 2


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
    spec_path = lab_root / "dynamic_workflows" / "task-dynamic" / "wf-runtime" / "dynamic_workflow_spec.json"
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


def test_dynamic_planning_failure_falls_back_to_static_mode(tmp_path: Path):
    lab_root = tmp_path / "lab"
    task_config = TaskConfig(
        task_id="task-dynamic",
        goal="Process synthetic item.",
        meta_agent=MetaAgentSpec(
            system_prompt="Return route JSON only.",
            llm_backend=BackendBinding(backend_id="meta"),
        ),
        roles={
            "StaticAgent": RoleSpec(
                name="StaticAgent",
                system_prompt="Static fallback.",
                llm_backend=BackendBinding(backend_id="worker"),
                allowed_tools=["read_text"],
            )
        },
        max_dispatch_steps=2,
        dynamic_subagents=_dynamic_config(max_planner_retries=0, fallback_to_static=True),
        runtime_policy=RuntimePolicy(max_tool_steps=2, metadata={}),
    )
    runtime = _base_runtime(
        lab_root,
        task_config=task_config,
        llm_runtimes={
            "planner": FakeLLMRuntime(default_content="not json"),
            "meta": FakeLLMRuntime(
                responses=[
                    LLMRuntimeResponse(
                        action=SubAgentAction(
                            action="final_answer",
                            content='{"route":"StaticAgent","instruction":"Use static fallback."}',
                        )
                    ),
                    LLMRuntimeResponse(
                        action=SubAgentAction(
                            action="final_answer",
                            content='{"route":"END","instruction":"done","metadata":{"final_answer":"static done"}}',
                        )
                    ),
                ]
            ),
            "worker": FakeLLMRuntime(default_content="static worker done"),
        },
    )

    result = runtime.run(_request())

    assert result.get("execution_mode") != "dynamic"
    assert result["final_answer"] == "static done"
    fallback_path = lab_root / "dynamic_workflows" / "task-dynamic" / "planning_failed" / "fallback_reason.json"
    assert fallback_path.exists()


def test_per_work_item_dynamic_planning_failure_after_partial_run_does_not_switch_to_static(tmp_path: Path):
    lab_root = tmp_path / "lab"
    first_payload = _workflow_payload_for_work_item("item-a")
    third_payload = _workflow_payload_for_work_item("item-c")
    task_config = TaskConfig(
        task_id="task-dynamic",
        goal="Process synthetic work items.",
        meta_agent=MetaAgentSpec(
            system_prompt="Static fallback should not be reached after partial dynamic execution.",
            llm_backend=BackendBinding(backend_id="meta"),
        ),
        roles={
            "StaticAgent": RoleSpec(
                name="StaticAgent",
                system_prompt="Static fallback.",
                llm_backend=BackendBinding(backend_id="worker"),
                allowed_tools=["read_text"],
            )
        },
        dynamic_subagents=_dynamic_config(scope="per_work_item", fallback_to_static=True, max_planner_retries=0),
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
            "meta": FakeLLMRuntime(default_content='{"route":"StaticAgent","instruction":"should not run"}'),
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
    assert result["runs"][2]["metadata"]["fallback_suppressed_after_partial_dynamic"] is True
    assert not any(run["role"] == "StaticAgent" for run in result["runs"])
    assert (lab_root / "dynamic_workflows" / request.task_id / "planning_failed" / "fallback_reason.json").exists()
    assert (lab_root / "dynamic_workflows" / request.task_id / "wf-item-a" / "dynamic_workflow_trace.json").exists()
    assert (lab_root / "dynamic_workflows" / request.task_id / "wf-item-c" / "dynamic_workflow_trace.json").exists()


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
    registry = FileLabStateRegistry(lab_root / "registries" / "lab_state")
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
    registry = FileLabStateRegistry(lab_root / "registries" / "lab_state")
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


def _base_runtime(lab_root: Path, *, task_config: TaskConfig, llm_runtimes: dict[str, FakeLLMRuntime]) -> TaskRuntime:
    registry = ToolRegistry()
    registry.register(ToolSpec(name="read_text", description="read", parameters_schema={}), lambda args: "text")
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
        task_registry=FileTaskRegistry(lab_root / "registries" / "task"),
        trajectory_registry=FileTrajectoryRegistry(lab_root / "registries" / "trajectory"),
        lab_state_registry=FileLabStateRegistry(lab_root / "registries" / "lab_state"),
        llm_runtimes=llm_runtimes,
        memory_runtimes={"memory": NullMemoryBackend()},
        skill_runtimes={"skill": skill},
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
        "fallback_to_static": False,
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
