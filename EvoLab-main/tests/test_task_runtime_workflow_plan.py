from pathlib import Path
import time

import pytest

from evolab.config.task_config import BackendBinding, MetaAgentSpec, RoleSpec, TaskConfig
from evolab.contracts.common import ArtifactRef, Message, RuntimePolicy
from evolab.contracts.llm import LLMGenerationConfig, LLMRuntimeResponse, SubAgentAction
from evolab.contracts.retrieval import MemoryBundle, RetrievalRequest, SkillBundle, SkillRef
from evolab.contracts.task import TaskOrigin, TaskPurpose, TaskRequest
from evolab.contracts.tools import ToolCall, ToolResult, ToolSpec
from evolab.registries.trajectory import FileTrajectoryRegistry
from evolab.runtime.prompt_builder import PromptBuilder
from evolab.runtime.task_runtime import TaskRuntime, _parse_dispatch_decision
from evolab.tools.runtime import ToolRegistry, ToolRuntime


class MemoryRuntime:
    def __init__(self):
        self.add_calls: list[tuple[str, str, list[Message]]] = []

    def search(self, request: RetrievalRequest) -> MemoryBundle:
        return MemoryBundle(backend_id="memory-local", items=[])

    def add(self, task_id: str, role: str, messages: list[Message]) -> dict[str, str]:
        self.add_calls.append((task_id, role, messages))
        return {"status": "updated", "state_ref": f"{role}-memory-after"}


class SkillRuntime:
    def __init__(self):
        self.look_at_events: list[dict] = []

    def get(self, request: RetrievalRequest) -> SkillBundle:
        schema = SkillRef(
            skill_id="skill.extraction_schema_interpretation.v1",
            name="Extraction Schema Interpretation",
            content="Description:\nInterpret schema.",
            required_tools=["lookup"],
        )
        mapping = SkillRef(
            skill_id="skill.schema_guided_field_mapping.v1",
            name="Schema Guided Field Mapping",
            content="Description:\nMap fields.",
            required_tools=["lookup"],
        )
        return SkillBundle(
            backend_id="skill-local",
            graph_version_ref="graph-v1",
            required_tools=["lookup"],
            skills=[mapping, schema],
            metadata={
                "graph_context_summary": {"graph_version": "graph-v1"},
                "retrieval_trace": {
                    "returned_skill_ids": [mapping.skill_id, schema.skill_id],
                    "directly_matched_skill_ids": [mapping.skill_id],
                    "dependency_added_skill_ids": [schema.skill_id],
                    "optional_expanded_skill_ids": [],
                    "relation_expansion_steps": [
                        {
                            "source_skill_id": mapping.skill_id,
                            "target_skill_id": schema.skill_id,
                            "relation": "depends_on",
                            "reason": "dependency_added",
                        }
                    ],
                },
            },
        )

    def look_at(self, event: dict) -> dict[str, str]:
        self.look_at_events.append(event)
        return {"status": "recorded"}


class ScriptedLLM:
    def __init__(self, responses: list[LLMRuntimeResponse]):
        self.responses = responses
        self.calls: list[tuple[list[Message], list[dict], LLMGenerationConfig]] = []

    def generate(self, messages: list[Message], tool_specs: list[dict], generation_config: LLMGenerationConfig):
        self.calls.append((messages, tool_specs, generation_config))
        return self.responses.pop(0)


def _request() -> TaskRequest:
    return TaskRequest(
        task_id="task-1",
        origin=TaskOrigin.HUMAN,
        purpose=TaskPurpose.SCIENCE,
        goal="Extract records.",
    )


def _runtime_with_static_role_dispatch(**kwargs) -> TaskRuntime:
    runtime = TaskRuntime(**kwargs)

    def dispatch(request: TaskRequest) -> dict:
        roles = runtime._roles()
        if len(roles) != 1:
            raise AssertionError("workflow-plan tests expect a single role")
        result = runtime._run_role(request, roles[0], 0)
        return {
            "task_id": request.task_id,
            "status": result.get("status", "completed"),
            "failure_reason": result.get("failure_reason") if isinstance(result.get("failure_reason"), str) else None,
            "run_ref": result["run_ref"],
            "run_refs": [result["run_ref"]],
            "runs": [result],
            "role": result["role"],
            "final_answer": result["final_answer"],
        }

    runtime.dispatch_loop = dispatch
    return runtime


def _runtime(tmp_path: Path, *, enable_workflow_planning: bool, llm: ScriptedLLM, skill: SkillRuntime):
    registry = ToolRegistry()
    artifact_path = tmp_path / "lookup.json"

    def lookup(arguments: dict) -> ToolResult:
        artifact_path.write_text("{}", encoding="utf-8")
        return ToolResult(
            call_id="handler-call",
            status="ok",
            content="lookup ok",
            artifact_refs=[ArtifactRef(uri=str(artifact_path), type="dataset")],
        )

    registry.register(ToolSpec(name="lookup", description="lookup", parameters_schema={"type": "object"}), lookup)
    task_config = TaskConfig(
        task_id="task-1",
        goal="Extract records.",
        runtime_policy=RuntimePolicy(enable_workflow_planning=enable_workflow_planning),
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="Solve.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["lookup"],
            )
        },
    )
    return _runtime_with_static_role_dispatch(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        tool_artifact_root_factory=lambda request, run_ref: tmp_path / "artifacts" / run_ref,
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": MemoryRuntime()},
        skill_runtimes={"skill-local": skill},
    )


def test_workflow_planning_disabled_keeps_flat_runtime_metadata(tmp_path: Path):
    skill = SkillRuntime()
    llm = ScriptedLLM([LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="flat answer"))])
    runtime = _runtime(tmp_path, enable_workflow_planning=False, llm=llm, skill=skill)

    result = runtime.run(_request())
    saved = runtime.trajectory_registry.get_subagent_run(result["run_ref"])

    assert saved is not None
    assert "workflow_plan" not in saved.metadata
    assert skill.look_at_events[0]["metadata"].get("workflow_plan") is None


def test_plan_aware_task_runtime_executes_nodes_records_trace_artifacts_and_look_at(tmp_path: Path):
    skill = SkillRuntime()
    llm = ScriptedLLM(
        [
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="schema interpreted")),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="call-1", name="lookup", arguments={"query": "fields"}),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="fields mapped")),
        ]
    )
    runtime = _runtime(tmp_path, enable_workflow_planning=True, llm=llm, skill=skill)

    result = runtime.run(_request())
    saved = runtime.trajectory_registry.get_subagent_run(result["run_ref"])

    assert saved is not None
    plan = saved.metadata["workflow_plan"]
    assert plan["metadata"]["topological_order"] == [
        "skill.extraction_schema_interpretation.v1",
        "skill.schema_guided_field_mapping.v1",
    ]
    assert [record["skill_id"] for record in saved.metadata["node_execution_records"]] == plan["metadata"][
        "topological_order"
    ]
    assert saved.metadata["plan_execution_trace"]["status"] == "completed"
    assert saved.metadata["tool_trace"]["calls"][0]["tool_call"]["name"] == "lookup"
    assert saved.artifact_refs
    assert saved.metadata["plan_execution_trace"]["artifact_refs"] == [ref.model_dump(mode="json") for ref in saved.artifact_refs]
    llm_calls = runtime.trajectory_registry.list_llm_calls()
    assert saved.llm_call_refs == [call.call_ref for call in llm_calls]
    assert len(llm_calls) == 3
    assert llm_calls[1].metadata["runtime_stage"] == "workflow_node"
    assert llm_calls[1].metadata["workflow_skill_id"] == "skill.schema_guided_field_mapping.v1"
    assert llm_calls[1].output_messages[0].metadata["tool_call"]["name"] == "lookup"
    assert llm_calls[2].input_messages[-1].role == "tool"
    assert llm_calls[2].input_messages[-1].content.startswith("lookup ok")
    assert "Tool result payload:" in llm_calls[2].input_messages[-1].content
    assert saved.artifact_refs[0].uri in llm_calls[2].input_messages[-1].content
    observation = skill.look_at_events[0]
    assert observation["metadata"]["workflow_plan"]["plan_id"] == plan["plan_id"]
    assert observation["metadata"]["plan_execution_trace"]["plan_id"] == plan["plan_id"]
    assert observation["tool_trace"]["calls"][0]["tool_call"]["name"] == "lookup"


def test_flat_task_runtime_executes_multiple_tool_calls_from_one_llm_step(tmp_path: Path):
    skill = SkillRuntime()
    llm = ScriptedLLM(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_calls=[
                        ToolCall(call_id="call-1", name="lookup", arguments={"query": "schema"}),
                        ToolCall(call_id="call-2", name="lookup", arguments={"query": "fields"}),
                    ],
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="used both tools")),
        ]
    )
    runtime = _runtime(tmp_path, enable_workflow_planning=False, llm=llm, skill=skill)

    result = runtime.run(_request())
    saved = runtime.trajectory_registry.get_subagent_run(result["run_ref"])
    llm_calls = runtime.trajectory_registry.list_llm_calls()

    assert saved is not None
    assert len(saved.tool_calls) == 2
    assert [record.tool_call.call_id for record in saved.tool_calls] == ["call-1", "call-2"]
    assert [record.result.call_id for record in saved.tool_calls] == ["call-1", "call-2"]
    assert len(llm_calls) == 2
    assert llm_calls[0].output_messages[0].metadata["tool_calls"] == [
        {"schema_version": "v1", "call_id": "call-1", "name": "lookup", "arguments": {"query": "schema"}},
        {"schema_version": "v1", "call_id": "call-2", "name": "lookup", "arguments": {"query": "fields"}},
    ]
    tool_messages = [message for message in llm_calls[1].input_messages if message.role == "tool"]
    assert [message.tool_call_id for message in tool_messages] == ["call-1", "call-2"]
    assert all("Tool result payload:" in message.content for message in tool_messages)


def test_flat_task_runtime_truncates_large_tool_results_before_next_llm_call(tmp_path: Path):
    registry = ToolRegistry()

    def large_lookup(arguments: dict) -> ToolResult:
        return ToolResult(
            call_id="handler-call",
            status="ok",
            content="lookup ok",
            metadata={"text": "x" * 2_000},
        )

    registry.register(ToolSpec(name="lookup", description="lookup", parameters_schema={"type": "object"}), large_lookup)
    llm = ScriptedLLM(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="call-1", name="lookup", arguments={"query": "large"}),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="used compact tool result")),
        ]
    )
    task_config = TaskConfig(
        task_id="task-1",
        goal="Extract records.",
        runtime_policy=RuntimePolicy(
            enable_workflow_planning=False,
            metadata={"tool_result_prompt_max_chars": 300},
        ),
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="Solve.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["lookup"],
            )
        },
    )
    runtime = _runtime_with_static_role_dispatch(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": MemoryRuntime()},
        skill_runtimes={"skill-local": SkillRuntime()},
    )

    runtime.run(_request())

    tool_messages = [message for message in llm.calls[1][0] if message.role == "tool"]
    assert len(tool_messages) == 1
    assert len(tool_messages[0].content) <= 360
    assert "[truncated" in tool_messages[0].content
    saved = runtime.trajectory_registry.list_tool_call_records()[0]
    assert saved.record.result.metadata["text"] == "x" * 2_000


def test_flat_task_runtime_suppresses_repeated_same_target_tool_calls(tmp_path: Path):
    registry = ToolRegistry()
    executed: list[dict] = []
    registry.register(
        ToolSpec(name="lookup", description="lookup", parameters_schema={"type": "object"}),
        lambda arguments: executed.append(arguments) or "lookup ok",
    )
    llm = ScriptedLLM(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="call-1", name="lookup", arguments={"path": "source.txt"}),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="call-2", name="lookup", arguments={"path": "source.txt"}),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="call-3", name="lookup", arguments={"path": "source.txt"}),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="used prior result")),
        ]
    )
    task_config = TaskConfig(
        task_id="task-1",
        goal="Extract records.",
        runtime_policy=RuntimePolicy(
            enable_workflow_planning=False,
            metadata={"max_repeated_tool_calls_per_run": 2},
        ),
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="Solve.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["lookup"],
            )
        },
    )
    runtime = _runtime_with_static_role_dispatch(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": MemoryRuntime()},
        skill_runtimes={"skill-local": SkillRuntime()},
    )

    runtime.run(_request())

    assert executed == [{"path": "source.txt"}, {"path": "source.txt"}]
    records = runtime.trajectory_registry.list_tool_call_records()
    assert [record.record.result.status for record in records] == ["ok", "ok", "error"]
    assert records[-1].record.result.metadata["error_type"] == "repeated_tool_call_suppressed"
    repeated_tool_messages = [message for message in llm.calls[3][0] if message.tool_call_id == "call-3"]
    assert "repeated tool call suppressed" in repeated_tool_messages[0].content


def test_flat_runtime_stops_after_role_completion_guards_pass(tmp_path: Path):
    registry = ToolRegistry()
    executed: list[str] = []
    report_path = tmp_path / "report.md"
    registry.register(
        ToolSpec(name="write_report", description="write report", parameters_schema={"type": "object"}),
        lambda arguments: executed.append("write_report")
        or ToolResult(
            call_id="write_report",
            status="ok",
            content="wrote report",
            artifact_refs=[ArtifactRef(uri=str(report_path), type="log", metadata={"filename": "report.md"})],
        ),
    )
    registry.register(
        ToolSpec(name="lookup", description="lookup", parameters_schema={"type": "object"}),
        lambda arguments: executed.append("lookup") or "lookup should not run",
    )
    llm = ScriptedLLM(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="report", name="write_report", arguments={"content": "done"}),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="lookup", name="lookup", arguments={}),
                )
            ),
        ]
    )
    task_config = TaskConfig(
        task_id="task-1",
        goal="Write a report.",
        runtime_policy=RuntimePolicy(
            enable_workflow_planning=False,
            metadata={"completion_guards_by_role": {"Reporter": {"required_tool_calls_before_final": ["write_report"]}}},
        ),
        roles={
            "Reporter": RoleSpec(
                name="Reporter",
                system_prompt="Report.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["write_report", "lookup"],
            )
        },
    )
    runtime = _runtime_with_static_role_dispatch(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": MemoryRuntime()},
        skill_runtimes={"skill-local": SkillRuntime()},
    )

    result = runtime.run(_request())
    saved = runtime.trajectory_registry.get_subagent_run(result["run_ref"])

    assert executed == ["write_report"]
    assert len(llm.calls) == 1
    assert saved is not None
    assert saved.output_messages[-1].content == "role completion guards satisfied after successful required tool calls"


def test_flat_runtime_reserves_remaining_budget_for_completion_guard_tools(tmp_path: Path):
    registry = ToolRegistry()
    executed: list[str] = []
    jsonl_path = tmp_path / "records.jsonl"
    report_path = tmp_path / "report.md"
    registry.register(
        ToolSpec(name="lookup", description="lookup", parameters_schema={"type": "object"}),
        lambda arguments: executed.append("lookup") or "lookup ok",
    )
    registry.register(
        ToolSpec(name="write_jsonl", description="write jsonl", parameters_schema={"type": "object"}),
        lambda arguments: executed.append("write_jsonl")
        or ToolResult(
            call_id="write_jsonl",
            status="ok",
            content="wrote records",
            artifact_refs=[ArtifactRef(uri=str(jsonl_path), type="dataset", metadata={"record_count": 1})],
            metadata={"record_count": 1},
        ),
    )
    registry.register(
        ToolSpec(name="write_report", description="write report", parameters_schema={"type": "object"}),
        lambda arguments: executed.append("write_report")
        or ToolResult(
            call_id="write_report",
            status="ok",
            content="wrote report",
            artifact_refs=[ArtifactRef(uri=str(report_path), type="log", metadata={"filename": "report.md"})],
        ),
    )
    llm = ScriptedLLM(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="lookup-1", name="lookup", arguments={"query": "one"}),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="lookup-2", name="lookup", arguments={"query": "two"}),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="lookup-3", name="lookup", arguments={"query": "three"}),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_calls=[
                        ToolCall(call_id="jsonl", name="write_jsonl", arguments={"records": [{"id": "1"}]}),
                        ToolCall(call_id="report", name="write_report", arguments={"content": "done"}),
                    ],
                )
            ),
        ]
    )
    task_config = TaskConfig(
        task_id="task-1",
        goal="Write records and a report.",
        runtime_policy=RuntimePolicy(
            enable_workflow_planning=False,
            max_tool_steps=4,
            metadata={
                "completion_guards_by_role": {
                    "Writer": {"required_tool_calls_before_final": ["write_jsonl", "write_report"]}
                }
            },
        ),
        roles={
            "Writer": RoleSpec(
                name="Writer",
                system_prompt="Write.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["lookup", "write_jsonl", "write_report"],
            )
        },
    )
    runtime = _runtime_with_static_role_dispatch(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": MemoryRuntime()},
        skill_runtimes={"skill-local": SkillRuntime()},
    )

    result = runtime.run(_request())
    saved = runtime.trajectory_registry.get_subagent_run(result["run_ref"])
    tool_records = runtime.trajectory_registry.list_tool_call_records()

    assert executed == ["lookup", "lookup", "write_jsonl", "write_report"]
    assert [record.record.tool_call.name for record in tool_records] == [
        "lookup",
        "lookup",
        "lookup",
        "write_jsonl",
        "write_report",
    ]
    assert tool_records[2].record.result.status == "error"
    assert tool_records[2].record.result.metadata["error_type"] == "completion_guard_budget_reserved"
    assert saved is not None
    assert saved.output_messages[-1].content == "role completion guards satisfied after successful required tool calls"


def test_flat_runtime_allows_required_guard_tool_after_reserved_budget_warning(tmp_path: Path):
    registry = ToolRegistry()
    executed: list[str] = []
    report_path = tmp_path / "report.md"
    registry.register(
        ToolSpec(name="lookup", description="lookup", parameters_schema={"type": "object"}),
        lambda arguments: executed.append("lookup") or "lookup ok",
    )
    registry.register(
        ToolSpec(name="write_report", description="write report", parameters_schema={"type": "object"}),
        lambda arguments: executed.append("write_report")
        or ToolResult(
            call_id="write_report",
            status="ok",
            content="wrote report",
            artifact_refs=[ArtifactRef(uri=str(report_path), type="log", metadata={"filename": "report.md"})],
        ),
    )
    llm = ScriptedLLM(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="lookup-1", name="lookup", arguments={"path": "source.txt"}),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="lookup-2", name="lookup", arguments={"path": "source.txt"}),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="report", name="write_report", arguments={"content": "done"}),
                )
            ),
        ]
    )
    task_config = TaskConfig(
        task_id="task-1",
        goal="Write a report.",
        runtime_policy=RuntimePolicy(
            enable_workflow_planning=False,
            max_tool_steps=2,
            metadata={
                "max_repeated_tool_calls_per_run": 1,
                "completion_guards_by_role": {"Reporter": {"required_tool_calls_before_final": ["write_report"]}},
            },
        ),
        roles={
            "Reporter": RoleSpec(
                name="Reporter",
                system_prompt="Report.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["lookup", "write_report"],
            )
        },
    )
    runtime = _runtime_with_static_role_dispatch(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": MemoryRuntime()},
        skill_runtimes={"skill-local": SkillRuntime()},
    )

    result = runtime.run(_request())
    records = runtime.trajectory_registry.list_tool_call_records()

    assert result["final_answer"] == "role completion guards satisfied after successful required tool calls"
    assert executed == ["lookup", "write_report"]
    assert [record.record.result.status for record in records] == ["ok", "error", "ok"]
    assert records[1].record.result.metadata["error_type"] == "completion_guard_budget_reserved"


def test_flat_runtime_does_not_spend_tool_budget_on_repeated_suppression(tmp_path: Path):
    registry = ToolRegistry()
    executed: list[str] = []
    jsonl_path = tmp_path / "records.jsonl"
    report_path = tmp_path / "report.md"
    registry.register(
        ToolSpec(name="lookup", description="lookup", parameters_schema={"type": "object"}),
        lambda arguments: executed.append(f"lookup:{arguments['path']}") or "lookup ok",
    )
    registry.register(
        ToolSpec(name="write_jsonl", description="write jsonl", parameters_schema={"type": "object"}),
        lambda arguments: executed.append("write_jsonl")
        or ToolResult(
            call_id="write_jsonl",
            status="ok",
            content="wrote records",
            artifact_refs=[ArtifactRef(uri=str(jsonl_path), type="dataset", metadata={"record_count": 1})],
            metadata={"record_count": 1},
        ),
    )
    registry.register(
        ToolSpec(name="write_report", description="write report", parameters_schema={"type": "object"}),
        lambda arguments: executed.append("write_report")
        or ToolResult(
            call_id="write_report",
            status="ok",
            content="wrote report",
            artifact_refs=[ArtifactRef(uri=str(report_path), type="log", metadata={"filename": "report.md"})],
        ),
    )
    llm = ScriptedLLM(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="lookup-1", name="lookup", arguments={"path": "source-a.txt"}),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="lookup-2", name="lookup", arguments={"path": "source-a.txt"}),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="lookup-3", name="lookup", arguments={"path": "source-b.txt"}),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_calls=[
                        ToolCall(call_id="jsonl", name="write_jsonl", arguments={"records": [{"id": "1"}]}),
                        ToolCall(call_id="report", name="write_report", arguments={"content": "done"}),
                    ],
                )
            ),
        ]
    )
    task_config = TaskConfig(
        task_id="task-1",
        goal="Write records and a report.",
        runtime_policy=RuntimePolicy(
            enable_workflow_planning=False,
            max_tool_steps=4,
            metadata={
                "max_repeated_tool_calls_per_run": 1,
                "completion_guards_by_role": {
                    "Writer": {"required_tool_calls_before_final": ["write_jsonl", "write_report"]}
                },
            },
        ),
        roles={
            "Writer": RoleSpec(
                name="Writer",
                system_prompt="Write.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["lookup", "write_jsonl", "write_report"],
            )
        },
    )
    runtime = _runtime_with_static_role_dispatch(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": MemoryRuntime()},
        skill_runtimes={"skill-local": SkillRuntime()},
    )

    result = runtime.run(_request())
    records = runtime.trajectory_registry.list_tool_call_records()

    assert result["final_answer"] == "role completion guards satisfied after successful required tool calls"
    assert executed == ["lookup:source-a.txt", "lookup:source-b.txt", "write_jsonl", "write_report"]
    assert [record.record.result.status for record in records] == ["ok", "error", "ok", "ok", "ok"]
    assert records[1].record.result.metadata["error_type"] == "repeated_tool_call_suppressed"


def test_flat_runtime_caps_non_required_tools_before_completion_guard_outputs(tmp_path: Path):
    registry = ToolRegistry()
    executed: list[str] = []
    report_path = tmp_path / "report.md"
    registry.register(
        ToolSpec(name="lookup", description="lookup", parameters_schema={"type": "object"}),
        lambda arguments: executed.append(arguments["query"]) or "lookup ok",
    )
    registry.register(
        ToolSpec(name="write_report", description="write report", parameters_schema={"type": "object"}),
        lambda arguments: executed.append("write_report")
        or ToolResult(
            call_id="write_report",
            status="ok",
            content="wrote report",
            artifact_refs=[ArtifactRef(uri=str(report_path), type="log", metadata={"filename": "report.md"})],
        ),
    )
    llm = ScriptedLLM(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="lookup-1", name="lookup", arguments={"query": "one"}),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="lookup-2", name="lookup", arguments={"query": "two"}),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="report", name="write_report", arguments={"content": "done"}),
                )
            ),
        ]
    )
    task_config = TaskConfig(
        task_id="task-1",
        goal="Write a report.",
        runtime_policy=RuntimePolicy(
            enable_workflow_planning=False,
            metadata={
                "completion_guards_by_role": {
                    "Reporter": {
                        "required_tool_calls_before_final": ["write_report"],
                        "max_non_required_tool_calls_before_required_outputs": 1,
                    }
                }
            },
        ),
        roles={
            "Reporter": RoleSpec(
                name="Reporter",
                system_prompt="Report.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["lookup", "write_report"],
            )
        },
    )
    runtime = _runtime_with_static_role_dispatch(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": MemoryRuntime()},
        skill_runtimes={"skill-local": SkillRuntime()},
    )

    result = runtime.run(_request())
    records = runtime.trajectory_registry.list_tool_call_records()

    assert result["final_answer"] == "role completion guards satisfied after successful required tool calls"
    assert executed == ["one", "write_report"]
    assert [record.record.result.status for record in records] == ["ok", "error", "ok"]
    assert records[1].record.result.metadata["error_type"] == "completion_guard_required_outputs_due"


def test_flat_runtime_caps_non_required_tools_when_minimum_jsonl_records_missing(tmp_path: Path):
    registry = ToolRegistry()
    executed: list[str] = []
    registry.register(
        ToolSpec(name="lookup", description="lookup", parameters_schema={"type": "object"}),
        lambda arguments: executed.append(arguments["query"]) or "lookup ok",
    )
    registry.register(
        ToolSpec(name="write_jsonl", description="write jsonl", parameters_schema={"type": "object"}),
        lambda arguments: executed.append(f"write_jsonl:{len(arguments.get('records') or [])}")
        or ToolResult(
            call_id="write_jsonl",
            status="ok",
            content="wrote records",
            artifact_refs=[ArtifactRef(uri=str(tmp_path / "records.jsonl"), type="dataset")],
            metadata={"record_count": len(arguments.get("records") or [])},
        ),
    )
    llm = ScriptedLLM(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="lookup-1", name="lookup", arguments={"query": "one"}),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="empty", name="write_jsonl", arguments={"records": []}),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="lookup-2", name="lookup", arguments={"query": "two"}),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="nonempty", name="write_jsonl", arguments={"records": [{"id": "1"}]}),
                )
            ),
        ]
    )
    task_config = TaskConfig(
        task_id="task-1",
        goal="Write records.",
        runtime_policy=RuntimePolicy(
            enable_workflow_planning=False,
            metadata={
                "completion_guards_by_role": {
                    "Writer": {
                        "required_tool_calls_before_final": ["write_jsonl"],
                        "minimum_jsonl_records_before_final": 1,
                        "max_non_required_tool_calls_before_required_outputs": 1,
                    }
                }
            },
        ),
        roles={
            "Writer": RoleSpec(
                name="Writer",
                system_prompt="Write.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["lookup", "write_jsonl"],
            )
        },
    )
    runtime = _runtime_with_static_role_dispatch(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": MemoryRuntime()},
        skill_runtimes={"skill-local": SkillRuntime()},
    )

    result = runtime.run(_request())
    records = runtime.trajectory_registry.list_tool_call_records()

    assert result["final_answer"] == "role completion guards satisfied after successful required tool calls"
    assert executed == ["one", "write_jsonl:0", "write_jsonl:1"]
    assert [record.record.result.status for record in records] == ["ok", "ok", "error", "ok"]
    assert records[2].record.result.metadata["error_type"] == "completion_guard_required_outputs_due"
    assert records[2].record.result.metadata["minimum_jsonl_records_before_final"] == 1


def test_workflow_runtime_fails_after_repeated_completion_guard_violations(tmp_path: Path):
    class GuardedSkillRuntime(SkillRuntime):
        def get(self, request: RetrievalRequest) -> SkillBundle:
            skill = SkillRef(
                skill_id="skill.write_records.v1",
                name="Write Records",
                content="Description:\nWrite records.",
                required_tools=["lookup", "write_jsonl"],
            )
            return SkillBundle(
                backend_id="skill-local",
                graph_version_ref="graph-v1",
                required_tools=["lookup", "write_jsonl"],
                skills=[skill],
            )

    registry = ToolRegistry()
    executed: list[str] = []
    registry.register(
        ToolSpec(name="lookup", description="lookup", parameters_schema={"type": "object"}),
        lambda arguments: executed.append(arguments["query"]) or "lookup ok",
    )
    registry.register(
        ToolSpec(name="write_jsonl", description="write jsonl", parameters_schema={"type": "object"}),
        lambda arguments: executed.append("write_jsonl")
        or ToolResult(call_id="write_jsonl", status="ok", content="wrote records"),
    )
    llm = ScriptedLLM(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="lookup-1", name="lookup", arguments={"query": "one"}),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="lookup-2", name="lookup", arguments={"query": "two"}),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="lookup-3", name="lookup", arguments={"query": "three"}),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="lookup-4", name="lookup", arguments={"query": "four"}),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="done without records")),
        ]
    )
    task_config = TaskConfig(
        task_id="task-1",
        goal="Write records.",
        runtime_policy=RuntimePolicy(
            enable_workflow_planning=True,
            max_tool_steps_per_node=8,
            metadata={
                "completion_guard_max_violations": 3,
                "completion_guards_by_role": {
                    "Writer": {
                        "required_tool_calls_before_final": ["write_jsonl"],
                        "max_non_required_tool_calls_before_required_outputs": 1,
                    }
                },
            },
        ),
        roles={
            "Writer": RoleSpec(
                name="Writer",
                system_prompt="Write.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["lookup", "write_jsonl"],
            )
        },
    )
    runtime = _runtime_with_static_role_dispatch(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": MemoryRuntime()},
        skill_runtimes={"skill-local": GuardedSkillRuntime()},
    )

    result = runtime._run_role(_request(), task_config.roles["Writer"], 0, return_failed_result=True)
    saved = runtime.trajectory_registry.get_subagent_run(result["run_ref"])
    records = runtime.trajectory_registry.list_tool_call_records()

    assert result["status"] == "guard_failed"
    assert result["failure_reason"] is not None
    assert "completion guard required outputs due" in result["failure_reason"]
    assert len(llm.calls) == 4
    assert executed == ["one"]
    assert [record.record.result.metadata.get("error_type") for record in records] == [
        None,
        "completion_guard_required_outputs_due",
        "completion_guard_required_outputs_due",
        "completion_guard_required_outputs_due",
    ]
    assert saved is not None
    node_records = saved.metadata["node_execution_records"]
    assert node_records[0]["status"] == "failed"


def test_workflow_completion_guard_does_not_block_preliminary_node_without_due_output_tool(tmp_path: Path):
    class TwoStepSkillRuntime(SkillRuntime):
        def get(self, request: RetrievalRequest) -> SkillBundle:
            locate = SkillRef(
                skill_id="skill.locate_sources.v1",
                name="Locate Sources",
                content="Description:\nFind relevant source material.",
                required_tools=["lookup"],
            )
            write = SkillRef(
                skill_id="skill.write_records.v1",
                name="Write Records",
                content="Description:\nWrite extracted records.",
                required_tools=["write_jsonl"],
            )
            return SkillBundle(
                backend_id="skill-local",
                graph_version_ref="graph-v1",
                required_tools=["lookup", "write_jsonl"],
                skills=[locate, write],
            )

    registry = ToolRegistry()
    executed: list[str] = []
    registry.register(
        ToolSpec(name="lookup", description="lookup", parameters_schema={"type": "object"}),
        lambda arguments: executed.append("lookup") or "lookup ok",
    )
    registry.register(
        ToolSpec(name="write_jsonl", description="write jsonl", parameters_schema={"type": "object"}),
        lambda arguments: executed.append("write_jsonl")
        or ToolResult(
            call_id="write_jsonl",
            status="ok",
            content="wrote records",
            artifact_refs=[ArtifactRef(uri=str(tmp_path / "records.jsonl"), type="dataset")],
            metadata={"record_count": 1},
        ),
    )
    llm = ScriptedLLM(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="lookup", name="lookup", arguments={"query": "source"}),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="sources localized")),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(
                        call_id="write",
                        name="write_jsonl",
                        arguments={"records": [{"id": "record-1"}]},
                    ),
                )
            ),
        ]
    )
    task_config = TaskConfig(
        task_id="task-1",
        goal="Write records.",
        runtime_policy=RuntimePolicy(
            enable_workflow_planning=True,
            max_tool_steps_per_node=1,
            metadata={
                "completion_guard_max_violations": 1,
                "completion_guards_by_role": {
                    "Writer": {"required_tool_calls_before_final": ["write_jsonl"]}
                },
            },
        ),
        roles={
            "Writer": RoleSpec(
                name="Writer",
                system_prompt="Write.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["lookup", "write_jsonl"],
            )
        },
    )
    runtime = _runtime_with_static_role_dispatch(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": MemoryRuntime()},
        skill_runtimes={"skill-local": TwoStepSkillRuntime()},
    )

    result = runtime.run(_request())
    saved = runtime.trajectory_registry.get_subagent_run(result["run_ref"])
    records = runtime.trajectory_registry.list_tool_call_records()

    assert result["status"] == "completed"
    assert executed == ["lookup", "write_jsonl"]
    assert [record.record.result.status for record in records] == ["ok", "ok"]
    assert saved is not None
    assert [node["status"] for node in saved.metadata["node_execution_records"]] == ["completed", "completed"]


def test_workflow_preliminary_node_tool_budget_does_not_skip_output_node(tmp_path: Path):
    class TwoStepSkillRuntime(SkillRuntime):
        def get(self, request: RetrievalRequest) -> SkillBundle:
            locate = SkillRef(
                skill_id="skill.locate_sources.v1",
                name="Locate Sources",
                content="Description:\nFind relevant source material.",
                required_tools=["lookup"],
            )
            write = SkillRef(
                skill_id="skill.write_records.v1",
                name="Write Records",
                content="Description:\nWrite extracted records.",
                required_tools=["write_jsonl"],
            )
            return SkillBundle(
                backend_id="skill-local",
                graph_version_ref="graph-v1",
                required_tools=["lookup", "write_jsonl"],
                skills=[locate, write],
            )

    registry = ToolRegistry()
    executed: list[str] = []
    registry.register(
        ToolSpec(name="lookup", description="lookup", parameters_schema={"type": "object"}),
        lambda arguments: executed.append(f"lookup:{arguments['query']}") or "lookup ok",
    )
    registry.register(
        ToolSpec(name="write_jsonl", description="write jsonl", parameters_schema={"type": "object"}),
        lambda arguments: executed.append("write_jsonl")
        or ToolResult(
            call_id="write_jsonl",
            status="ok",
            content="wrote records",
            artifact_refs=[ArtifactRef(uri=str(tmp_path / "records.jsonl"), type="dataset")],
            metadata={"record_count": 1},
        ),
    )
    llm = ScriptedLLM(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="lookup-1", name="lookup", arguments={"query": "source-1"}),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="lookup-2", name="lookup", arguments={"query": "source-2"}),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(
                        call_id="write",
                        name="write_jsonl",
                        arguments={"records": [{"id": "record-1"}]},
                    ),
                )
            ),
        ]
    )
    task_config = TaskConfig(
        task_id="task-1",
        goal="Write records.",
        runtime_policy=RuntimePolicy(
            enable_workflow_planning=True,
            max_tool_steps_per_node=1,
            metadata={
                "completion_guards_by_role": {
                    "Writer": {"required_tool_calls_before_final": ["write_jsonl"]}
                },
            },
        ),
        roles={
            "Writer": RoleSpec(
                name="Writer",
                system_prompt="Write.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["lookup", "write_jsonl"],
            )
        },
    )
    runtime = _runtime_with_static_role_dispatch(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": MemoryRuntime()},
        skill_runtimes={"skill-local": TwoStepSkillRuntime()},
    )

    result = runtime.run(_request())
    saved = runtime.trajectory_registry.get_subagent_run(result["run_ref"])

    assert result["status"] == "completed"
    assert executed == ["lookup:source-1", "write_jsonl"]
    assert saved is not None
    node_records = saved.metadata["node_execution_records"]
    assert [node["status"] for node in node_records] == ["completed", "completed"]
    assert "continuing with gathered context" in node_records[0]["output_summary"]


def test_workflow_preliminary_node_repeated_suppression_does_not_spin_or_skip_output_node(tmp_path: Path):
    class TwoStepSkillRuntime(SkillRuntime):
        def get(self, request: RetrievalRequest) -> SkillBundle:
            locate = SkillRef(
                skill_id="skill.locate_sources.v1",
                name="Locate Sources",
                content="Description:\nFind relevant source material.",
                required_tools=["lookup"],
            )
            write = SkillRef(
                skill_id="skill.write_records.v1",
                name="Write Records",
                content="Description:\nWrite extracted records.",
                required_tools=["write_jsonl"],
            )
            return SkillBundle(
                backend_id="skill-local",
                graph_version_ref="graph-v1",
                required_tools=["lookup", "write_jsonl"],
                skills=[locate, write],
            )

    registry = ToolRegistry()
    executed: list[str] = []
    registry.register(
        ToolSpec(name="lookup", description="lookup", parameters_schema={"type": "object"}),
        lambda arguments: executed.append(f"lookup:{arguments['query']}") or "lookup ok",
    )
    registry.register(
        ToolSpec(name="write_jsonl", description="write jsonl", parameters_schema={"type": "object"}),
        lambda arguments: executed.append("write_jsonl")
        or ToolResult(
            call_id="write_jsonl",
            status="ok",
            content="wrote records",
            artifact_refs=[ArtifactRef(uri=str(tmp_path / "records.jsonl"), type="dataset")],
            metadata={"record_count": 1},
        ),
    )
    llm = ScriptedLLM(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="lookup-1", name="lookup", arguments={"query": "same"}),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="lookup-2", name="lookup", arguments={"query": "same"}),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="lookup-other", name="lookup", arguments={"query": "other"}),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="lookup-3", name="lookup", arguments={"query": "same"}),
                )
            ),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(
                        call_id="write",
                        name="write_jsonl",
                        arguments={"records": [{"id": "record-1"}]},
                    ),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="late node final")),
        ]
    )
    task_config = TaskConfig(
        task_id="task-1",
        goal="Write records.",
        runtime_policy=RuntimePolicy(
            enable_workflow_planning=True,
            max_tool_steps_per_node=8,
            metadata={
                "max_repeated_tool_calls_per_run": 1,
                "repeated_tool_call_suppression_max_violations": 2,
                "completion_guards_by_role": {
                    "Writer": {"required_tool_calls_before_final": ["write_jsonl"]}
                },
            },
        ),
        roles={
            "Writer": RoleSpec(
                name="Writer",
                system_prompt="Write.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["lookup", "write_jsonl"],
            )
        },
    )
    runtime = _runtime_with_static_role_dispatch(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": MemoryRuntime()},
        skill_runtimes={"skill-local": TwoStepSkillRuntime()},
    )

    result = runtime.run(_request())
    saved = runtime.trajectory_registry.get_subagent_run(result["run_ref"])
    records = runtime.trajectory_registry.list_tool_call_records()

    assert result["status"] == "completed"
    assert len(llm.calls) == 5
    assert executed == ["lookup:same", "lookup:other", "write_jsonl"]
    assert [record.record.result.metadata.get("error_type") for record in records] == [
        None,
        "repeated_tool_call_suppressed",
        None,
        "repeated_tool_call_suppressed",
        None,
    ]
    assert saved is not None
    assert "suppressed repeated tool calls" in saved.metadata["node_execution_records"][0]["output_summary"]
    writer_prompt = llm.calls[4][0][1].content
    assert "lookup ok" in writer_prompt


def test_flat_task_runtime_counts_each_parallel_tool_call_against_tool_budget(tmp_path: Path):
    tool_calls: list[str] = []
    registry = ToolRegistry()
    registry.register(
        ToolSpec(name="lookup", description="lookup", parameters_schema={"type": "object"}),
        lambda arguments: tool_calls.append(arguments["query"]) or "ok",
    )
    llm = ScriptedLLM(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_calls=[
                        ToolCall(call_id="call-1", name="lookup", arguments={"query": "one"}),
                        ToolCall(call_id="call-2", name="lookup", arguments={"query": "two"}),
                        ToolCall(call_id="call-3", name="lookup", arguments={"query": "three"}),
                    ],
                )
            )
        ]
    )
    task_config = TaskConfig(
        task_id="task-1",
        goal="Extract records.",
        runtime_policy=RuntimePolicy(max_tool_steps=2),
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="Solve.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["lookup"],
            )
        },
    )
    runtime = _runtime_with_static_role_dispatch(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": MemoryRuntime()},
        skill_runtimes={"skill-local": SkillRuntime()},
    )

    with pytest.raises(RuntimeError, match="max_tool_steps"):
        runtime.run(_request())

    assert tool_calls == ["one", "two"]


def test_workflow_task_runtime_executes_multiple_tool_calls_from_one_llm_step(tmp_path: Path):
    class OneSkillRuntime(SkillRuntime):
        def get(self, request: RetrievalRequest) -> SkillBundle:
            skill = SkillRef(
                skill_id="skill.lookup_pair.v1",
                name="Lookup Pair",
                content="Description:\nLookup two related inputs.",
                required_tools=["lookup"],
            )
            return SkillBundle(
                backend_id="skill-local",
                graph_version_ref="graph-v1",
                required_tools=["lookup"],
                skills=[skill],
                metadata={"retrieval_trace": {"returned_skill_ids": [skill.skill_id]}},
            )

    llm = ScriptedLLM(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_calls=[
                        ToolCall(call_id="call-1", name="lookup", arguments={"query": "schema"}),
                        ToolCall(call_id="call-2", name="lookup", arguments={"query": "fields"}),
                    ],
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="used both tools")),
        ]
    )
    runtime = _runtime(tmp_path, enable_workflow_planning=True, llm=llm, skill=OneSkillRuntime())

    result = runtime.run(_request())
    saved = runtime.trajectory_registry.get_subagent_run(result["run_ref"])
    llm_calls = runtime.trajectory_registry.list_llm_calls()

    assert saved is not None
    assert [record.tool_call.call_id for record in saved.tool_calls] == ["call-1", "call-2"]
    node_record = saved.metadata["node_execution_records"][0]
    assert [record["tool_call"]["call_id"] for record in node_record["tool_calls"]] == ["call-1", "call-2"]
    tool_messages = [message for message in llm_calls[1].input_messages if message.role == "tool"]
    assert [message.tool_call_id for message in tool_messages] == ["call-1", "call-2"]


def test_workflow_task_runtime_counts_each_parallel_tool_call_against_node_budget(tmp_path: Path):
    class OneSkillRuntime(SkillRuntime):
        def get(self, request: RetrievalRequest) -> SkillBundle:
            skill = SkillRef(
                skill_id="skill.lookup_many.v1",
                name="Lookup Many",
                content="Description:\nLookup inputs.",
                required_tools=["lookup"],
            )
            return SkillBundle(
                backend_id="skill-local",
                graph_version_ref="graph-v1",
                required_tools=["lookup"],
                skills=[skill],
                metadata={"retrieval_trace": {"returned_skill_ids": [skill.skill_id]}},
            )

    tool_calls: list[str] = []
    registry = ToolRegistry()
    registry.register(
        ToolSpec(name="lookup", description="lookup", parameters_schema={"type": "object"}),
        lambda arguments: tool_calls.append(arguments["query"]) or "ok",
    )
    llm = ScriptedLLM(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_calls=[
                        ToolCall(call_id="call-1", name="lookup", arguments={"query": "one"}),
                        ToolCall(call_id="call-2", name="lookup", arguments={"query": "two"}),
                        ToolCall(call_id="call-3", name="lookup", arguments={"query": "three"}),
                    ],
                )
            )
        ]
    )
    task_config = TaskConfig(
        task_id="task-1",
        goal="Extract records.",
        runtime_policy=RuntimePolicy(enable_workflow_planning=True, max_tool_steps_per_node=2),
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="Solve.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["lookup"],
            )
        },
    )
    runtime = _runtime_with_static_role_dispatch(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": MemoryRuntime()},
        skill_runtimes={"skill-local": OneSkillRuntime()},
    )

    with pytest.raises(RuntimeError, match="max tool calls"):
        runtime.run(_request())

    assert tool_calls == ["one", "two"]
    saved = runtime.trajectory_registry.list_subagent_runs()[0]
    node_record = saved.metadata["node_execution_records"][0]
    assert node_record["status"] == "failed"
    assert node_record["output_summary"] == "workflow node exceeded max tool calls before final answer"


def test_workflow_runtime_skips_remaining_nodes_after_role_completion_guards_pass(tmp_path: Path):
    class ReportThenLookupSkillRuntime(SkillRuntime):
        def get(self, request: RetrievalRequest) -> SkillBundle:
            report = SkillRef(
                skill_id="skill.report.v1",
                name="A Report",
                content="Description:\nWrite a report.",
                required_tools=["write_report"],
            )
            lookup = SkillRef(
                skill_id="skill.lookup.v1",
                name="B Lookup",
                content="Description:\nDo follow-up lookup.",
                required_tools=["lookup"],
            )
            return SkillBundle(
                backend_id="skill-local",
                graph_version_ref="graph-v1",
                required_tools=["write_report", "lookup"],
                skills=[report, lookup],
                metadata={"retrieval_trace": {"returned_skill_ids": [report.skill_id, lookup.skill_id]}},
            )

    registry = ToolRegistry()
    report_path = tmp_path / "report.md"
    registry.register(
        ToolSpec(name="write_report", description="write report", parameters_schema={"type": "object"}),
        lambda arguments: ToolResult(
            call_id="write_report",
            status="ok",
            content="wrote report",
            artifact_refs=[ArtifactRef(uri=str(report_path), type="log", metadata={"filename": "report.md"})],
        ),
    )
    registry.register(
        ToolSpec(name="lookup", description="lookup", parameters_schema={"type": "object"}),
        lambda arguments: "lookup should not run",
    )
    llm = ScriptedLLM(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="report", name="write_report", arguments={"content": "done"}),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="report complete")),
        ]
    )
    task_config = TaskConfig(
        task_id="task-1",
        goal="Write a report, then stop.",
        runtime_policy=RuntimePolicy(
            enable_workflow_planning=True,
            metadata={"completion_guards_by_role": {"Reporter": {"required_tool_calls_before_final": ["write_report"]}}},
        ),
        roles={
            "Reporter": RoleSpec(
                name="Reporter",
                system_prompt="Report.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["write_report", "lookup"],
            )
        },
    )
    runtime = _runtime_with_static_role_dispatch(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": MemoryRuntime()},
        skill_runtimes={"skill-local": ReportThenLookupSkillRuntime()},
    )

    result = runtime.run(_request())
    saved = runtime.trajectory_registry.get_subagent_run(result["run_ref"])

    assert saved is not None
    assert [record["status"] for record in saved.metadata["node_execution_records"]] == ["completed", "skipped"]
    assert (
        saved.metadata["node_execution_records"][0]["output_summary"]
        == "role completion guards satisfied after successful required tool calls"
    )
    assert saved.metadata["node_execution_records"][1]["output_summary"] == "skipped after role completion guards were satisfied"
    assert [record.tool_call.name for record in saved.tool_calls] == ["write_report"]
    assert len(llm.calls) == 1


@pytest.mark.skip(reason="removed default MetaAgent dispatch loop; dynamic role-pool planner is required for default execution")
def test_meta_dispatch_can_request_direct_execution_to_bypass_internal_workflow(tmp_path: Path):
    class MetaLLM:
        def __init__(self):
            self.calls = 0

        def generate(self, messages: list[Message], tool_specs: list[dict], generation_config: LLMGenerationConfig):
            self.calls += 1
            if self.calls == 1:
                return LLMRuntimeResponse(
                    action=SubAgentAction(
                        action="final_answer",
                        content='{"route":"ExecAgent","instruction":"Run directly without using the DAG.","metadata":{"execution_mode":"direct"}}',
                    )
                )
            return LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content='{"route":"END","instruction":"Done.","metadata":{"final_answer":"Done."}}',
                )
            )

    role_llm = ScriptedLLM([LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="direct done"))])
    task_config = TaskConfig(
        task_id="task-1",
        goal="Extract records.",
        max_dispatch_steps=3,
        meta_agent=MetaAgentSpec(
            name="meta",
            system_prompt="Route.",
            llm_backend=BackendBinding(backend_id="meta-llm"),
        ),
        runtime_policy=RuntimePolicy(enable_workflow_planning=True),
        roles={
            "ExecAgent": RoleSpec(
                name="ExecAgent",
                system_prompt="Execute.",
                llm_backend=BackendBinding(backend_id="role-llm"),
                allowed_tools=["lookup"],
            )
        },
    )
    registry = ToolRegistry()
    registry.register(ToolSpec(name="lookup", description="lookup", parameters_schema={"type": "object"}), lambda _: "ok")
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"meta-llm": MetaLLM(), "role-llm": role_llm},
        memory_runtimes={"memory-local": MemoryRuntime()},
        skill_runtimes={"skill-local": SkillRuntime()},
    )

    result = runtime.run(_request())
    saved = runtime.trajectory_registry.get_subagent_run(result["run_refs"][0])

    assert saved is not None
    assert "workflow_plan" not in saved.metadata
    assert len(role_llm.calls) == 1


@pytest.mark.skip(reason="removed default MetaAgent dispatch loop; dynamic role-pool planner is required for default execution")
def test_meta_dispatch_direct_recovery_phrase_bypasses_internal_workflow(tmp_path: Path):
    class MetaLLM:
        def __init__(self):
            self.calls = 0

        def generate(self, messages: list[Message], tool_specs: list[dict], generation_config: LLMGenerationConfig):
            self.calls += 1
            if self.calls == 1:
                return LLMRuntimeResponse(
                    action=SubAgentAction(
                        action="final_answer",
                        content=(
                            '{"route":"CriticAgent","instruction":"Prior internal DAG node exceeded tool budget; '
                            'retry with direct mode.","metadata":{}}'
                        ),
                    )
                )
            return LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content='{"route":"END","instruction":"Done.","metadata":{"final_answer":"Done."}}',
                )
            )

    role_llm = ScriptedLLM([LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="direct review done"))])
    task_config = TaskConfig(
        task_id="task-1",
        goal="Review failure.",
        max_dispatch_steps=3,
        meta_agent=MetaAgentSpec(
            name="meta",
            system_prompt="Route.",
            llm_backend=BackendBinding(backend_id="meta-llm"),
        ),
        runtime_policy=RuntimePolicy(enable_workflow_planning=True),
        roles={
            "CriticAgent": RoleSpec(
                name="CriticAgent",
                system_prompt="Review.",
                llm_backend=BackendBinding(backend_id="role-llm"),
                allowed_tools=["lookup"],
            )
        },
    )
    registry = ToolRegistry()
    registry.register(ToolSpec(name="lookup", description="lookup", parameters_schema={"type": "object"}), lambda _: "ok")
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"meta-llm": MetaLLM(), "role-llm": role_llm},
        memory_runtimes={"memory-local": MemoryRuntime()},
        skill_runtimes={"skill-local": SkillRuntime()},
    )

    result = runtime.run(_request())
    saved = runtime.trajectory_registry.get_subagent_run(result["run_refs"][0])

    assert saved is not None
    assert "workflow_plan" not in saved.metadata
    assert len(role_llm.calls) == 1


def test_workflow_runtime_stops_at_total_subagent_llm_budget_and_records_run(tmp_path: Path):
    skill = SkillRuntime()
    llm = ScriptedLLM(
        [
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="schema interpreted")),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="fields mapped")),
        ]
    )
    runtime = _runtime(tmp_path, enable_workflow_planning=True, llm=llm, skill=skill)
    runtime.task_config.runtime_policy.metadata["subagent_budget"] = {"max_subagent_llm_calls": 1}

    result = runtime.run(_request())
    saved = runtime.trajectory_registry.get_subagent_run(result["run_ref"])

    assert result["status"] in {"failed", "interrupted", "budget_exceeded"}
    assert "budget_exceeded" in result["failure_reason"]
    assert saved is not None
    assert saved.metadata["status"] == result["status"]
    assert saved.metadata["budget"]["llm_calls"] == 1
    assert saved.metadata["budget"]["max_subagent_llm_calls"] == 1
    assert saved.metadata["plan_execution_trace"]["status"] == "partial"
    assert saved.metadata["node_execution_records"][-1]["status"] == "skipped"


def test_workflow_runtime_stops_at_total_subagent_tool_budget_and_records_run(tmp_path: Path):
    skill = SkillRuntime()
    llm = ScriptedLLM(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="call-1", name="lookup", arguments={"query": "schema"}),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="schema interpreted")),
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="call-2", name="lookup", arguments={"query": "fields"}),
                )
            ),
        ]
    )
    runtime = _runtime(tmp_path, enable_workflow_planning=True, llm=llm, skill=skill)
    runtime.task_config.runtime_policy.metadata["subagent_budget"] = {"max_subagent_tool_calls": 1}

    result = runtime.run(_request())
    saved = runtime.trajectory_registry.get_subagent_run(result["run_ref"])

    assert result["status"] in {"failed", "interrupted", "budget_exceeded"}
    assert "budget_exceeded" in result["failure_reason"]
    assert saved is not None
    assert len(saved.tool_calls) == 1
    assert saved.metadata["budget"]["tool_calls"] == 1
    assert saved.metadata["budget"]["max_subagent_tool_calls"] == 1


def test_workflow_runtime_stops_at_total_subagent_runtime_budget_and_records_run(tmp_path: Path):
    class SlowLLM(ScriptedLLM):
        def generate(self, messages: list[Message], tool_specs: list[dict], generation_config: LLMGenerationConfig):
            time.sleep(0.02)
            return super().generate(messages, tool_specs, generation_config)

    skill = SkillRuntime()
    llm = SlowLLM(
        [
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="schema interpreted")),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="fields mapped")),
        ]
    )
    runtime = _runtime(tmp_path, enable_workflow_planning=True, llm=llm, skill=skill)
    runtime.task_config.runtime_policy.metadata["subagent_budget"] = {"max_subagent_runtime_seconds": 0.001}

    result = runtime.run(_request())
    saved = runtime.trajectory_registry.get_subagent_run(result["run_ref"])

    assert result["status"] in {"failed", "interrupted", "budget_exceeded"}
    assert "budget_exceeded" in result["failure_reason"]
    assert saved is not None
    assert saved.metadata["budget"]["max_subagent_runtime_seconds"] == 0.001
    assert saved.metadata["plan_execution_trace"]["status"] == "partial"


def test_workflow_memory_update_stores_compact_run_summary_not_full_node_prompts(tmp_path: Path):
    skill = SkillRuntime()
    huge_payload = "x" * 200_000
    llm = ScriptedLLM(
        [
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content=huge_payload)),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="fields mapped")),
        ]
    )
    memory = MemoryRuntime()
    registry = ToolRegistry()
    registry.register(ToolSpec(name="lookup", description="lookup", parameters_schema={"type": "object"}), lambda _: "ok")
    task_config = TaskConfig(
        task_id="task-1",
        goal="Extract records.",
        runtime_policy=RuntimePolicy(enable_workflow_planning=True),
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="Solve.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["lookup"],
            )
        },
    )
    runtime = _runtime_with_static_role_dispatch(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        tool_artifact_root_factory=lambda request, run_ref: tmp_path / "artifacts" / run_ref,
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": skill},
    )

    runtime.run(_request())

    assert memory.add_calls
    task_memory_messages = [messages for _task_id, role, messages in memory.add_calls if role == "task"][0]
    serialized = "\n".join(message.content for message in task_memory_messages)
    assert len(serialized) < 20_000
    assert huge_payload not in serialized
    assert "workflow_node_summaries" in serialized
    assert "truncated" in serialized


def test_flat_memory_update_stores_compact_run_summary_not_full_prompt_history(tmp_path: Path):
    registry = ToolRegistry()
    huge_payload = "x" * 200_000

    def lookup(arguments: dict) -> ToolResult:
        return ToolResult(call_id="handler-call", status="ok", content="lookup ok", metadata={"payload": huge_payload})

    registry.register(ToolSpec(name="lookup", description="lookup", parameters_schema={"type": "object"}), lookup)
    llm = ScriptedLLM(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="call-1", name="lookup", arguments={"query": "large"}),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content=huge_payload)),
        ]
    )
    memory = MemoryRuntime()
    task_config = TaskConfig(
        task_id="task-1",
        goal="Extract records.",
        runtime_policy=RuntimePolicy(enable_workflow_planning=False),
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="Solve.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["lookup"],
            )
        },
    )
    runtime = _runtime_with_static_role_dispatch(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": memory},
        skill_runtimes={"skill-local": SkillRuntime()},
    )

    runtime.run(_request())

    task_memory_messages = [messages for _task_id, role, messages in memory.add_calls if role == "task"][0]
    serialized = "\n".join(message.content for message in task_memory_messages)
    assert len(serialized) < 20_000
    assert huge_payload not in serialized
    assert "subagent_flat_summary" in serialized
    assert "tool_call_count" in serialized
    assert "truncated" in serialized


def test_workflow_node_prompt_compacts_large_retrieved_memory_items(tmp_path: Path):
    class LargeMemoryRuntime(MemoryRuntime):
        def search(self, request: RetrievalRequest) -> MemoryBundle:
            from evolab.contracts.retrieval import MemoryItem

            return MemoryBundle(
                backend_id="memory-local",
                items=[
                    MemoryItem(
                        memory_id=f"{request.role}-large",
                        content="large-memory-" + ("x" * 200_000),
                    )
                ],
            )

    skill = SkillRuntime()
    llm = ScriptedLLM(
        [
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="done")),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="done")),
        ]
    )
    task_config = TaskConfig(
        task_id="task-1",
        goal="Extract records.",
        runtime_policy=RuntimePolicy(enable_workflow_planning=True),
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="Solve.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["lookup"],
            )
        },
    )
    registry = ToolRegistry()
    registry.register(ToolSpec(name="lookup", description="lookup", parameters_schema={"type": "object"}), lambda _: "ok")
    runtime = _runtime_with_static_role_dispatch(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": LargeMemoryRuntime()},
        skill_runtimes={"skill-local": skill},
    )

    runtime.run(_request())

    first_prompt = llm.calls[0][0][1].content
    assert len(first_prompt) < 50_000
    assert "large-memory-" in first_prompt
    assert "truncated" in first_prompt


def test_workflow_node_prompt_includes_lab_resource_paths(tmp_path: Path):
    skill = SkillRuntime()
    llm = ScriptedLLM(
        [
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="done")),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="done")),
        ]
    )
    lab_root = tmp_path / "lab"
    task_config = TaskConfig(
        task_id="task-1",
        goal="Use domain_packages/biology_component_extraction_v1/biology_component_schema.json.",
        runtime_policy=RuntimePolicy(enable_workflow_planning=True),
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="Solve.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["lookup"],
            )
        },
    )
    registry = ToolRegistry()
    registry.register(ToolSpec(name="lookup", description="lookup", parameters_schema={"type": "object"}), lambda _: "ok")
    runtime = _runtime_with_static_role_dispatch(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(lab_root / "registries" / "trajectory"),
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": MemoryRuntime()},
        skill_runtimes={"skill-local": skill},
    )

    runtime.run(_request())

    first_prompt = llm.calls[0][0][1].content
    assert f'"lab_root": "{lab_root}"' in first_prompt
    assert f"{lab_root}/domain_packages/biology_component_extraction_v1" in first_prompt
    assert f"{lab_root}/domain_packages/biology_component_extraction_v1/biology_component_schema.json" in first_prompt
    assert '"tool_step_budget"' in first_prompt
    assert '"max_tool_calls": 20' in first_prompt
    assert "call the appropriate output tool before final_answer" in first_prompt


def test_plan_aware_runtime_prepares_tools_per_workflow_node(tmp_path: Path):
    class MixedToolSkillRuntime(SkillRuntime):
        def get(self, request: RetrievalRequest) -> SkillBundle:
            first = SkillRef(
                skill_id="skill.document_reading.v1",
                name="Document Reading",
                content="Description:\nRead documents.",
                required_tools=["read_text"],
            )
            second = SkillRef(
                skill_id="skill.final_writing.v1",
                name="Final Writing",
                content="Description:\nWrite final artifacts.",
                required_tools=["write_jsonl"],
            )
            return SkillBundle(
                backend_id="skill-local",
                graph_version_ref="graph-v1",
                required_tools=["read_text", "write_jsonl"],
                skills=[first, second],
                metadata={"graph_context_summary": {"graph_version": "graph-v1"}},
            )

    registry = ToolRegistry()
    registry.register(
        ToolSpec(name="read_text", description="read", parameters_schema={"type": "object"}),
        lambda arguments: "text",
    )
    registry.register(
        ToolSpec(name="write_jsonl", description="write", parameters_schema={"type": "object"}),
        lambda arguments: "written",
    )
    llm = ScriptedLLM(
        [
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="read done")),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="write done")),
        ]
    )
    task_config = TaskConfig(
        task_id="task-1",
        goal="Extract records.",
        runtime_policy=RuntimePolicy(
            enable_workflow_planning=True,
            metadata={"subagent_policy": {"tool_preparation_scope": "per_internal_dag_node_or_retrieved_skill"}},
        ),
        roles={
            "GenericAgent": RoleSpec(
                name="GenericAgent",
                system_prompt="Solve.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["read_text", "write_jsonl"],
            )
        },
    )
    runtime = _runtime_with_static_role_dispatch(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": MemoryRuntime()},
        skill_runtimes={"skill-local": MixedToolSkillRuntime()},
    )

    result = runtime.run(_request())
    saved = runtime.trajectory_registry.get_subagent_run(result["run_ref"])

    assert [spec["name"] for spec in llm.calls[0][1]] == ["read_text"]
    assert [spec["name"] for spec in llm.calls[1][1]] == ["write_jsonl"]
    assert saved is not None
    node_records = saved.metadata["node_execution_records"]
    assert node_records[0]["metadata"]["prepared_tool_names"] == ["read_text"]
    assert node_records[1]["metadata"]["prepared_tool_names"] == ["write_jsonl"]


def test_output_node_prompt_requires_explicit_candidate_handoff(tmp_path: Path):
    class OutputSkillRuntime(SkillRuntime):
        def get(self, request: RetrievalRequest) -> SkillBundle:
            writer = SkillRef(
                skill_id="skill.final_record_writing.v1",
                name="Final Record Writing",
                content="Description:\nSerialize final candidate records.",
                required_tools=["write_jsonl"],
                metadata={
                    "required_inputs": ["candidate_rows.json"],
                    "expected_outputs": ["final_records.jsonl"],
                },
            )
            return SkillBundle(
                backend_id="skill-local",
                graph_version_ref="graph-v1",
                required_tools=["write_jsonl"],
                skills=[writer],
                metadata={"graph_context_summary": {"graph_version": "graph-v1"}},
            )

    registry = ToolRegistry()
    registry.register(
        ToolSpec(name="write_jsonl", description="write", parameters_schema={"type": "object"}),
        lambda arguments: "written",
    )
    llm = ScriptedLLM([LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="needs candidates"))])
    task_config = TaskConfig(
        task_id="task-1",
        goal="Extract records.",
        runtime_policy=RuntimePolicy(enable_workflow_planning=True),
        roles={
            "WriteAgent": RoleSpec(
                name="WriteAgent",
                system_prompt="Write.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["write_jsonl"],
            )
        },
    )
    runtime = _runtime_with_static_role_dispatch(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": MemoryRuntime()},
        skill_runtimes={"skill-local": OutputSkillRuntime()},
    )

    runtime.run(_request())

    first_prompt = llm.calls[0][0][1].content
    assert '"generic_scientific_extraction_artifact_contracts"' in first_prompt
    assert "document_inventory.json" in first_prompt
    assert "candidate_tables.json" in first_prompt
    assert "final_records.jsonl" in first_prompt
    assert '"output_node_contract"' in first_prompt
    assert '"requires_concrete_candidates": true' in first_prompt
    assert '"candidate_artifacts_present": false' in first_prompt
    assert "request upstream Survey/Discovery" in first_prompt


def test_survey_exec_write_handoff_preserves_candidate_artifact_refs(tmp_path: Path):
    class HandoffSkillRuntime(SkillRuntime):
        def get(self, request: RetrievalRequest) -> SkillBundle:
            survey = SkillRef(
                skill_id="skill.document_intake.v1",
                name="Scientific Document Survey",
                content="Description:\nDiscover candidate source rows.",
                required_tools=["lookup"],
                metadata={"expected_outputs": ["candidate_rows.json"]},
            )
            writer = SkillRef(
                skill_id="skill.structured_record_construction.v1",
                name="Final Record Writing",
                content="Description:\nSerialize final candidate records.",
                required_tools=["write_jsonl"],
                metadata={
                    "required_inputs": ["candidate_rows.json"],
                    "expected_outputs": ["final_records.jsonl"],
                },
            )
            return SkillBundle(
                backend_id="skill-local",
                graph_version_ref="graph-v1",
                required_tools=["lookup", "write_jsonl"],
                skills=[survey, writer],
                metadata={
                    "graph_context_summary": {"graph_version": "graph-v1"},
                    "retrieval_trace": {
                        "relation_expansion_steps": [
                            {
                                "source_skill_id": survey.skill_id,
                                "target_skill_id": writer.skill_id,
                                "relation": "produces_input_for",
                                "reason": "candidate rows feed final writing",
                            }
                        ]
                    },
                },
            )

    candidate_path = tmp_path / "candidate_rows.json"

    def lookup(arguments: dict) -> ToolResult:
        candidate_path.write_text("[]", encoding="utf-8")
        return ToolResult(
            call_id="lookup",
            status="ok",
            content="candidate rows ready",
            artifact_refs=[
                ArtifactRef(
                    uri=str(candidate_path),
                    type="dataset",
                    metadata={"filename": "candidate_rows.json", "status": "candidate"},
                )
            ],
        )

    registry = ToolRegistry()
    registry.register(ToolSpec(name="lookup", description="lookup", parameters_schema={"type": "object"}), lookup)
    registry.register(
        ToolSpec(name="write_jsonl", description="write", parameters_schema={"type": "object"}),
        lambda arguments: "written",
    )
    llm = ScriptedLLM(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="candidate-rows", name="lookup", arguments={}),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="candidate rows discovered")),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="write saw candidates")),
        ]
    )
    task_config = TaskConfig(
        task_id="task-1",
        goal="Extract records.",
        runtime_policy=RuntimePolicy(enable_workflow_planning=True),
        roles={
            "WriteAgent": RoleSpec(
                name="WriteAgent",
                system_prompt="Write.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["lookup", "write_jsonl"],
            )
        },
    )
    runtime = _runtime_with_static_role_dispatch(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        tool_artifact_root_factory=lambda request, run_ref: tmp_path / "artifacts" / run_ref,
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": MemoryRuntime()},
        skill_runtimes={"skill-local": HandoffSkillRuntime()},
    )

    runtime.run(_request())

    write_prompt = llm.calls[2][0][1].content
    assert '"candidate_artifacts_present": true' in write_prompt
    assert "candidate_rows.json" in write_prompt
    assert str(tmp_path / "artifacts") in write_prompt


def test_exec_agent_workflow_nodes_get_role_support_tools_without_globalizing_other_roles(tmp_path: Path):
    class ExecSkillRuntime(SkillRuntime):
        def get(self, request: RetrievalRequest) -> SkillBundle:
            skill = SkillRef(
                skill_id="skill.task_relevant_section_localization.v1",
                name="Task Relevant Section Localization",
                content="Read source documents and locate relevant sections.",
                required_tools=["read_text", "search_text"],
            )
            return SkillBundle(
                backend_id="skill-local",
                graph_version_ref="graph-v1",
                required_tools=["read_text", "search_text"],
                skills=[skill],
            )

    registry = ToolRegistry()
    for name in ["read_text", "search_text", "list_files", "inspect_table", "read_table_slice", "write_jsonl", "write_report"]:
        registry.register(ToolSpec(name=name, description=name, parameters_schema={"type": "object"}), lambda _: "ok")
    llm = ScriptedLLM([LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="done"))])
    task_config = TaskConfig(
        task_id="task-1",
        goal="Extract records.",
        runtime_policy=RuntimePolicy(
            enable_workflow_planning=True,
            metadata={"subagent_policy": {"tool_preparation_scope": "per_internal_dag_node_or_retrieved_skill"}},
        ),
        roles={
            "ExecAgent": RoleSpec(
                name="ExecAgent",
                system_prompt="Execute.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=[
                    "read_text",
                    "search_text",
                    "list_files",
                    "inspect_table",
                    "read_table_slice",
                    "write_jsonl",
                    "write_report",
                ],
            )
        },
    )
    runtime = _runtime_with_static_role_dispatch(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": MemoryRuntime()},
        skill_runtimes={"skill-local": ExecSkillRuntime()},
    )

    result = runtime.run(_request())
    saved = runtime.trajectory_registry.get_subagent_run(result["run_ref"])

    assert saved is not None
    prepared = saved.metadata["node_execution_records"][0]["metadata"]["prepared_tool_names"]
    assert "read_text" in prepared
    assert "search_text" in prepared
    assert "list_files" in prepared
    assert "inspect_table" in prepared
    assert "read_table_slice" in prepared
    assert "write_report" in prepared
    assert "write_jsonl" not in prepared


def test_workflow_planning_scopes_survey_agent_internal_dag_to_assignment(tmp_path: Path):
    class BroadSkillRuntime(SkillRuntime):
        def get(self, request: RetrievalRequest) -> SkillBundle:
            intake = SkillRef(
                skill_id="skill.scientific_document_intake.v1",
                name="Scientific Document Intake",
                content="Survey document packages.",
                required_tools=[],
            )
            validation = SkillRef(
                skill_id="skill.extraction_result_validation.v1",
                name="Extraction Result Validation",
                content="Validate extracted records.",
                required_tools=[],
            )
            evaluation = SkillRef(
                skill_id="skill.ground_truth_based_evaluation.v1",
                name="Ground Truth Based Evaluation",
                content="Evaluate predictions.",
                required_tools=[],
            )
            return SkillBundle(
                backend_id="skill-local",
                graph_version_ref="graph-v1",
                skills=[intake, validation, evaluation],
                metadata={
                    "retrieval_trace": {
                        "returned_skill_ids": [intake.skill_id, validation.skill_id, evaluation.skill_id],
                    }
                },
            )

    llm = ScriptedLLM([LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="survey done"))])
    task_config = TaskConfig(
        task_id="task-1",
        goal="Survey inputs.",
        runtime_policy=RuntimePolicy(
            enable_workflow_planning=True,
            metadata={"subagent_policy": {"skill_retrieval_scope": "per_internal_dag_node"}},
        ),
        roles={
            "SurveyAgent": RoleSpec(
                name="SurveyAgent",
                system_prompt="Survey.",
                llm_backend=BackendBinding(backend_id="llm-local"),
            )
        },
    )
    runtime = _runtime_with_static_role_dispatch(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(ToolRegistry()),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": MemoryRuntime()},
        skill_runtimes={"skill-local": BroadSkillRuntime()},
    )

    result = runtime.run(_request())
    saved = runtime.trajectory_registry.get_subagent_run(result["run_ref"])

    assert saved is not None
    node_skill_ids = [node["skill_id"] for node in saved.metadata["workflow_plan"]["nodes"]]
    assert node_skill_ids == ["skill.scientific_document_intake.v1"]
    assert saved.metadata["skill_context"]["filtered_out_skill_ids"] == [
        "skill.extraction_result_validation.v1",
        "skill.ground_truth_based_evaluation.v1",
    ]


def test_meta_agent_selected_subagent_dag_response_parses_to_next_dispatch():
    decision = _parse_dispatch_decision(
        """
        {
          "decision_type": "run_subagent",
          "selected_subagents": [
            {
              "node_id": "survey_1",
              "generic_agent_type": "SurveyAgent",
              "assigned_task": "Survey inputs.",
              "input_dependencies": [],
              "expected_outputs": ["inventory"],
              "completion_criteria": "inventory complete",
              "recovery_policy": "rerun survey",
              "stage_index": 0,
              "workflow_dag": [{"node_id": "survey_1", "depends_on": [], "agent": "SurveyAgent"}]
            },
            {
              "node_id": "write_1",
              "generic_agent_type": "WriteAgent",
              "assigned_task": "Write outputs.",
              "input_dependencies": ["survey_1"],
              "expected_outputs": ["records"],
              "completion_criteria": "records written",
              "recovery_policy": "rerun writer",
              "stage_index": 1,
              "workflow_dag": [{"node_id": "survey_1", "depends_on": [], "agent": "SurveyAgent"}, {"node_id": "write_1", "depends_on": ["survey_1"], "agent": "WriteAgent"}]
            }
          ],
          "dispatch_rationale": "Survey then write."
        }
        """
    )

    assert decision.action.value == "run_subagent"
    assert decision.target_role == "SurveyAgent"
    assert decision.instruction == "Survey inputs."
    assert decision.retrieval_query == "Survey inputs."
    assert decision.metadata["meta_workflow_node_id"] == "survey_1"
    assert decision.metadata["agent_level_workflow_dag"][1]["node_id"] == "write_1"
    assert decision.metadata["dispatch_rationale"] == "Survey then write."


def test_meta_agent_route_contract_parses_to_subagent_dispatch():
    decision = _parse_dispatch_decision(
        """
        {
          "route": "SurveyAgent",
          "instruction": "Survey the lab files and report coverage.",
          "metadata": {
            "node_id": "survey_1",
            "stage_index": 0
          }
        }
        """
    )

    assert decision.action.value == "run_subagent"
    assert decision.target_role == "SurveyAgent"
    assert decision.instruction == "Survey the lab files and report coverage."
    assert decision.retrieval_query == "Survey the lab files and report coverage."
    assert decision.metadata["source_decision_type"] == "route"
    assert decision.metadata["route"] == "SurveyAgent"
    assert decision.metadata["generic_agent_type"] == "SurveyAgent"
    assert decision.metadata["meta_workflow_node_id"] == "survey_1"


def test_meta_agent_route_contract_parses_end_to_finish_task():
    decision = _parse_dispatch_decision(
        """
        {
          "route": "END",
          "instruction": "Workflow complete.",
          "metadata": {"final_answer": "Final artifacts are ready."}
        }
        """
    )

    assert decision.action.value == "finish_task"
    assert decision.target_role is None
    assert decision.metadata["source_decision_type"] == "route"
    assert decision.metadata["route"] == "END"
    assert decision.metadata["final_answer"] == "Final artifacts are ready."


def test_meta_agent_selected_subagent_dag_response_dispatches_next_incomplete_node():
    content = """
    {
      "decision_type": "run_subagent",
      "selected_subagents": [
        {
          "node_id": "survey_1",
          "generic_agent_type": "SurveyAgent",
          "assigned_task": "Survey inputs.",
          "input_dependencies": [],
          "expected_outputs": ["inventory"],
          "completion_criteria": "inventory complete"
        },
        {
          "node_id": "write_1",
          "generic_agent_type": "WriteAgent",
          "assigned_task": "Write outputs.",
          "input_dependencies": ["survey_1"],
          "expected_outputs": ["records"],
          "completion_criteria": "records written"
        }
      ]
    }
    """

    decision = _parse_dispatch_decision(
        content,
        completed_runs=[
            {
                "status": "completed",
                "role": "SurveyAgent",
                "meta_workflow_node_id": "survey_1",
            }
        ],
    )

    assert decision.action.value == "run_subagent"
    assert decision.target_role == "WriteAgent"
    assert decision.metadata["meta_workflow_node_id"] == "write_1"


def test_meta_agent_dispatch_list_response_parses_to_next_incomplete_node():
    decision = _parse_dispatch_decision(
        """
        {
          "dispatch": [
            {
              "meta_workflow_node_id": "design_1",
              "generic_agent_type": "DesignAgent",
              "assigned_task": "Design the plan.",
              "input_dependencies": ["survey_1"],
              "expected_outputs": ["plan"],
              "completion_criteria": "plan complete"
            },
            {
              "meta_workflow_node_id": "write_1",
              "generic_agent_type": "WriteAgent",
              "assigned_task": "Write outputs.",
              "input_dependencies": ["design_1"],
              "expected_outputs": ["records"],
              "completion_criteria": "records written"
            }
          ],
          "completion_policy": "Proceed with the DAG."
        }
        """,
        completed_runs=[
            {"status": "completed", "role": "SurveyAgent", "meta_workflow_node_id": "survey_1"},
        ],
    )

    assert decision.action.value == "run_subagent"
    assert decision.target_role == "DesignAgent"
    assert decision.instruction == "Design the plan."
    assert decision.metadata["meta_workflow_node_id"] == "design_1"
    assert decision.metadata["agent_level_workflow_dag"][0]["node_id"] == "design_1"


def test_meta_agent_single_node_decision_alias_parses_to_dispatch():
    decision = _parse_dispatch_decision(
        """
        {
          "decision": "run_subagent",
          "node_id": "survey_1",
          "generic_agent_type": "SurveyAgent",
          "assigned_task": "Survey inputs.",
          "input_dependencies": [],
          "expected_outputs": ["inventory"],
          "completion_criteria": "inventory complete",
          "workflow_dag": {
            "nodes": [
              {"node_id": "survey_1", "agent": "SurveyAgent", "depends_on": []}
            ],
            "edges": []
          }
        }
        """
    )

    assert decision.action.value == "run_subagent"
    assert decision.target_role == "SurveyAgent"
    assert decision.instruction == "Survey inputs."
    assert decision.metadata["meta_workflow_node_id"] == "survey_1"
    assert decision.metadata["agent_level_workflow_dag"]["nodes"][0]["node_id"] == "survey_1"


def test_meta_agent_dispatch_decision_wrapper_parses_to_dispatch():
    decision = _parse_dispatch_decision(
        """
        {
          "dispatch_decision": {
            "action": "run_subagent",
            "generic_agent_type": "DesignAgent",
            "meta_workflow_node_id": "design_1",
            "stage_index": 1,
            "assigned_task": "Design the plan.",
            "input_dependencies": ["survey_1"],
            "expected_outputs": ["plan"],
            "completion_criteria": ["plan covers both articles"],
            "recovery_policy": {"on_partial_coverage": "retry design"}
          }
        }
        """
    )

    assert decision.action.value == "run_subagent"
    assert decision.target_role == "DesignAgent"
    assert decision.instruction == "Design the plan."
    assert decision.metadata["meta_workflow_node_id"] == "design_1"


def test_meta_agent_dispatch_run_subagent_wrapper_parses_to_dispatch():
    decision = _parse_dispatch_decision(
        """
        {
          "decision": "dispatch",
          "selected_role": "SurveyAgent",
          "workflow_dag": {
            "nodes": [
              {
                "node_id": "survey_1",
                "generic_agent_type": "SurveyAgent",
                "assigned_task": "Survey inputs.",
                "input_dependencies": [],
                "expected_outputs": ["inventory"],
                "completion_criteria": "inventory complete"
              }
            ]
          },
          "run_subagent": {
            "node_id": "survey_1",
            "generic_agent_type": "SurveyAgent",
            "assigned_task": "Survey inputs.",
            "input_dependencies": [],
            "expected_outputs": ["inventory"],
            "completion_criteria": "inventory complete"
          }
        }
        """
    )

    assert decision.action.value == "run_subagent"
    assert decision.target_role == "SurveyAgent"
    assert decision.instruction == "Survey inputs."
    assert decision.metadata["meta_workflow_node_id"] == "survey_1"


def test_meta_agent_finish_alias_parses_to_finish_task():
    decision = _parse_dispatch_decision(
        """
        {
          "dispatch": false,
          "finish": true,
          "reason": "Selected workflow completed.",
          "selected_dag": [
            {"meta_workflow_node_id": "survey_1", "generic_agent_type": "SurveyAgent"},
            {"meta_workflow_node_id": "exec_1", "generic_agent_type": "ExecAgent"}
          ]
        }
        """
    )

    assert decision.action.value == "finish_task"
    assert decision.metadata["final_answer"] == "Selected workflow completed."
    assert decision.metadata["agent_level_workflow_dag"][1]["meta_workflow_node_id"] == "exec_1"


def test_meta_agent_run_subagent_metadata_wrapper_parses_to_dispatch():
    decision = _parse_dispatch_decision(
        """
        {
          "decision": "run_subagent",
          "selected_role": "SurveyAgent",
          "metadata": {
            "node_id": "survey_dataset_inventory",
            "generic_agent_type": "SurveyAgent",
            "assigned_task": "Survey inputs.",
            "input_dependencies": [],
            "expected_outputs": ["inventory"],
            "completion_criteria": "inventory complete",
            "workflow_dag": {
              "nodes": [
                {"node_id": "survey_dataset_inventory", "agent": "SurveyAgent", "depends_on": []}
              ]
            }
          }
        }
        """
    )

    assert decision.action.value == "run_subagent"
    assert decision.target_role == "SurveyAgent"
    assert decision.instruction == "Survey inputs."
    assert decision.metadata["meta_workflow_node_id"] == "survey_dataset_inventory"


def test_meta_agent_extra_text_json_can_be_safely_extracted():
    decision = _parse_dispatch_decision(
        """
        Here is the dispatch:
        ```json
        {"action":"run_subagent","target_role":"SurveyAgent","instruction":"Survey inputs."}
        ```
        """
    )

    assert decision.action.value == "run_subagent"
    assert decision.target_role == "SurveyAgent"
    assert decision.instruction == "Survey inputs."


def test_plan_aware_runtime_creates_fallback_node_when_no_skill_matches(tmp_path: Path):
    class EmptySkillRuntime(SkillRuntime):
        def get(self, request: RetrievalRequest) -> SkillBundle:
            return SkillBundle(
                backend_id="skill-local",
                graph_version_ref="graph-v1",
                required_tools=[],
                skills=[],
                metadata={"graph_context_summary": {"graph_version": "graph-v1"}},
            )

    registry = ToolRegistry()
    registry.register(
        ToolSpec(name="read_text", description="read", parameters_schema={"type": "object"}),
        lambda arguments: "text",
    )
    llm = ScriptedLLM(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(call_id="read", name="read_text", arguments={"path": "input.md"}),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="surveyed")),
        ]
    )
    task_config = TaskConfig(
        task_id="task-1",
        goal="Survey inputs.",
        runtime_policy=RuntimePolicy(
            enable_workflow_planning=True,
            metadata={
                "subagent_policy": {"tool_preparation_scope": "per_internal_dag_node_or_retrieved_skill"},
                "completion_guards_by_role": {"SurveyAgent": {"required_tool_calls_before_final": ["read_text"]}},
            },
        ),
        roles={
            "SurveyAgent": RoleSpec(
                name="SurveyAgent",
                system_prompt="Survey.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["read_text"],
            )
        },
    )
    runtime = _runtime_with_static_role_dispatch(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": MemoryRuntime()},
        skill_runtimes={"skill-local": EmptySkillRuntime()},
    )

    result = runtime.run(_request())
    saved = runtime.trajectory_registry.get_subagent_run(result["run_ref"])

    assert [spec["name"] for spec in llm.calls[0][1]] == ["read_text"]
    assert saved is not None
    assert saved.metadata["workflow_plan"]["nodes"][0]["skill_id"] == "runtime.assigned_task.SurveyAgent"
    assert saved.metadata["node_execution_records"][0]["metadata"]["fallback_node"] is True


def test_runtime_limits_retrieved_skills_and_internal_dag_nodes_per_role_budget(tmp_path: Path):
    class ManySkillRuntime(SkillRuntime):
        def get(self, request: RetrievalRequest) -> SkillBundle:
            skills = [
                SkillRef(
                    skill_id=f"skill.generic_{index}.v1",
                    name=f"Generic Skill {index}",
                    content="Description:\nGeneric assigned-task skill.",
                    required_tools=[],
                )
                for index in range(5)
            ]
            return SkillBundle(
                backend_id="skill-local",
                graph_version_ref="graph-v1",
                skills=skills,
                metadata={"retrieval_trace": {"returned_skill_ids": [skill.skill_id for skill in skills]}},
            )

    llm = ScriptedLLM(
        [
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="node 1")),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="node 2")),
        ]
    )
    task_config = TaskConfig(
        task_id="task-1",
        goal="Run generic assignment.",
        runtime_policy=RuntimePolicy(
            enable_workflow_planning=True,
            max_workflow_nodes=5,
            metadata={
                "subagent_budgets_by_role": {
                    "GenericAgent": {
                        "max_retrieved_skills": 3,
                        "max_internal_dag_nodes": 2,
                    }
                }
            },
        ),
        roles={
            "GenericAgent": RoleSpec(
                name="GenericAgent",
                system_prompt="Do assigned work.",
                llm_backend=BackendBinding(backend_id="llm-local"),
            )
        },
    )
    runtime = _runtime_with_static_role_dispatch(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(ToolRegistry()),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": MemoryRuntime()},
        skill_runtimes={"skill-local": ManySkillRuntime()},
    )

    result = runtime.run(_request())
    saved = runtime.trajectory_registry.get_subagent_run(result["run_ref"])

    assert saved is not None
    assert [skill["skill_id"] for skill in saved.metadata["skill_context"]["selected_skills"]] == [
        "skill.generic_0.v1",
        "skill.generic_1.v1",
        "skill.generic_2.v1",
    ]
    assert len(saved.metadata["workflow_plan"]["nodes"]) == 3
    assert len(saved.metadata["node_execution_records"]) == 3
    assert saved.metadata["node_execution_records"][-1]["status"] == "skipped"
    warnings = saved.metadata["workflow_plan"]["metadata"]["planning_warnings"]
    assert "retrieved skill budget limited GenericAgent to 3 skill(s); pruned 2" in warnings
    assert "workflow execution limited to 2 nodes; skipped 1 nodes" in warnings
