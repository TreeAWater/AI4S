from pathlib import Path

from evolab.config.task_config import BackendBinding, RoleSpec, TaskConfig
from evolab.contracts.common import Message, RuntimePolicy
from evolab.contracts.llm import LLMGenerationConfig, LLMRuntimeResponse, SubAgentAction
from evolab.contracts.retrieval import MemoryBundle, RetrievalRequest, SkillBundle, SkillRef
from evolab.contracts.task import TaskOrigin, TaskPurpose, TaskRequest
from evolab.contracts.tools import ToolCall
from evolab.registries.trajectory import FileTrajectoryRegistry
from evolab.runtime.prompt_builder import PromptBuilder
from evolab.runtime.skill_retrieval import prepare_skill_runtime_context
from evolab.runtime.task_runtime import TaskRuntime
from evolab.tools.human import register_human_tools
from evolab.tools.runtime import ToolRegistry, ToolRuntime


class EmptyMemory:
    def search(self, request: RetrievalRequest) -> MemoryBundle:
        return MemoryBundle(backend_id="memory-local")

    def add(self, task_id: str, role: str, messages: list[Message]) -> dict[str, str]:
        return {"status": "updated", "state_ref": "memory-after"}


class HumanSkillRuntime:
    def __init__(self):
        self.look_at_events: list[dict] = []

    def get(self, request: RetrievalRequest) -> SkillBundle:
        return SkillBundle(
            backend_id="skill-local",
            required_tools=[],
            skills=[
                SkillRef(
                    skill_id="skill.human_feedback_integration.v1",
                    name="Human Feedback Integration",
                    content="Description:\nIntegrate optional human feedback.",
                    required_tools=[],
                )
            ],
            metadata={"graph_context_summary": {}, "retrieval_trace": {"returned_skill_ids": []}},
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


def test_register_human_tools_registers_policy_gated_specs_and_mock_results():
    registry = ToolRegistry()
    register_human_tools(registry)

    specs = [registry.get_spec(name) for name in ["ask_human", "request_human_review", "notify_human"]]
    assert [spec.name for spec in specs if spec is not None] == ["ask_human", "request_human_review", "notify_human"]
    assert all(spec.metadata["requires_human"] is True for spec in specs if spec is not None)

    runtime = ToolRuntime(registry)
    hidden = runtime.prepare(
        required_tools=[],
        optional_tools=["ask_human", "request_human_review", "notify_human"],
        allowed_tools=["ask_human", "request_human_review", "notify_human"],
        policy=RuntimePolicy(allow_human_tools=False),
    )
    exposed = runtime.prepare(
        required_tools=[],
        optional_tools=["ask_human", "request_human_review", "notify_human"],
        allowed_tools=["ask_human", "request_human_review", "notify_human"],
        policy=RuntimePolicy(allow_human_tools=True),
    )

    assert hidden.tool_specs == []
    assert [spec.name for spec in exposed.tool_specs] == ["ask_human", "request_human_review", "notify_human"]
    ask = runtime.execute_tool_name(
        call_id="human-1",
        name="ask_human",
        arguments={"question": "Proceed?", "context": "test"},
    )
    review = runtime.execute_tool_name(
        call_id="human-2",
        name="request_human_review",
        arguments={
            "artifact_ref": "artifact://1",
            "review_type": "conflict_resolution",
            "instructions": "resolve conflict",
            "blocking": True,
        },
    )
    notify = runtime.execute_tool_name(
        call_id="human-3",
        name="notify_human",
        arguments={"message": "done"},
    )

    assert "MOCK_HUMAN_RESPONSE" in ask.content
    assert review.metadata["review_status"] == "needs_revision"
    assert notify.metadata["delivered"] is True


def test_prepare_skill_runtime_context_exposes_human_tools_as_optional_without_required_tools():
    registry = ToolRegistry()
    register_human_tools(registry)
    skill = HumanSkillRuntime()
    prepared = prepare_skill_runtime_context(
        retrieval_request=RetrievalRequest(task_id="task-1", role="solver", query="review"),
        skill_backend=skill,
        tool_runtime=ToolRuntime(registry),
        allowed_tools=["ask_human", "request_human_review", "notify_human"],
        policy=RuntimePolicy(allow_human_tools=True),
    )

    assert prepared.skill_bundle.required_tools == []
    assert [spec.name for spec in prepared.tool_bundle.tool_specs] == [
        "ask_human",
        "request_human_review",
        "notify_human",
    ]

    no_registry_prepared = prepare_skill_runtime_context(
        retrieval_request=RetrievalRequest(task_id="task-1", role="solver", query="review"),
        skill_backend=skill,
        tool_runtime=ToolRuntime(ToolRegistry()),
        allowed_tools=["ask_human"],
        policy=RuntimePolicy(allow_human_tools=True),
    )
    assert no_registry_prepared.tool_bundle.tool_specs == []


def test_plan_aware_node_can_call_human_tool_through_normal_tool_trace(tmp_path: Path):
    registry = ToolRegistry()
    register_human_tools(registry)
    skill = HumanSkillRuntime()
    llm = ScriptedLLM(
        [
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="tool_call",
                    tool_call=ToolCall(
                        call_id="human-call-1",
                        name="ask_human",
                        arguments={"question": "Proceed?", "context": "node"},
                    ),
                )
            ),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="human response used")),
        ]
    )
    task_config = TaskConfig(
        task_id="task-1",
        goal="Review extraction.",
        runtime_policy=RuntimePolicy(enable_workflow_planning=True, allow_human_tools=True),
        roles={
            "solver": RoleSpec(
                name="solver",
                system_prompt="Solve.",
                llm_backend=BackendBinding(backend_id="llm-local"),
                allowed_tools=["ask_human", "request_human_review", "notify_human"],
            )
        },
    )
    runtime = TaskRuntime(
        task_config=task_config,
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        llm_runtimes={"llm-local": llm},
        memory_runtimes={"memory-local": EmptyMemory()},
        skill_runtimes={"skill-local": skill},
    )

    result = runtime.run(TaskRequest(task_id="task-1", origin=TaskOrigin.HUMAN, purpose=TaskPurpose.SCIENCE, goal="Review."))
    saved = runtime.trajectory_registry.get_subagent_run(result["run_ref"])

    assert saved is not None
    assert saved.metadata["tool_trace"]["calls"][0]["tool_call"]["name"] == "ask_human"
    assert saved.metadata["node_execution_records"][0]["tool_calls"][0]["tool_call"]["name"] == "ask_human"
    assert all(
        node["skill_id"] not in {"ask_human", "request_human_review", "notify_human"}
        for node in saved.metadata["workflow_plan"]["nodes"]
    )
