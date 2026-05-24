from __future__ import annotations

import pytest

from evolab.backends.skills import FakeSkillBackend
from evolab.config.task_config import BackendBinding
from evolab.contracts.dynamic_workflow import DynamicSubAgentSpec, DynamicSubagentsConfig
from evolab.contracts.retrieval import SkillItem
from evolab.contracts.tools import ToolCall, ToolSpec
from evolab.runtime.dynamic_workflow import DynamicSubAgentFactory
from evolab.tools.runtime import ToolRegistry, ToolRuntime


def test_factory_creates_ephemeral_role_with_default_worker_backend_and_skill_tools():
    tool_runtime = _tool_runtime("read_text", "profile_table")
    skill_backend = FakeSkillBackend(
        skills=[SkillItem(skill_id="skill.profile", name="Profile", content="Profile tables.", required_tools=["profile_table"])]
    )
    factory = DynamicSubAgentFactory(
        config=_config(allowed_tool_names=["read_text", "profile_table"]),
        tool_runtime=tool_runtime,
        skill_backend=skill_backend,
        available_llm_backend_ids={"planner", "openrouter-qwen30b"},
    )

    runtime_agent = factory.create(
        spec=_spec(skill_retrieval_request={"query": "profile tables"}),
        task_id="task-1",
        work_item_id="item-1",
        workflow_id="wf-1",
        planner_backend_id="planner",
        static_fallback_available=True,
    )

    assert runtime_agent.role.name == "TableAgent"
    assert runtime_agent.role.llm_backend.backend_id == "openrouter-qwen30b"
    assert runtime_agent.role.allowed_tools == ["read_text", "profile_table"]
    assert runtime_agent.provenance["resolved_skill_ids"] == ["skill.profile"]
    assert runtime_agent.provenance["workflow_id"] == "wf-1"


def test_factory_rejects_unknown_tool_and_backend():
    tool_runtime = _tool_runtime("read_text")
    factory = DynamicSubAgentFactory(
        config=_config(allowed_tool_names=["read_text"]),
        tool_runtime=tool_runtime,
        skill_backend=FakeSkillBackend(skills=[]),
        available_llm_backend_ids={"planner", "openrouter-qwen30b"},
    )

    with pytest.raises(ValueError, match="unknown backend"):
        factory.create(
            spec=_spec(llm_backend_id="missing"),
            task_id="task-1",
            work_item_id=None,
            workflow_id="wf-1",
            planner_backend_id="planner",
            static_fallback_available=True,
        )
    with pytest.raises(ValueError, match="outside dynamic_subagents.allowed_tool_names"):
        factory.create(
            spec=_spec(allowed_tools=["write_report"]),
            task_id="task-1",
            work_item_id=None,
            workflow_id="wf-1",
            planner_backend_id="planner",
            static_fallback_available=True,
        )


def test_dynamic_agent_uses_toolruntime_preparation_and_cannot_bypass_prepared_tools():
    tool_runtime = _tool_runtime("read_text", "write_report")
    factory = DynamicSubAgentFactory(
        config=_config(allowed_tool_names=["read_text", "write_report"]),
        tool_runtime=tool_runtime,
        skill_backend=FakeSkillBackend(skills=[]),
        available_llm_backend_ids={"planner", "openrouter-qwen30b"},
    )
    runtime_agent = factory.create(
        spec=_spec(allowed_tools=["read_text"]),
        task_id="task-1",
        work_item_id=None,
        workflow_id="wf-1",
        planner_backend_id="planner",
        static_fallback_available=True,
    )

    assert runtime_agent.role.allowed_tools == ["read_text"]
    result = tool_runtime.execute(ToolCall(call_id="blocked", name="write_report", arguments={}))
    assert result.status == "error"
    assert result.metadata["error_type"] == "unprepared_tool"


def test_factory_binds_default_agent_memory_backend_for_ephemeral_roles():
    tool_runtime = _tool_runtime("read_text")
    factory = DynamicSubAgentFactory(
        config=_config(allowed_tool_names=["read_text"]),
        tool_runtime=tool_runtime,
        skill_backend=FakeSkillBackend(skills=[]),
        available_llm_backend_ids={"planner", "openrouter-qwen30b"},
        agent_memory_backend=BackendBinding(backend_id="mem0-agent-memory"),
    )

    runtime_agent = factory.create(
        spec=_spec(),
        task_id="task-1",
        work_item_id=None,
        workflow_id="wf-1",
        planner_backend_id="planner",
        static_fallback_available=True,
    )

    assert runtime_agent.role.agent_memory_backend is not None
    assert runtime_agent.role.agent_memory_backend.backend_id == "mem0-agent-memory"


def _config(**updates):
    payload = {
        "enabled": True,
        "planner_backend": {"backend_id": "planner"},
        "default_worker_backend": {"backend_id": "openrouter-qwen30b"},
        "allowed_tool_names": ["read_text"],
    }
    payload.update(updates)
    return DynamicSubagentsConfig.model_validate(payload)


def _spec(**updates):
    payload = {
        "subagent_id": "agent-table",
        "role_name": "TableAgent",
        "goal": "Inspect table evidence.",
        "system_prompt": "Inspect table evidence.",
        "input_schema": {"type": "object"},
        "output_schema": {"type": "object"},
        "allowed_tools": ["read_text"],
    }
    payload.update(updates)
    return DynamicSubAgentSpec.model_validate(payload)


def _tool_runtime(*names: str) -> ToolRuntime:
    registry = ToolRegistry()
    for name in names:
        registry.register(ToolSpec(name=name, description=name, parameters_schema={}), lambda args: "ok")
    return ToolRuntime(registry)
