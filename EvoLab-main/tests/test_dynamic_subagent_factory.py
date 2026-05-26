from __future__ import annotations

import pytest

from evolab.backends.skills import FakeSkillBackend
from evolab.config.task_config import BackendBinding, RoleSpec
from evolab.contracts.dynamic_workflow import DynamicSubAgentSpec, DynamicSubagentsConfig
from evolab.contracts.generated_tools import TaskEffectiveToolCatalog
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
        )
    with pytest.raises(ValueError, match="outside dynamic_subagents.allowed_tool_names"):
        factory.create(
            spec=_spec(allowed_tools=["write_report"]),
            task_id="task-1",
            work_item_id=None,
            workflow_id="wf-1",
            planner_backend_id="planner",
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
    )

    assert runtime_agent.role.allowed_tools == ["read_text"]
    result = tool_runtime.execute(ToolCall(call_id="blocked", name="write_report", arguments={}))
    assert result.status == "error"
    assert result.metadata["error_type"] == "unprepared_tool"


def test_factory_creates_dynamic_subagent_with_generated_effective_tool_and_prepares_runtime():
    tool_runtime = ToolRuntime(ToolRegistry())
    generated_spec = ToolSpec(
        name="gt_task_extract",
        description="Generated extract.",
        parameters_schema={"type": "object"},
        metadata={"generated_tool": True},
    )
    tool_runtime.activate_generated_tool_scope("task-1")
    tool_runtime.register_task_generated_tool(
        generated_spec,
        lambda args: "generated",
        provenance={"code_hash": "abc"},
    )
    catalog = TaskEffectiveToolCatalog(
        task_id="task-1",
        builtin_allowed_tool_names=[],
        generated_tool_names=["gt_task_extract"],
        tool_specs_by_name={"gt_task_extract": generated_spec},
        provenance_by_name={"gt_task_extract": {"code_hash": "abc"}},
    )
    factory = DynamicSubAgentFactory(
        config=_config(allowed_tool_names=[]),
        tool_runtime=tool_runtime,
        skill_backend=FakeSkillBackend(skills=[]),
        available_llm_backend_ids={"planner", "openrouter-qwen30b"},
        effective_tool_catalog=catalog,
    )

    runtime_agent = factory.create(
        spec=_spec(allowed_tools=["gt_task_extract"]),
        task_id="task-1",
        work_item_id=None,
        workflow_id="wf-1",
        planner_backend_id="planner",
    )

    assert runtime_agent.role.allowed_tools == ["gt_task_extract"]
    assert runtime_agent.tool_names == ["gt_task_extract"]
    result = tool_runtime.execute_tool_name("call-1", "gt_task_extract", {})
    assert result.status == "ok"
    assert result.content == "generated"
    assert result.metadata["generated_tool"]["code_hash"] == "abc"


def test_factory_strips_agent_memory_backend_for_ephemeral_roles():
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
    )

    assert runtime_agent.role.agent_memory_backend is None


def test_factory_forces_dynamic_agent_to_consume_matching_agents_md_role():
    tool_runtime = _tool_runtime("read_text", "write_report")
    static_role = RoleSpec(
        name="TableAgent",
        system_prompt="Evolved agents.md prompt. Require explicit evidence before accepting records.",
        llm_backend=BackendBinding(backend_id="openrouter-qwen30b"),
        allowed_tools=["write_report"],
        required_skills=["skill.evidence"],
        memory_policy={"prefer_recent_failures": True},
        metadata={"evolved_by": "meta"},
    )
    factory = DynamicSubAgentFactory(
        config=_config(allowed_tool_names=["read_text", "write_report"]),
        tool_runtime=tool_runtime,
        skill_backend=FakeSkillBackend(skills=[]),
        available_llm_backend_ids={"planner", "openrouter-qwen30b"},
        static_roles=[static_role],
    )

    runtime_agent = factory.create(
        spec=_spec(system_prompt="Planner-generated table inspection prompt."),
        task_id="task-1",
        work_item_id="item-1",
        workflow_id="wf-1",
        planner_backend_id="planner",
    )

    assert runtime_agent.role.system_prompt.startswith("Evolved agents.md prompt.")
    assert "Dynamic workflow assignment context from planner" in runtime_agent.role.system_prompt
    assert "Planner-generated table inspection prompt." in runtime_agent.role.system_prompt
    assert runtime_agent.role.allowed_tools == ["read_text", "write_report"]
    assert runtime_agent.role.required_skills == ["skill.evidence"]
    assert runtime_agent.role.agent_memory_backend is None
    assert runtime_agent.role.memory_policy == {}
    assert runtime_agent.role.metadata["consumed_static_agents_md_role"] is True
    assert runtime_agent.provenance["consumed_static_agents_md_role"] is True


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
