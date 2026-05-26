from __future__ import annotations

import pytest

from evolab.backends.skills import FakeSkillBackend
from evolab.contracts.dynamic_workflow import DynamicSubagentsConfig, DynamicWorkflowSpec
from evolab.contracts.retrieval import SkillItem
from evolab.contracts.tools import ToolSpec
from evolab.runtime.dynamic_workflow import validate_dynamic_workflow_spec
from evolab.tools.runtime import ToolRegistry, ToolRuntime


def test_valid_dynamic_workflow_schema_validates_and_fills_default_backend():
    spec = _workflow_spec()
    prepared, report = validate_dynamic_workflow_spec(
        spec,
        config=_dynamic_config(),
        available_llm_backend_ids={"planner", "worker"},
        tool_runtime=_tool_runtime("read_text", "write_report"),
        skill_backend=FakeSkillBackend(skills=[]),
        task_id="task-1",
    )

    assert report.valid is True
    assert report.errors == []
    assert prepared.dynamic_subagents[0].llm_backend_id == "worker"


def test_missing_required_fields_fail_schema_validation():
    payload = _workflow_spec().model_dump(mode="json")
    del payload["workflow_id"]

    with pytest.raises(Exception):
        DynamicWorkflowSpec.model_validate(payload)


def test_unknown_tool_is_rejected_by_dynamic_validation():
    spec = _workflow_spec(agent_updates={"allowed_tools": ["unknown_tool"]})
    _prepared, report = validate_dynamic_workflow_spec(
        spec,
        config=_dynamic_config(allowed_tool_names=["unknown_tool"]),
        available_llm_backend_ids={"planner", "worker"},
        tool_runtime=_tool_runtime("read_text"),
        skill_backend=FakeSkillBackend(skills=[]),
        task_id="task-1",
    )

    assert report.valid is False
    assert any("unknown tool" in error for error in report.errors)


def test_invalid_dag_edge_is_rejected_by_schema():
    payload = _workflow_spec().model_dump(mode="json")
    payload["workflow_edges"] = [{"source_node_id": "node-a", "target_node_id": "missing", "relation": "depends_on"}]

    with pytest.raises(Exception):
        DynamicWorkflowSpec.model_validate(payload)


def test_required_skill_tools_are_resolved_when_allowed():
    spec = _workflow_spec(agent_updates={"skill_retrieval_request": {"query": "table reader"}})
    skill = SkillItem(
        skill_id="skill.table_reader",
        name="Table Reader",
        content="Read tables.",
        required_tools=["profile_table"],
    )
    prepared, report = validate_dynamic_workflow_spec(
        spec,
        config=_dynamic_config(allowed_tool_names=["read_text", "write_report", "profile_table"]),
        available_llm_backend_ids={"planner", "worker"},
        tool_runtime=_tool_runtime("read_text", "write_report", "profile_table"),
        skill_backend=FakeSkillBackend(skills=[skill]),
        task_id="task-1",
    )

    assert report.valid is True
    assert report.resolved_skill_ids == ["skill.table_reader"]
    assert prepared.dynamic_subagents[0].llm_backend_id == "worker"


def _dynamic_config(**updates):
    payload = {
        "enabled": True,
        "mode": "dynamic",
        "scope": "per_task",
        "planner_backend": {"backend_id": "planner"},
        "default_worker_backend": {"backend_id": "worker"},
        "allowed_tool_names": ["read_text", "write_report"],
    }
    payload.update(updates)
    return DynamicSubagentsConfig.model_validate(payload)


def _workflow_spec(*, agent_updates=None):
    agent = {
        "subagent_id": "agent-text",
        "role_name": "TextContextAgent",
        "goal": "Read text evidence.",
        "system_prompt": "Read text evidence.",
        "input_schema": {"type": "object"},
        "output_schema": {"type": "object"},
        "allowed_tools": ["read_text"],
        "artifact_outputs": ["context.json"],
    }
    agent.update(agent_updates or {})
    return DynamicWorkflowSpec.model_validate(
        {
            "workflow_id": "wf-1",
            "task_summary": "Process one synthetic item.",
            "article_context_summary": "text only",
            "dynamic_subagents": [agent],
            "workflow_nodes": [
                {
                    "node_id": "node-a",
                    "subagent_id": "agent-text",
                    "output_artifacts": ["context.json"],
                }
            ],
            "workflow_edges": [],
            "artifact_contracts": {"context.json": {"type": "object"}},
            "validation_rules": ["outputs must be JSON"],
            "planner_rationale_summary": "Text-only item needs one context agent.",
        }
    )


def _tool_runtime(*names: str) -> ToolRuntime:
    registry = ToolRegistry()
    for name in names:
        registry.register(ToolSpec(name=name, description=name, parameters_schema={}), lambda args: "ok")
    return ToolRuntime(registry)
