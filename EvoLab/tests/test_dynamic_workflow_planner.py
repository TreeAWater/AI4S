from __future__ import annotations

import json

from evolab.backends.llm.fake import FakeLLMRuntime
from evolab.backends.skills import FakeSkillBackend
from evolab.contracts.dynamic_workflow import DynamicSubagentsConfig
from evolab.contracts.llm import LLMRuntimeResponse, SubAgentAction
from evolab.contracts.task import TaskOrigin, TaskPurpose, TaskRequest
from evolab.contracts.tools import ToolSpec
from evolab.runtime.dynamic_workflow import DynamicWorkflowPlanner
from evolab.tools.runtime import ToolRegistry, ToolRuntime


def test_fake_planner_returns_valid_dynamic_workflow_json_and_defaults_worker_backend():
    planner_runtime = FakeLLMRuntime(default_content=json.dumps(_workflow_payload()))
    planner = DynamicWorkflowPlanner(
        planner_llm=planner_runtime,
        config=_config(),
        tool_runtime=_tool_runtime("read_text", "write_report"),
        skill_backend=FakeSkillBackend(skills=[]),
        available_llm_backend_ids={"gpt-planner-nano", "openrouter-qwen30b"},
    )

    outcome = planner.plan(request=_request())

    assert outcome.spec is not None
    assert outcome.validation_report.valid is True
    assert outcome.spec.dynamic_subagents[0].llm_backend_id == "openrouter-qwen30b"
    assert outcome.validation_report.planner_backend_id == "gpt-planner-nano"
    assert outcome.validation_report.default_worker_backend_id == "openrouter-qwen30b"
    assert planner_runtime.requests[0].generation_config.response_json_schema is None


def test_invalid_planner_json_retries_then_falls_back_without_static_breakage():
    planner_runtime = FakeLLMRuntime(
        responses=[
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="not json")),
            LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content=json.dumps(_workflow_payload(tool="not_allowed")))),
        ]
    )
    planner = DynamicWorkflowPlanner(
        planner_llm=planner_runtime,
        config=_config(max_planner_retries=1, allowed_tool_names=["read_text"]),
        tool_runtime=_tool_runtime("read_text"),
        skill_backend=FakeSkillBackend(skills=[]),
        available_llm_backend_ids={"gpt-planner-nano", "openrouter-qwen30b"},
    )

    outcome = planner.plan(request=_request())

    assert outcome.spec is None
    assert outcome.fallback_reason is not None
    assert outcome.fallback_reason["fallback_to_static"] is True
    assert len(planner_runtime.requests) == 2


def test_planner_rejects_chain_of_thought_field():
    payload = _workflow_payload()
    payload["chain_of_thought"] = "private reasoning"
    planner_runtime = FakeLLMRuntime(default_content=json.dumps(payload))
    planner = DynamicWorkflowPlanner(
        planner_llm=planner_runtime,
        config=_config(max_planner_retries=0),
        tool_runtime=_tool_runtime("read_text"),
        skill_backend=FakeSkillBackend(skills=[]),
        available_llm_backend_ids={"gpt-planner-nano", "openrouter-qwen30b"},
    )

    outcome = planner.plan(request=_request())

    assert outcome.spec is None
    assert any("chain_of_thought" in error or "extra" in error for error in outcome.validation_report.errors)


def test_planner_respects_max_subagent_limit():
    payload = _workflow_payload()
    second = dict(payload["dynamic_subagents"][0])
    second["subagent_id"] = "agent-extra"
    second["role_name"] = "ExtraAgent"
    payload["dynamic_subagents"].append(second)
    payload["workflow_nodes"].append({"node_id": "node-extra", "subagent_id": "agent-extra"})
    planner_runtime = FakeLLMRuntime(default_content=json.dumps(payload))
    planner = DynamicWorkflowPlanner(
        planner_llm=planner_runtime,
        config=_config(max_subagents=1, max_subagents_per_work_item=1, max_planner_retries=0),
        tool_runtime=_tool_runtime("read_text"),
        skill_backend=FakeSkillBackend(skills=[]),
        available_llm_backend_ids={"gpt-planner-nano", "openrouter-qwen30b"},
    )

    outcome = planner.plan(request=_request())

    assert outcome.spec is None
    assert any("exceeds limit" in error for error in outcome.validation_report.errors)


def test_planner_normalizes_common_json_shape_deviations():
    payload = _workflow_payload()
    payload.pop("planner_rationale_summary")
    payload["task_id"] = "task-1"
    payload["task_goal"] = "Summarize documents."
    payload["work_item"] = {"work_item_id": "item-1"}
    payload["top_level_keys"] = ["workflow_id", "dynamic_subagents"]
    payload["static_fallback_subagents"] = [{"name": "StaticAgent"}]
    payload["artifact_contracts"] = [{"artifact_name": "context.json", "type": "object"}]
    payload["dynamic_subagents"][0].pop("goal")
    payload["dynamic_subagents"][0].pop("role_name")
    payload["dynamic_subagents"][0].pop("system_prompt")
    payload["dynamic_subagents"][0].pop("output_schema")
    payload["dynamic_subagents"][0]["task"] = "Read text context."
    payload["workflow_nodes"][0]["execution_constraints"] = ["bounded"]
    payload["workflow_edges"] = [
        {
            "source_node_id": "node-text",
            "target_node_id": "node-text-2",
            "artifact_names": ["context.json"],
        }
    ]
    second = dict(payload["workflow_nodes"][0])
    second["node_id"] = "node-text-2"
    second["dependencies"] = ["node-text"]
    payload["workflow_nodes"].append(second)
    planner_runtime = FakeLLMRuntime(default_content=json.dumps(payload))
    planner = DynamicWorkflowPlanner(
        planner_llm=planner_runtime,
        config=_config(),
        tool_runtime=_tool_runtime("read_text", "write_report"),
        skill_backend=FakeSkillBackend(skills=[]),
        available_llm_backend_ids={"gpt-planner-nano", "openrouter-qwen30b"},
    )

    outcome = planner.plan(request=_request())

    assert outcome.spec is not None
    assert outcome.spec.dynamic_subagents[0].goal == "Read text context."
    assert outcome.spec.dynamic_subagents[0].role_name == "AgentText"
    assert "Read text context" in outcome.spec.dynamic_subagents[0].system_prompt
    assert outcome.spec.dynamic_subagents[0].output_schema == {"type": "object", "additionalProperties": True}
    assert outcome.spec.artifact_contracts["context.json"]["type"] == "object"
    assert outcome.spec.workflow_edges[0].metadata["artifact_names"] == ["context.json"]
    assert outcome.spec.metadata["task_id"] == "task-1"
    assert outcome.spec.metadata["task_goal"] == "Summarize documents."
    assert outcome.spec.metadata["work_item"] == {"work_item_id": "item-1"}
    assert outcome.spec.metadata["top_level_keys"] == ["workflow_id", "dynamic_subagents"]
    assert outcome.spec.metadata["static_fallback_subagents"] == [{"name": "StaticAgent"}]
    assert outcome.spec.planner_rationale_summary == "Dynamic workflow generated from task and work-item context."


def test_extraction_workflow_requires_extraction_validation_and_writer_nodes():
    payload = _workflow_payload()
    payload["task_goal"] = "Extract structured records from source evidence."
    payload["work_item"] = {"work_item_id": "item-1"}
    planner_runtime = FakeLLMRuntime(default_content=json.dumps(payload))
    planner = DynamicWorkflowPlanner(
        planner_llm=planner_runtime,
        config=_config(metadata={"extraction_task": True}, max_planner_retries=0),
        tool_runtime=_tool_runtime("read_text", "write_report"),
        skill_backend=FakeSkillBackend(skills=[]),
        available_llm_backend_ids={"gpt-planner-nano", "openrouter-qwen30b"},
    )

    outcome = planner.plan(request=_request(goal="Extract structured records."))

    assert outcome.spec is None
    assert any("extraction-capable node" in error for error in outcome.validation_report.errors)
    assert any("validation" in error for error in outcome.validation_report.errors)
    assert any("writer" in error for error in outcome.validation_report.errors)


def test_text_only_extraction_workflow_with_extraction_validation_writer_is_valid():
    payload = _extraction_workflow_payload()
    planner_runtime = FakeLLMRuntime(default_content=json.dumps(payload))
    planner = DynamicWorkflowPlanner(
        planner_llm=planner_runtime,
        config=_config(metadata={"extraction_task": True}, max_planner_retries=0),
        tool_runtime=_tool_runtime("read_text", "write_report"),
        skill_backend=FakeSkillBackend(skills=[]),
        available_llm_backend_ids={"gpt-planner-nano", "openrouter-qwen30b"},
    )

    outcome = planner.plan(request=_request(goal="Extract structured records."))

    assert outcome.spec is not None
    assert outcome.validation_report.valid is True


def test_workbook_extraction_workflow_requires_table_or_workbook_inspection_node():
    payload = _extraction_workflow_payload()
    payload["work_item"] = {"work_item_id": "item-1", "exact_source_files": ["/tmp/article/supplementary/table.xlsx"]}
    planner_runtime = FakeLLMRuntime(default_content=json.dumps(payload))
    planner = DynamicWorkflowPlanner(
        planner_llm=planner_runtime,
        config=_config(metadata={"extraction_task": True}, max_planner_retries=0),
        tool_runtime=_tool_runtime("read_text", "write_report"),
        skill_backend=FakeSkillBackend(skills=[]),
        available_llm_backend_ids={"gpt-planner-nano", "openrouter-qwen30b"},
    )

    outcome = planner.plan(request=_request(goal="Extract structured records."))

    assert outcome.spec is None
    assert any("table/workbook inspection" in error for error in outcome.validation_report.errors)


def test_workbook_extraction_workflow_with_table_inspection_node_is_valid():
    payload = _extraction_workflow_payload(structured=True)
    payload["work_item"] = {"work_item_id": "item-1", "exact_source_files": ["/tmp/article/supplementary/table.xlsx"]}
    planner_runtime = FakeLLMRuntime(default_content=json.dumps(payload))
    planner = DynamicWorkflowPlanner(
        planner_llm=planner_runtime,
        config=_config(
            allowed_tool_names=["read_text", "write_report", "inspect_table"],
            metadata={"extraction_task": True},
            max_planner_retries=0,
        ),
        tool_runtime=_tool_runtime("read_text", "write_report", "inspect_table"),
        skill_backend=FakeSkillBackend(skills=[]),
        available_llm_backend_ids={"gpt-planner-nano", "openrouter-qwen30b"},
    )

    outcome = planner.plan(request=_request(goal="Extract structured records."))

    assert outcome.spec is not None
    assert outcome.validation_report.valid is True


def _config(**updates):
    payload = {
        "enabled": True,
        "mode": "dynamic",
        "scope": "per_work_item",
        "planner_backend": {"backend_id": "gpt-planner-nano"},
        "default_worker_backend": {"backend_id": "openrouter-qwen30b"},
        "allowed_tool_names": ["read_text", "write_report"],
        "max_planner_retries": 0,
        "fallback_to_static": True,
    }
    payload.update(updates)
    return DynamicSubagentsConfig.model_validate(payload)


def _workflow_payload(tool: str = "read_text"):
    return {
        "workflow_id": "wf-text",
        "work_item_id": "item-1",
        "task_summary": "Process a text-only item.",
        "article_context_summary": "main text only",
        "dynamic_subagents": [
            {
                "subagent_id": "agent-text",
                "role_name": "TextContextAgent",
                "goal": "Read text context.",
                "system_prompt": "Read text context.",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
                "allowed_tools": [tool],
                "artifact_outputs": ["context.json"],
            }
        ],
        "workflow_nodes": [{"node_id": "node-text", "subagent_id": "agent-text"}],
        "workflow_edges": [],
        "artifact_contracts": {"context.json": {"type": "object"}},
        "validation_rules": ["output must cite evidence"],
        "planner_rationale_summary": "A text-only item needs a text context agent.",
    }


def _request(goal: str = "Process synthetic documents."):
    return TaskRequest(
        task_id="task-1",
        origin=TaskOrigin.HUMAN,
        purpose=TaskPurpose.SCIENCE,
        goal=goal,
    )


def _tool_runtime(*names: str) -> ToolRuntime:
    registry = ToolRegistry()
    for name in names:
        registry.register(ToolSpec(name=name, description=name, parameters_schema={}), lambda args: "ok")
    return ToolRuntime(registry)


def _extraction_workflow_payload(*, structured: bool = False):
    tools = ["read_text", "write_report"]
    if structured:
        tools.append("inspect_table")
    agents = [
        {
            "subagent_id": "context",
            "role_name": "TextContextAgent",
            "goal": "Survey source context.",
            "system_prompt": "Survey source context.",
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"},
            "allowed_tools": ["read_text", "write_report"],
            "artifact_outputs": ["context.json"],
        },
        {
            "subagent_id": "extract",
            "role_name": "EvidenceExtractionAgent",
            "goal": "Extract candidate records from source evidence.",
            "system_prompt": "Extract candidate records from source evidence.",
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"},
            "allowed_tools": tools,
            "artifact_inputs": ["context.json"],
            "artifact_outputs": ["candidate_records.json"],
        },
        {
            "subagent_id": "validate",
            "role_name": "EvidenceValidatorAgent",
            "goal": "Validate candidate records.",
            "system_prompt": "Validate candidate records.",
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"},
            "allowed_tools": ["read_text", "write_report"],
            "artifact_inputs": ["candidate_records.json"],
            "artifact_outputs": ["validated_records.json"],
        },
        {
            "subagent_id": "writer",
            "role_name": "SchemaWriterAgent",
            "goal": "Write final records.",
            "system_prompt": "Write final records.",
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"},
            "allowed_tools": ["write_report"],
            "artifact_inputs": ["validated_records.json"],
            "artifact_outputs": ["final_records.jsonl"],
        },
    ]
    if structured:
        agents.insert(
            1,
            {
                "subagent_id": "table",
                "role_name": "TableTriageAgent",
                "goal": "Inspect workbook or table sources.",
                "system_prompt": "Inspect workbook or table sources.",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
                "allowed_tools": ["inspect_table", "write_report"],
                "artifact_inputs": ["context.json"],
                "artifact_outputs": ["candidate_tables.json"],
            },
        )
    nodes = [
        {"node_id": "context", "subagent_id": "context", "output_artifacts": ["context.json"]},
    ]
    if structured:
        nodes.append(
            {
                "node_id": "table",
                "subagent_id": "table",
                "input_artifacts": ["context.json"],
                "output_artifacts": ["candidate_tables.json"],
                "dependencies": ["context"],
            }
        )
        extract_inputs = ["candidate_tables.json"]
        extract_deps = ["table"]
    else:
        extract_inputs = ["context.json"]
        extract_deps = ["context"]
    nodes.extend(
        [
            {
                "node_id": "extract",
                "subagent_id": "extract",
                "input_artifacts": extract_inputs,
                "output_artifacts": ["candidate_records.json"],
                "dependencies": extract_deps,
            },
            {
                "node_id": "validate",
                "subagent_id": "validate",
                "input_artifacts": ["candidate_records.json"],
                "output_artifacts": ["validated_records.json"],
                "dependencies": ["extract"],
            },
            {
                "node_id": "write",
                "subagent_id": "writer",
                "input_artifacts": ["validated_records.json"],
                "output_artifacts": ["final_records.jsonl"],
                "dependencies": ["validate"],
            },
        ]
    )
    return {
        "workflow_id": "wf-extract",
        "work_item_id": "item-1",
        "task_summary": "Extract structured records from source evidence.",
        "article_context_summary": "main text only" if not structured else "main text plus supplementary workbook",
        "dynamic_subagents": agents,
        "workflow_nodes": nodes,
        "workflow_edges": [],
        "artifact_contracts": {"final_records.jsonl": {"type": "jsonl"}},
        "validation_rules": ["records must be source grounded"],
        "planner_rationale_summary": "Extraction requires survey, extraction, validation, and writing.",
    }
