from __future__ import annotations

import json
from pathlib import Path

from evolab.backends.llm.fake import FakeLLMRuntime
from evolab.backends.memory import NullMemoryBackend
from evolab.backends.skills import FakeSkillBackend
from evolab.config.task_config import TaskConfig
from evolab.contracts.common import RuntimePolicy
from evolab.contracts.dynamic_workflow import DynamicSubagentsConfig
from evolab.contracts.llm import LLMRuntimeResponse, SubAgentAction
from evolab.contracts.retrieval import SkillItem
from evolab.contracts.task import TaskOrigin, TaskPurpose, TaskRequest
from evolab.contracts.tools import ToolSpec
from evolab.registries.lab_state import FileLabStateRegistry
from evolab.registries.task import FileTaskRegistry
from evolab.registries.trajectory import FileTrajectoryRegistry
from evolab.runtime.prompt_builder import PromptBuilder
from evolab.runtime.task_runtime import TaskRuntime, _dynamic_work_items_for_request
from evolab.tools.runtime import ToolRegistry, ToolRuntime


def test_per_work_item_dynamic_planning_creates_different_workflows(tmp_path: Path):
    lab_root = tmp_path / "lab"
    runtime = _runtime(
        lab_root,
        planner_responses=[_text_only_workflow(), _workbook_workflow()],
        worker_response_count=7,
    )
    request = TaskRequest(
        task_id="task-per-item",
        origin=TaskOrigin.HUMAN,
        purpose=TaskPurpose.SCIENCE,
        goal="Process two synthetic document items.",
        metadata={
            "work_items": [
                {"work_item_id": "text-only", "files": ["main_text.md"]},
                {"work_item_id": "workbook", "files": ["main_text.md", "supplement.xlsx"]},
            ]
        },
    )

    result = runtime.run(request)

    assert result["execution_mode"] == "dynamic"
    assert result["status"] == "completed"
    workflows = result["dynamic_workflows"]
    assert [workflow["work_item_id"] for workflow in workflows] == ["text-only", "workbook"]
    first_roles = [run["role"] for run in workflows[0]["runs"]]
    second_roles = [run["role"] for run in workflows[1]["runs"]]
    assert first_roles == ["TextContextAgent", "EvidenceWriterAgent"]
    assert second_roles == [
        "TextContextAgent",
        "WorkbookInspectorAgent",
        "TableTriageAgent",
        "EvidenceValidatorAgent",
        "SchemaWriterAgent",
    ]
    report_path = lab_root / "dynamic_workflows" / "task-per-item" / "per_work_item_demo_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "text_only_roles": first_roles,
                "workbook_roles": second_roles,
                "workflow_ids": [workflow["workflow_id"] for workflow in workflows],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    assert report_path.exists()


def test_dynamic_work_items_parse_article_package_context_from_task_goal():
    article_root = "/tmp/example article package"
    request = TaskRequest(
        task_id="task-per-item",
        origin=TaskOrigin.HUMAN,
        purpose=TaskPurpose.SCIENCE,
        goal=(
            "1. work_item_id: item_a\n"
            f"   article_package: {article_root}\n"
            "   exact_source_files:\n"
            f"     - {article_root}/manifest.json\n"
            f"     - {article_root}/main_text.md\n"
        ),
    )
    policy_metadata = {
        "work_item_routing": {
            "enabled": True,
            "executor_roles": ["ExecAgent"],
            "reviewer_roles": ["CriticAgent"],
            "finalizer_roles": ["WriteAgent"],
            "required_work_item_ids": ["item_a"],
        }
    }

    items = _dynamic_work_items_for_request(request, policy_metadata, "per_work_item")

    assert items == [
        {
            "work_item_id": "item_a",
            "article_package": article_root,
            "lab_path": article_root,
            "exact_source_files": [f"{article_root}/manifest.json", f"{article_root}/main_text.md"],
            "source_files": [f"{article_root}/manifest.json", f"{article_root}/main_text.md"],
        }
    ]


def _runtime(lab_root: Path, *, planner_responses: list[dict], worker_response_count: int) -> TaskRuntime:
    registry = ToolRegistry()
    for name in ["read_text", "inspect_excel_workbook", "inspect_table", "write_report"]:
        registry.register(ToolSpec(name=name, description=name, parameters_schema={}), lambda args: "ok")
    return TaskRuntime(
        task_config=TaskConfig(
            task_id="task-per-item",
            goal="Process two synthetic document items.",
            dynamic_subagents=DynamicSubagentsConfig.model_validate(
                {
                    "enabled": True,
                    "mode": "dynamic",
                    "scope": "per_work_item",
                    "planner_backend": {"backend_id": "planner"},
                    "default_worker_backend": {"backend_id": "worker"},
                    "allowed_tool_names": ["read_text", "inspect_excel_workbook", "inspect_table", "write_report"],
                    "max_planner_retries": 0,
                }
            ),
            runtime_policy=RuntimePolicy(max_tool_steps=1, metadata={}),
        ),
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
        task_registry=FileTaskRegistry(lab_root / "registries" / "task"),
        trajectory_registry=FileTrajectoryRegistry(lab_root / "registries" / "trajectory"),
        lab_state_registry=FileLabStateRegistry(lab_root / "registries" / "lab_state"),
        llm_runtimes={
            "planner": FakeLLMRuntime(
                responses=[
                    LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content=json.dumps(payload)))
                    for payload in planner_responses
                ]
            ),
            "worker": FakeLLMRuntime(
                responses=[
                    LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content=f"node {index} done"))
                    for index in range(worker_response_count)
                ]
            ),
        },
        memory_runtimes={"memory": NullMemoryBackend()},
        skill_runtimes={
            "skill": FakeSkillBackend(
                skills=[SkillItem(skill_id="skill.generic", name="Generic", content="Generic.", required_tools=[])]
            )
        },
    )


def _text_only_workflow():
    return _workflow(
        "wf-text-only",
        "text-only",
        [
            ("text", "TextContextAgent", ["read_text"]),
            ("writer", "EvidenceWriterAgent", ["write_report"]),
        ],
    )


def _workbook_workflow():
    return _workflow(
        "wf-workbook",
        "workbook",
        [
            ("text", "TextContextAgent", ["read_text"]),
            ("workbook", "WorkbookInspectorAgent", ["inspect_excel_workbook"]),
            ("triage", "TableTriageAgent", ["inspect_table"]),
            ("validator", "EvidenceValidatorAgent", ["read_text"]),
            ("writer", "SchemaWriterAgent", ["write_report"]),
        ],
    )


def _workflow(workflow_id: str, work_item_id: str, agents: list[tuple[str, str, list[str]]]):
    dynamic_subagents = []
    nodes = []
    edges = []
    previous_node_id = None
    for index, (subagent_id, role_name, tools) in enumerate(agents, start=1):
        dynamic_subagents.append(
            {
                "subagent_id": subagent_id,
                "role_name": role_name,
                "goal": f"{role_name} goal",
                "system_prompt": f"{role_name} prompt",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
                "allowed_tools": tools,
            }
        )
        node_id = f"node-{index}-{subagent_id}"
        nodes.append(
            {
                "node_id": node_id,
                "subagent_id": subagent_id,
                "dependencies": [previous_node_id] if previous_node_id else [],
            }
        )
        if previous_node_id:
            edges.append({"source_node_id": previous_node_id, "target_node_id": node_id, "relation": "depends_on"})
        previous_node_id = node_id
    return {
        "workflow_id": workflow_id,
        "work_item_id": work_item_id,
        "task_summary": "Process synthetic item.",
        "article_context_summary": work_item_id,
        "dynamic_subagents": dynamic_subagents,
        "workflow_nodes": nodes,
        "workflow_edges": edges,
        "artifact_contracts": {},
        "validation_rules": ["complete node outputs"],
        "planner_rationale_summary": f"{work_item_id} needs {len(agents)} dynamic roles.",
    }
