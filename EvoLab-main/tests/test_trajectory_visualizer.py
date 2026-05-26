import json
import re
from pathlib import Path

from evolab.cli import main
from evolab.contracts.common import ArtifactRef, Message
from evolab.contracts.dispatch import DispatchAction, DispatchDecision
from evolab.contracts.evolution import LLMEvolutionMode, LLMEvolutionResult
from evolab.contracts.records import (
    EvolutionRunRecord,
    LLMCallRecord,
    MetaAgentRunRecord,
    SubagentRunRecord,
    ToolCallTrajectoryRecord,
    TrajectoryEventRecord,
)
from evolab.contracts.retrieval import (
    MemoryBundle,
    MemoryItem,
    RetrievalRequest,
    SkillBundle,
    SkillItem,
)
from evolab.contracts.task import TaskOrigin, TaskPurpose
from evolab.contracts.tools import ToolCall, ToolCallRecord, ToolResult
from evolab.registries.trajectory import FileTrajectoryRegistry
from evolab.runtime.trajectory_visualizer import visualize_trajectory


def test_visualize_trajectory_writes_interactive_dashboard(tmp_path: Path):
    lab_root = tmp_path / "lab"
    registry = _seed_trajectory(lab_root / "registries" / "trajectory")

    result = visualize_trajectory(lab_root=lab_root)

    output_path = Path(result["output_path"])
    assert output_path == lab_root / "artifacts" / "trajectory_view.html"
    assert result["counts"] == {
        "meta_agent_runs": 1,
        "subagent_runs": 1,
        "llm_calls": 1,
        "tool_calls": 1,
        "events": 1,
        "evolution_runs": 1,
    }
    html = output_path.read_text(encoding="utf-8")
    assert 'id="trajectory-data"' in html
    assert '<section class="metrics" id="metrics" aria-label="Trajectory summary"><div class="metric">' in html
    assert "Static fallback: scripts not running" in html
    assert "Interactive ready" in html
    assert "Copy JSON" in html
    assert "Subagent Runs" in html
    assert "Evolution Overview" in html
    assert "MetaAgent Routing" in html
    assert "Skill And Memory" in html
    assert "Tool Calls" in html
    assert "LLM Calls" in html
    assert "solver" in html
    assert "skill.extract.v1" in html
    assert "extract_records" in html
    assert "retrieved memory" in html
    assert registry.list_subagent_runs()[0].run_ref in html
    dashboard_data = _dashboard_data_from_html(html)
    assert dashboard_data["subagentRuns"][0]["role"] == "solver"
    assert dashboard_data["subagentRuns"][0]["skillCount"] == 1
    assert dashboard_data["toolCalls"][0]["tool"] == "extract_records"
    assert dashboard_data["llmCalls"][0]["model"] == "model-x"
    assert "evolutionOverview" in dashboard_data


def test_visualize_trajectory_cli_accepts_trajectory_dir(tmp_path: Path, capsys):
    trajectory_dir = tmp_path / "trajectory"
    _seed_trajectory(trajectory_dir)
    output_path = tmp_path / "view.html"

    exit_code = main(
        [
            "visualize-trajectory",
            "--trajectory-dir",
            str(trajectory_dir),
            "--output",
            str(output_path),
            "--title",
            "Smoke Trajectory",
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    assert "Smoke Trajectory" in output_path.read_text(encoding="utf-8")
    assert str(output_path) in capsys.readouterr().out


def test_visualize_trajectory_includes_dynamic_subagents(tmp_path: Path):
    lab_root = tmp_path / "lab"
    _seed_trajectory(lab_root / "registries" / "trajectory")
    workflow_dir = (
        lab_root
        / "dynamic_workflows"
        / "task-1"
        / "item-1__workflow-1"
    )
    workflow_dir.mkdir(parents=True)
    dynamic_subagents = [
        {
            "subagent_id": "solver_agent",
            "role_name": "solver",
            "goal": "Extract records with source-grounded evidence.",
            "system_prompt": "Use only assigned tools and cite evidence.",
            "allowed_tools": ["read_text", "extract_records"],
            "required_skills": ["skill.extract.v1"],
            "artifact_inputs": ["source.md"],
            "artifact_outputs": ["records.jsonl"],
            "llm_backend_id": "qwen-local",
            "max_turns": 4,
            "max_tool_calls": 6,
        }
    ]
    workflow_spec = {
        "workflow_id": "workflow-1",
        "work_item_id": "item-1",
        "task_summary": "Extract evidence-backed records.",
        "planner_rationale_summary": "Use a focused dynamic solver.",
        "dynamic_subagents": dynamic_subagents,
        "workflow_nodes": [
            {
                "node_id": "extract",
                "subagent_id": "solver_agent",
                "input_artifacts": ["source.md"],
                "output_artifacts": ["records.jsonl"],
            }
        ],
        "workflow_edges": [],
    }
    workflow_trace = {
        "workflow_id": "workflow-1",
        "work_item_id": "item-1",
        "status": "completed",
        "execution_mode": "dynamic",
        "planner_backend_id": "planner-nano",
        "default_worker_backend_id": "qwen-local",
        "node_results": [
            {
                "node_id": "extract",
                "role": "solver",
                "run_ref": "subagent-1",
                "status": "completed",
                "artifact_refs": [
                    {
                        "uri": "/tmp/records.jsonl",
                        "type": "dataset",
                        "metadata": {"filename": "records.jsonl"},
                    }
                ],
            }
        ],
    }
    validation_report = {
        "valid": True,
        "planner_backend_id": "planner-nano",
        "default_worker_backend_id": "qwen-local",
        "errors": [],
        "warnings": [],
        "metadata": {"workflow_id": "workflow-1", "work_item_id": "item-1"},
    }
    (workflow_dir / "dynamic_subagents.json").write_text(
        json.dumps(dynamic_subagents),
        encoding="utf-8",
    )
    (workflow_dir / "dynamic_workflow_spec.json").write_text(
        json.dumps(workflow_spec),
        encoding="utf-8",
    )
    (workflow_dir / "dynamic_workflow_trace.json").write_text(
        json.dumps(workflow_trace),
        encoding="utf-8",
    )
    (workflow_dir / "planner_validation_report.json").write_text(
        json.dumps(validation_report),
        encoding="utf-8",
    )
    (workflow_dir / "dynamic_subagent_records.jsonl").write_text(
        json.dumps(workflow_trace["node_results"][0]) + "\n",
        encoding="utf-8",
    )

    result = visualize_trajectory(lab_root=lab_root)

    assert result["counts"]["dynamic_workflows"] == 1
    assert result["counts"]["dynamic_subagent_specs"] == 1
    assert result["counts"]["dynamic_subagent_records"] == 1
    html = Path(result["output_path"]).read_text(encoding="utf-8")
    assert "Dynamic Subagents" in html
    assert "Evolution Overview" in html
    assert "extract_records" in html
    assert "workflow-1" in html
    dashboard_data = _dashboard_data_from_html(html)
    dynamic_subagent = dashboard_data["dynamicSubagents"][0]
    assert dynamic_subagent["role"] == "solver"
    assert dynamic_subagent["nodeId"] == "extract"
    assert dynamic_subagent["allowedTools"] == ["read_text", "extract_records"]
    overview = dashboard_data["evolutionOverview"][0]
    assert overview["workItemId"] == "item-1"
    assert overview["nodeCount"] == 1
    assert overview["nodes"][0]["selectedSkills"] == ["skill.extract.v1 - Extraction Skill"]
    assert overview["nodes"][0]["memoryUpdateStatus"] == "updated"
    assert dashboard_data["subagentRuns"][0]["dynamicWorkflowId"] == "workflow-1"
    assert dashboard_data["subagentRuns"][0]["dynamicNode"] == "extract"


def _seed_trajectory(root: Path) -> FileTrajectoryRegistry:
    registry = FileTrajectoryRegistry(root)
    meta = MetaAgentRunRecord(
        run_ref="meta-1",
        task_id="task-1",
        decision=DispatchDecision(
            action=DispatchAction.RUN_SUBAGENT,
            target_role="solver",
            instruction="Extract candidate records.",
            metadata={"work_item_id": "item-1"},
        ),
    )
    subagent = SubagentRunRecord(
        run_ref="subagent-1",
        task_id="task-1",
        task_origin=TaskOrigin.HUMAN,
        task_purpose=TaskPurpose.SCIENCE,
        stage_index=0,
        role="solver",
        instruction="Extract candidate records.",
        retrieval_request=RetrievalRequest(task_id="task-1", role="solver", query="extract"),
        memory_bundle=MemoryBundle(
            backend_id="memory-local",
            items=[MemoryItem(memory_id="mem-1", content="retrieved memory", score=0.8)],
        ),
        skill_bundle=SkillBundle(
            backend_id="skill-local",
            skills=[
                SkillItem(
                    skill_id="skill.extract.v1",
                    name="Extraction Skill",
                    content="Use evidence-backed extraction.",
                    required_tools=["extract_records"],
                )
            ],
            required_tools=["extract_records"],
        ),
        prompt_messages=[Message(role="user", content="Extract.")],
        llm_call_refs=["llm-1"],
        llm_backend_id="llm-api",
        artifact_refs=[ArtifactRef(uri="/tmp/records.jsonl", type="dataset")],
        metadata={
            "status": "completed",
            "skill_update_result": {"status": "updated"},
            "memory_update_result": {"status": "updated"},
            "node_execution_records": [
                {
                    "node_id": "node-1",
                    "skill_id": "skill.extract.v1",
                    "status": "completed",
                    "tool_calls": [{"name": "extract_records"}],
                    "output_summary": "records extracted",
                }
            ],
        },
    )
    llm_call = LLMCallRecord(
        call_ref="llm-1",
        run_ref=subagent.run_ref,
        backend_id="llm-api",
        model="model-x",
        input_messages=[Message(role="user", content="Extract.")],
        output_messages=[Message(role="assistant", content="Done.")],
        metadata={"runtime_stage": "subagent", "action": "final_answer"},
    )
    tool_call = ToolCallTrajectoryRecord(
        record_ref="tool-1",
        run_ref=subagent.run_ref,
        task_id="task-1",
        tool_call_id="call-1",
        tool_name="extract_records",
        role="solver",
        runtime_stage="subagent",
        step_index=0,
        record=ToolCallRecord(
            tool_call=ToolCall(call_id="call-1", name="extract_records", arguments={"path": "input.md"}),
            result=ToolResult(call_id="call-1", status="ok", content="wrote records"),
        ),
    )
    event = TrajectoryEventRecord(
        event_ref="event-1",
        event_type="subagent_started",
        subject_type="subagent",
        subject_ref=subagent.run_ref,
        task_id="task-1",
        run_ref=subagent.run_ref,
        metadata={"role": "solver"},
    )
    evolution = EvolutionRunRecord(
        run_ref="evo-1",
        mode=LLMEvolutionMode.BASICS,
        backend_id="llm-api",
        result_status="not_recommended",
        result=LLMEvolutionResult(status="not_recommended", recommend_for_promotion=False),
        training_trajectory_refs=[subagent.run_ref],
    )
    registry.save_meta_agent_run(meta)
    registry.save_subagent_run(subagent)
    registry.save_llm_call(llm_call)
    registry.save_tool_call_record(tool_call)
    registry.save_event(event)
    registry.save_evolution_run(evolution)
    return registry


def _dashboard_data_from_html(html: str) -> dict:
    match = re.search(
        r'<script id="trajectory-data" type="application/json">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))
