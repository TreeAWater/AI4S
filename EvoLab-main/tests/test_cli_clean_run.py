import json
import shutil
from pathlib import Path

import pytest

from evolab.backends.embeddings import FakeEmbeddingBackend
from evolab.backends.llm import ApiLLMBackend, LocalTrainableLLMBackend
from evolab.backends.llm.api import OpenAIChatCompletionsRuntime, OpenAIResponsesRuntime
from evolab.backends.memory import MethodMemoryBackend, NullMemoryBackend
from evolab.backends.memory.methods.everos import EverOSMemoryMethod
from evolab.backends.memory.methods.mem0 import Mem0MemoryMethod
from evolab.backends.skills import GraphSkillBackend
from evolab.backends.trainers import SFTTrainer
from evolab.cli import (
    _compile_experiment_config,
    _copy_agents_config,
    _build_embedding_backends,
    _build_evolution_backends,
    _build_llm_backends,
    _build_memory_backends,
    _build_skill_backends,
    run_clean_demo,
    run_export_sft,
    run_train_sft,
)
from evolab.config.agents import load_agents_file, render_agents_markdown
from evolab.config.task_config import BackendBinding, RoleSpec, TaskConfig
from evolab.contracts.common import ArtifactRef, Message
from evolab.contracts.llm import LLMGenerationConfig
from evolab.contracts.state import BackendStateRecord
from evolab.lab.layout import LabLayout
from evolab.registries.backend_state import FileBackendStateRegistry
from evolab.registries.trajectory import FileTrajectoryRegistry


ROOT = Path(__file__).resolve().parents[1]
DEV_CONFIGS = ROOT / "dev" / "configs"
DEV_SEED_GRAPH = DEV_CONFIGS / "skills" / "graphs" / "scientific_ie_seed_graph_v1.json"


def _layout(lab_root: Path) -> LabLayout:
    return LabLayout(lab_root)


def _trajectory_registry(lab_root: Path) -> FileTrajectoryRegistry:
    return FileTrajectoryRegistry(_layout(lab_root).registries_dir / "trajectory")


def _backend_state_registry(lab_root: Path) -> FileBackendStateRegistry:
    return FileBackendStateRegistry(_layout(lab_root).registries_dir / "backend_state")


def _clear_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "ALL_PROXY",
        "all_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
    ):
        monkeypatch.delenv(key, raising=False)


def _write_agents_file(path: Path, *roles: RoleSpec) -> None:
    path.write_text(render_agents_markdown(list(roles)), encoding="utf-8")


def test_compile_experiment_config_defaults_to_dynamic_role_pool(tmp_path: Path):
    config = {
        "lab_root": str(tmp_path / "lab"),
        "task": "Inspect inputs and write a report.",
        "meta_agent": {"system_prompt": "Manage roles and route JSON only."},
        "backends": {
            "llm": {"fake-llm": {"type": "fake", "responses": []}},
            "memory": {"mem0-meta-memory": {"type": "null"}, "mem0-task-memory": {"type": "null"}},
            "skill": {"fake-skill": {"type": "fake", "skills": []}},
        },
    }

    compiled = _compile_experiment_config(config, config_path=tmp_path / "experiment.yaml")

    task_config = compiled["task_config"]
    assert task_config["agents_ref"] == "agents.md"
    assert task_config["dynamic_subagents"]["enabled"] is True
    assert task_config["dynamic_subagents"]["planner_backend"]["backend_id"] == "fake-llm"
    assert task_config["dynamic_subagents"]["default_worker_backend"]["backend_id"] == "fake-llm"
    assert task_config["task_memory_backend"]["backend_id"] == "mem0-task-memory"
    assert task_config["roles"]["GeneralistAgent"].get("agent_memory_backend") is None
    assert "subagents" not in compiled


def test_compile_experiment_config_omitted_agents_ref_ignores_sibling_agents_md(tmp_path: Path):
    _write_agents_file(
        tmp_path / "agents.md",
        RoleSpec(
            name="SiblingAgent",
            system_prompt="Loaded only when agents_ref is explicit.",
            llm_backend=BackendBinding(backend_id="fake-llm"),
        ),
    )
    base_config = {
        "lab_root": str(tmp_path / "lab"),
        "task": "Inspect inputs and write a report.",
        "meta_agent": {"system_prompt": "Route JSON only."},
        "backends": {
            "llm": {"fake-llm": {"type": "fake", "responses": []}},
            "memory": {"mem0-task-memory": {"type": "null"}},
            "skill": {"fake-skill": {"type": "fake", "skills": []}},
        },
    }

    default_compiled = _compile_experiment_config(base_config, config_path=tmp_path / "experiment.yaml")
    seed_compiled = _compile_experiment_config(
        {
            **base_config,
            "seed_agents": {"SeedAgent": {"system_prompt": "Use the explicit seed role."}},
        },
        config_path=tmp_path / "seed.yaml",
    )

    assert list(default_compiled["task_config"]["roles"]) == ["GeneralistAgent"]
    assert list(seed_compiled["task_config"]["roles"]) == ["SeedAgent"]
    assert default_compiled["task_config"]["runtime_policy"]["metadata"]["agents_ref_materialized_from_seed"] is True


def test_compile_experiment_config_explicit_agents_ref_loads_sibling_agents_md(tmp_path: Path):
    _write_agents_file(
        tmp_path / "agents.md",
        RoleSpec(
            name="SiblingAgent",
            system_prompt="Loaded when agents_ref is explicit.",
            llm_backend=BackendBinding(backend_id="fake-llm"),
        ),
    )
    config = {
        "lab_root": str(tmp_path / "lab"),
        "task": "Inspect inputs and write a report.",
        "agents_ref": "agents.md",
        "meta_agent": {"system_prompt": "Route JSON only."},
        "backends": {
            "llm": {"fake-llm": {"type": "fake", "responses": []}},
            "memory": {"mem0-task-memory": {"type": "null"}},
            "skill": {"fake-skill": {"type": "fake", "skills": []}},
        },
    }

    compiled = _compile_experiment_config(config, config_path=tmp_path / "experiment.yaml")

    assert list(compiled["task_config"]["roles"]) == ["SiblingAgent"]
    assert compiled["task_config"]["runtime_policy"]["metadata"]["agents_ref_materialized_from_seed"] is False


def test_compile_experiment_config_explicit_missing_agents_ref_raises(tmp_path: Path):
    config = {
        "lab_root": str(tmp_path / "lab"),
        "task": "Inspect inputs and write a report.",
        "agents_ref": "missing-agents.md",
        "meta_agent": {"system_prompt": "Route JSON only."},
        "backends": {
            "llm": {"fake-llm": {"type": "fake", "responses": []}},
            "memory": {"mem0-task-memory": {"type": "null"}},
            "skill": {"fake-skill": {"type": "fake", "skills": []}},
        },
    }

    with pytest.raises(ValueError, match="agents_ref"):
        _compile_experiment_config(config, config_path=tmp_path / "experiment.yaml")


def test_compile_experiment_config_rejects_top_level_subagents(tmp_path: Path):
    config = {
        "lab_root": str(tmp_path / "lab"),
        "task": "Inspect inputs and write a report.",
        "meta_agent": {"system_prompt": "Route JSON only."},
        "subagents": {"LegacyAgent": {"system_prompt": "Legacy."}},
        "backends": {
            "llm": {"fake-llm": {"type": "fake", "responses": []}},
            "memory": {"mem0-task-memory": {"type": "null"}},
            "skill": {"fake-skill": {"type": "fake", "skills": []}},
        },
    }

    with pytest.raises(
        ValueError,
        match="subagents is no longer a supported execution entry; use agents_ref or seed_agents",
    ):
        _compile_experiment_config(config, config_path=tmp_path / "experiment.yaml")


def test_compile_experiment_config_validates_dynamic_allowed_worker_backend_ids(tmp_path: Path):
    config = {
        "lab_root": str(tmp_path / "lab"),
        "task": "Inspect inputs and write a report.",
        "meta_agent": {"system_prompt": "Route JSON only."},
        "dynamic_subagents": {
            "enabled": True,
            "planner_backend": {"backend_id": "fake-llm"},
            "default_worker_backend": {"backend_id": "fake-llm"},
            "allowed_worker_backend_ids": ["missing-worker"],
        },
        "backends": {
            "llm": {"fake-llm": {"type": "fake", "responses": []}},
            "memory": {"mem0-task-memory": {"type": "null"}},
            "skill": {"fake-skill": {"type": "fake", "skills": []}},
        },
    }

    with pytest.raises(ValueError, match="missing-worker"):
        _compile_experiment_config(config, config_path=tmp_path / "experiment.yaml")


def test_compile_experiment_config_defaults_explicit_dynamic_allowed_tools(tmp_path: Path):
    config = {
        "lab_root": str(tmp_path / "lab"),
        "task": "Read inputs/source.txt.",
        "allowed_tools": ["read_text"],
        "meta_agent": {"system_prompt": "Route JSON only."},
        "dynamic_subagents": {
            "enabled": True,
            "mode": "dynamic",
            "planner_backend": {"backend_id": "fake-llm"},
            "default_worker_backend": {"backend_id": "fake-llm"},
            "require_output_schema": False,
        },
        "backends": {
            "llm": {"fake-llm": {"type": "fake", "responses": []}},
            "memory": {"mem0-task-memory": {"type": "null"}},
            "skill": {"fake-skill": {"type": "fake", "skills": []}},
        },
    }

    compiled = _compile_experiment_config(config, config_path=tmp_path / "experiment.yaml")

    assert compiled["task_config"]["dynamic_subagents"]["allowed_tool_names"] == ["read_text"]


def test_copy_agents_config_rejects_missing_source_without_roles(tmp_path: Path):
    task_config = TaskConfig(task_id="task", goal="Goal.", agents_ref="missing-agents.md")

    with pytest.raises(ValueError, match="agents_ref does not exist and no roles are available"):
        _copy_agents_config(
            task_config=task_config,
            config_dir=tmp_path,
            lab_root=tmp_path / "lab",
        )

    assert not (tmp_path / "lab" / "configs" / "missing-agents.md").exists()


def test_clean_run_materializes_seed_agents_md(tmp_path: Path):
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        """
lab_root: ignored
task: Write a short report.
meta_agent:
  system_prompt: Return role-pool decisions as JSON.
backends:
  llm:
    fake-llm:
      type: fake
      responses:
        - action:
            action: final_answer
            content: '{"route":"END","instruction":"No generated tool needed.","metadata":{"no_generated_tool_reason":"Built-in tools are sufficient."}}'
        - action:
            action: final_answer
            content: '{"route":"END","instruction":"Use the existing role pool.","metadata":{"no_role_pool_update_reason":"Seed role pool is sufficient."}}'
        - action:
            action: final_answer
            content: '{"workflow_id":"wf-seed","task_summary":"Write a short report.","article_context_summary":"none","dynamic_subagents":[{"subagent_id":"writer","role_name":"GeneralistAgent","goal":"Write a short report.","system_prompt":"Write the report.","input_schema":{"type":"object"},"output_schema":{"type":"object"},"allowed_tools":[]}],"workflow_nodes":[{"node_id":"node-writer","subagent_id":"writer"}],"workflow_edges":[],"artifact_contracts":{},"validation_rules":[],"planner_rationale_summary":"One generalist worker is sufficient."}'
        - action:
            action: final_answer
            content: "Short report complete."
  memory:
    mem0-meta-memory:
      type: null
    mem0-task-memory:
      type: null
  skill:
    fake-skill:
      type: fake
      skills: []
""",
        encoding="utf-8",
    )

    result = run_clean_demo(config_path, lab_root=tmp_path / "lab")

    assert result["status"] == "completed"
    assert result["final_answer"] == "Short report complete."
    assert [run["role"] for run in result["runs"]] == ["GeneralistAgent"]

    agents_path = _layout(tmp_path / "lab").configs_dir / "agents.md"
    assert agents_path.exists()
    assert "GeneralistAgent" in agents_path.read_text(encoding="utf-8")
    roles = load_agents_file(agents_path)
    assert roles["GeneralistAgent"].llm_backend.backend_id == "fake-llm"
    assert roles["GeneralistAgent"].agent_memory_backend is None


def test_clean_run_omitted_agents_ref_materializes_seed_despite_sibling_agents_md(tmp_path: Path):
    _write_agents_file(
        tmp_path / "agents.md",
        RoleSpec(
            name="WrongAgent",
            system_prompt="This sibling file must not be copied for implicit agents_ref.",
            llm_backend=BackendBinding(backend_id="fake-llm"),
            agent_memory_backend=BackendBinding(backend_id="mem0-task-memory"),
        ),
    )
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        """
lab_root: ignored
task: Write a short report.
meta_agent:
  system_prompt: Return role-pool decisions as JSON.
backends:
  llm:
    fake-llm:
      type: fake
      responses:
        - action:
            action: final_answer
            content: '{"route":"END","instruction":"No generated tool needed.","metadata":{"no_generated_tool_reason":"Built-in tools are sufficient."}}'
        - action:
            action: final_answer
            content: '{"route":"END","instruction":"Use the existing role pool.","metadata":{"no_role_pool_update_reason":"Seed role pool is sufficient."}}'
        - action:
            action: final_answer
            content: '{"workflow_id":"wf-seed","task_summary":"Write a short report.","article_context_summary":"none","dynamic_subagents":[{"subagent_id":"writer","role_name":"GeneralistAgent","goal":"Write a short report.","system_prompt":"Write the report.","input_schema":{"type":"object"},"output_schema":{"type":"object"},"allowed_tools":[]}],"workflow_nodes":[{"node_id":"node-writer","subagent_id":"writer"}],"workflow_edges":[],"artifact_contracts":{},"validation_rules":[],"planner_rationale_summary":"One generalist worker is sufficient."}'
        - action:
            action: final_answer
            content: "Short report complete."
  memory:
    mem0-task-memory:
      type: null
  skill:
    fake-skill:
      type: fake
      skills: []
""",
        encoding="utf-8",
    )

    result = run_clean_demo(config_path, lab_root=tmp_path / "lab")

    assert result["status"] == "completed"
    roles = load_agents_file(_layout(tmp_path / "lab").configs_dir / "agents.md")
    assert "GeneralistAgent" in roles
    assert "WrongAgent" not in roles


def test_clean_run_explicit_agents_ref_renders_compiled_memory_compatible_roles(tmp_path: Path):
    _write_agents_file(
        tmp_path / "agents.md",
        RoleSpec(
            name="ExplicitAgent",
            system_prompt="Write the requested report.",
            llm_backend=BackendBinding(backend_id="fake-llm"),
        ),
    )
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        """
lab_root: ignored
task: Write a short report.
agents_ref: agents.md
meta_agent:
  system_prompt: Return role-pool decisions as JSON.
backends:
  llm:
    fake-llm:
      type: fake
      responses:
        - action:
            action: final_answer
            content: '{"route":"END","instruction":"No generated tool needed.","metadata":{"no_generated_tool_reason":"Built-in tools are sufficient."}}'
        - action:
            action: final_answer
            content: '{"route":"END","instruction":"Use the existing role pool.","metadata":{"no_role_pool_update_reason":"Explicit role pool is sufficient."}}'
        - action:
            action: final_answer
            content: '{"workflow_id":"wf-explicit","task_summary":"Write a short report.","article_context_summary":"none","dynamic_subagents":[{"subagent_id":"writer","role_name":"ExplicitAgent","goal":"Write a short report.","system_prompt":"Write the report.","input_schema":{"type":"object"},"output_schema":{"type":"object"},"allowed_tools":[]}],"workflow_nodes":[{"node_id":"node-writer","subagent_id":"writer"}],"workflow_edges":[],"artifact_contracts":{},"validation_rules":[],"planner_rationale_summary":"The explicit role pool supplies the worker."}'
        - action:
            action: final_answer
            content: "Explicit report complete."
  memory:
    mem0-task-memory:
      type: null
  skill:
    fake-skill:
      type: fake
      skills: []
""",
        encoding="utf-8",
    )

    result = run_clean_demo(config_path, lab_root=tmp_path / "lab")

    assert result["status"] == "completed"
    assert result["final_answer"] == "Explicit report complete."
    roles = load_agents_file(_layout(tmp_path / "lab").configs_dir / "agents.md")
    assert roles["ExplicitAgent"].agent_memory_backend is None


def test_compile_experiment_config_accepts_natural_language_task_and_route_pool(tmp_path: Path):
    config = {
        "lab_root": str(tmp_path / "lab"),
        "task": "Read the lab input and write a short report. Success means a report artifact exists.",
        "meta_agent": {
            "system_prompt": "Route to a subagent or END. Return JSON only.",
        },
        "seed_agents": {
            "SurveyAgent": {"system_prompt": "Survey available files."},
            "WriteAgent": {"system_prompt": "Write final artifacts."},
        },
        "backends": {
            "llm": {
                "fake-llm": {
                    "type": "fake",
                    "responses": [],
                }
            },
            "memory": {
                "mem0-agent-memory": {"type": "null"},
                "mem0-task-memory": {"type": "null"},
            },
            "skill": {"fake-skill": {"type": "fake", "skills": []}},
        },
    }

    compiled = _compile_experiment_config(config, config_path=tmp_path / "experiment.yaml")

    assert compiled["task"]["goal"] == config["task"]
    assert compiled["task"]["metadata"]["task_description"] == config["task"]
    task_config = compiled["task_config"]
    assert task_config["goal"] == config["task"]
    assert task_config["meta_agent"]["system_prompt"] == config["meta_agent"]["system_prompt"]
    assert task_config["meta_agent"]["llm_backend"]["backend_id"] == "fake-llm"
    assert list(task_config["roles"]) == ["SurveyAgent", "WriteAgent"]
    assert task_config["roles"]["SurveyAgent"].get("agent_memory_backend") is None
    assert task_config["roles"]["WriteAgent"].get("agent_memory_backend") is None
    assert task_config["roles"]["SurveyAgent"]["allowed_tools"] == [
        "list_files",
        "read_text",
        "inspect_file_metadata",
        "extract_sections",
        "search_text",
        "inspect_table",
        "read_table_slice",
        "inspect_excel_workbook",
        "read_excel_sheet",
        "detect_table_header",
        "normalize_table",
        "profile_table",
        "build_document_inventory",
        "discover_candidate_source_files",
        "discover_candidate_tables",
        "extract_candidate_rows",
        "build_candidate_records",
        "validate_candidate_records",
        "serialize_final_records",
        "json_schema_validate",
        "write_jsonl",
        "write_report",
    ]
    assert task_config["runtime_policy"]["enable_workflow_planning"] is True
    assert task_config["runtime_policy"]["metadata"]["route_contract"]["end_route"] == "END"
    assert task_config["runtime_policy"]["metadata"]["required_final_artifacts"] == []
    assert compiled["tools"]["scientific_ie"]["enabled"] is True


def test_compile_experiment_config_seed_agents_preserve_explicit_agent_memory_backend(tmp_path: Path):
    config = {
        "lab_root": str(tmp_path / "lab"),
        "task": "Inspect inputs and write a report.",
        "meta_agent": {"system_prompt": "Route JSON only."},
        "seed_agents": {
            "SurveyAgent": {
                "system_prompt": "Survey available files.",
                "agent_memory_backend": {"backend_id": "mem0-agent-memory"},
            },
            "WriteAgent": {"system_prompt": "Write final artifacts."},
        },
        "backends": {
            "llm": {"fake-llm": {"type": "fake", "responses": []}},
            "memory": {
                "mem0-agent-memory": {"type": "null"},
                "mem0-task-memory": {"type": "null"},
            },
            "skill": {"fake-skill": {"type": "fake", "skills": []}},
        },
    }

    compiled = _compile_experiment_config(config, config_path=tmp_path / "experiment.yaml")

    roles = compiled["task_config"]["roles"]
    assert roles["SurveyAgent"]["agent_memory_backend"]["backend_id"] == "mem0-agent-memory"
    assert roles["WriteAgent"].get("agent_memory_backend") is None


def test_compile_experiment_config_accepts_meta_agent_prompt_ref(tmp_path: Path):
    prompt_path = tmp_path / "meta_prompt.md"
    prompt_path.write_text("Route through the configured agents.", encoding="utf-8")
    config = {
        "task": "Solve the task.",
        "meta_agent": {
            "prompt_ref": "meta_prompt.md",
        },
        "seed_agents": {"Solver": {"system_prompt": "Solve."}},
        "backends": {
            "llm": {"fake-llm": {"type": "fake", "responses": []}},
            "memory": {
                "mem0-agent-memory": {"type": "null"},
                "mem0-task-memory": {"type": "null"},
            },
            "skill": {"fake-skill": {"type": "fake", "skills": []}},
        },
    }

    compiled = _compile_experiment_config(config, config_path=tmp_path / "experiment.yaml")

    meta_agent = compiled["task_config"]["meta_agent"]
    assert meta_agent["prompt_ref"] == "meta_prompt.md"
    assert meta_agent["system_prompt"] == ""


def test_compile_experiment_config_accepts_task_evaluator(tmp_path: Path):
    ground_truth_path = tmp_path / "ground_truth.json"
    ground_truth_path.write_text(json.dumps({"answer": "A"}), encoding="utf-8")
    config = {
        "task": "Solve the task.",
        "meta_agent": {"system_prompt": "Route JSON only."},
        "evaluator": {
            "system_prompt": "Evaluate after task completion.",
            "ground_truth_ref": "ground_truth.json",
            "rubric": "Exact answer match.",
        },
        "seed_agents": {"Solver": {"system_prompt": "Solve."}},
        "backends": {
            "llm": {"fake-llm": {"type": "fake", "responses": []}},
            "memory": {
                "mem0-agent-memory": {"type": "null"},
                "mem0-task-memory": {"type": "null"},
            },
            "skill": {"fake-skill": {"type": "fake", "skills": []}},
        },
    }

    compiled = _compile_experiment_config(config, config_path=tmp_path / "experiment.yaml")

    reflector = compiled["task_config"]["reflector"]
    assert reflector["name"] == "reflector"
    assert reflector["system_prompt"] == "Evaluate after task completion."
    assert reflector["ground_truth_ref"] == "ground_truth.json"
    assert reflector["rubric"] == "Exact answer match."
    assert reflector["llm_backend"]["backend_id"] == "fake-llm"


def test_compile_experiment_config_accepts_meta_agent_memory_backend(tmp_path: Path):
    config = {
        "lab_root": str(tmp_path / "lab"),
        "task": "Route one subagent and remember routing decisions.",
        "meta_agent": {
            "system_prompt": "Route JSON only.",
            "memory_backend": {"backend_id": "mem0-meta-memory", "state_ref": "meta-state-v1"},
        },
        "seed_agents": {"SurveyAgent": {"system_prompt": "Survey."}},
        "backends": {
            "llm": {"fake-llm": {"type": "fake", "responses": []}},
            "memory": {
                "mem0-meta-memory": {"type": "null"},
                "mem0-agent-memory": {"type": "null"},
                "mem0-task-memory": {"type": "null"},
            },
            "skill": {"fake-skill": {"type": "fake", "skills": []}},
        },
    }

    compiled = _compile_experiment_config(config, config_path=tmp_path / "experiment.yaml")

    assert compiled["task_config"]["meta_agent"]["memory_backend"] == {
        "schema_version": "v1",
        "backend_id": "mem0-meta-memory",
        "config_ref": None,
        "state_ref": "meta-state-v1",
    }


def test_compile_experiment_config_derives_required_final_artifacts_from_task(tmp_path: Path):
    config = {
        "lab_root": str(tmp_path / "lab"),
        "task": (
            "Extract records. Final outputs should be biology_component_records.jsonl "
            "and biology_component_report.md."
        ),
        "meta_agent": {"system_prompt": "Route JSON only."},
        "seed_agents": {"WriteAgent": {"system_prompt": "Write final artifacts."}},
        "backends": {
            "llm": {"fake-llm": {"type": "fake", "responses": []}},
            "memory": {
                "mem0-agent-memory": {"type": "null"},
                "mem0-task-memory": {"type": "null"},
            },
            "skill": {"fake-skill": {"type": "fake", "skills": []}},
        },
    }

    compiled = _compile_experiment_config(config, config_path=tmp_path / "experiment.yaml")

    assert compiled["task_config"]["runtime_policy"]["metadata"]["required_final_artifacts"] == [
        "biology_component_records.jsonl",
        "biology_component_report.md",
    ]


def test_compile_experiment_config_accepts_generic_runtime_budget_and_completion_guards(tmp_path: Path):
    config = {
        "lab_root": str(tmp_path / "lab"),
        "task": "Process work items and write final artifacts.",
        "max_tool_steps": 20,
        "max_tool_steps_per_node": 12,
        "max_workflow_nodes": 8,
        "tool_result_prompt_max_chars": 4096,
        "max_repeated_tool_calls_per_run": 2,
        "completion_guards_by_role": {
            "WriteAgent": {"required_tool_calls_before_final": ["write_jsonl", "write_report"]}
        },
        "subagent_budget": {"max_subagent_runtime_seconds": 120},
        "subagent_budgets_by_role": {
            "ExecAgent": {"max_internal_dag_nodes": 6},
            "WriteAgent": {"max_internal_dag_nodes": 1},
        },
        "meta_agent": {"system_prompt": "Route JSON only."},
        "seed_agents": {
            "ExecAgent": {"system_prompt": "Execute one work item."},
            "WriteAgent": {"system_prompt": "Write final artifacts."},
        },
        "backends": {
            "llm": {"fake-llm": {"type": "fake", "responses": []}},
            "memory": {
                "mem0-agent-memory": {"type": "null"},
                "mem0-task-memory": {"type": "null"},
            },
            "skill": {"fake-skill": {"type": "fake", "skills": []}},
        },
    }

    compiled = _compile_experiment_config(config, config_path=tmp_path / "experiment.yaml")
    runtime_policy = compiled["task_config"]["runtime_policy"]

    assert runtime_policy["max_tool_steps"] == 20
    assert runtime_policy["max_tool_steps_per_node"] == 12
    assert runtime_policy["max_workflow_nodes"] == 8
    metadata = runtime_policy["metadata"]
    assert metadata["completion_guards_by_role"] == config["completion_guards_by_role"]
    assert metadata["subagent_budget"] == config["subagent_budget"]
    assert metadata["subagent_budgets_by_role"] == config["subagent_budgets_by_role"]
    assert metadata["tool_result_prompt_max_chars"] == 4096
    assert metadata["max_repeated_tool_calls_per_run"] == 2


def test_clean_run_short_config_routes_to_subagent_and_end(tmp_path: Path):
    config_path = tmp_path / "short.yaml"
    config_path.write_text(
        """
lab_root: unused
files:
  inputs/source.txt: "alpha evidence"
allowed_tools: ["read_text"]
task: |
  Read inputs/source.txt and then end after the survey answer is available.
meta_agent:
  system_prompt: |
    Inspect agents.md and return END with role-pool update metadata only.
  llm_backend:
    backend_id: fake-meta-llm
dynamic_subagents:
  enabled: true
  mode: dynamic
  planner_backend:
    backend_id: fake-planner-llm
  default_worker_backend:
    backend_id: fake-worker-llm
  allowed_worker_backend_ids: ["fake-worker-llm"]
  allowed_tool_names: ["read_text"]
  require_output_schema: false
seed_agents:
  SurveyAgent:
    system_prompt: |
      Read the requested input and summarize it.
    llm_backend:
      backend_id: fake-worker-llm
    allowed_tools: ["read_text"]
backends:
  llm:
    fake-meta-llm:
      type: fake
      responses:
        - action:
            action: final_answer
            content: '{"route":"END","instruction":"No generated tool needed.","metadata":{"no_generated_tool_reason":"read_text is sufficient."}}'
        - action:
            action: final_answer
            content: '{"route":"END","instruction":"No reusable role update needed.","metadata":{"no_role_pool_update_reason":"SurveyAgent is sufficient."}}'
    fake-planner-llm:
      type: fake
      responses:
        - action:
            action: final_answer
            content: '{"workflow_id":"wf-short","task_summary":"Read inputs/source.txt and then end after the survey answer is available.","article_context_summary":"inputs/source.txt contains alpha evidence","dynamic_subagents":[{"subagent_id":"survey","role_name":"SurveyAgent","goal":"Read inputs/source.txt and summarize it.","system_prompt":"Read inputs/source.txt and summarize it.","input_schema":{"type":"object"},"output_schema":{},"allowed_tools":["read_text"]}],"workflow_nodes":[{"node_id":"node-survey","subagent_id":"survey"}],"workflow_edges":[],"artifact_contracts":{},"validation_rules":[],"planner_rationale_summary":"One survey worker is sufficient."}'
    fake-worker-llm:
      type: fake
      responses:
        - action:
            action: tool_call
            tool_call:
              call_id: read-source
              name: read_text
              arguments:
                path: inputs/source.txt
        - action:
            action: final_answer
            content: "Saw alpha evidence."
  memory:
    mem0-task-memory:
      type: "null"
  skill:
    fake-skill:
      type: fake
      skills:
        - skill_id: read-source
          name: Read source
          content: Read the source file.
          required_tools: ["read_text"]
""",
        encoding="utf-8",
    )
    lab_root = tmp_path / "lab"

    result = run_clean_demo(config_path, lab_root)

    assert result["execution_mode"] == "dynamic"
    assert result["final_answer"] == "Saw alpha evidence."
    trajectory_registry = _trajectory_registry(lab_root)
    meta_runs = trajectory_registry.list_meta_agent_runs()
    assert [run.decision.action.value for run in meta_runs] == ["finish_task", "finish_task"]
    assert meta_runs[0].decision.metadata["no_generated_tool_reason"] == "read_text is sufficient."
    assert meta_runs[1].decision.metadata["no_role_pool_update_reason"] == "SurveyAgent is sufficient."
    subagent_runs = trajectory_registry.list_subagent_runs()
    assert [run.role for run in subagent_runs] == ["SurveyAgent"]
    assert subagent_runs[0].tool_calls[0].tool_call.name == "read_text"


def test_clean_run_demo_config_initializes_fresh_lab_and_runs_v0_demo(tmp_path: Path):
    lab_root = tmp_path / "demo-lab"
    lab_root.mkdir()
    stale_file = lab_root / "stale.txt"
    stale_file.write_text("old", encoding="utf-8")

    result = run_clean_demo(DEV_CONFIGS / "demo_v0.yaml", lab_root)

    assert result["task_id"] == "demo-v0"
    assert not stale_file.exists()
    assert sorted((_layout(lab_root).tasks_queue_dir / "done").glob("*.json"))
    assert result["execution_mode"] == "dynamic"
    trajectory_registry = _trajectory_registry(lab_root)
    meta_runs = trajectory_registry.list_meta_agent_runs()
    assert [run.decision.action.value for run in meta_runs] == ["finish_task", "finish_task"]
    assert meta_runs[0].decision.metadata["no_generated_tool_reason"].startswith("The built-in read_file")
    assert meta_runs[1].decision.metadata["no_role_pool_update_reason"].startswith("SolverAgent")
    subagent_runs = trajectory_registry.list_subagent_runs()
    assert [run.role for run in subagent_runs] == ["SolverAgent", "ReviewerAgent"]
    assert [run.stage_index for run in subagent_runs] == [0, 1]
    assert subagent_runs[0].tool_calls[0].tool_call.name == "read_file"
    assert subagent_runs[0].tool_calls[0].result.status == "ok"
    assert "Catalyst A yield: 91%" in subagent_runs[0].tool_calls[0].result.content
    evolution_runs = trajectory_registry.list_evolution_runs()
    assert len(evolution_runs) == 2
    assert all(run.result_status == "promoted_candidate" for run in evolution_runs)
    assert all(run.result.metadata["agent0_sage"]["trainer_id"] == "fake_agent0" for run in evolution_runs)
    assert all(run.result.metadata["trainer"] == "sage" for run in evolution_runs)
    assert all(
        any(artifact.metadata.get("role") == "accepted_samples" for artifact in run.result.artifact_refs)
        for run in evolution_runs
    )
    backend_state_registry = _backend_state_registry(lab_root)
    assert backend_state_registry.resolve_active_state("fake-llm") is not None


def test_clean_run_resolves_meta_agent_instruction_relative_to_config_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config_path = (DEV_CONFIGS / "demo_v0.yaml").resolve()
    lab_root = tmp_path / "demo-lab"
    monkeypatch.chdir(tmp_path)

    result = run_clean_demo(config_path, lab_root)

    assert result["task_id"] == "demo-v0"
    copied_instruction = _layout(lab_root).configs_dir / "demo_v0_agent.md"
    assert copied_instruction.is_file()
    trajectory_registry = _trajectory_registry(lab_root)
    assert len(trajectory_registry.list_meta_agent_runs()) == 2


def test_clean_run_refuses_to_delete_repository_root(tmp_path: Path):
    lab_root = tmp_path / "repo-root"
    lab_root.mkdir()
    marker = lab_root / ".git"
    marker.mkdir()

    with pytest.raises(ValueError, match="refusing to clean unsafe lab root"):
        run_clean_demo(DEV_CONFIGS / "demo_v0.yaml", lab_root)

    assert marker.exists()


def test_clean_run_config_builds_api_llm_backend_from_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _clear_proxy_env(monkeypatch)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "AIGOCODE_GPT_API=openai-responses",
                "AIGOCODE_GPT_BASE_URL=https://api.example.test",
                "AIGOCODE_GPT_API_KEY=test-secret",
            ]
        ),
        encoding="utf-8",
    )
    config = {
        "backends": {
            "llm": {
                "aigocode-gpt": {
                    "type": "api",
                    "env_ref": "aigocode-gpt",
                    "model": "gpt-4.1-mini",
                }
            }
        },
    }

    backends = _build_llm_backends(config, config_dir=tmp_path)

    backend = backends["aigocode-gpt"]
    assert isinstance(backend, ApiLLMBackend)
    assert backend.backend_id == "aigocode-gpt"
    assert backend.config.model == "gpt-4.1-mini"
    assert backend.config.base_url == "https://api.example.test"
    assert backend.client.api_key == "test-secret"
    assert isinstance(backend.instantiate(state_ref=None), OpenAIResponsesRuntime)


def test_clean_run_config_builds_openrouter_deepseek_chat_backend_from_openai_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _clear_proxy_env(monkeypatch)
    (tmp_path / ".env").write_text("OPENAI_API_KEY=test-openrouter-secret\n", encoding="utf-8")
    config = {
        "backends": {
            "llm": {
                "openrouter-deepseek-v4-flash": {
                    "type": "api",
                    "api": "openai-chat-completions",
                    "api_key_env": "OPENAI_API_KEY",
                    "base_url": "https://openrouter.ai/api/v1",
                    "model": "deepseek/deepseek-v4-flash",
                }
            }
        },
    }

    backends = _build_llm_backends(config, config_dir=tmp_path)

    backend = backends["openrouter-deepseek-v4-flash"]
    assert isinstance(backend, ApiLLMBackend)
    assert backend.config.api == "openai-chat-completions"
    assert backend.config.model == "deepseek/deepseek-v4-flash"
    assert backend.config.base_url == "https://openrouter.ai/api/v1"
    assert backend.client.api_key == "test-openrouter-secret"
    assert isinstance(backend.instantiate(state_ref=None), OpenAIChatCompletionsRuntime)


def test_clean_run_config_accepts_api_backend_max_output_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _clear_proxy_env(monkeypatch)
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=test-openrouter-secret\n", encoding="utf-8")
    config = {
        "backends": {
            "llm": {
                "openrouter-qwen30b": {
                    "type": "api",
                    "api": "openai-chat-completions",
                    "api_key_env": "OPENROUTER_API_KEY",
                    "base_url": "https://openrouter.ai/api/v1",
                    "model": "qwen/qwen3-30b-a3b-instruct-2507",
                    "max_output_tokens": 4096,
                }
            }
        },
    }

    backends = _build_llm_backends(config, config_dir=tmp_path)

    backend = backends["openrouter-qwen30b"]
    assert isinstance(backend, ApiLLMBackend)
    assert backend.config.max_output_tokens == 4096
    assert backend.config.api_key_env == "OPENROUTER_API_KEY"


def test_clean_run_config_builds_local_openai_compatible_backend_without_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)
    _clear_proxy_env(monkeypatch)
    config = {
        "backends": {
            "llm": {
                "local-vllm": {
                    "type": "api",
                    "hosting": "local",
                    "api": "openai-chat-completions",
                    "api_key_env": "LOCAL_LLM_API_KEY",
                    "base_url": "http://127.0.0.1:8000/v1",
                    "model": "evolab-local",
                }
            }
        },
    }

    backends = _build_llm_backends(config, config_dir=tmp_path)

    backend = backends["local-vllm"]
    assert backend.config.hosting == "local"
    assert backend.client.api_key == "dummy-local-key"
    assert isinstance(backend.instantiate(state_ref=None), OpenAIChatCompletionsRuntime)


def test_clean_run_config_rejects_legacy_api_env_json_path():
    config = {"api_env": {"json_path": "env.json"}}

    with pytest.raises(ValueError, match="api_env.json_path is no longer supported"):
        _build_llm_backends(config)


def test_clean_run_config_rejects_inline_api_keys(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = {
        "backends": {
            "llm": {
                "aigocode-gpt": {
                    "type": "api",
                    "apiKey": "inline-secret",
                    "model": "gpt-4.1-mini",
                }
            }
        },
    }

    with pytest.raises(ValueError, match="must not include inline api keys"):
        _build_llm_backends(config)


def test_clean_run_config_builds_fake_embedding_backend():
    config = {
        "backends": {
            "embedding": {
                "memory-embedding": {
                    "type": "fake",
                    "dimensions": 5,
                }
            }
        }
    }

    backends = _build_embedding_backends(config)

    backend = backends["memory-embedding"]
    assert isinstance(backend, FakeEmbeddingBackend)
    assert backend.backend_id == "memory-embedding"
    assert backend.dimensions == 5


def test_clean_run_config_builds_native_mem0_method_memory_backend(tmp_path: Path):
    config = {
        "backends": {
            "memory": {
                "mem0-agent-memory": {
                    "type": "method",
                    "method": "mem0",
                    "store_path": "registries/memory/agent.sqlite",
                    "audit_log_path": "registries/memory/agent.audit.jsonl",
                    "llm_backend": "fake-llm",
                    "embedding_backend": "fake-embedding",
                    "default_search_top_k": 5,
                    "default_search_threshold": 0.2,
                }
            }
        }
    }

    backends = _build_memory_backends(config, config_dir=tmp_path)

    backend = backends["mem0-agent-memory"]
    assert isinstance(backend, MethodMemoryBackend)
    assert backend.backend_id == "mem0-agent-memory"
    assert backend.default_search_top_k == 5
    assert backend.default_search_threshold == 0.2
    assert isinstance(backend.method, Mem0MemoryMethod)
    assert backend.method.llm_backend_id == "fake-llm"
    assert backend.method.embedding_backend_id == "fake-embedding"
    assert backend.method.store.path == tmp_path / "registries" / "memory" / "agent.sqlite"
    assert backend.method.store.audit_log_path == tmp_path / "registries" / "memory" / "agent.audit.jsonl"


def test_clean_run_config_builds_native_mem0_method_memory_backend_from_nested_config(tmp_path: Path):
    config = {
        "backends": {
            "memory": {
                "mem0-agent-memory": {
                    "type": "method",
                    "method": "mem0",
                    "config": {
                        "store_path": "registries/memory/agent.sqlite",
                        "audit_log_path": "registries/memory/agent.audit.jsonl",
                        "llm_backend": "fake-llm",
                        "embedding_backend": "fake-embedding",
                        "default_search_top_k": 5,
                        "default_search_threshold": 0.2,
                        "defaults": {"search_top_k": 5},
                    },
                }
            }
        }
    }

    backends = _build_memory_backends(config, config_dir=tmp_path)

    backend = backends["mem0-agent-memory"]
    assert isinstance(backend, MethodMemoryBackend)
    assert backend.backend_id == "mem0-agent-memory"
    assert backend.default_search_top_k == 5
    assert backend.default_search_threshold == 0.2
    assert isinstance(backend.method, Mem0MemoryMethod)
    assert backend.method.llm_backend_id == "fake-llm"
    assert backend.method.embedding_backend_id == "fake-embedding"
    assert backend.method.store.path == tmp_path / "registries" / "memory" / "agent.sqlite"
    assert backend.method.store.audit_log_path == tmp_path / "registries" / "memory" / "agent.audit.jsonl"


def test_clean_run_config_builds_native_everos_method_memory_backend(tmp_path: Path):
    config = {
        "backends": {
            "memory": {
                "everos-agent-memory": {
                    "type": "method",
                    "method": "everos",
                    "store_path": "registries/memory/everos-agent.sqlite",
                    "audit_log_path": "registries/memory/everos-agent.audit.jsonl",
                    "llm_backend": "fake-llm",
                    "embedding_backend": "fake-embedding",
                    "scene_similarity_threshold": 0.7,
                    "extraction_recent_message_limit": 12,
                    "max_scene_candidates": 6,
                    "recollection_mode": "agentic",
                    "recollection_candidate_limit": 9,
                    "default_search_top_k": 4,
                    "default_search_threshold": 0.1,
                }
            }
        }
    }

    backends = _build_memory_backends(config, config_dir=tmp_path)

    backend = backends["everos-agent-memory"]
    assert isinstance(backend, MethodMemoryBackend)
    assert backend.backend_id == "everos-agent-memory"
    assert backend.default_search_top_k == 4
    assert backend.default_search_threshold == 0.1
    assert isinstance(backend.method, EverOSMemoryMethod)
    assert backend.method.llm_backend_id == "fake-llm"
    assert backend.method.embedding_backend_id == "fake-embedding"
    assert backend.method.scene_similarity_threshold == 0.7
    assert backend.method.extraction_recent_message_limit == 12
    assert backend.method.max_scene_candidates == 6
    assert backend.method.recollection_mode == "agentic"
    assert backend.method.recollection_candidate_limit == 9
    assert backend.method.store.path == tmp_path / "registries" / "memory" / "everos-agent.sqlite"
    assert backend.method.store.audit_log_path == tmp_path / "registries" / "memory" / "everos-agent.audit.jsonl"


def test_clean_run_config_builds_native_everos_memory_backend_from_type_shorthand(tmp_path: Path):
    config = {
        "backends": {
            "memory": {
                "everos-task-memory": {
                    "type": "everos",
                    "store_path": "registries/memory/everos-task.sqlite",
                    "llm_backend": "fake-llm",
                    "embedding_backend": "fake-embedding",
                }
            }
        }
    }

    backends = _build_memory_backends(config, config_dir=tmp_path)

    backend = backends["everos-task-memory"]
    assert isinstance(backend, MethodMemoryBackend)
    assert isinstance(backend.method, EverOSMemoryMethod)
    assert backend.method.store.path == tmp_path / "registries" / "memory" / "everos-task.sqlite"


def test_clean_run_config_rejects_everos_http_service_options(tmp_path: Path):
    config = {
        "backends": {
            "memory": {
                "everos-agent-memory": {
                    "type": "method",
                    "method": "everos",
                    "base_url": "http://localhost:1995",
                    "store_path": "registries/memory/everos-agent.sqlite",
                    "llm_backend": "fake-llm",
                    "embedding_backend": "fake-embedding",
                }
            }
        }
    }

    with pytest.raises(ValueError, match="must not configure EverOS HTTP service endpoints"):
        _build_memory_backends(config, config_dir=tmp_path)


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_clean_run_config_rejects_native_mem0_default_search_threshold_out_of_range(
    tmp_path: Path, threshold: float
):
    config = {
        "backends": {
            "memory": {
                "mem0-agent-memory": {
                    "type": "method",
                    "method": "mem0",
                    "store_path": "registries/memory/agent.sqlite",
                    "llm_backend": "fake-llm",
                    "embedding_backend": "fake-embedding",
                    "default_search_threshold": threshold,
                }
            }
        }
    }

    with pytest.raises(ValueError, match="default_search_threshold must be between 0 and 1"):
        _build_memory_backends(config, config_dir=tmp_path)


def test_clean_run_config_builds_native_mem0_method_memory_backend_from_compat_shorthand(tmp_path: Path):
    config = {
        "backends": {
            "memory": {
                "mem0-task-memory": {
                    "type": "mem0",
                    "implementation": "native",
                    "store_path": "registries/memory/task.sqlite",
                    "llm_backend": "fake-llm",
                    "embedding_backend": "fake-embedding",
                }
            }
        }
    }

    backends = _build_memory_backends(config, config_dir=tmp_path)

    backend = backends["mem0-task-memory"]
    assert isinstance(backend, MethodMemoryBackend)
    assert isinstance(backend.method, Mem0MemoryMethod)
    assert backend.method.store.path == tmp_path / "registries" / "memory" / "task.sqlite"


def test_clean_run_config_defaults_mem0_compat_shorthand_to_native_method_backend(tmp_path: Path):
    config = {
        "backends": {
            "memory": {
                "mem0-agent-memory": {
                    "type": "mem0",
                    "store_path": "registries/memory/agent.sqlite",
                    "llm_backend": "fake-llm",
                    "embedding_backend": "fake-embedding",
                }
            }
        }
    }

    backends = _build_memory_backends(config, config_dir=tmp_path)

    backend = backends["mem0-agent-memory"]
    assert isinstance(backend, MethodMemoryBackend)
    assert isinstance(backend.method, Mem0MemoryMethod)
    assert backend.method.store.path == tmp_path / "registries" / "memory" / "agent.sqlite"


def test_clean_run_config_rejects_removed_in_memory_mem0_client():
    config = {
        "backends": {
            "memory": {
                "mem0-agent-memory": {
                    "type": "mem0",
                    "client": "in_memory",
                }
            }
        }
    }

    with pytest.raises(ValueError, match="use type: method, method: mem0"):
        _build_memory_backends(config)


def test_clean_run_config_rejects_removed_in_memory_mem0_client_on_method_mem0(tmp_path: Path):
    config = {
        "backends": {
            "memory": {
                "mem0-agent-memory": {
                    "type": "method",
                    "method": "mem0",
                    "client": "in_memory",
                    "store_path": "memory/agent.sqlite",
                    "llm_backend": "fake-llm",
                    "embedding_backend": "fake-embedding",
                }
            }
        }
    }

    with pytest.raises(ValueError, match="use type: method, method: mem0"):
        _build_memory_backends(config, config_dir=tmp_path)


@pytest.mark.parametrize("client_key", ["client", "client_type"])
def test_clean_run_config_rejects_unsupported_client_on_method_mem0(tmp_path: Path, client_key: str):
    config = {
        "backends": {
            "memory": {
                "mem0-agent-memory": {
                    "type": "method",
                    "method": "mem0",
                    client_key: "external",
                    "store_path": "memory/agent.sqlite",
                    "llm_backend": "fake-llm",
                    "embedding_backend": "fake-embedding",
                }
            }
        }
    }

    with pytest.raises(ValueError, match="native mem0 uses llm_backend and embedding_backend"):
        _build_memory_backends(config, config_dir=tmp_path)


@pytest.mark.parametrize("client_key", ["client", "client_type"])
def test_clean_run_config_rejects_nested_unsupported_client_on_method_mem0(tmp_path: Path, client_key: str):
    config = {
        "backends": {
            "memory": {
                "mem0-agent-memory": {
                    "type": "method",
                    "method": "mem0",
                    "config": {
                        client_key: "external",
                        "store_path": "memory/agent.sqlite",
                        "llm_backend": "fake-llm",
                        "embedding_backend": "fake-embedding",
                    },
                }
            }
        }
    }

    with pytest.raises(ValueError, match="native mem0 uses llm_backend and embedding_backend"):
        _build_memory_backends(config, config_dir=tmp_path)


@pytest.mark.parametrize("client_key", ["client", "client_type"])
def test_clean_run_config_rejects_nested_unsupported_client_on_mem0_shorthand(tmp_path: Path, client_key: str):
    config = {
        "backends": {
            "memory": {
                "mem0-agent-memory": {
                    "type": "mem0",
                    "config": {
                        client_key: "external",
                        "store_path": "memory/agent.sqlite",
                        "llm_backend": "fake-llm",
                        "embedding_backend": "fake-embedding",
                    },
                }
            }
        }
    }

    with pytest.raises(ValueError, match="native mem0 uses llm_backend and embedding_backend"):
        _build_memory_backends(config, config_dir=tmp_path)


@pytest.mark.parametrize(
    ("backend_type", "extra_payload"),
    [
        ("method", {"method": "mem0"}),
        ("mem0", {}),
    ],
)
def test_clean_run_config_rejects_nested_removed_in_memory_mem0_client(
    tmp_path: Path, backend_type: str, extra_payload: dict[str, str]
):
    config = {
        "backends": {
            "memory": {
                "mem0-agent-memory": {
                    "type": backend_type,
                    **extra_payload,
                    "config": {
                        "client": "in_memory",
                        "store_path": "memory/agent.sqlite",
                        "llm_backend": "fake-llm",
                        "embedding_backend": "fake-embedding",
                    },
                }
            }
        }
    }

    with pytest.raises(ValueError, match="use type: method, method: mem0"):
        _build_memory_backends(config, config_dir=tmp_path)


@pytest.mark.parametrize(
    ("backend_type", "extra_payload"),
    [
        ("method", {"method": "mem0"}),
        ("mem0", {}),
    ],
)
def test_clean_run_config_rejects_nested_unsupported_mem0_implementation(
    tmp_path: Path, backend_type: str, extra_payload: dict[str, str]
):
    config = {
        "backends": {
            "memory": {
                "mem0-agent-memory": {
                    "type": backend_type,
                    **extra_payload,
                    "config": {
                        "implementation": "external",
                        "store_path": "memory/agent.sqlite",
                        "llm_backend": "fake-llm",
                        "embedding_backend": "fake-embedding",
                    },
                }
            }
        }
    }

    with pytest.raises(ValueError, match="unsupported Mem0 implementation 'external'"):
        _build_memory_backends(config, config_dir=tmp_path)


@pytest.mark.parametrize(
    ("backend_type", "extra_payload"),
    [
        ("method", {"method": "mem0"}),
        ("mem0", {}),
    ],
)
def test_clean_run_config_rejects_nested_unsupported_mem0_method(
    tmp_path: Path, backend_type: str, extra_payload: dict[str, str]
):
    config = {
        "backends": {
            "memory": {
                "mem0-agent-memory": {
                    "type": backend_type,
                    **extra_payload,
                    "config": {
                        "method": "external",
                        "store_path": "memory/agent.sqlite",
                        "llm_backend": "fake-llm",
                        "embedding_backend": "fake-embedding",
                    },
                }
            }
        }
    }

    with pytest.raises(ValueError, match="unsupported Mem0 method 'external'"):
        _build_memory_backends(config, config_dir=tmp_path)


def test_clean_run_config_builds_null_memory_backend():
    config = {
        "backends": {
            "memory": {
                "memory-off": {
                    "type": "null",
                }
            }
        }
    }

    backends = _build_memory_backends(config)

    backend = backends["memory-off"]
    assert isinstance(backend, NullMemoryBackend)
    assert backend.backend_id == "memory-off"


def test_clean_run_config_rejects_inline_mem0_api_keys():
    config = {
        "backends": {
            "memory": {
                "mem0-agent-memory": {
                    "type": "mem0",
                    "apiKey": "inline-secret",
                }
            }
        }
    }

    with pytest.raises(ValueError, match="must not include inline Mem0 api keys"):
        _build_memory_backends(config)


@pytest.mark.parametrize("api_key", ["apiKey", "api_key"])
@pytest.mark.parametrize(
    ("backend_type", "extra_payload"),
    [
        ("method", {"method": "mem0"}),
        ("mem0", {}),
    ],
)
def test_clean_run_config_rejects_nested_inline_mem0_api_keys(
    tmp_path: Path, backend_type: str, extra_payload: dict[str, str], api_key: str
):
    config = {
        "backends": {
            "memory": {
                "mem0-agent-memory": {
                    "type": backend_type,
                    **extra_payload,
                    "config": {
                        api_key: "inline-secret",
                        "store_path": "memory/agent.sqlite",
                        "llm_backend": "fake-llm",
                        "embedding_backend": "fake-embedding",
                    },
                }
            }
        }
    }

    with pytest.raises(ValueError, match="must not include inline Mem0 api keys"):
        _build_memory_backends(config, config_dir=tmp_path)


def test_clean_run_demo_v1_records_mem0_memory_lineage(tmp_path: Path):
    lab_root = tmp_path / "demo-v1-lab"
    config = json.loads((DEV_CONFIGS / "demo_v1_ci.yaml").read_text(encoding="utf-8"))

    assert config["task_config"]["agents_ref"] == "agents/demo_v1_ci_agents.md"
    assert config["task_config"]["dynamic_subagents"] == {
        "enabled": True,
        "mode": "dynamic",
        "scope": "per_task",
        "planner_backend": {"backend_id": "fake-llm"},
        "default_worker_backend": {"backend_id": "fake-llm"},
        "allowed_tool_names": [],
    }
    assert "mem0-agent-memory" not in config["backends"]["memory"]

    result = run_clean_demo(DEV_CONFIGS / "demo_v1_ci.yaml", lab_root)

    assert result["task_id"] == "demo-v1"
    assert result["execution_mode"] == "dynamic"
    trajectory_registry = _trajectory_registry(lab_root)
    subagent_runs = trajectory_registry.list_subagent_runs()
    assert [run.role for run in subagent_runs] == ["GeneralistAgent"]
    saved = subagent_runs[0]
    assert "agent_memory_bundle" not in saved.metadata
    assert "agent_memory_update_result" not in saved.metadata
    assert saved.metadata["task_memory_bundle"]["backend_id"] == "mem0-task-memory"
    assert saved.metadata["task_memory_update_result"]["metadata"]["memory_method"] == "mem0"
    assert saved.metadata["task_memory_update_result"]["status"] == "updated"
    backend_state_registry = _backend_state_registry(lab_root)
    records = backend_state_registry.list_states()
    memory_records = [record for record in records if record.backend_type == "memory"]
    assert {(record.backend_id, record.metadata["memory_scope"]) for record in memory_records} == {
        ("mem0-task-memory", "task"),
    }
    assert (_layout(lab_root).registries_dir / "memory" / "mem0-task.sqlite").is_file()
    assert not Path("configs").exists()


def test_clean_run_config_builds_graph_skill_backend():
    backends = _build_skill_backends(
        {
            "backends": {
                "skill": {
                    "scientific-ie-skills": {
                        "type": "graph",
                        "graph_path": str(DEV_SEED_GRAPH),
                    }
                }
            }
        }
    )

    assert isinstance(backends["scientific-ie-skills"], GraphSkillBackend)
    assert backends["scientific-ie-skills"].backend_id == "graph_skill"


def test_clean_run_graph_skill_backend_writes_update_log_inside_lab(tmp_path: Path):
    source_root = tmp_path / "source"
    graph_source = source_root / "configs" / "skills" / "graphs" / "scientific_ie_seed_graph_v1.json"
    graph_source.parent.mkdir(parents=True)
    graph_source.write_text(
        DEV_SEED_GRAPH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    config_path = source_root / "configs" / "demo_graph.json"
    graph_agents_path = source_root / "configs" / "agents" / "graph_agents.md"
    graph_agents_path.parent.mkdir(parents=True)
    _write_agents_file(
        graph_agents_path,
        RoleSpec(
            name="solver",
            system_prompt="Solve.",
            llm_backend=BackendBinding(backend_id="fake-llm"),
        ),
    )
    config_path.write_text(
        """
        {
          "lab_root": "lab/demo-graph",
          "task": {
            "task_id": "demo-graph",
            "origin": "human",
            "purpose": "science",
            "goal": "zzzz-no-skill-match",
            "task_config_ref": "configs/demo_graph.json"
          },
              "task_config": {
                "task_id": "demo-graph",
                "goal": "zzzz-no-skill-match",
                "agents_ref": "agents/graph_agents.md",
                "dynamic_subagents": {
                  "enabled": true,
              "planner_backend": {"backend_id": "fake-llm"},
              "default_worker_backend": {"backend_id": "fake-llm"}
            }
          },
          "backends": {
            "llm": {
              "fake-llm": {
                "type": "fake",
                "responses": [
                  {"action": {"action": "final_answer", "content": "{\\"workflow_id\\":\\"wf-graph\\",\\"task_summary\\":\\"zzzz-no-skill-match\\",\\"article_context_summary\\":\\"none\\",\\"dynamic_subagents\\":[{\\"subagent_id\\":\\"solver\\",\\"role_name\\":\\"solver\\",\\"goal\\":\\"zzzz-no-skill-match\\",\\"system_prompt\\":\\"Solve.\\",\\"input_schema\\":{\\"type\\":\\"object\\"},\\"output_schema\\":{\\"type\\":\\"object\\"},\\"allowed_tools\\":[]}],\\"workflow_nodes\\":[{\\"node_id\\":\\"node-solver\\",\\"subagent_id\\":\\"solver\\"}],\\"workflow_edges\\":[],\\"artifact_contracts\\":{},\\"validation_rules\\":[],\\"planner_rationale_summary\\":\\"One solver worker is sufficient.\\"}"}},
                  {"action": {"action": "final_answer", "content": "done"}}
                ]
              }
            },
            "memory": {
              "fake-memory": {"type": "fake"}
            },
            "skill": {
              "scientific-ie-skills": {
                "type": "graph",
                "graph_path": "configs/skills/graphs/scientific_ie_seed_graph_v1.json"
              }
            }
          }
        }
        """,
        encoding="utf-8",
    )
    lab_root = tmp_path / "lab"

    run_clean_demo(config_path, lab_root)

    assert (_layout(lab_root).configs_dir / "skills" / "graphs" / "scientific_ie_seed_graph_v1.updates.jsonl").is_file()
    assert not graph_source.with_suffix(".updates.jsonl").exists()


def test_clean_run_copies_domain_packages_referenced_by_skill_graph(tmp_path: Path):
    source_root = tmp_path / "source"
    graph_source = source_root / "configs" / "skills" / "graphs" / "scientific_ie_seed_graph_v1.json"
    graph_source.parent.mkdir(parents=True)
    graph_source.write_text(
        DEV_SEED_GRAPH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    package_source = source_root / "domain_packages" / "biology_component_extraction_v1"
    package_source.mkdir(parents=True)
    (package_source / "biology_component_schema.json").write_text('{"type":"object"}', encoding="utf-8")
    skill_source = source_root / "skills" / "scientific_ie" / "scientific_document_intake"
    shutil.copytree(ROOT / "dev" / "skills" / "scientific_ie" / "scientific_document_intake", skill_source)
    config_path = source_root / "configs" / "demo_graph.json"
    graph_agents_path = source_root / "configs" / "agents" / "graph_agents.md"
    graph_agents_path.parent.mkdir(parents=True)
    _write_agents_file(
        graph_agents_path,
        RoleSpec(
            name="solver",
            system_prompt="Solve.",
            llm_backend=BackendBinding(backend_id="fake-llm"),
        ),
    )
    config_path.write_text(
        """
        {
          "lab_root": "lab/demo-graph",
          "task": {
            "task_id": "demo-graph",
            "origin": "human",
            "purpose": "science",
            "goal": "zzzz-no-skill-match",
            "task_config_ref": "configs/demo_graph.json"
          },
              "task_config": {
                "task_id": "demo-graph",
                "goal": "zzzz-no-skill-match",
                "agents_ref": "agents/graph_agents.md",
                "dynamic_subagents": {
                  "enabled": true,
              "planner_backend": {"backend_id": "fake-llm"},
              "default_worker_backend": {"backend_id": "fake-llm"}
            }
          },
          "backends": {
            "llm": {
              "fake-llm": {
                "type": "fake",
                "responses": [
                  {"action": {"action": "final_answer", "content": "{\\"workflow_id\\":\\"wf-graph\\",\\"task_summary\\":\\"zzzz-no-skill-match\\",\\"article_context_summary\\":\\"none\\",\\"dynamic_subagents\\":[{\\"subagent_id\\":\\"solver\\",\\"role_name\\":\\"solver\\",\\"goal\\":\\"zzzz-no-skill-match\\",\\"system_prompt\\":\\"Solve.\\",\\"input_schema\\":{\\"type\\":\\"object\\"},\\"output_schema\\":{\\"type\\":\\"object\\"},\\"allowed_tools\\":[]}],\\"workflow_nodes\\":[{\\"node_id\\":\\"node-solver\\",\\"subagent_id\\":\\"solver\\"}],\\"workflow_edges\\":[],\\"artifact_contracts\\":{},\\"validation_rules\\":[],\\"planner_rationale_summary\\":\\"One solver worker is sufficient.\\"}"}},
                  {"action": {"action": "final_answer", "content": "done"}}
                ]
              }
            },
            "memory": {"fake-memory": {"type": "fake"}},
            "skill": {
              "scientific-ie-skills": {
                "type": "graph",
                "graph_path": "configs/skills/graphs/scientific_ie_seed_graph_v1.json"
              }
            }
          }
        }
        """,
        encoding="utf-8",
    )
    lab_root = tmp_path / "lab"

    run_clean_demo(config_path, lab_root)

    assert (
        _layout(lab_root).state_root
        / "domain_packages"
        / "biology_component_extraction_v1"
        / "biology_component_schema.json"
    ).is_file()
    assert (
        _layout(lab_root).state_root
        / "skills"
        / "scientific_ie"
        / "scientific_document_intake"
        / "metadata.yaml"
    ).is_file()


def test_clean_run_builds_local_trainable_llm_backend(tmp_path: Path):
    backends = _build_llm_backends(
        {
            "backends": {
                "llm": {
                    "aigocode-gpt": {
                        "type": "local_trainable",
                        "default_content": "local rollout",
                    }
                }
            }
        },
        config_dir=tmp_path,
    )

    assert isinstance(backends["aigocode-gpt"], LocalTrainableLLMBackend)
    assert backends["aigocode-gpt"].backend_id == "aigocode-gpt"


def test_clean_run_local_trainable_llm_backend_resolves_registry_state(tmp_path: Path):
    state_ref = "local-trainable://aigocode-gpt/state/promoted"
    state_manifest_path = tmp_path / "local_trainable_state.json"
    state_manifest_path.write_text(
        json.dumps(
            {
                "backend_id": "aigocode-gpt",
                "state_ref": state_ref,
                "created_by_trainer": "sft",
                "default_content": "promoted local rollout",
                "metadata": {"training_backend": "dry_run"},
            }
        ),
        encoding="utf-8",
    )
    state_registry = FileBackendStateRegistry(tmp_path / "backend_state")
    state_registry.register_candidate(
        BackendStateRecord(
            state_ref=state_ref,
            backend_id="aigocode-gpt",
            backend_type="llm",
            artifact_refs=[
                ArtifactRef(
                    uri=str(state_manifest_path),
                    type="model_adapter",
                    metadata={"role": "local_trainable_state"},
                )
            ],
        )
    )
    state_registry.promote("aigocode-gpt", state_ref, "evolution-run-1")

    backends = _build_llm_backends(
        {"backends": {"llm": {"aigocode-gpt": {"type": "local_trainable"}}}},
        config_dir=tmp_path,
        backend_state_registry=state_registry,
    )
    runtime = backends["aigocode-gpt"].instantiate(
        state_registry.resolve_active_state("aigocode-gpt")
    )
    response = runtime.generate(
        messages=[Message(role="user", content="hello")],
        tool_specs=[],
        generation_config=LLMGenerationConfig(model="local"),
    )

    assert response.action.content == "promoted local rollout"
    assert response.raw_response["state_manifest"]["created_by_trainer"] == "sft"


def test_local_trainable_is_not_an_evolution_backend(tmp_path: Path):
    with pytest.raises(ValueError, match="configure an SFT, OPSD, or Agent0SAGE trainer"):
        _build_evolution_backends(
            {
                "evolution": {
                    "backends": {
                        "aigocode-gpt": {
                            "type": "local_trainable",
                        }
                    }
                }
            },
            trajectory_registry=FileTrajectoryRegistry(tmp_path / "trajectory"),
        )


def test_dotenv_resolution_does_not_walk_all_parents(tmp_path: Path):
    config_dir = tmp_path / "outer" / "repo" / "configs"
    config_dir.mkdir(parents=True)
    (tmp_path / "outer" / ".env").write_text(
        "AIGOCODE_GPT_API_KEY=test-secret\n",
        encoding="utf-8",
    )
    config = {
        "backends": {
            "llm": {
                "aigocode-gpt": {
                    "type": "api",
                    "env_ref": "aigocode-gpt",
                    "model": "gpt-4.1-mini",
                }
            }
        }
    }

    with pytest.raises(ValueError, match=r"missing \.env entry"):
        _build_llm_backends(config, config_dir=config_dir)


def test_export_sft_command_function_exports_clean_run_trajectory(tmp_path: Path):
    lab_root = tmp_path / "demo-lab"
    run_clean_demo(DEV_CONFIGS / "demo_v0.yaml", lab_root)

    result = run_export_sft(
        lab_root=lab_root,
        output_dir=tmp_path / "sft",
        teacher_backend_ids=["fake-llm"],
    )

    assert result["manifest"].sample_count == 2
    assert result["manifest"].train_count == 2
    assert Path(result["train_path"]).is_file()
    assert Path(result["manifest_path"]).is_file()


def test_clean_run_builds_sft_evolution_backend(tmp_path: Path):
    trajectory_registry = FileTrajectoryRegistry(tmp_path / "trajectory")
    backends = _build_evolution_backends(
        {
            "evolution": {
                "backends": {
                    "teacher-api": {
                        "type": "sft",
                        "promote_dry_run": True,
                    }
                }
            }
        },
        trajectory_registry=trajectory_registry,
    )

    assert isinstance(backends["teacher-api"], SFTTrainer)


def test_train_sft_command_function_runs_dry_run_from_clean_run_trajectory(tmp_path: Path):
    lab_root = tmp_path / "demo-lab"
    run_clean_demo(DEV_CONFIGS / "demo_v0.yaml", lab_root)

    result = run_train_sft(
        lab_root=lab_root,
        backend_id="fake-llm",
        artifact_root=tmp_path / "sft-train",
        promote_dry_run=True,
    )

    assert result["result"].status == "promoted_candidate"
    assert result["promoted"] is True
    assert result["promotion_errors"] == []
    assert result["result"].standard_metrics.n_train_samples == 2
    assert (Path(result["artifact_root"]) / "dataset" / "train.jsonl").is_file()
    state_registry = _backend_state_registry(lab_root)
    assert state_registry.resolve_active_state("fake-llm") == result["result"].new_state_ref
    trajectory_registry = _trajectory_registry(lab_root)
    sft_record = trajectory_registry.get_evolution_run(result["run_ref"])
    assert sft_record is not None
    assert sft_record.training_trajectory_refs
    assert sft_record.result_status == "promoted_candidate"
