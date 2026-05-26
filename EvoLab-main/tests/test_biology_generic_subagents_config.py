import json
from pathlib import Path

import pytest
import yaml

from evolab.config.agents import load_agents_file
from evolab.cli import _compile_experiment_config
from evolab.config.task_config import TaskConfig


ROOT = Path(__file__).resolve().parents[1]
DEV_CONFIGS = ROOT / "dev" / "configs"
NEW_CONFIG_PATH = DEV_CONFIGS / "biology_component_extraction_v1_generic_subagents.yaml"
TWO_PAPER_CONFIG_PATH = DEV_CONFIGS / "biology_component_extraction_v1_two_paper_work_items.yaml"
DEMO_V1_CONFIG_PATH = DEV_CONFIGS / "demo_v1.yaml"
ROLE_POOL_PATH = DEV_CONFIGS / "agents" / "scientific_ie_agents.md"

ALLOWED_GENERIC_AGENT_TYPES = [
    "SurveyAgent",
    "DesignAgent",
    "ExecAgent",
    "CriticAgent",
    "WriteAgent",
]

LEGACY_BIOLOGY_ROLE_NAMES = [
    "intake_agent",
    "table_triage_agent",
    "table_s1_extractor_agent",
    "table_s2_extractor_agent",
    "record_builder_agent",
    "validator_writer_agent",
]

FORBIDDEN_STRUCTURED_TASK_FIELDS = [
    "task_config",
    "target_article_dir",
    "target_article_id",
    "article_root",
    "dataset_root",
    "input_scope",
    "execution_scope",
    "generic_workflow",
    "available_generic_subagents",
]


def _load_yaml_config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _compiled_config() -> dict:
    return _compile_experiment_config(_load_yaml_config(NEW_CONFIG_PATH), config_path=NEW_CONFIG_PATH)


def _role_pool_tool_names() -> set[str]:
    roles = load_agents_file(ROLE_POOL_PATH)
    return {tool for role in roles.values() for tool in role.allowed_tools}


def test_only_short_generic_biology_config_exists() -> None:
    assert NEW_CONFIG_PATH.exists()
    assert not (DEV_CONFIGS / "biology_component_extraction_v1.yaml").exists()


def test_generic_config_is_human_short_form_not_internal_runtime_schema() -> None:
    config = _load_yaml_config(NEW_CONFIG_PATH)

    assert isinstance(config["task"], str)
    assert "work" in config["task"]
    assert "item" in config["task"]
    assert "Do not treat pre-existing biology_component_records.jsonl" in config["task"]
    assert "Final artifacts must be produced as" in config["task"]
    assert "lab artifacts by the current run" in config["task"]
    assert isinstance(config["meta_agent"]["system_prompt"], str)
    assert config["agents_ref"] == "agents/scientific_ie_agents.md"
    assert config["dynamic_subagents"]["enabled"] is True
    assert config["dynamic_subagents"]["mode"] == "dynamic"
    assert set(config["dynamic_subagents"]["allowed_tool_names"]) >= _role_pool_tool_names()
    assert set(config) >= {"task", "meta_agent", "agents_ref", "dynamic_subagents", "work_item_routing", "lab_root", "backends"}
    assert "subagents" not in config
    assert "roles" not in config
    assert "runtime_policy" not in config
    assert "tools" not in config

    serialized = NEW_CONFIG_PATH.read_text(encoding="utf-8")
    for field in FORBIDDEN_STRUCTURED_TASK_FIELDS:
        assert field not in serialized


def test_shared_scientific_ie_role_pool_defines_generic_roles_without_worker_agent_memory() -> None:
    roles = load_agents_file(ROLE_POOL_PATH)

    assert list(roles) == ALLOWED_GENERIC_AGENT_TYPES
    assert "OtherAgent" not in roles
    assert "solver" not in roles
    for legacy_role in LEGACY_BIOLOGY_ROLE_NAMES:
        assert legacy_role not in roles

    for name, role in roles.items():
        assert role.llm_backend.backend_id == "aigocode-gpt", name
        assert role.agent_memory_backend is None, name
        assert role.allowed_tools, name
        assert role.metadata["role_pool_seed"] is True


def test_short_config_backends_are_declared_without_exposing_runtime_task_config() -> None:
    config = _load_yaml_config(NEW_CONFIG_PATH)

    assert "aigocode-gpt" in config["backends"]["llm"]
    assert config["backends"]["llm"]["aigocode-gpt"]["type"] == "api"
    assert config["backends"]["llm"]["aigocode-gpt"]["model"] == "gpt-5.4-mini"
    assert "mem0-agent-memory" not in config["backends"]["memory"]
    assert "mem0-meta-memory" in config["backends"]["memory"]
    assert "mem0-task-memory" in config["backends"]["memory"]
    assert "scientific-ie-skills" in config["backends"]["skill"]
    assert config["evolve_worker"]["enabled"] is False


def test_short_config_compiles_to_runtime_task_config_with_route_pool() -> None:
    compiled = _compiled_config()
    task_config = TaskConfig.model_validate(compiled["task_config"])

    assert compiled["task"]["goal"] == _load_yaml_config(NEW_CONFIG_PATH)["task"]
    assert task_config.goal == compiled["task"]["goal"]
    assert task_config.meta_agent is not None
    assert task_config.meta_agent.memory_backend is not None
    assert task_config.meta_agent.memory_backend.backend_id == "mem0-meta-memory"
    assert "route" in task_config.meta_agent.system_prompt
    assert "END" in task_config.meta_agent.system_prompt
    assert task_config.agents_ref == "agents/scientific_ie_agents.md"
    assert task_config.dynamic_subagents is not None
    assert task_config.dynamic_subagents.enabled is True
    assert task_config.dynamic_subagents.mode == "dynamic"
    assert task_config.dynamic_subagents.planner_backend is not None
    assert task_config.dynamic_subagents.planner_backend.backend_id == "aigocode-gpt"
    assert task_config.dynamic_subagents.default_worker_backend is not None
    assert task_config.dynamic_subagents.default_worker_backend.backend_id == "aigocode-gpt"
    assert set(task_config.dynamic_subagents.allowed_tool_names) >= _role_pool_tool_names()
    assert list(task_config.roles) == ALLOWED_GENERIC_AGENT_TYPES
    assert task_config.max_dispatch_steps >= 20
    assert task_config.runtime_policy.enable_workflow_planning is True
    assert task_config.runtime_policy.metadata["route_contract"] == {
        "route_field": "route",
        "end_route": "END",
        "subagent_routes": ALLOWED_GENERIC_AGENT_TYPES,
    }
    assert task_config.runtime_policy.metadata["work_item_routing"] == {
        "enabled": True,
        "executor_roles": ["ExecAgent"],
        "reviewer_roles": ["CriticAgent"],
        "finalizer_roles": ["WriteAgent"],
        "required_work_item_ids": [],
        "work_item_id_field": "work_item_id",
    }
    assert task_config.runtime_policy.metadata["agents_ref_materialized_from_seed"] is False
    assert task_config.runtime_policy.metadata["agents_ref_render_compiled_roles"] is False


def test_compiled_runtime_roles_do_not_include_solver_or_legacy_biology_roles() -> None:
    compiled = _compiled_config()
    roles = compiled["task_config"]["roles"]

    assert "OtherAgent" not in roles
    assert "solver" not in roles
    for legacy_role in LEGACY_BIOLOGY_ROLE_NAMES:
        assert legacy_role not in roles


def test_compiler_provides_default_scientific_tools_without_user_config_complexity() -> None:
    compiled = _compiled_config()
    roles = compiled["task_config"]["roles"]

    assert compiled["tools"]["scientific_ie"] == {"enabled": True, "include_human_tools": False}
    assert "discover_candidate_source_files" in roles["SurveyAgent"]["allowed_tools"]
    assert "discover_candidate_tables" in roles["DesignAgent"]["allowed_tools"]
    assert "build_candidate_records" in roles["ExecAgent"]["allowed_tools"]
    assert "validate_candidate_records" in roles["CriticAgent"]["allowed_tools"]
    assert "serialize_final_records" in roles["WriteAgent"]["allowed_tools"]


def test_dynamic_tool_allow_lists_cover_shared_role_pool_tools() -> None:
    expected_tools = _role_pool_tool_names()
    generic_config = _load_yaml_config(NEW_CONFIG_PATH)
    two_paper_config = _load_yaml_config(TWO_PAPER_CONFIG_PATH)
    demo_v1_config = json.loads(DEMO_V1_CONFIG_PATH.read_text(encoding="utf-8"))

    assert set(generic_config["dynamic_subagents"]["allowed_tool_names"]) >= expected_tools
    assert set(two_paper_config["dynamic_subagents"]["allowed_tool_names"]) >= expected_tools
    assert set(demo_v1_config["task_config"]["dynamic_subagents"]["allowed_tool_names"]) >= expected_tools


def test_compiled_runtime_policy_describes_internal_subagent_planning() -> None:
    compiled = _compiled_config()
    metadata = compiled["task_config"]["runtime_policy"]["metadata"]

    assert metadata["subagent_policy"] == {
        "internal_planning": True,
        "must_build_internal_dag": True,
        "skill_retrieval_scope": "per_internal_dag_node",
        "tool_preparation_scope": "per_internal_dag_node_or_retrieved_skill",
    }


def test_compiled_biology_config_requires_records_and_one_report_artifact() -> None:
    compiled = _compiled_config()
    metadata = compiled["task_config"]["runtime_policy"]["metadata"]

    assert metadata["required_final_artifacts"] == ["biology_component_records.jsonl"]
    assert metadata["required_final_artifact_groups"] == [
        {
            "one_of": ["biology_component_report.md", "biology_component_report.json"],
            "description": "biology component final report or audit artifact",
        }
    ]


def test_two_paper_config_compiles_required_work_items_and_runtime_budget() -> None:
    source_config = _load_yaml_config(TWO_PAPER_CONFIG_PATH)
    compiled = _compile_experiment_config(
        source_config,
        config_path=TWO_PAPER_CONFIG_PATH,
    )
    task_config = TaskConfig.model_validate(compiled["task_config"])
    llm_backends = source_config["backends"]["llm"]

    assert source_config["dotenv_path"] == ".env"
    assert "subagents" not in source_config
    assert source_config["agents_ref"] == "agents/scientific_ie_agents.md"
    assert source_config["dynamic_subagents"]["enabled"] is True
    assert source_config["dynamic_subagents"]["mode"] == "dynamic"
    assert set(source_config["dynamic_subagents"]["allowed_tool_names"]) >= _role_pool_tool_names()
    assert list(llm_backends) == ["aigocode-gpt"]
    assert llm_backends["aigocode-gpt"] == {
        "type": "api",
        "env_ref": "aigocode-gpt",
        "model": "gpt-5.4-mini",
        "max_retries": 3,
        "retry_initial_delay_seconds": 5.0,
        "retry_max_delay_seconds": 30.0,
    }
    assert task_config.runtime_policy.max_tool_steps == 30
    assert task_config.runtime_policy.max_tool_steps_per_node == 30
    assert task_config.runtime_policy.max_workflow_nodes == 30
    assert task_config.max_dispatch_steps == 30
    assert task_config.runtime_policy.metadata["max_meta_dispatch_parse_retries"] == 4
    assert task_config.runtime_policy.metadata["tool_result_prompt_max_chars"] == 6000
    assert task_config.runtime_policy.metadata["max_repeated_tool_calls_per_run"] == 2
    assert task_config.runtime_policy.metadata["work_item_routing"] == {
        "enabled": True,
        "executor_roles": ["ExecAgent"],
        "reviewer_roles": ["CriticAgent"],
        "finalizer_roles": ["WriteAgent"],
        "required_work_item_ids": [
            "synthetic_promoter_multinomial_diffusion",
            "ai_knowledge_sigma70_design",
        ],
        "work_item_id_field": "work_item_id",
    }
    assert task_config.dynamic_subagents is not None
    assert task_config.dynamic_subagents.enabled is True
    assert set(task_config.dynamic_subagents.allowed_tool_names) >= _role_pool_tool_names()
    assert task_config.roles["DesignAgent"].allowed_tools
    assert task_config.runtime_policy.metadata["completion_guards_by_role"] == {
        "SurveyAgent": {"required_tool_calls_before_final": ["write_report"]},
        "DesignAgent": {"required_tool_calls_before_final": ["write_report"]},
        "ExecAgent": {
            "required_tool_calls_before_final": ["write_jsonl", "write_report"],
            "minimum_jsonl_records_before_final": 1,
        },
        "CriticAgent": {"required_tool_calls_before_final": ["write_report"]},
        "WriteAgent": {
            "required_tool_calls_before_final": ["write_jsonl", "write_report"],
            "minimum_jsonl_records_before_final": 1,
            "max_non_required_tool_calls_before_required_outputs": 8,
        },
    }
    assert task_config.runtime_policy.metadata["subagent_budgets_by_role"]["ExecAgent"] == {
        "max_internal_dag_nodes": 30,
        "max_subagent_tool_calls": 30,
        "max_subagent_llm_calls": 30,
        "max_subagent_runtime_seconds": 900,
    }


@pytest.mark.parametrize("role_name", ["MetaAgent", "UnknownAgent"])
def test_subagent_budgets_reject_non_dispatchable_role_names(role_name: str) -> None:
    source_config = _load_yaml_config(TWO_PAPER_CONFIG_PATH)
    source_config["subagent_budgets_by_role"] = dict(source_config["subagent_budgets_by_role"])
    source_config["subagent_budgets_by_role"][role_name] = {"max_subagent_llm_calls": 30}

    with pytest.raises(ValueError, match="subagent_budgets_by_role contains unknown role"):
        _compile_experiment_config(
            source_config,
            config_path=TWO_PAPER_CONFIG_PATH,
        )


def test_full_run_config_article_package_paths_exist() -> None:
    full_config_path = DEV_CONFIGS / "biology_component_extraction_v1_28_article_work_items.yaml"
    source_config = _load_yaml_config(full_config_path)
    article_paths = [
        line.split("article_package:", 1)[1].strip()
        for line in source_config["task"].splitlines()
        if "article_package:" in line
    ]

    assert len(article_paths) == 28
    missing = [path for path in article_paths if not Path(path).is_dir()]
    assert missing == []


def test_full_run_config_exact_source_files_exist() -> None:
    full_config_path = DEV_CONFIGS / "biology_component_extraction_v1_28_article_work_items.yaml"
    source_config = _load_yaml_config(full_config_path)
    source_paths = []
    for line in source_config["task"].splitlines():
        stripped = line.strip()
        if stripped.startswith("- /"):
            source_paths.append(stripped[2:].strip())

    assert source_paths
    missing = [path for path in source_paths if not Path(path).is_file()]
    assert missing == []
