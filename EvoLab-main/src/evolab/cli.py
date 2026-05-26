from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel

from evolab.backends.embeddings import ApiEmbeddingBackend, ApiEmbeddingBackendConfig, FakeEmbeddingBackend
from evolab.backends.evolution import (
    FakeAgent0Trainer,
    FakeEvolutionBackend,
    FakeSAGETrainer,
    PromptOverlayEvolutionTrainer,
)
from evolab.backends.llm import ApiLLMBackend, ApiLLMBackendConfig, FakeLLMBackend, LocalTrainableLLMBackend
from evolab.backends.memory import (
    FakeMemoryBackend,
    MethodMemoryBackend,
    NullMemoryBackend,
)
from evolab.backends.memory.methods.everos import EverOSMemoryMethod
from evolab.backends.memory.methods.mem0 import Mem0MemoryMethod
from evolab.backends.skills import FakeSkillBackend, GraphSkillBackend
from evolab.backends.trainers import OPSDTrainer, OPSDTrainerConfig, SFTTrainer, SFTTrainerConfig
from evolab.config.agents import default_seed_roles, load_agents_file, render_agents_markdown
from evolab.config.env import env_ref_prefix, lookup_env_value, parse_dotenv
from evolab.config.task_config import BackendBinding, MetaAgentSpec, ReflectorSpec, RoleSpec, TaskConfig
from evolab.contracts.common import RuntimePolicy
from evolab.contracts.dynamic_workflow import DynamicSubagentsConfig
from evolab.contracts.evolution import LLMEvolutionMode, LLMEvolutionRequest
from evolab.contracts.llm import LLMRuntimeResponse
from evolab.contracts.records import EvolutionRunRecord
from evolab.contracts.retrieval import SkillItem
from evolab.contracts.task import ProposerInputRef, TaskOrigin, TaskPurpose, TaskRequest
from evolab.contracts.tools import ToolSpec
from evolab.lab.layout import LabLayout
from evolab.lab.resolver import LabResolver
from evolab.runtime.evolution_executor import EvolutionExecutor
from evolab.runtime.evolve_worker import EvolveWorker
from evolab.runtime.opsd_exporter import OPSDExportConfig, export_opsd_dataset
from evolab.runtime.sft_exporter import SFTExportConfig, export_sft_dataset
from evolab.runtime.task_worker import TaskWorker
from evolab.runtime.trajectory_collector import TrajectoryCollector
from evolab.runtime.trajectory_visualizer import visualize_trajectory
from evolab.tools.runtime import ToolRegistry
from evolab.tools.scientific_ie import register_scientific_ie_tools

ModelT = TypeVar("ModelT", bound=BaseModel)
_DEFAULT_AGENTS_REF = "agents.md"
_SEED_AGENTS_NOTE = "Materialized seed role pool for automatic EvoLab role evolution."


def run_clean_demo(config_path: Path | str, lab_root: Path | str | None = None) -> dict[str, Any]:
    config_path = Path(config_path)
    config = _load_yaml(config_path)
    config = _compile_experiment_config(config, config_path=config_path)
    root = Path(lab_root or config.get("lab_root") or "lab/demo_v0")
    _clean_lab_root(root)

    layout = LabLayout(root)
    resolver = LabResolver(layout)
    resolver.ensure()
    _seed_lab_files(config, layout.root)
    layout.configs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, layout.configs_dir / config_path.name)
    _copy_skill_graph_files(config, config_path.resolve().parent, layout.state_root)

    task_config = _model_from_payload(TaskConfig, config["task_config"])
    task_config = _copy_meta_agent_instruction(
        task_config=task_config,
        config_dir=config_path.resolve().parent,
        lab_root=layout.state_root,
    )
    task_config = _copy_agents_config(
        task_config=task_config,
        config_dir=config_path.resolve().parent,
        lab_root=layout.state_root,
    )
    _validate_clean_run_dynamic_role_pool(task_config)
    task_config = _copy_reflector_config_refs(
        task_config=task_config,
        config_dir=config_path.resolve().parent,
        lab_root=layout.state_root,
    )
    task_request = _model_from_payload(TaskRequest, config["task"])
    evolution_backends = _build_evolution_backends(
        config,
        trajectory_registry=resolver.trajectory_registry(),
    )
    task_path = resolver.task_registry().save_task_request(task_request)
    resolver.task_queue().enqueue(
        f"task-{task_request.task_id}",
        {"request_payload_uri": str(task_path)},
    )

    llm_backends = _build_llm_backends(
        config,
        config_dir=config_path.resolve().parent,
        backend_state_registry=resolver.backend_state_registry(),
    )
    embedding_backends = _build_embedding_backends(config, config_dir=config_path.resolve().parent)
    memory_backends = _build_memory_backends(config, config_dir=layout.state_root)

    worker = TaskWorker(
        layout=layout,
        worker_id="clean-run",
        task_config=task_config,
        llm_backends=llm_backends,
        embedding_backends=embedding_backends,
        memory_backends=memory_backends,
        skill_backends=_build_skill_backends(config, config_dir=layout.configs_dir),
        evolution_backends=evolution_backends,
        tool_registry=_build_tool_registry(config, layout.root),
        llm_backend_bindings=_llm_bindings(task_config),
        memory_backend_bindings=_memory_bindings(task_config),
        progress_callback=_print_progress,
    )
    worker.startup()
    result = worker.run_once()
    if result is None:
        raise RuntimeError("clean-run task failed")
    if evolution_backends:
        evolve_worker = EvolveWorker(
            resolver.evolve_queue(),
            evolution_backends,
            resolver.backend_state_registry(),
            worker_id="clean-run-evolve",
            trajectory_registry=resolver.trajectory_registry(),
        )
        while evolve_worker.run_once():
            pass
    return result


def _print_progress(message: str) -> None:
    print(message, flush=True)


def _compile_experiment_config(config: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    if "task_config" in config:
        return config
    task_description = config.get("task")
    if not isinstance(task_description, str) or not task_description.strip():
        raise ValueError("short experiment config requires task as a natural language string")
    meta_agent_payload = config.get("meta_agent")
    if not isinstance(meta_agent_payload, dict):
        raise ValueError("short experiment config requires meta_agent mapping")
    meta_prompt_ref = meta_agent_payload.get("prompt_ref", meta_agent_payload.get("system_prompt_ref"))
    if meta_prompt_ref is not None and (not isinstance(meta_prompt_ref, str) or not meta_prompt_ref.strip()):
        raise ValueError("meta_agent.prompt_ref must be a non-empty string when provided")
    meta_system_prompt = meta_agent_payload.get("system_prompt")
    if meta_prompt_ref is None and (not isinstance(meta_system_prompt, str) or not meta_system_prompt.strip()):
        raise ValueError("short experiment config requires meta_agent.prompt_ref or meta_agent.system_prompt")
    agents_ref_explicit = "agents_ref" in config
    if agents_ref_explicit:
        agents_ref = config.get("agents_ref")
        if not isinstance(agents_ref, str) or not agents_ref.strip():
            raise ValueError("agents_ref must be a non-empty string when provided")
    else:
        agents_ref = _DEFAULT_AGENTS_REF
    if "subagents" in config:
        raise ValueError("subagents is no longer a supported execution entry; use agents_ref or seed_agents")
    seed_agents_payload = config.get("seed_agents")
    if seed_agents_payload is not None and (not isinstance(seed_agents_payload, dict) or not seed_agents_payload):
        raise ValueError("seed_agents must be a non-empty mapping when provided")
    reflector_payload = config.get("reflector", config.get("evaluator", config.get("task_evaluator")))
    if reflector_payload is not None and not isinstance(reflector_payload, dict):
        raise ValueError("reflector/evaluator config must be a mapping when provided")
    backends = config.get("backends")
    if not isinstance(backends, dict):
        raise ValueError("short experiment config requires backends mapping")

    llm_backend_id = _default_backend_id(backends, "llm")
    task_memory_backend_id = _default_backend_id(backends, "memory", preferred=("mem0-task-memory", "task"))
    task_id = _short_task_id(config_path, config)
    allowed_tools = _default_allowed_tools(config)
    dynamic_subagents = _dynamic_subagents_from_config(
        config,
        backends,
        default_llm_backend_id=llm_backend_id,
        allowed_tools=allowed_tools,
    )
    if seed_agents_payload is not None:
        role_map = _role_map_from_seed_agents(
            seed_agents_payload,
            llm_backend_id=llm_backend_id,
            allowed_tools=allowed_tools,
        )
    else:
        role_map = {
            name: role.model_dump(mode="json")
            for name, role in default_seed_roles(
                llm_backend_id=llm_backend_id,
                allowed_tools=allowed_tools,
            ).items()
        }
    agents_path = _resolve_config_ref(config_path.resolve().parent, agents_ref)
    agents_ref_render_compiled_roles = False
    uses_dynamic_role_pool = _uses_dynamic_role_pool(dynamic_subagents)
    if agents_ref_explicit:
        if not agents_path.is_file():
            raise ValueError(f"agents_ref does not resolve to a file: {agents_ref}")
        role_map = {
            name: role.model_dump(mode="json")
            for name, role in load_agents_file(agents_path).items()
        }
        if not uses_dynamic_role_pool and _role_map_needs_transitional_agent_memory_backend(role_map):
            role_map = _role_map_with_transitional_agent_memory_backend(
                role_map,
                backend_id=task_memory_backend_id,
                overwrite_existing=False,
            )
            agents_ref_render_compiled_roles = True
    elif not uses_dynamic_role_pool:
        role_map = _role_map_with_transitional_agent_memory_backend(
            role_map,
            backend_id=task_memory_backend_id,
            overwrite_existing=False,
        )
    runtime_metadata: dict[str, Any] = {
        "task_description": task_description,
        "agents_ref_materialized_from_seed": not agents_ref_explicit,
        "agents_ref_render_compiled_roles": agents_ref_render_compiled_roles,
        "route_contract": {
            "route_field": "route",
            "end_route": "END",
            "subagent_routes": list(role_map.keys()),
        },
        "max_meta_dispatch_parse_retries": _nonnegative_int(
            config.get("max_meta_dispatch_parse_retries"),
            default=2,
        ),
        "subagent_policy": {
            "internal_planning": True,
            "must_build_internal_dag": True,
            "skill_retrieval_scope": "per_internal_dag_node",
            "tool_preparation_scope": "per_internal_dag_node_or_retrieved_skill",
        },
        "required_final_artifacts": _required_final_artifacts_from_task(task_description),
        "required_final_artifact_groups": _required_final_artifact_groups_from_task(task_description),
    }
    work_item_routing = _work_item_routing_from_config(config, role_map.keys())
    if work_item_routing is not None:
        runtime_metadata["work_item_routing"] = work_item_routing
    _copy_runtime_metadata_mapping(
        config,
        runtime_metadata,
        role_names=set(role_map),
        key="completion_guards_by_role",
        validate_role_keys=True,
    )
    _copy_runtime_metadata_mapping(
        config,
        runtime_metadata,
        role_names=set(role_map),
        key="subagent_budgets_by_role",
        validate_role_keys=True,
    )
    _copy_runtime_metadata_mapping(
        config,
        runtime_metadata,
        role_names=set(role_map),
        key="subagent_budget",
        validate_role_keys=False,
    )
    _copy_runtime_metadata_nonnegative_int(
        config,
        runtime_metadata,
        key="tool_result_prompt_max_chars",
    )
    _copy_runtime_metadata_nonnegative_int(
        config,
        runtime_metadata,
        key="max_repeated_tool_calls_per_run",
    )
    _copy_runtime_metadata_nonnegative_int(
        config,
        runtime_metadata,
        key="scientific_max_rows_per_table",
    )
    _copy_runtime_metadata_string(
        config,
        runtime_metadata,
        key="scientific_sequence_extraction_profile",
    )
    _copy_runtime_metadata_bool(
        config,
        runtime_metadata,
        key="scientific_primary_component_tables_only",
    )
    _copy_runtime_metadata_bool(
        config,
        runtime_metadata,
        key="dynamic_zero_output_recovery_enabled",
    )

    task_config = TaskConfig(
        task_id=task_id,
        goal=task_description,
        task_memory_backend=BackendBinding(backend_id=task_memory_backend_id),
        agents_ref=agents_ref,
        meta_agent=MetaAgentSpec(
            name=str(meta_agent_payload.get("name") or "meta"),
            system_prompt=str(meta_system_prompt or ""),
            prompt_ref=meta_prompt_ref,
            llm_backend=_backend_binding_from_config(
                meta_agent_payload.get("llm_backend"),
                default_backend_id=llm_backend_id,
                field_name="meta_agent.llm_backend",
            ),
            memory_backend=_optional_backend_binding_from_config(
                meta_agent_payload.get("memory_backend", meta_agent_payload.get("meta_memory_backend")),
                field_name="meta_agent.memory_backend",
            ),
            instruction_ref=meta_agent_payload.get("instruction_ref"),
        ),
        reflector=_reflector_spec_from_config(
            reflector_payload,
            default_backend_id=llm_backend_id,
        ),
        roles=role_map,
        dynamic_subagents=dynamic_subagents,
        max_dispatch_steps=_positive_int(config.get("max_dispatch_steps"), default=20),
        runtime_policy=RuntimePolicy(
            max_tool_steps=_nonnegative_int(config.get("max_tool_steps"), default=20),
            max_tool_steps_per_node=_optional_nonnegative_int(config.get("max_tool_steps_per_node")),
            max_workflow_nodes=_positive_int(config.get("max_workflow_nodes"), default=20),
            allow_human_tools=bool(config.get("allow_human_tools", False)),
            enable_workflow_planning=bool(config.get("enable_workflow_planning", True)),
            metadata=runtime_metadata,
        ),
    ).model_dump(mode="json")
    task = TaskRequest(
        task_id=task_id,
        origin=TaskOrigin.HUMAN,
        purpose=TaskPurpose.SCIENCE,
        goal=task_description,
        task_config_ref=str(config_path),
        metadata={"task_description": task_description},
    ).model_dump(mode="json")
    compiled = dict(config)
    compiled["task"] = task
    compiled["task_config"] = task_config
    compiled.setdefault("tools", {"scientific_ie": {"enabled": True, "include_human_tools": False}})
    return compiled


def _dynamic_subagents_from_config(
    config: dict[str, Any],
    backends: dict[str, Any],
    *,
    default_llm_backend_id: str,
    allowed_tools: list[str],
) -> DynamicSubagentsConfig | None:
    raw = config.get("dynamic_subagents")
    if raw is None:
        parsed = DynamicSubagentsConfig.model_validate(
            {
                "enabled": True,
                "mode": "dynamic",
                "scope": "per_task",
                "planner_backend": {"backend_id": default_llm_backend_id},
                "default_worker_backend": {"backend_id": default_llm_backend_id},
                "allowed_tool_names": list(allowed_tools),
            }
        )
    else:
        if not isinstance(raw, dict):
            raise ValueError("dynamic_subagents must be a mapping")
        parsed = DynamicSubagentsConfig.model_validate(raw)
        if "allowed_tool_names" not in raw:
            parsed = parsed.model_copy(update={"allowed_tool_names": list(allowed_tools)})
    if not parsed.enabled or parsed.mode != "dynamic":
        raise ValueError("dynamic_subagents must be enabled with mode=dynamic for clean-run execution")
    llm_backends = backends.get("llm")
    if not isinstance(llm_backends, dict):
        raise ValueError("dynamic_subagents requires configured backends.llm")
    planner_id = parsed.planner_backend.backend_id if parsed.planner_backend else None
    worker_id = parsed.default_worker_backend.backend_id if parsed.default_worker_backend else None
    checked_backend_ids = [
        backend_id
        for backend_id in (planner_id, worker_id, *parsed.allowed_worker_backend_ids)
        if backend_id
    ]
    missing = [backend_id for backend_id in checked_backend_ids if backend_id not in llm_backends]
    if missing:
        raise ValueError(f"dynamic_subagents references unknown LLM backend(s): {', '.join(missing)}")
    return parsed


def _role_map_from_seed_agents(
    seed_agents: dict[str, Any],
    *,
    llm_backend_id: str,
    allowed_tools: list[str],
) -> dict[str, dict[str, Any]]:
    role_map: dict[str, dict[str, Any]] = {}
    for name, payload in seed_agents.items():
        if not isinstance(name, str) or not name:
            raise ValueError("seed_agents names must be non-empty strings")
        if not isinstance(payload, dict):
            raise ValueError(f"seed_agents.{name} must be a mapping")
        system_prompt = payload.get("system_prompt")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError(f"seed_agents.{name}.system_prompt is required")
        role_map[name] = RoleSpec(
            name=name,
            system_prompt=system_prompt,
            llm_backend=_backend_binding_from_config(
                payload.get("llm_backend"),
                default_backend_id=llm_backend_id,
                field_name=f"seed_agents.{name}.llm_backend",
            ),
            agent_memory_backend=_optional_backend_binding_from_config(
                payload.get("agent_memory_backend"),
                field_name=f"seed_agents.{name}.agent_memory_backend",
            ),
            allowed_tools=_string_list_or_default(payload.get("allowed_tools"), allowed_tools),
            required_skills=_string_list_or_default(payload.get("required_skills", payload.get("skillset")), []),
            memory_policy=dict(payload.get("memory_policy") or {}),
            metadata=dict(payload.get("metadata") or {}),
        ).model_dump(mode="json")
    return role_map


def _role_map_needs_transitional_agent_memory_backend(role_map: dict[str, dict[str, Any]]) -> bool:
    return any(payload.get("agent_memory_backend") is None for payload in role_map.values())


def _uses_dynamic_role_pool(dynamic_subagents: DynamicSubagentsConfig | None) -> bool:
    return dynamic_subagents is not None and dynamic_subagents.enabled and dynamic_subagents.mode != "static"


def _role_map_with_transitional_agent_memory_backend(
    role_map: dict[str, dict[str, Any]],
    *,
    backend_id: str,
    overwrite_existing: bool = True,
) -> dict[str, dict[str, Any]]:
    binding = BackendBinding(backend_id=backend_id).model_dump(mode="json")
    updated: dict[str, dict[str, Any]] = {}
    for name, payload in role_map.items():
        if overwrite_existing or payload.get("agent_memory_backend") is None:
            updated[name] = {**payload, "agent_memory_backend": binding}
        else:
            updated[name] = dict(payload)
    return updated


def _short_task_id(config_path: Path, config: dict[str, Any]) -> str:
    explicit = config.get("task_id")
    if isinstance(explicit, str) and explicit:
        return explicit
    stem = config_path.stem if config_path.stem else "task"
    return _safe_config_id(stem)


def _safe_config_id(value: str) -> str:
    safe = "".join(char.lower() if char.isalnum() else "-" for char in value)
    safe = "-".join(part for part in safe.split("-") if part)
    return safe or "task"


def _default_backend_id(
    backends: dict[str, Any],
    section_name: str,
    *,
    preferred: tuple[str, ...] = (),
) -> str:
    section = backends.get(section_name)
    if not isinstance(section, dict) or not section:
        raise ValueError(f"short experiment config requires backends.{section_name}")
    for name in preferred:
        if name in section:
            return name
    return str(next(iter(section.keys())))


def _default_allowed_tools(config: dict[str, Any]) -> list[str]:
    raw_tools = config.get("allowed_tools")
    if isinstance(raw_tools, list) and all(isinstance(item, str) for item in raw_tools):
        return list(raw_tools)
    return [
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


def _work_item_routing_from_config(config: dict[str, Any], role_names: Any) -> dict[str, Any] | None:
    raw = config.get("work_item_routing")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("work_item_routing must be a mapping")
    enabled = bool(raw.get("enabled", False))
    role_set = {str(name) for name in role_names}
    policy = {
        "enabled": enabled,
        "executor_roles": _validated_policy_roles(raw.get("executor_roles"), role_set, "executor_roles"),
        "reviewer_roles": _validated_policy_roles(raw.get("reviewer_roles"), role_set, "reviewer_roles"),
        "finalizer_roles": _validated_policy_roles(raw.get("finalizer_roles"), role_set, "finalizer_roles"),
        "required_work_item_ids": _validated_work_item_ids(raw.get("required_work_item_ids")),
        "work_item_id_field": str(raw.get("work_item_id_field") or "work_item_id"),
    }
    max_failed_attempts = _optional_nonnegative_int(raw.get("max_failed_executor_attempts_per_work_item"))
    if max_failed_attempts is not None and max_failed_attempts > 0:
        policy["max_failed_executor_attempts_per_work_item"] = max_failed_attempts
    if enabled and not policy["executor_roles"]:
        raise ValueError("enabled work_item_routing requires at least one executor role")
    if enabled and not policy["reviewer_roles"]:
        raise ValueError("enabled work_item_routing requires at least one reviewer role")
    return policy


def _validated_work_item_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError("work_item_routing.required_work_item_ids must be a list of non-empty strings")
    return list(value)


def _validated_policy_roles(value: Any, role_names: set[str], field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"work_item_routing.{field_name} must be a list of role names")
    unknown = sorted(set(value) - role_names)
    if unknown:
        raise ValueError(f"work_item_routing.{field_name} contains unknown role(s): {', '.join(unknown)}")
    return list(value)


def _copy_runtime_metadata_mapping(
    config: dict[str, Any],
    runtime_metadata: dict[str, Any],
    *,
    role_names: set[str],
    key: str,
    validate_role_keys: bool,
) -> None:
    value = config.get(key)
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    if validate_role_keys:
        unknown = sorted(item for item in value if not isinstance(item, str) or item not in role_names)
        if unknown:
            raise ValueError(f"{key} contains unknown role(s): {', '.join(str(item) for item in unknown)}")
    runtime_metadata[key] = json.loads(json.dumps(value))


def _copy_runtime_metadata_nonnegative_int(
    config: dict[str, Any],
    runtime_metadata: dict[str, Any],
    *,
    key: str,
) -> None:
    value = config.get(key)
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    runtime_metadata[key] = value


def _copy_runtime_metadata_bool(config: dict[str, Any], runtime_metadata: dict[str, Any], *, key: str) -> None:
    value = config.get(key)
    if isinstance(value, bool):
        runtime_metadata[key] = value


def _copy_runtime_metadata_string(config: dict[str, Any], runtime_metadata: dict[str, Any], *, key: str) -> None:
    value = config.get(key)
    if isinstance(value, str) and value:
        runtime_metadata[key] = value


def _required_final_artifacts_from_task(task_description: str) -> list[str]:
    candidates = ["biology_component_records.jsonl"]
    report_names = _biology_report_artifacts_in_task(task_description)
    if len(report_names) == 1:
        candidates.append(report_names[0])
    return [name for name in candidates if name in task_description]


def _required_final_artifact_groups_from_task(task_description: str) -> list[dict[str, Any]]:
    report_names = _biology_report_artifacts_in_task(task_description)
    if len(report_names) < 2:
        return []
    return [
        {
            "one_of": report_names,
            "description": "biology component final report or audit artifact",
        }
    ]


def _biology_report_artifacts_in_task(task_description: str) -> list[str]:
    return [
        name
        for name in ("biology_component_report.md", "biology_component_report.json")
        if name in task_description
    ]


def _string_list_or_default(value: Any, default: list[str]) -> list[str]:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    return list(default)


def _reflector_spec_from_config(
    payload: dict[str, Any] | None,
    *,
    default_backend_id: str,
) -> ReflectorSpec | None:
    if payload is None:
        return None
    prompt_ref = payload.get("prompt_ref", payload.get("system_prompt_ref"))
    if prompt_ref is not None and (not isinstance(prompt_ref, str) or not prompt_ref.strip()):
        raise ValueError("reflector.prompt_ref must be a non-empty string when provided")
    system_prompt = payload.get("system_prompt")
    if prompt_ref is None and (not isinstance(system_prompt, str) or not system_prompt.strip()):
        raise ValueError("reflector requires prompt_ref or system_prompt")
    return ReflectorSpec(
        name=str(payload.get("name") or "reflector"),
        system_prompt=str(system_prompt or ""),
        prompt_ref=prompt_ref,
        llm_backend=_backend_binding_from_config(
            payload.get("llm_backend"),
            default_backend_id=default_backend_id,
            field_name="reflector.llm_backend",
        ),
        instruction_ref=payload.get("instruction_ref"),
        ground_truth=payload.get("ground_truth"),
        ground_truth_ref=payload.get("ground_truth_ref"),
        rubric=payload.get("rubric"),
        rubric_ref=payload.get("rubric_ref"),
        metadata=dict(payload.get("metadata") or {}),
    )


def _positive_int(value: Any, *, default: int) -> int:
    parsed = _nonnegative_int(value, default=default)
    return parsed if parsed > 0 else default


def _nonnegative_int(value: Any, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, int) and value >= 0:
        return value
    return default


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evolab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    clean_run = subparsers.add_parser("clean-run", help="Developer helper for legacy dev configs.")
    clean_run.add_argument("config", nargs="?", default="dev/configs/demo_v0.yaml")
    clean_run.add_argument("--lab-root", default=None)
    export_sft = subparsers.add_parser("export-sft")
    export_sft.add_argument("--lab-root", required=True)
    export_sft.add_argument("--output-dir", default=None)
    export_sft.add_argument("--teacher-backend-id", action="append", default=[])
    export_sft.add_argument("--runtime-stage", action="append", default=[])
    export_sft.add_argument("--action", action="append", default=[])
    export_sft.add_argument("--source-run-ref", action="append", default=[])
    export_sft.add_argument("--source-llm-call-ref", action="append", default=[])
    export_sft.add_argument("--include-meta-agent", action="store_true")
    export_sft.add_argument("--include-tool-call-samples", action="store_true")
    export_sft.add_argument("--val-fraction", type=float, default=0.0)
    export_sft.add_argument("--shuffle", action="store_true")
    export_sft.add_argument("--seed", type=int, default=0)
    export_opsd = subparsers.add_parser("export-opsd")
    export_opsd.add_argument("--lab-root", required=True)
    export_opsd.add_argument("--output-dir", default=None)
    export_opsd.add_argument("--teacher-backend-id", action="append", default=[])
    export_opsd.add_argument("--source-run-ref", action="append", default=[])
    export_opsd.add_argument("--source-llm-call-ref", action="append", default=[])
    export_opsd.add_argument("--include-subagent-choices", action="store_true")
    export_opsd.add_argument("--exclude-tool-choices", action="store_true")
    export_opsd.add_argument("--val-fraction", type=float, default=0.0)
    export_opsd.add_argument("--shuffle", action="store_true")
    export_opsd.add_argument("--seed", type=int, default=0)
    train_sft = subparsers.add_parser("train-sft")
    train_sft.add_argument("--lab-root", required=True)
    train_sft.add_argument("--backend-id", required=True)
    train_sft.add_argument("--artifact-root", default=None)
    train_sft.add_argument("--training-backend", choices=["dry_run", "transformers"], default="dry_run")
    train_sft.add_argument("--base-model-ref", default=None)
    train_sft.add_argument("--previous-state-ref", default=None)
    train_sft.add_argument("--teacher-backend-id", action="append", default=[])
    train_sft.add_argument("--runtime-stage", action="append", default=[])
    train_sft.add_argument("--action", action="append", default=[])
    train_sft.add_argument("--source-run-ref", action="append", default=[])
    train_sft.add_argument("--source-llm-call-ref", action="append", default=[])
    train_sft.add_argument("--include-meta-agent", action="store_true")
    train_sft.add_argument("--include-tool-call-samples", action="store_true")
    train_sft.add_argument("--val-fraction", type=float, default=0.0)
    train_sft.add_argument("--shuffle", action="store_true")
    train_sft.add_argument("--seed", type=int, default=0)
    train_sft.add_argument("--max-length", type=int, default=2048)
    train_sft.add_argument("--min-train-samples", type=int, default=1)
    train_sft.add_argument("--promote-dry-run", action="store_true")
    train_sft.add_argument("--training-arg", action="append", default=[])
    train_opsd = subparsers.add_parser("train-opsd")
    train_opsd.add_argument("--lab-root", required=True)
    train_opsd.add_argument("--backend-id", required=True)
    train_opsd.add_argument("--artifact-root", default=None)
    train_opsd.add_argument("--previous-state-ref", default=None)
    train_opsd.add_argument("--teacher-backend-id", action="append", default=[])
    train_opsd.add_argument("--source-run-ref", action="append", default=[])
    train_opsd.add_argument("--source-llm-call-ref", action="append", default=[])
    train_opsd.add_argument("--include-subagent-choices", action="store_true")
    train_opsd.add_argument("--exclude-tool-choices", action="store_true")
    train_opsd.add_argument("--val-fraction", type=float, default=0.0)
    train_opsd.add_argument("--shuffle", action="store_true")
    train_opsd.add_argument("--seed", type=int, default=0)
    train_opsd.add_argument("--min-train-samples", type=int, default=1)
    train_opsd.add_argument("--promote-dry-run", action="store_true")
    train_opsd.add_argument("--training-arg", action="append", default=[])
    visualize = subparsers.add_parser("visualize-trajectory")
    visualize_source = visualize.add_mutually_exclusive_group(required=True)
    visualize_source.add_argument("--lab-root")
    visualize_source.add_argument("--trajectory-dir")
    visualize.add_argument("--output", default=None)
    visualize.add_argument("--title", default=None)

    args = parser.parse_args(argv)
    if args.command == "clean-run":
        result = run_clean_demo(Path(args.config), Path(args.lab_root) if args.lab_root else None)
        print(json.dumps(_json_compatible(result), indent=2, sort_keys=True))
        return 0
    if args.command == "export-sft":
        result = run_export_sft(
            lab_root=Path(args.lab_root),
            output_dir=Path(args.output_dir) if args.output_dir else None,
            teacher_backend_ids=args.teacher_backend_id,
            runtime_stages=args.runtime_stage,
            actions=args.action,
            source_run_refs=args.source_run_ref,
            source_llm_call_refs=args.source_llm_call_ref,
            include_meta_agent=args.include_meta_agent,
            include_tool_call_samples=args.include_tool_call_samples,
            val_fraction=args.val_fraction,
            shuffle=args.shuffle,
            seed=args.seed,
        )
        print(json.dumps(_json_compatible(result), indent=2, sort_keys=True))
        return 0
    if args.command == "export-opsd":
        result = run_export_opsd(
            lab_root=Path(args.lab_root),
            output_dir=Path(args.output_dir) if args.output_dir else None,
            teacher_backend_ids=args.teacher_backend_id,
            source_run_refs=args.source_run_ref,
            source_llm_call_refs=args.source_llm_call_ref,
            include_tool_choices=not args.exclude_tool_choices,
            include_subagent_choices=args.include_subagent_choices,
            val_fraction=args.val_fraction,
            shuffle=args.shuffle,
            seed=args.seed,
        )
        print(json.dumps(_json_compatible(result), indent=2, sort_keys=True))
        return 0
    if args.command == "train-sft":
        result = run_train_sft(
            lab_root=Path(args.lab_root),
            backend_id=args.backend_id,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
            training_backend=args.training_backend,
            base_model_ref=args.base_model_ref,
            previous_state_ref=args.previous_state_ref,
            teacher_backend_ids=args.teacher_backend_id,
            runtime_stages=args.runtime_stage,
            actions=args.action,
            source_run_refs=args.source_run_ref,
            source_llm_call_refs=args.source_llm_call_ref,
            include_meta_agent=args.include_meta_agent,
            include_tool_call_samples=args.include_tool_call_samples,
            val_fraction=args.val_fraction,
            shuffle=args.shuffle,
            seed=args.seed,
            max_length=args.max_length,
            min_train_samples=args.min_train_samples,
            promote_dry_run=args.promote_dry_run,
            training_args=_parse_training_args(args.training_arg),
        )
        print(json.dumps(_json_compatible(result), indent=2, sort_keys=True))
        return 0
    if args.command == "train-opsd":
        result = run_train_opsd(
            lab_root=Path(args.lab_root),
            backend_id=args.backend_id,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
            previous_state_ref=args.previous_state_ref,
            teacher_backend_ids=args.teacher_backend_id,
            source_run_refs=args.source_run_ref,
            source_llm_call_refs=args.source_llm_call_ref,
            include_tool_choices=not args.exclude_tool_choices,
            include_subagent_choices=args.include_subagent_choices,
            val_fraction=args.val_fraction,
            shuffle=args.shuffle,
            seed=args.seed,
            min_train_samples=args.min_train_samples,
            promote_dry_run=args.promote_dry_run,
            training_args=_parse_training_args(args.training_arg),
        )
        print(json.dumps(_json_compatible(result), indent=2, sort_keys=True))
        return 0
    if args.command == "visualize-trajectory":
        result = visualize_trajectory(
            lab_root=Path(args.lab_root) if args.lab_root else None,
            trajectory_dir=Path(args.trajectory_dir) if args.trajectory_dir else None,
            output_path=Path(args.output) if args.output else None,
            title=args.title,
        )
        print(json.dumps(_json_compatible(result), indent=2, sort_keys=True))
        return 0
    raise ValueError(f"unsupported command: {args.command}")


def run_export_sft(
    *,
    lab_root: Path | str,
    output_dir: Path | str | None = None,
    teacher_backend_ids: list[str] | None = None,
    runtime_stages: list[str] | None = None,
    actions: list[str] | None = None,
    source_run_refs: list[str] | None = None,
    source_llm_call_refs: list[str] | None = None,
    include_meta_agent: bool = False,
    include_tool_call_samples: bool = False,
    val_fraction: float = 0.0,
    shuffle: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    layout = LabLayout(lab_root)
    resolver = LabResolver(layout)
    destination = Path(output_dir) if output_dir is not None else layout.root / "artifacts" / "sft"
    result = export_sft_dataset(
        trajectory_registry=resolver.trajectory_registry(),
        output_dir=destination,
        config=SFTExportConfig(
            teacher_backend_ids=teacher_backend_ids or [],
            runtime_stages=runtime_stages or ["subagent_flat", "workflow_node"],
            actions=actions or ["final_answer"],
            source_run_refs=source_run_refs or [],
            source_llm_call_refs=source_llm_call_refs or [],
            include_meta_agent=include_meta_agent,
            include_tool_call_samples=include_tool_call_samples,
            val_fraction=val_fraction,
            shuffle=shuffle,
            seed=seed,
        ),
    )
    return {
        "manifest": result.manifest,
        "manifest_path": str(result.manifest_path),
        "train_path": str(result.train_path),
        "val_path": str(result.val_path),
    }


def run_train_sft(
    *,
    lab_root: Path | str,
    backend_id: str,
    artifact_root: Path | str | None = None,
    training_backend: str = "dry_run",
    base_model_ref: str | None = None,
    previous_state_ref: str | None = None,
    teacher_backend_ids: list[str] | None = None,
    runtime_stages: list[str] | None = None,
    actions: list[str] | None = None,
    source_run_refs: list[str] | None = None,
    source_llm_call_refs: list[str] | None = None,
    include_meta_agent: bool = False,
    include_tool_call_samples: bool = False,
    val_fraction: float = 0.0,
    shuffle: bool = False,
    seed: int = 0,
    max_length: int = 2048,
    min_train_samples: int = 1,
    promote_dry_run: bool = False,
    training_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    layout = LabLayout(lab_root)
    resolver = LabResolver(layout)
    destination = (
        Path(artifact_root)
        if artifact_root is not None
        else layout.root / "artifacts" / "sft-train"
    )
    trainer = SFTTrainer(
        trajectory_registry=resolver.trajectory_registry(),
        config=SFTTrainerConfig(
            training_backend=training_backend,
            base_model_ref=base_model_ref,
            max_length=max_length,
            min_train_samples=min_train_samples,
            promote_dry_run=promote_dry_run,
            export=SFTExportConfig(
                teacher_backend_ids=teacher_backend_ids or [],
                runtime_stages=runtime_stages or ["subagent_flat", "workflow_node"],
                actions=actions or ["final_answer"],
                source_run_refs=source_run_refs or [],
                source_llm_call_refs=source_llm_call_refs or [],
                include_meta_agent=include_meta_agent,
                include_tool_call_samples=include_tool_call_samples,
                val_fraction=val_fraction,
                shuffle=shuffle,
                seed=seed,
            ),
            training_args=training_args or {},
        ),
    )
    run_ref = f"sft-{uuid4()}"
    request = LLMEvolutionRequest(
        mode=LLMEvolutionMode.BASICS,
        backend_id=backend_id,
        previous_state_ref=previous_state_ref,
        artifact_root_uri=str(destination),
        trigger_trajectory_ref=_single_source_run_ref(source_run_refs),
        proposer_input_refs=_source_run_proposer_refs(source_run_refs or []),
        metadata={"trigger": "train-sft"},
    )
    executor = EvolutionExecutor(resolver.backend_state_registry(), worker_id="train-sft")
    outcome = executor.run(request=request, trainer=trainer, run_ref=run_ref)
    TrajectoryCollector(resolver.trajectory_registry()).save_evolution_run(
        EvolutionRunRecord(
            run_ref=run_ref,
            mode=request.mode,
            backend_id=request.backend_id,
            result_status=outcome.result.status,
            result=outcome.result,
            training_trajectory_refs=_sft_result_source_run_refs(outcome.result) or _request_training_refs(request),
            output_snapshot_refs=_metadata_string_list(outcome.result.metadata, "output_snapshot_refs"),
            lora_role=outcome.result.lora_role,
            metadata={
                "worker_id": "train-sft",
                "trainer_id": trainer.trainer_id,
                "request": request.model_dump(mode="json"),
                "promotion_errors": outcome.promotion_errors,
                "promoted": outcome.promoted,
            },
        )
    )
    return {
        "artifact_root": str(destination),
        "run_ref": run_ref,
        "result": outcome.result,
        "promotion_errors": outcome.promotion_errors,
        "promoted": outcome.promoted,
    }


def run_export_opsd(
    *,
    lab_root: Path | str,
    output_dir: Path | str | None = None,
    teacher_backend_ids: list[str] | None = None,
    source_run_refs: list[str] | None = None,
    source_llm_call_refs: list[str] | None = None,
    include_tool_choices: bool = True,
    include_subagent_choices: bool = False,
    val_fraction: float = 0.0,
    shuffle: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    layout = LabLayout(lab_root)
    resolver = LabResolver(layout)
    destination = Path(output_dir) if output_dir is not None else layout.root / "artifacts" / "opsd"
    result = export_opsd_dataset(
        trajectory_registry=resolver.trajectory_registry(),
        output_dir=destination,
        config=OPSDExportConfig(
            teacher_backend_ids=teacher_backend_ids or [],
            source_run_refs=source_run_refs or [],
            source_llm_call_refs=source_llm_call_refs or [],
            include_tool_choices=include_tool_choices,
            include_subagent_choices=include_subagent_choices,
            val_fraction=val_fraction,
            shuffle=shuffle,
            seed=seed,
        ),
    )
    return {
        "manifest": result.manifest,
        "manifest_path": str(result.manifest_path),
        "train_path": str(result.train_path),
        "val_path": str(result.val_path),
    }


def run_train_opsd(
    *,
    lab_root: Path | str,
    backend_id: str,
    artifact_root: Path | str | None = None,
    previous_state_ref: str | None = None,
    teacher_backend_ids: list[str] | None = None,
    source_run_refs: list[str] | None = None,
    source_llm_call_refs: list[str] | None = None,
    include_tool_choices: bool = True,
    include_subagent_choices: bool = False,
    val_fraction: float = 0.0,
    shuffle: bool = False,
    seed: int = 0,
    min_train_samples: int = 1,
    promote_dry_run: bool = False,
    training_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    layout = LabLayout(lab_root)
    resolver = LabResolver(layout)
    destination = (
        Path(artifact_root)
        if artifact_root is not None
        else layout.root / "artifacts" / "opsd-train"
    )
    trainer = OPSDTrainer(
        trajectory_registry=resolver.trajectory_registry(),
        config=OPSDTrainerConfig(
            min_train_samples=min_train_samples,
            promote_dry_run=promote_dry_run,
            export=OPSDExportConfig(
                teacher_backend_ids=teacher_backend_ids or [],
                source_run_refs=source_run_refs or [],
                source_llm_call_refs=source_llm_call_refs or [],
                include_tool_choices=include_tool_choices,
                include_subagent_choices=include_subagent_choices,
                val_fraction=val_fraction,
                shuffle=shuffle,
                seed=seed,
            ),
            training_args=training_args or {},
        ),
    )
    run_ref = f"opsd-{uuid4()}"
    request = LLMEvolutionRequest(
        mode=LLMEvolutionMode.BASICS,
        backend_id=backend_id,
        previous_state_ref=previous_state_ref,
        artifact_root_uri=str(destination),
        trigger_trajectory_ref=_single_source_run_ref(source_run_refs),
        proposer_input_refs=_training_source_proposer_refs(
            source_run_refs or [],
            role="opsd_source",
            source="train-opsd",
        ),
        metadata={"trigger": "train-opsd"},
    )
    executor = EvolutionExecutor(resolver.backend_state_registry(), worker_id="train-opsd")
    outcome = executor.run(request=request, trainer=trainer, run_ref=run_ref)
    TrajectoryCollector(resolver.trajectory_registry()).save_evolution_run(
        EvolutionRunRecord(
            run_ref=run_ref,
            mode=request.mode,
            backend_id=request.backend_id,
            result_status=outcome.result.status,
            result=outcome.result,
            training_trajectory_refs=_opsd_result_source_run_refs(outcome.result) or _request_training_refs(request),
            output_snapshot_refs=_metadata_string_list(outcome.result.metadata, "output_snapshot_refs"),
            lora_role=outcome.result.lora_role,
            metadata={
                "worker_id": "train-opsd",
                "trainer_id": trainer.trainer_id,
                "request": request.model_dump(mode="json"),
                "promotion_errors": outcome.promotion_errors,
                "promoted": outcome.promoted,
            },
        )
    )
    return {
        "artifact_root": str(destination),
        "run_ref": run_ref,
        "result": outcome.result,
        "promotion_errors": outcome.promotion_errors,
        "promoted": outcome.promoted,
    }


def _clean_lab_root(root: Path) -> None:
    _validate_clean_lab_root(root)
    if root.exists():
        shutil.rmtree(root)


def _validate_clean_lab_root(root: Path) -> None:
    resolved = root.resolve()
    cwd = Path.cwd().resolve()
    home = Path.home().resolve()
    protected_roots = {Path(resolved.anchor).resolve(), cwd, home}
    if (
        resolved in protected_roots
        or len(resolved.parts) <= 2
        or ".git" in resolved.parts
        or (resolved / ".git").exists()
    ):
        raise ValueError(f"refusing to clean unsafe lab root: {root}")


def _load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ModuleNotFoundError:
        payload = json.loads(text)
    else:
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return payload


def _model_from_payload(model_type: type[ModelT], payload: Any) -> ModelT:
    return model_type.model_validate_json(json.dumps(payload))


def _copy_meta_agent_instruction(
    *,
    task_config: TaskConfig,
    config_dir: Path,
    lab_root: Path,
) -> TaskConfig:
    if task_config.meta_agent is None:
        return task_config
    meta_agent = task_config.meta_agent
    prompt_ref = _copy_meta_agent_ref(meta_agent.prompt_ref, config_dir=config_dir, lab_root=lab_root)
    instruction_ref = _copy_meta_agent_ref(meta_agent.instruction_ref, config_dir=config_dir, lab_root=lab_root)
    if prompt_ref == meta_agent.prompt_ref and instruction_ref == meta_agent.instruction_ref:
        return task_config
    return task_config.model_copy(
        update={
            "meta_agent": meta_agent.model_copy(
                update={
                    "prompt_ref": prompt_ref,
                    "instruction_ref": instruction_ref,
                }
            )
        }
    )


def _copy_meta_agent_ref(ref: str | None, *, config_dir: Path, lab_root: Path) -> str | None:
    if ref is None:
        return None
    source = _resolve_config_ref(config_dir, ref)
    destination = _lab_relative_path(lab_root, _lab_config_ref(ref, source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination)


def _copy_agents_config(
    *,
    task_config: TaskConfig,
    config_dir: Path,
    lab_root: Path,
) -> TaskConfig:
    if task_config.agents_ref is None:
        return task_config
    agents_ref = task_config.agents_ref
    if (
        task_config.runtime_policy.metadata.get("agents_ref_materialized_from_seed") is True
        or task_config.runtime_policy.metadata.get("agents_ref_render_compiled_roles") is True
    ):
        if not task_config.roles:
            raise ValueError(f"agents_ref is marked for compiled role rendering but no roles are available: {agents_ref}")
        destination_ref = _lab_config_ref(agents_ref, Path(agents_ref))
        destination = _lab_relative_path(lab_root, destination_ref)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            render_agents_markdown(task_config.roles, note=_SEED_AGENTS_NOTE),
            encoding="utf-8",
        )
        return task_config.model_copy(update={"agents_ref": destination_ref})
    source = _resolve_config_ref(config_dir, agents_ref)
    destination_ref = _lab_config_ref(agents_ref, source)
    destination = _lab_relative_path(lab_root, destination_ref)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_file():
        shutil.copy2(source, destination)
        roles = load_agents_file(destination)
        return task_config.model_copy(update={"agents_ref": destination_ref, "roles": roles})
    else:
        if not task_config.roles:
            raise ValueError(f"agents_ref does not exist and no roles are available: {source}")
        destination.write_text(
            render_agents_markdown(task_config.roles, note=_SEED_AGENTS_NOTE),
            encoding="utf-8",
        )
    return task_config.model_copy(update={"agents_ref": destination_ref})


def _validate_clean_run_dynamic_role_pool(task_config: TaskConfig) -> None:
    dynamic = task_config.dynamic_subagents
    if dynamic is None or not dynamic.enabled or dynamic.mode != "dynamic":
        raise ValueError("clean-run requires dynamic_subagents.enabled=true with mode=dynamic")
    if task_config.agents_ref is None:
        raise ValueError("clean-run requires agents_ref so agents.md is the active role pool")


def _copy_reflector_config_refs(
    *,
    task_config: TaskConfig,
    config_dir: Path,
    lab_root: Path,
) -> TaskConfig:
    if task_config.reflector is None:
        return task_config
    reflector = task_config.reflector
    updates = {
        "prompt_ref": _copy_meta_agent_ref(reflector.prompt_ref, config_dir=config_dir, lab_root=lab_root),
        "instruction_ref": _copy_meta_agent_ref(reflector.instruction_ref, config_dir=config_dir, lab_root=lab_root),
        "ground_truth_ref": _copy_meta_agent_ref(reflector.ground_truth_ref, config_dir=config_dir, lab_root=lab_root),
        "rubric_ref": _copy_meta_agent_ref(reflector.rubric_ref, config_dir=config_dir, lab_root=lab_root),
    }
    if all(getattr(reflector, key) == value for key, value in updates.items()):
        return task_config
    return task_config.model_copy(
        update={
            "reflector": reflector.model_copy(update=updates)
        }
    )


def _resolve_config_ref(config_dir: Path, ref: str) -> Path:
    path = Path(ref)
    if path.is_absolute():
        return path
    candidates = [config_dir / path]
    if path.parts and path.parts[0] == config_dir.name:
        candidates.append(config_dir.parent / path)
    candidates.append(Path.cwd() / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _lab_config_ref(ref: str, source: Path) -> str:
    path = Path(ref)
    if path.is_absolute():
        return str(Path("configs") / source.name)
    if path.parts and path.parts[0] == "configs":
        return str(path)
    return str(Path("configs") / path)


def _seed_lab_files(config: dict[str, Any], lab_root: Path) -> None:
    files = config.get("files", {})
    if not isinstance(files, dict):
        raise ValueError("files must be a mapping")
    for relative_path, content in files.items():
        if not isinstance(relative_path, str):
            raise ValueError("file path keys must be strings")
        path = _lab_relative_path(lab_root, relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(content), encoding="utf-8")


def _copy_skill_graph_files(config: dict[str, Any], config_dir: Path, lab_root: Path) -> None:
    for payload in _backend_section(config, "skill").values():
        if payload.get("type") != "graph":
            continue
        graph_path_value = payload.get("graph_path") or payload.get("path")
        if not isinstance(graph_path_value, str) or not graph_path_value:
            continue
        source = _resolve_config_ref(config_dir, graph_path_value)
        destination = _lab_relative_path(lab_root, _lab_config_ref(graph_path_value, source))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        _copy_skill_packages_from_graph(source, config_dir=config_dir, lab_root=lab_root)
        _copy_skill_group_configs_from_graph(source, config_dir=config_dir, lab_root=lab_root)
        _copy_domain_packages_from_graph(source, config_dir=config_dir, lab_root=lab_root)


def _copy_skill_packages_from_graph(graph_source: Path, *, config_dir: Path, lab_root: Path) -> None:
    try:
        payload = json.loads(graph_source.read_text(encoding="utf-8"))
    except Exception:
        return
    refs: list[str] = []
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        roots = metadata.get("skill_roots", [])
        if isinstance(roots, list):
            refs.extend(ref for ref in roots if isinstance(ref, str) and ref)
    for raw_skill in payload.get("skills", []):
        if not isinstance(raw_skill, dict):
            continue
        package_ref = raw_skill.get("package_ref")
        if isinstance(package_ref, str) and package_ref:
            refs.append(package_ref)
    _copy_directory_refs(refs, config_dir=config_dir, lab_root=lab_root)


def _copy_skill_group_configs_from_graph(graph_source: Path, *, config_dir: Path, lab_root: Path) -> None:
    try:
        payload = json.loads(graph_source.read_text(encoding="utf-8"))
    except Exception:
        return
    group_names: set[str] = set()
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        groups = metadata.get("skill_groups", [])
        if isinstance(groups, str):
            group_names.add(groups)
        elif isinstance(groups, list):
            group_names.update(group for group in groups if isinstance(group, str) and group)
    for raw_skill in payload.get("skills", []):
        if isinstance(raw_skill, dict) and isinstance(raw_skill.get("group"), str):
            group_names.add(raw_skill["group"])
    for group_name in sorted(group_names):
        source = _resolve_config_ref(config_dir, f"configs/skills/groups/{group_name}.yaml")
        if not source.is_file():
            continue
        destination = _lab_relative_path(lab_root, f"configs/skills/groups/{group_name}.yaml")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _copy_domain_packages_from_graph(graph_source: Path, *, config_dir: Path, lab_root: Path) -> None:
    try:
        payload = json.loads(graph_source.read_text(encoding="utf-8"))
    except Exception:
        return
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return
    package_refs = metadata.get("domain_packages", [])
    if not isinstance(package_refs, list):
        return
    _copy_directory_refs(package_refs, config_dir=config_dir, lab_root=lab_root)


def _copy_directory_refs(refs: list[Any], *, config_dir: Path, lab_root: Path) -> None:
    copied: set[Path] = set()
    for ref in refs:
        if not isinstance(ref, str) or not ref:
            continue
        source = _resolve_config_dir_or_cwd_ref(config_dir, ref)
        if not source.exists() or not source.is_dir():
            continue
        destination = _lab_relative_path(lab_root, ref)
        try:
            if source.resolve() == destination.resolve():
                continue
        except FileNotFoundError:
            pass
        if destination in copied:
            continue
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
        copied.add(destination)


def _resolve_config_dir_or_cwd_ref(config_dir: Path, ref: str) -> Path:
    path = Path(ref)
    if path.is_absolute():
        return path
    for candidate in (config_dir / path, config_dir.parent / path, Path.cwd() / path):
        if candidate.exists():
            return candidate
    return config_dir / path


def _build_tool_registry(config: dict[str, Any], lab_root: Path) -> ToolRegistry:
    registry = ToolRegistry()
    tools = config.get("tools", {})
    if not isinstance(tools, dict):
        raise ValueError("tools must be a mapping")
    scientific_ie_config = tools.get("scientific_ie", {})
    if scientific_ie_config is True:
        scientific_ie_config = {"enabled": True}
    if not isinstance(scientific_ie_config, dict):
        raise ValueError("tools.scientific_ie must be a mapping")
    if scientific_ie_config.get("enabled") is True:
        register_scientific_ie_tools(
            registry,
            artifact_root=lab_root / "artifacts" / "tools",
            base_dir=lab_root,
            include_human_tools=scientific_ie_config.get("include_human_tools", True),
        )
    read_file_config = tools.get("read_file", {})
    if not isinstance(read_file_config, dict):
        raise ValueError("tools.read_file must be a mapping")
    if read_file_config.get("enabled") is True:
        registry.register(
            ToolSpec(
                name="read_file",
                description="Read a UTF-8 text file from the demo lab root.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            ),
            lambda arguments: _read_lab_file(lab_root, arguments),
        )
    return registry


def _read_lab_file(lab_root: Path, arguments: dict[str, Any]) -> str:
    raw_path = arguments.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("read_file requires a non-empty string path")
    return _lab_relative_path(lab_root, raw_path).read_text(encoding="utf-8")


def _lab_relative_path(lab_root: Path, relative_path: str) -> Path:
    if Path(relative_path).is_absolute():
        raise ValueError(f"lab file path must be relative: {relative_path!r}")
    root = lab_root.resolve()
    path = (root / relative_path).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"lab file path escapes lab root: {relative_path!r}")
    return path


def _build_llm_backends(
    config: dict[str, Any],
    config_dir: Path | None = None,
    *,
    backend_state_registry: Any | None = None,
) -> dict[str, Any]:
    backends = {}
    env_values = _load_dotenv_values(config, config_dir=config_dir)
    for backend_id, payload in _backend_section(config, "llm").items():
        backend_type = payload.get("type")
        if backend_type == "fake":
            backends[backend_id] = FakeLLMBackend(
                backend_id=backend_id,
                default_content=payload.get("default_content", "fake response"),
                responses=[
                    _model_from_payload(LLMRuntimeResponse, item)
                    for item in payload.get("responses", [])
                ],
            )
            continue
        if backend_type == "api":
            backends[backend_id] = _build_api_llm_backend(
                backend_id=backend_id,
                payload=payload,
                env_values=env_values,
            )
            continue
        if backend_type == "local_trainable":
            backends[backend_id] = LocalTrainableLLMBackend(
                backend_id=backend_id,
                default_content=payload.get("default_content", "local trainable mock response"),
                state_registry=backend_state_registry,
            )
            continue
        raise ValueError(f"backend {backend_id!r} has unsupported llm type {backend_type!r}")
    return backends


def _build_api_llm_backend(
    *,
    backend_id: str,
    payload: dict[str, Any],
    env_values: dict[str, str],
) -> ApiLLMBackend:
    env_ref = payload.get("env_ref")
    env_entry = _api_env_entry(env_values, env_ref, backend_id)
    api_kind = payload.get("api") or env_entry.get("api") or "openai-responses"
    if api_kind not in {"openai-responses", "openai-chat-completions"}:
        raise ValueError(f"backend {backend_id!r} has unsupported api {api_kind!r}")
    hosting = payload.get("hosting") or env_entry.get("hosting") or "remote"
    if hosting not in {"remote", "local"}:
        raise ValueError(f"backend {backend_id!r} has unsupported hosting {hosting!r}")
    model = payload.get("model") or env_entry.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError(f"backend {backend_id!r} requires a non-empty model")
    base_url = payload.get("base_url") or payload.get("baseUrl") or env_entry.get("base_url") or env_entry.get("baseUrl")
    if base_url is not None and not isinstance(base_url, str):
        raise ValueError(f"backend {backend_id!r} base_url must be a string")
    if "api_key" in payload or "apiKey" in payload:
        raise ValueError(f"backend {backend_id!r} must not include inline api keys; use env_ref or api_key_env")
    api_key_env = payload.get("api_key_env", "OPENAI_API_KEY")
    api_key = env_entry.get("api_key") or env_entry.get("apiKey") or _first_env_value(env_values, api_key_env)
    if api_key is not None and not isinstance(api_key, str):
        raise ValueError(f"backend {backend_id!r} api key must be a string")
    return ApiLLMBackend(
        ApiLLMBackendConfig(
            provider="openai",
            api=api_kind,
            hosting=hosting,
            model=model,
            api_key_env=api_key_env,
            base_url=base_url,
            max_output_tokens=_optional_int_backend_option(payload, env_entry, "max_output_tokens"),
            timeout_seconds=_optional_float_backend_option(payload, env_entry, "timeout_seconds"),
            max_retries=_int_backend_option(payload, env_entry, "max_retries", 2),
            retry_initial_delay_seconds=_float_backend_option(
                payload,
                env_entry,
                "retry_initial_delay_seconds",
                1.0,
            ),
            retry_max_delay_seconds=_float_backend_option(
                payload,
                env_entry,
                "retry_max_delay_seconds",
                30.0,
            ),
            extra_body=_dict_backend_option(payload, env_entry, "extra_body"),
        ),
        backend_id=backend_id,
        api_key=api_key,
    )


def _build_embedding_backends(config: dict[str, Any], config_dir: Path | None = None) -> dict[str, Any]:
    backends = {}
    env_values = _load_dotenv_values(config, config_dir=config_dir)
    for backend_id, payload in _backend_section(config, "embedding").items():
        backend_type = payload.get("type")
        if backend_type == "fake":
            backends[backend_id] = FakeEmbeddingBackend(
                backend_id=backend_id,
                dimensions=_int_backend_option(payload, {}, "dimensions", 8),
            )
            continue
        if backend_type == "api":
            backends[backend_id] = _build_api_embedding_backend(
                backend_id=backend_id,
                payload=payload,
                env_values=env_values,
            )
            continue
        raise ValueError(f"backend {backend_id!r} has unsupported embedding type {backend_type!r}")
    return backends


def _build_api_embedding_backend(
    *,
    backend_id: str,
    payload: dict[str, Any],
    env_values: dict[str, str],
) -> ApiEmbeddingBackend:
    env_ref = payload.get("env_ref")
    env_entry = _api_env_entry(env_values, env_ref, backend_id)
    api_kind = payload.get("api") or env_entry.get("api") or "openai-embeddings"
    if api_kind != "openai-embeddings":
        raise ValueError(f"backend {backend_id!r} has unsupported embedding api {api_kind!r}")
    model = payload.get("model") or env_entry.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError(f"backend {backend_id!r} requires a non-empty model")
    base_url = payload.get("base_url") or payload.get("baseUrl") or env_entry.get("base_url") or env_entry.get("baseUrl")
    if base_url is not None and not isinstance(base_url, str):
        raise ValueError(f"backend {backend_id!r} base_url must be a string")
    if "api_key" in payload or "apiKey" in payload:
        raise ValueError(f"backend {backend_id!r} must not include inline api keys; use env_ref or api_key_env")
    api_key_env = payload.get("api_key_env", "OPENAI_API_KEY")
    if not isinstance(api_key_env, str) or not api_key_env:
        raise ValueError(f"backend {backend_id!r} api_key_env must be a non-empty string")
    api_key = env_entry.get("api_key") or env_entry.get("apiKey") or _first_env_value(env_values, api_key_env)
    if api_key is not None and not isinstance(api_key, str):
        raise ValueError(f"backend {backend_id!r} api key must be a string")
    return ApiEmbeddingBackend(
        ApiEmbeddingBackendConfig(
            provider="openai",
            api=api_kind,
            model=model,
            api_key_env=api_key_env,
            base_url=base_url,
            timeout_seconds=_optional_float_backend_option(payload, env_entry, "timeout_seconds"),
        ),
        backend_id=backend_id,
        api_key=api_key,
    )


def _int_backend_option(
    payload: dict[str, Any],
    env_entry: dict[str, Any],
    key: str,
    default: int,
) -> int:
    value = payload.get(key, env_entry.get(key, default))
    if isinstance(value, bool):
        raise ValueError(f"backend option {key!r} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    raise ValueError(f"backend option {key!r} must be an integer")


def _optional_int_backend_option(
    payload: dict[str, Any],
    env_entry: dict[str, Any],
    key: str,
) -> int | None:
    if key not in payload and key not in env_entry:
        return None
    return _int_backend_option(payload, env_entry, key, 0)


def _float_backend_option(
    payload: dict[str, Any],
    env_entry: dict[str, Any],
    key: str,
    default: float,
) -> float:
    value = payload.get(key, env_entry.get(key, default))
    if isinstance(value, bool):
        raise ValueError(f"backend option {key!r} must be a number")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    raise ValueError(f"backend option {key!r} must be a number")


def _optional_float_backend_option(
    payload: dict[str, Any],
    env_entry: dict[str, Any],
    key: str,
) -> float | None:
    if key not in payload and key not in env_entry:
        return None
    return _float_backend_option(payload, env_entry, key, 0.0)


def _dict_backend_option(
    payload: dict[str, Any],
    env_entry: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    if key not in payload and key not in env_entry:
        return {}
    value = payload.get(key, env_entry.get(key, {}))
    if isinstance(value, dict):
        return json.loads(json.dumps(value))
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"backend option {key!r} must be a JSON object") from exc
        if isinstance(loaded, dict):
            return loaded
    raise ValueError(f"backend option {key!r} must be an object")


def _api_env_entry(
    env_values: dict[str, str],
    env_ref: Any,
    backend_id: str,
) -> dict[str, Any]:
    if env_ref is None:
        return {
            key: value
            for key, value in {
                "api": _first_env_value(env_values, "OPENAI_API"),
                "base_url": _first_env_value(env_values, "OPENAI_BASE_URL", "OPENAI_BASEURL"),
                "api_key": _first_env_value(env_values, "OPENAI_API_KEY", "OPENAI_APIKEY"),
                "model": _first_env_value(env_values, "OPENAI_MODEL"),
            }.items()
            if value is not None
        }
    if not isinstance(env_ref, str) or not env_ref:
        raise ValueError(f"backend {backend_id!r} env_ref must be a non-empty string")
    prefix = env_ref_prefix(env_ref)
    entry = {
        "api": _first_env_value(env_values, f"{prefix}_API"),
        "hosting": _first_env_value(env_values, f"{prefix}_HOSTING"),
        "base_url": _first_env_value(env_values, f"{prefix}_BASE_URL", f"{prefix}_BASEURL"),
        "api_key": _first_env_value(env_values, f"{prefix}_API_KEY", f"{prefix}_APIKEY"),
        "model": _first_env_value(env_values, f"{prefix}_MODEL"),
    }
    entry = {key: value for key, value in entry.items() if value is not None}
    if not entry:
        raise ValueError(f"backend {backend_id!r} references missing .env entry {env_ref!r}")
    return entry


def _first_env_value(values: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        value = lookup_env_value(values, key)
        if value is not None:
            return value
    return None


def _load_dotenv_values(config: dict[str, Any], config_dir: Path | None = None) -> dict[str, str]:
    api_env_config = config.get("api_env")
    if api_env_config:
        if not isinstance(api_env_config, dict):
            raise ValueError("api_env must be a mapping")
        if api_env_config.get("json_path") or api_env_config.get("path"):
            raise ValueError("api_env.json_path is no longer supported; store API credentials in .env")
    dotenv_path = config.get("dotenv_path")
    if dotenv_path is not None and (not isinstance(dotenv_path, str) or not dotenv_path):
        raise ValueError("dotenv_path must be a non-empty string")
    path = _resolve_dotenv_path(dotenv_path or ".env", config_dir=config_dir)
    return parse_dotenv(path) if path is not None else {}


def _resolve_dotenv_path(path_value: str, config_dir: Path | None = None) -> Path | None:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        if path.is_file():
            return path
        return None
    search_roots: list[Path] = []
    if config_dir is not None:
        resolved_config_dir = Path(config_dir).resolve()
        search_roots.extend([resolved_config_dir, resolved_config_dir.parent])
    else:
        search_roots.append(Path.cwd().resolve())
    for root in search_roots:
        candidate = (root / path).resolve()
        if candidate.is_file():
            return candidate
    return None


def _build_memory_backends(config: dict[str, Any], config_dir: Path | None = None) -> dict[str, Any]:
    backends = {}
    for backend_id, payload in _backend_section(config, "memory").items():
        backend_type = payload.get("type")
        if backend_type == "fake":
            backends[backend_id] = FakeMemoryBackend(backend_id=backend_id)
            continue
        if backend_type is None or backend_type == "null":
            backends[backend_id] = NullMemoryBackend(backend_id=backend_id)
            continue
        if backend_type == "method":
            backends[backend_id] = _build_method_memory_backend(
                backend_id=backend_id,
                payload=payload,
                config_dir=config_dir,
            )
            continue
        if backend_type == "mem0":
            backends[backend_id] = _build_mem0_compat_memory_backend(
                backend_id=backend_id,
                payload=payload,
                config_dir=config_dir,
            )
            continue
        if backend_type == "everos":
            backends[backend_id] = _build_native_everos_memory_backend(
                backend_id=backend_id,
                payload=payload,
                config_dir=config_dir,
            )
            continue
        raise ValueError(f"backend {backend_id!r} has unsupported memory type {backend_type!r}")
    return backends


def _build_method_memory_backend(
    *,
    backend_id: str,
    payload: dict[str, Any],
    config_dir: Path | None = None,
) -> MethodMemoryBackend:
    method = payload.get("method")
    if method == "mem0":
        _reject_mem0_client_options(backend_id, payload)
        return _build_native_mem0_memory_backend(backend_id=backend_id, payload=payload, config_dir=config_dir)
    if method == "everos":
        return _build_native_everos_memory_backend(backend_id=backend_id, payload=payload, config_dir=config_dir)
    raise ValueError(f"backend {backend_id!r} has unsupported memory method {method!r}")


def _build_mem0_compat_memory_backend(
    *,
    backend_id: str,
    payload: dict[str, Any],
    config_dir: Path | None = None,
) -> MethodMemoryBackend:
    _reject_mem0_inline_api_keys(backend_id, payload)
    _reject_mem0_client_options(backend_id, payload)
    implementation = payload.get("implementation", "native")
    if implementation != "native":
        raise ValueError(
            f"backend {backend_id!r} has unsupported Mem0 implementation {implementation!r}; "
            "use implementation: native or type: method, method: mem0"
        )
    return _build_native_mem0_memory_backend(backend_id=backend_id, payload=payload, config_dir=config_dir)


def _raise_removed_in_memory_mem0_client(backend_id: str) -> None:
    raise ValueError(
        f"backend {backend_id!r} uses removed Mem0 client 'in_memory'; "
        "use type: method, method: mem0 plus fake LLM and fake embedding backends for tests; "
        "native mem0 uses llm_backend and embedding_backend"
    )


def _reject_mem0_client_options(backend_id: str, source: dict[str, Any]) -> None:
    for key in ("client", "client_type"):
        if key not in source:
            continue
        client_kind = source[key]
        if client_kind == "in_memory":
            _raise_removed_in_memory_mem0_client(backend_id)
        raise ValueError(
            f"backend {backend_id!r} has unsupported native Mem0 option {key}: {client_kind!r}; "
            "the removed in_memory Mem0 client is not supported; native mem0 uses llm_backend and embedding_backend"
        )


def _build_native_mem0_memory_backend(
    *,
    backend_id: str,
    payload: dict[str, Any],
    config_dir: Path | None = None,
) -> MethodMemoryBackend:
    config_payload = _native_mem0_config_payload(
        backend_id=backend_id,
        payload=payload,
        config_dir=config_dir,
    )
    return MethodMemoryBackend(
        backend_id=backend_id,
        method=Mem0MemoryMethod(
            store_path=config_payload["store_path"],
            audit_log_path=config_payload.get("audit_log_path"),
            llm_backend_id=config_payload["llm_backend"],
            embedding_backend_id=config_payload["embedding_backend"],
            top_k_existing=config_payload.get("top_k_existing", 10),
        ),
        default_search_top_k=config_payload.get("default_search_top_k"),
        default_search_threshold=config_payload.get("default_search_threshold"),
    )


def _native_mem0_config_payload(
    *,
    backend_id: str,
    payload: dict[str, Any],
    config_dir: Path | None = None,
) -> dict[str, Any]:
    nested = payload.get("config", {})
    if nested is None:
        nested = {}
    if not isinstance(nested, dict):
        raise ValueError("mem0 method memory backend config must be a mapping")
    for source in (payload, nested):
        _reject_mem0_inline_api_keys(backend_id, source)
        method = source.get("method")
        if method is not None and method != "mem0":
            raise ValueError(f"backend {backend_id!r} has unsupported Mem0 method {method!r}")
        _reject_mem0_client_options(backend_id, source)
        implementation = source.get("implementation")
        if implementation is not None and implementation != "native":
            raise ValueError(
                f"backend {backend_id!r} has unsupported Mem0 implementation {implementation!r}; "
                "use implementation: native or type: method, method: mem0"
            )
    allowed = {
        "store_path",
        "audit_log_path",
        "llm_backend",
        "embedding_backend",
        "top_k_existing",
        "default_search_top_k",
        "default_search_threshold",
    }
    config_payload = {key: value for key, value in nested.items() if key in allowed}
    for key in allowed:
        if key in payload:
            config_payload[key] = payload[key]
    store_path = _required_path_config(config_payload, "store_path", "mem0 method memory backend")
    config_payload["store_path"] = str(_resolve_config_path(store_path, config_dir=config_dir))
    audit_log_path = config_payload.get("audit_log_path")
    if audit_log_path is not None:
        if not isinstance(audit_log_path, str) or not audit_log_path:
            raise ValueError("mem0 method memory backend audit_log_path must be a non-empty string")
        config_payload["audit_log_path"] = str(_resolve_config_path(audit_log_path, config_dir=config_dir))
    for key in ("llm_backend", "embedding_backend"):
        value = config_payload.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"mem0 method memory backend {key} must be a non-empty string")
    if "top_k_existing" in config_payload:
        config_payload["top_k_existing"] = _positive_int_config(
            config_payload["top_k_existing"],
            "mem0 method memory backend top_k_existing",
        )
    if "default_search_top_k" in config_payload:
        config_payload["default_search_top_k"] = _positive_int_config(
            config_payload["default_search_top_k"],
            "mem0 method memory backend default_search_top_k",
        )
    if "default_search_threshold" in config_payload:
        config_payload["default_search_threshold"] = _bounded_float_config(
            config_payload["default_search_threshold"],
            "mem0 method memory backend default_search_threshold",
            minimum=0.0,
            maximum=1.0,
        )
    return config_payload


def _reject_mem0_inline_api_keys(backend_id: str, source: dict[str, Any]) -> None:
    if "api_key" in source or "apiKey" in source:
        raise ValueError(f"backend {backend_id!r} must not include inline Mem0 api keys; use api_key_env")


def _build_native_everos_memory_backend(
    *,
    backend_id: str,
    payload: dict[str, Any],
    config_dir: Path | None = None,
) -> MethodMemoryBackend:
    config_payload = _native_everos_config_payload(
        backend_id=backend_id,
        payload=payload,
        config_dir=config_dir,
    )
    return MethodMemoryBackend(
        backend_id=backend_id,
        method=EverOSMemoryMethod(
            store_path=config_payload["store_path"],
            audit_log_path=config_payload.get("audit_log_path"),
            llm_backend_id=config_payload["llm_backend"],
            embedding_backend_id=config_payload["embedding_backend"],
            scene_similarity_threshold=config_payload.get("scene_similarity_threshold", 0.78),
            extraction_recent_message_limit=config_payload.get("extraction_recent_message_limit", 20),
            max_scene_candidates=config_payload.get("max_scene_candidates", 8),
            recollection_mode=config_payload.get("recollection_mode", "scene"),
            recollection_candidate_limit=config_payload.get("recollection_candidate_limit", 16),
        ),
        default_search_top_k=config_payload.get("default_search_top_k"),
        default_search_threshold=config_payload.get("default_search_threshold"),
    )


def _native_everos_config_payload(
    *,
    backend_id: str,
    payload: dict[str, Any],
    config_dir: Path | None = None,
) -> dict[str, Any]:
    nested = payload.get("config", {})
    if nested is None:
        nested = {}
    if not isinstance(nested, dict):
        raise ValueError("EverOS method memory backend config must be a mapping")
    for source in (payload, nested):
        method = source.get("method")
        if method is not None and method != "everos":
            raise ValueError(f"backend {backend_id!r} has unsupported EverOS method {method!r}")
        if "api_key" in source or "apiKey" in source:
            raise ValueError(f"backend {backend_id!r} must not include inline EverOS api keys")
        if "base_url" in source or "endpoint" in source:
            raise ValueError(
                f"backend {backend_id!r} must not configure EverOS HTTP service endpoints; "
                "native EverOS uses local EvoLab store_path plus llm_backend and embedding_backend"
            )
    allowed = {
        "store_path",
        "audit_log_path",
        "llm_backend",
        "embedding_backend",
        "scene_similarity_threshold",
        "extraction_recent_message_limit",
        "max_scene_candidates",
        "recollection_mode",
        "recollection_candidate_limit",
        "default_search_top_k",
        "default_search_threshold",
    }
    config_payload = {key: value for key, value in nested.items() if key in allowed}
    for key in allowed:
        if key in payload:
            config_payload[key] = payload[key]
    store_path = _required_path_config(config_payload, "store_path", "EverOS method memory backend")
    config_payload["store_path"] = str(_resolve_config_path(store_path, config_dir=config_dir))
    audit_log_path = config_payload.get("audit_log_path")
    if audit_log_path is not None:
        if not isinstance(audit_log_path, str) or not audit_log_path:
            raise ValueError("EverOS method memory backend audit_log_path must be a non-empty string")
        config_payload["audit_log_path"] = str(_resolve_config_path(audit_log_path, config_dir=config_dir))
    for key in ("llm_backend", "embedding_backend"):
        value = config_payload.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"EverOS method memory backend {key} must be a non-empty string")
    if "scene_similarity_threshold" in config_payload:
        config_payload["scene_similarity_threshold"] = _bounded_float_config(
            config_payload["scene_similarity_threshold"],
            "EverOS method memory backend scene_similarity_threshold",
            minimum=0.0,
            maximum=1.0,
        )
    for key in (
        "extraction_recent_message_limit",
        "max_scene_candidates",
        "recollection_candidate_limit",
        "default_search_top_k",
    ):
        if key in config_payload:
            config_payload[key] = _positive_int_config(
                config_payload[key],
                f"EverOS method memory backend {key}",
            )
    if "default_search_threshold" in config_payload:
        config_payload["default_search_threshold"] = _bounded_float_config(
            config_payload["default_search_threshold"],
            "EverOS method memory backend default_search_threshold",
            minimum=0.0,
            maximum=1.0,
        )
    recollection_mode = config_payload.get("recollection_mode", "scene")
    if recollection_mode not in {"scene", "agentic"}:
        raise ValueError("EverOS method memory backend recollection_mode must be 'scene' or 'agentic'")
    config_payload["recollection_mode"] = recollection_mode
    return config_payload


def _required_path_config(config_payload: dict[str, Any], key: str, label: str) -> str:
    value = config_payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} {key} must be a non-empty string")
    return value


def _resolve_config_path(path_value: str, config_dir: Path | None = None) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    root = Path(config_dir).resolve() if config_dir is not None else Path.cwd().resolve()
    return (root / path).resolve()


def _positive_int_config(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return parsed


def _float_config(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number") from exc


def _bounded_float_config(value: Any, label: str, *, minimum: float, maximum: float) -> float:
    parsed = _float_config(value, label)
    if not math.isfinite(parsed) or parsed < minimum or parsed > maximum:
        raise ValueError(f"{label} must be between {minimum:g} and {maximum:g}")
    return parsed


def _build_skill_backends(config: dict[str, Any], config_dir: Path | None = None) -> dict[str, Any]:
    backends = {}
    for backend_id, payload in _backend_section(config, "skill").items():
        backend_type = payload.get("type")
        if backend_type == "fake":
            skills = [
                _model_from_payload(SkillItem, item)
                for item in payload.get("skills", [])
            ]
            backends[backend_id] = FakeSkillBackend(
                backend_id=backend_id,
                skills=skills,
                graph_version_ref=payload.get("graph_version_ref", "fake-skill-graph-v1"),
                skill_state_ref=payload.get("skill_state_ref", "fake-skill-state-v1"),
                next_skill_state_ref=payload.get("next_skill_state_ref"),
            )
            continue
        if backend_type == "graph":
            graph_path_value = payload.get("graph_path") or payload.get("path")
            if not isinstance(graph_path_value, str) or not graph_path_value:
                raise ValueError(f"backend {backend_id!r} requires graph_path")
            config_root = config_dir or Path.cwd()
            graph_path = _resolve_config_ref(config_root, graph_path_value)
            repo_root_value = payload.get("repo_root")
            repo_root = Path(repo_root_value).expanduser() if isinstance(repo_root_value, str) else None
            backends[backend_id] = GraphSkillBackend(
                graph_path,
                repo_root=repo_root,
                strict_packages=bool(payload.get("strict_packages", False)),
            )
            continue
        raise ValueError(f"backend {backend_id!r} has unsupported skill type {backend_type!r}")
    return backends


def _build_evolution_backends(config: dict[str, Any], trajectory_registry: Any | None = None) -> dict[str, Any]:
    backends = {}
    for backend_id, payload in config.get("evolution", {}).get("backends", {}).items():
        backend_type = payload.get("type")
        if backend_type in {"fake", "fake_evolution"}:
            backends[backend_id] = FakeEvolutionBackend(
                scenario=payload.get("scenario", "promoted_candidate")
            )
        elif backend_type == "fake_sage":
            backends[backend_id] = FakeSAGETrainer(
                scenario=payload.get("scenario", "promoted_candidate")
            )
        elif backend_type == "fake_agent0":
            backends[backend_id] = FakeAgent0Trainer(
                solver_scenario=payload.get("scenario", "promoted_candidate")
            )
        elif backend_type == "prompt_overlay":
            backends[backend_id] = PromptOverlayEvolutionTrainer()
        elif backend_type == "sft":
            if trajectory_registry is None:
                raise ValueError("SFT evolution backend requires a trajectory registry")
            backends[backend_id] = SFTTrainer(
                trajectory_registry=trajectory_registry,
                config=_model_from_payload(
                    SFTTrainerConfig,
                    {key: value for key, value in payload.items() if key != "type"},
                ),
            )
        elif backend_type == "opsd":
            if trajectory_registry is None:
                raise ValueError("OPSD evolution backend requires a trajectory registry")
            backends[backend_id] = OPSDTrainer(
                trajectory_registry=trajectory_registry,
                config=_model_from_payload(
                    OPSDTrainerConfig,
                    {key: value for key, value in payload.items() if key != "type"},
                ),
            )
        elif backend_type == "local_trainable":
            raise ValueError(
                f"backend {backend_id!r} has unsupported evolution type 'local_trainable'; "
                "configure an SFT, OPSD, or Agent0SAGE trainer for evolution"
            )
        else:
            raise ValueError(f"backend {backend_id!r} has unsupported evolution type {backend_type!r}")
    return backends


def _backend_section(config: dict[str, Any], name: str) -> dict[str, dict[str, Any]]:
    section = config.get("backends", {}).get(name, {})
    if not isinstance(section, dict):
        raise ValueError(f"backends.{name} must be a mapping")
    return section


def _optional_backend_binding_from_config(value: Any, *, field_name: str) -> BackendBinding | None:
    if value is None:
        return None
    return _backend_binding_from_config(value, default_backend_id=None, field_name=field_name)


def _backend_binding_from_config(
    value: Any,
    *,
    default_backend_id: str | None,
    field_name: str,
) -> BackendBinding:
    if value is None:
        if default_backend_id is None:
            raise ValueError(f"{field_name} requires a backend id")
        return BackendBinding(backend_id=default_backend_id)
    if isinstance(value, str) and value:
        return BackendBinding(backend_id=value)
    if isinstance(value, dict):
        return _model_from_payload(BackendBinding, value)
    raise ValueError(f"{field_name} must be a backend id string or BackendBinding mapping")


def _require_type(payload: dict[str, Any], expected: str, backend_id: str) -> None:
    if payload.get("type") != expected:
        raise ValueError(f"backend {backend_id!r} must use type={expected!r}")


def _llm_bindings(task_config: TaskConfig) -> list[BackendBinding]:
    bindings = [role.llm_backend for role in task_config.roles.values()]
    if task_config.meta_agent is not None:
        bindings.append(task_config.meta_agent.llm_backend)
    if task_config.dynamic_subagents is not None and task_config.dynamic_subagents.enabled:
        dynamic = task_config.dynamic_subagents
        if dynamic.planner_backend is not None:
            bindings.append(BackendBinding(backend_id=dynamic.planner_backend.backend_id))
        if dynamic.default_worker_backend is not None:
            bindings.append(BackendBinding(backend_id=dynamic.default_worker_backend.backend_id))
        bindings.extend(BackendBinding(backend_id=backend_id) for backend_id in dynamic.allowed_worker_backend_ids)
    return _unique_bindings(bindings)


def _memory_bindings(task_config: TaskConfig) -> list[BackendBinding]:
    bindings = [
        role.agent_memory_backend
        for role in task_config.roles.values()
        if role.agent_memory_backend is not None
    ]
    if task_config.meta_agent is not None and task_config.meta_agent.memory_backend is not None:
        bindings.append(task_config.meta_agent.memory_backend)
    if task_config.task_memory_backend is not None:
        bindings.append(task_config.task_memory_backend)
    return _unique_bindings(bindings)


def _unique_bindings(bindings: list[BackendBinding] | Any) -> list[BackendBinding]:
    unique: dict[str, BackendBinding] = {}
    for binding in bindings:
        if binding.backend_id not in unique:
            unique[binding.backend_id] = binding
    return list(unique.values())


def _parse_training_args(values: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for value in values:
        key, separator, raw = value.partition("=")
        if not separator or not key:
            raise ValueError(f"training args must use key=value form: {value!r}")
        try:
            parsed[key] = json.loads(raw)
        except json.JSONDecodeError:
            parsed[key] = raw
    return parsed


def _single_source_run_ref(source_run_refs: list[str] | None) -> str | None:
    if source_run_refs is None or len(source_run_refs) != 1:
        return None
    return source_run_refs[0]


def _source_run_proposer_refs(source_run_refs: list[str]) -> list[ProposerInputRef]:
    return _training_source_proposer_refs(source_run_refs, role="sft_source", source="train-sft")


def _training_source_proposer_refs(
    source_run_refs: list[str],
    *,
    role: str,
    source: str,
) -> list[ProposerInputRef]:
    return [
        ProposerInputRef(
            ref_type="trajectory",
            ref_id=run_ref,
            role=role,
            metadata={"source": source},
        )
        for run_ref in source_run_refs
    ]


def _request_training_refs(request: LLMEvolutionRequest) -> list[str]:
    refs: list[str] = []
    if request.trigger_trajectory_ref is not None:
        refs.append(request.trigger_trajectory_ref)
    for ref in request.proposer_input_refs:
        if ref.ref_type == "trajectory" and ref.ref_id not in refs:
            refs.append(ref.ref_id)
    return refs


def _sft_result_source_run_refs(result: Any) -> list[str]:
    return _result_source_run_refs(result)


def _opsd_result_source_run_refs(result: Any) -> list[str]:
    return _result_source_run_refs(result)


def _result_source_run_refs(result: Any) -> list[str]:
    manifest_uri = getattr(result, "metadata", {}).get("manifest_uri")
    if not isinstance(manifest_uri, str):
        return []
    manifest_path = Path(manifest_uri)
    if not manifest_path.is_file():
        return []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    refs = payload.get("source_run_refs")
    if not isinstance(refs, list):
        return []
    return [ref for ref in refs if isinstance(ref, str)]


def _metadata_string_list(metadata: dict[str, Any], key: str) -> list[str]:
    value = metadata.get(key)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _json_compatible(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_compatible(item) for item in value]
    if isinstance(value, tuple):
        return [_json_compatible(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
