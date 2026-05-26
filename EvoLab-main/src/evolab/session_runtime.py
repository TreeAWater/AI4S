from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from evolab.backends.memory import NullMemoryBackend
from evolab.config.agents import default_seed_roles, render_agents_markdown
from evolab.config.agents import load_agents_file
from evolab.config.task_config import BackendBinding, RoleSpec
from evolab.config.task_config import MetaAgentSpec, TaskConfig
from evolab.contracts.common import RuntimePolicy
from evolab.contracts.dynamic_workflow import DynamicSubagentsConfig
from evolab.contracts.task import TaskOrigin, TaskPurpose, TaskRequest
from evolab.lab.layout import LabLayout
from evolab.lab.resolver import LabResolver
from evolab.runtime.task_worker import TaskWorker
from evolab.sdk import SessionConfig
from evolab.tools import (
    ToolRegistry,
    register_file_tools,
    register_output_tools,
    register_schema_tools,
    register_scientific_artifact_tools,
    register_table_tools,
    register_text_tools,
)


def initialize_lab(config: SessionConfig) -> LabLayout:
    layout = LabLayout(config.lab_dir)
    LabResolver(layout).ensure()
    _ensure_agents_file(config, layout)
    return layout


def run_session(config: SessionConfig) -> None:
    layout = initialize_lab(config)
    task_id = _task_id(config)
    task_prompt = _task_prompt(config)
    registry = _build_tool_registry(config, layout)
    task_config = _build_task_config(
        config=config,
        layout=layout,
        task_id=task_id,
        task_prompt=task_prompt,
        allowed_tool_names=registry.tool_names(),
    )
    request = TaskRequest(
        task_id=task_id,
        origin=TaskOrigin.HUMAN,
        purpose=TaskPurpose.SCIENCE,
        goal=task_prompt,
        task_config_ref=None,
        metadata={"task_description": task_prompt, "sdk_session": True},
    )
    resolver = LabResolver(layout)
    task_path = resolver.task_registry().save_task_request(request)
    resolver.task_queue().enqueue(
        f"task-{task_id}",
        {"request_payload_uri": str(task_path)},
    )
    worker = TaskWorker(
        layout=layout,
        worker_id="sdk-session",
        task_config=task_config,
        llm_backends=_build_llm_backends(config),
        embedding_backends=_build_embedding_backends(config),
        memory_backends=_build_memory_backends(config),
        skill_backends=_build_skill_backends(config, layout),
        tool_registry=registry,
    )
    worker.startup()
    worker.run_once()


def _ensure_agents_file(config: SessionConfig, layout: LabLayout) -> None:
    if layout.agents_path.exists():
        return
    roles = _seed_roles(config)
    layout.agents_path.parent.mkdir(parents=True, exist_ok=True)
    layout.agents_path.write_text(
        render_agents_markdown(
            roles,
            note="Materialized seed role pool for EvoLab SDK role evolution.",
        ),
        encoding="utf-8",
    )


def _seed_roles(config: SessionConfig) -> dict[str, RoleSpec]:
    if config.seed_roles:
        return {
            name: _role_from_payload(name, payload, config)
            for name, payload in config.seed_roles.items()
        }
    return default_seed_roles(
        llm_backend_id=_default_llm_backend_id(config),
        allowed_tools=_allowed_tools(config),
    )


def _role_from_payload(name: str, payload: Any, config: SessionConfig) -> RoleSpec:
    if not isinstance(payload, dict):
        raise ValueError(f"seed role {name!r} must be a mapping")
    role_payload = {
        "name": name,
        "system_prompt": payload.get("system_prompt")
        or payload.get("prompt")
        or "You are an EvoLab task worker.",
        "llm_backend": payload.get("llm_backend")
        or BackendBinding(backend_id=_default_llm_backend_id(config)).model_dump(mode="json"),
        "allowed_tools": payload.get("allowed_tools", _allowed_tools(config)),
        "required_skills": payload.get("required_skills", []),
        "memory_policy": payload.get("memory_policy", {}),
        "metadata": payload.get("metadata", {}),
    }
    return RoleSpec.model_validate(role_payload)


def _default_llm_backend_id(config: SessionConfig) -> str:
    if not config.llm:
        raise ValueError("SessionConfig.llm requires at least one backend")
    return str(next(iter(config.llm)))


def _allowed_tools(config: SessionConfig) -> list[str]:
    raw = config.tools.get("allowed_tools")
    if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
        return list(raw)
    return []


def _task_id(config: SessionConfig) -> str:
    raw = config.runtime.get("task_id")
    if isinstance(raw, str) and raw:
        return _safe_id(raw)
    return f"sdk-{uuid4().hex[:12]}"


def _safe_id(value: str) -> str:
    safe = "".join(char.lower() if char.isalnum() else "-" for char in value)
    safe = "-".join(part for part in safe.split("-") if part)
    return safe or "sdk-task"


def _task_prompt(config: SessionConfig) -> str:
    if isinstance(config.task, str):
        return config.task
    return config.task.to_prompt()


def _build_task_config(
    *,
    config: SessionConfig,
    layout: LabLayout,
    task_id: str,
    task_prompt: str,
    allowed_tool_names: list[str],
) -> TaskConfig:
    llm_backend_id = _default_llm_backend_id(config)
    memory_backend_id = _default_memory_backend_id(config)
    roles = load_agents_file(layout.agents_path)
    return TaskConfig(
        task_id=task_id,
        goal=task_prompt,
        task_memory_backend=BackendBinding(backend_id=memory_backend_id),
        meta_agent=MetaAgentSpec(
            name="meta",
            system_prompt=_meta_agent_prompt(config),
            llm_backend=BackendBinding(backend_id=llm_backend_id),
            memory_backend=BackendBinding(backend_id=memory_backend_id),
        ),
        agents_ref=str(layout.agents_path),
        roles=roles,
        dynamic_subagents=DynamicSubagentsConfig.model_validate(
            {
                "enabled": True,
                "mode": "dynamic",
                "scope": "per_task",
                "planner_backend": {"backend_id": llm_backend_id},
                "default_worker_backend": {"backend_id": llm_backend_id},
                "allowed_tool_names": sorted(allowed_tool_names),
            }
        ),
        runtime_policy=RuntimePolicy(
            max_tool_steps=_nonnegative_int(config.runtime.get("max_tool_steps"), default=20),
            max_tool_steps_per_node=_optional_nonnegative_int(config.runtime.get("max_tool_steps_per_node")),
            max_workflow_nodes=_positive_int(config.runtime.get("max_workflow_nodes"), default=20),
            allow_human_tools=bool(config.runtime.get("allow_human_tools", False)),
            enable_workflow_planning=True,
            metadata={
                "task_description": task_prompt,
                "sdk_session": True,
                "max_meta_dispatch_parse_retries": _nonnegative_int(
                    config.runtime.get("max_meta_dispatch_parse_retries"),
                    default=2,
                ),
            },
        ),
    )


def _meta_agent_prompt(config: SessionConfig) -> str:
    raw = config.meta_agent or {}
    prompt = raw.get("system_prompt") if isinstance(raw, dict) else None
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    return (
        "You are EvoLab's MetaAgent. During preplanning, return JSON only. "
        "Use metadata.no_generated_tool_reason when built-in tools are enough, "
        "metadata.generated_tool_package when a task-local Python tool is needed, "
        "metadata.no_role_pool_update_reason when the current role pool is enough, "
        "and metadata.role_pool_update when AGENTS.md should evolve."
    )


def _default_memory_backend_id(config: SessionConfig) -> str:
    if config.memory:
        return str(next(iter(config.memory)))
    return "task"


def _build_llm_backends(config: SessionConfig) -> dict[str, Any]:
    from evolab.cli import _build_llm_backends as build_llm_backends

    return build_llm_backends(_backend_config(config))


def _build_embedding_backends(config: SessionConfig) -> dict[str, Any]:
    if not config.embeddings:
        return {}
    from evolab.cli import _build_embedding_backends as build_embedding_backends

    return build_embedding_backends(_backend_config(config))


def _build_memory_backends(config: SessionConfig) -> dict[str, Any]:
    if not config.memory:
        return {"task": NullMemoryBackend(backend_id="task")}
    from evolab.cli import _build_memory_backends as build_memory_backends

    return build_memory_backends(_backend_config(config), config_dir=Path(config.lab_dir) / ".evolab")


def _build_skill_backends(config: SessionConfig, layout: LabLayout) -> dict[str, Any]:
    if not config.skills:
        from evolab.backends.skills import FakeSkillBackend

        return {"default": FakeSkillBackend(backend_id="default")}
    from evolab.cli import _build_skill_backends as build_skill_backends

    return build_skill_backends(_backend_config(config), config_dir=layout.state_root)


def _backend_config(config: SessionConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "backends": {
            "llm": config.llm,
            "embedding": config.embeddings,
            "memory": config.memory,
            "skill": config.skills,
        }
    }
    if config.env_file is not None:
        payload["dotenv_path"] = str(config.env_file)
    return payload


def _build_tool_registry(config: SessionConfig, layout: LabLayout) -> ToolRegistry:
    registry = ToolRegistry()
    if config.tools.get("builtin", True) is False:
        return registry
    register_file_tools(registry, base_dir=layout.root, excluded_roots=[layout.state_root])
    register_text_tools(registry, base_dir=layout.root)
    register_table_tools(registry, base_dir=layout.root)
    register_schema_tools(registry, base_dir=layout.root)
    register_scientific_artifact_tools(
        registry,
        artifact_root=layout.user_artifacts_dir / "tools",
        base_dir=layout.root,
    )
    register_output_tools(registry, artifact_root=layout.root)
    return registry


def _positive_int(value: Any, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int) and value > 0:
        return value
    return default


def _nonnegative_int(value: Any, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int) and value >= 0:
        return value
    return default


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None
