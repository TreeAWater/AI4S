from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import Path
from typing import Any

from evolab.config.task_config import BackendBinding, RoleSpec

_JSON_FENCE_RE = re.compile(r"```(?:json|agents\.json)?\s*\n(?P<payload>.*?)\n```", re.DOTALL | re.IGNORECASE)


def parse_agents_markdown(text: str, *, source: str = "agents.md") -> dict[str, RoleSpec]:
    payload = _load_agents_payload(text, source=source)
    return parse_agents_payload(payload, source=source)


def parse_agents_payload(payload: Any, *, source: str = "agents.md") -> dict[str, RoleSpec]:
    raw_roles = _raw_roles_payload(payload, source=source)
    roles: dict[str, RoleSpec] = {}
    for index, item in enumerate(raw_roles):
        name, role_payload = _role_name_and_payload(item, index=index, source=source)
        role = RoleSpec.model_validate(_normalise_role_payload(name, role_payload))
        if role.name in roles:
            raise ValueError(f"{source} contains duplicate agent name {role.name!r}")
        roles[role.name] = role
    if not roles:
        raise ValueError(f"{source} must define at least one agent")
    return roles


def load_agents_file(path: Path | str) -> dict[str, RoleSpec]:
    source = Path(path)
    return parse_agents_markdown(source.read_text(encoding="utf-8"), source=str(source))


def default_seed_roles(*, llm_backend_id: str, allowed_tools: list[str]) -> dict[str, RoleSpec]:
    return {
        "GeneralistAgent": RoleSpec(
            name="GeneralistAgent",
            system_prompt=(
                "You are a general EvoLab worker. Inspect the assigned task, use the prepared tools, "
                "produce traceable artifacts when requested, and report failures explicitly."
            ),
            llm_backend=BackendBinding(backend_id=llm_backend_id),
            allowed_tools=list(allowed_tools),
            required_skills=[],
            metadata={
                "role_pool_seed": True,
                "role_pool_generation": 0,
                "specialization": "general task execution",
            },
        )
    }


def render_agents_markdown(roles: dict[str, RoleSpec] | list[RoleSpec], *, note: str | None = None) -> str:
    role_items = list(roles.values()) if isinstance(roles, dict) else list(roles)
    payload = {
        "schema_version": "v1",
        "agents": [role.model_dump(mode="json", exclude_none=True) for role in role_items],
    }
    lines = [
        "# EvoLab Agents",
        "",
        "This file is part of the task config. The runtime reads the latest version before each MetaAgent dispatch.",
    ]
    if note:
        lines.extend(["", note])
    lines.extend(["", "```json", json.dumps(payload, indent=2, sort_keys=True), "```", ""])
    return "\n".join(lines)


def agents_markdown_revision(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _load_agents_payload(text: str, *, source: str) -> Any:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    match = _JSON_FENCE_RE.search(text)
    if match is None:
        raise ValueError(f"{source} must contain JSON or a fenced json block")
    try:
        return json.loads(match.group("payload"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source} contains invalid agents JSON") from exc


def _raw_roles_payload(payload: Any, *, source: str) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError(f"{source} agents payload must be a mapping or list")
    agents = payload.get("agents")
    if isinstance(agents, list):
        return agents
    roles = payload.get("roles")
    if isinstance(roles, dict):
        return [{**_dict(value, source=source, name=name), "name": name} for name, value in roles.items()]
    if isinstance(roles, list):
        return roles
    reserved = {"schema_version", "metadata"}
    role_items = {name: value for name, value in payload.items() if name not in reserved}
    if role_items and all(isinstance(value, dict) for value in role_items.values()):
        return [
            {**_dict(value, source=source, name=name), "name": name}
            for name, value in role_items.items()
        ]
    raise ValueError(f"{source} must define agents as a list or roles as a mapping")


def _role_name_and_payload(item: Any, *, index: int, source: str) -> tuple[str, dict[str, Any]]:
    if not isinstance(item, dict):
        raise ValueError(f"{source} agent entry #{index + 1} must be a mapping")
    name = item.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{source} agent entry #{index + 1} requires non-empty name")
    return name, item


def _normalise_role_payload(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["name"] = name
    if "prompt" in result and "system_prompt" not in result:
        result["system_prompt"] = result.pop("prompt")
    if "toolset" in result and "allowed_tools" not in result:
        result["allowed_tools"] = result.pop("toolset")
    if "skillset" in result and "required_skills" not in result:
        result["required_skills"] = result.pop("skillset")
    for key in ("llm_backend", "agent_memory_backend"):
        if isinstance(result.get(key), str):
            result[key] = {"backend_id": result[key]}
    return result


def _dict(value: Any, *, source: str, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{source} role {name!r} must be a mapping")
    return value
