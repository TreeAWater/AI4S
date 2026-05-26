# Dynamic Role Self-Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make EvoLab execute through a dynamic, automatically evolving role pool in `agents.md`, with worker subagents using task-level memory only.

**Architecture:** Add a focused role-pool runtime that validates and writes MetaAgent role updates to `agents.md`, then have dynamic workflow planning consume the updated role templates. Short configs materialize `agents.md` and dynamic-subagent defaults, while the old static role-order runtime path is removed from the default execution flow.

**Tech Stack:** Python 3.11, Pydantic v2, pytest, file-backed Lab registries, existing EvoLab `RoleSpec`/`TaskConfig`/`DynamicWorkflowSpec` contracts.

---

## File Map

- Create `evolab/runtime/role_pool.py`
  - Owns role-pool update extraction, validation, atomic write, history append, and result models.
- Modify `evolab/config/agents.py`
  - Adds seed role rendering helpers for configs that do not provide `agents_ref`.
- Modify `evolab/cli.py`
  - Compiles short configs into dynamic execution by default.
  - Materializes `agents.md` into the Lab config directory.
  - Stops treating `subagents` as the user-facing execution entry.
- Modify `evolab/runtime/task_runtime.py`
  - Invokes role-pool evolution before dynamic planning.
  - Removes default static role-order fallback.
  - Changes dynamic worker memory retrieval/update to task-memory-only.
- Modify `evolab/runtime/dynamic_workflow.py`
  - Renames planner input semantics from static fallback subagents to role pool templates.
  - Keeps dynamic runtime role creation compatible with role templates from `agents.md`.
- Modify `evolab/runtime/README.md`, `docs/configuration.md`, and `docs/dynamic_subagent_workflows.md`
  - Documents the new role-pool execution path and memory model.
- Test files to create or modify:
  - Create `tests/test_role_pool_runtime.py`
  - Modify `tests/test_cli_clean_run.py`
  - Modify `tests/test_dynamic_workflow_planner.py`
  - Modify `tests/test_dynamic_workflow_runtime.py`
  - Modify `tests/test_memory_replay.py`
  - Modify or remove static-default expectations in `tests/test_static_dynamic_mode_compatibility.py`
  - Modify static-runtime fixtures in `tests/test_task_runtime_workflow_plan.py`
  - Modify config assertions in `tests/test_biology_generic_subagents_config.py`

## Task 1: Role-Pool Update Runtime

**Files:**
- Create: `evolab/runtime/role_pool.py`
- Test: `tests/test_role_pool_runtime.py`

- [ ] **Step 1: Write failing tests for add, edit, delete, and invalid delete-all**

Add `tests/test_role_pool_runtime.py`:

```python
import json
from pathlib import Path

from evolab.config.agents import render_agents_markdown
from evolab.config.task_config import BackendBinding, RoleSpec
from evolab.runtime.role_pool import apply_role_pool_update, role_pool_update_payload


def _role(name: str, prompt: str = "Base prompt.") -> RoleSpec:
    return RoleSpec(
        name=name,
        system_prompt=prompt,
        llm_backend=BackendBinding(backend_id="planner-llm"),
        allowed_tools=["read_text", "write_report"],
    )


def test_role_pool_update_payload_prefers_canonical_key():
    metadata = {
        "agent_config_update": {"reason": "legacy"},
        "role_pool_update": {"reason": "canonical"},
    }

    assert role_pool_update_payload(metadata) == {"reason": "canonical"}


def test_apply_role_pool_update_adds_and_edits_roles(tmp_path: Path):
    agents_path = tmp_path / "agents.md"
    agents_path.write_text(render_agents_markdown({"SurveyAgent": _role("SurveyAgent")}), encoding="utf-8")

    result = apply_role_pool_update(
        agents_path=agents_path,
        payload={
            "reason": "Need table triage.",
            "roles": {
                "SurveyAgent": {"system_prompt_append": "Also report missing files."},
                "TableEvidenceTriageAgent": {
                    "system_prompt": "Inspect tables before extraction.",
                    "llm_backend": {"backend_id": "planner-llm"},
                    "allowed_tools": ["read_text", "write_report"],
                    "required_skills": ["scientific_table_structure_understanding"],
                },
            },
        },
        task_id="task-1",
        run_ref="meta-1",
        known_llm_backend_ids={"planner-llm"},
        allowed_tool_names={"read_text", "write_report"},
    )

    assert result.status == "updated"
    assert result.added_roles == ["TableEvidenceTriageAgent"]
    assert result.modified_roles == ["SurveyAgent"]
    assert result.removed_roles == []
    text = agents_path.read_text(encoding="utf-8")
    assert "TableEvidenceTriageAgent" in text
    assert "Also report missing files." in text
    history = (tmp_path / "agents.md.updates.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(history) == 1
    assert json.loads(history[0])["result"]["status"] == "updated"


def test_apply_role_pool_update_removes_role(tmp_path: Path):
    agents_path = tmp_path / "agents.md"
    agents_path.write_text(
        render_agents_markdown(
            {
                "SurveyAgent": _role("SurveyAgent"),
                "OldAgent": _role("OldAgent"),
            }
        ),
        encoding="utf-8",
    )

    result = apply_role_pool_update(
        agents_path=agents_path,
        payload={"reason": "Old role is obsolete.", "remove_roles": ["OldAgent"]},
        task_id="task-1",
        run_ref="meta-2",
        known_llm_backend_ids={"planner-llm"},
        allowed_tool_names={"read_text", "write_report"},
    )

    assert result.status == "updated"
    assert result.removed_roles == ["OldAgent"]
    assert "OldAgent" not in agents_path.read_text(encoding="utf-8")


def test_apply_role_pool_update_rejects_delete_all_without_overwriting(tmp_path: Path):
    agents_path = tmp_path / "agents.md"
    original = render_agents_markdown({"SurveyAgent": _role("SurveyAgent")})
    agents_path.write_text(original, encoding="utf-8")

    result = apply_role_pool_update(
        agents_path=agents_path,
        payload={"reason": "bad update", "remove_roles": ["SurveyAgent"]},
        task_id="task-1",
        run_ref="meta-3",
        known_llm_backend_ids={"planner-llm"},
        allowed_tool_names={"read_text", "write_report"},
    )

    assert result.status == "rejected"
    assert "at least one active role" in result.errors[0]
    assert agents_path.read_text(encoding="utf-8") == original
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_role_pool_runtime.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'evolab.runtime.role_pool'`.

- [ ] **Step 3: Implement `evolab/runtime/role_pool.py`**

Create `evolab/runtime/role_pool.py`:

```python
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from evolab.config.agents import agents_markdown_revision, parse_agents_payload, render_agents_markdown
from evolab.config.task_config import BackendBinding, RoleSpec

RolePoolUpdateStatus = Literal["updated", "no_op", "rejected"]

ROLE_POOL_UPDATE_KEYS = (
    "role_pool_update",
    "agent_config_update",
    "agents_update",
    "subagent_config_update",
)

PRIVATE_REASONING_KEYS = {"chain_of_thought", "reasoning", "hidden_reasoning"}


@dataclass(frozen=True)
class RolePoolUpdateResult:
    status: RolePoolUpdateStatus
    agents_ref: str
    before_revision: str | None
    after_revision: str | None
    added_roles: list[str] = field(default_factory=list)
    modified_roles: list[str] = field(default_factory=list)
    removed_roles: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    history_ref: str | None = None
    reason: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "agents_ref": self.agents_ref,
            "before_revision": self.before_revision,
            "after_revision": self.after_revision,
            "added_roles": list(self.added_roles),
            "modified_roles": list(self.modified_roles),
            "removed_roles": list(self.removed_roles),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "history_ref": self.history_ref,
            "reason": self.reason,
        }


def role_pool_update_payload(metadata: dict[str, Any]) -> dict[str, Any] | None:
    for key in ROLE_POOL_UPDATE_KEYS:
        value = metadata.get(key)
        if isinstance(value, dict):
            return value
    return None


def apply_role_pool_update(
    *,
    agents_path: Path,
    payload: dict[str, Any],
    task_id: str,
    run_ref: str,
    known_llm_backend_ids: set[str],
    allowed_tool_names: set[str] | None = None,
) -> RolePoolUpdateResult:
    agents_path = Path(agents_path)
    before_text = agents_path.read_text(encoding="utf-8")
    before_revision = agents_markdown_revision(before_text)
    before_roles = parse_agents_payload(_payload_from_markdown(before_text), source=str(agents_path))

    try:
        after_roles, added, modified, removed, warnings = _updated_roles(
            before_roles=before_roles,
            payload=payload,
            known_llm_backend_ids=known_llm_backend_ids,
            allowed_tool_names=allowed_tool_names or set(),
        )
    except ValueError as exc:
        result = RolePoolUpdateResult(
            status="rejected",
            agents_ref=str(agents_path),
            before_revision=before_revision,
            after_revision=before_revision,
            errors=[str(exc)],
            reason=_optional_string(payload.get("reason")),
        )
        _append_history(agents_path, task_id, run_ref, payload, result)
        return result

    if not added and not modified and not removed:
        result = RolePoolUpdateResult(
            status="no_op",
            agents_ref=str(agents_path),
            before_revision=before_revision,
            after_revision=before_revision,
            warnings=warnings,
            reason=_optional_string(payload.get("reason")),
        )
        _append_history(agents_path, task_id, run_ref, payload, result)
        return result

    after_text = render_agents_markdown(
        after_roles,
        note=f"Last updated automatically by MetaAgent run {run_ref}.",
    )
    parse_agents_payload(_payload_from_markdown(after_text), source="role_pool_update")
    after_revision = agents_markdown_revision(after_text)
    tmp = agents_path.with_suffix(agents_path.suffix + ".tmp")
    tmp.write_text(after_text, encoding="utf-8")
    os.replace(tmp, agents_path)
    result = RolePoolUpdateResult(
        status="updated",
        agents_ref=str(agents_path),
        before_revision=before_revision,
        after_revision=after_revision,
        added_roles=added,
        modified_roles=modified,
        removed_roles=removed,
        warnings=warnings,
        reason=_optional_string(payload.get("reason")),
    )
    _append_history(agents_path, task_id, run_ref, payload, result)
    return result
```

Then add the helper functions in the same file:

```python
def _updated_roles(
    *,
    before_roles: dict[str, RoleSpec],
    payload: dict[str, Any],
    known_llm_backend_ids: set[str],
    allowed_tool_names: set[str],
) -> tuple[dict[str, RoleSpec], list[str], list[str], list[str], list[str]]:
    roles = dict(before_roles)
    added: list[str] = []
    modified: list[str] = []
    removed: list[str] = []
    warnings: list[str] = []

    for role_name, item in _role_update_items(payload):
        if role_name in roles:
            base_payload = roles[role_name].model_dump(mode="json")
        else:
            base_payload = {"name": role_name}
        updated_role = _role_from_update(role_name, base_payload, item)
        _validate_role(updated_role, known_llm_backend_ids, allowed_tool_names)
        if role_name in roles:
            if roles[role_name].model_dump(mode="json") != updated_role.model_dump(mode="json"):
                modified.append(role_name)
        else:
            added.append(role_name)
        roles[role_name] = updated_role

    for role_name in _remove_roles(payload):
        if role_name in roles:
            del roles[role_name]
            removed.append(role_name)

    if not roles:
        raise ValueError("role_pool_update must leave at least one active role")
    return roles, sorted(added), sorted(modified), sorted(removed), warnings


def _role_update_items(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    raw = payload.get("roles", payload.get("agents"))
    if raw is None and isinstance(payload.get("name"), str):
        raw = [payload]
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [(str(name), _dict(value, f"role {name!r}")) for name, value in raw.items()]
    if isinstance(raw, list):
        result = []
        for index, value in enumerate(raw):
            item = _dict(value, f"role item #{index + 1}")
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"role item #{index + 1} requires non-empty name")
            result.append((name, item))
        return result
    raise ValueError("role_pool_update.roles must be an object or list")


def _remove_roles(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("remove_roles", [])
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("role_pool_update.remove_roles must be a list")
    return [item for item in raw if isinstance(item, str) and item]


def _role_from_update(role_name: str, base_payload: dict[str, Any], item: dict[str, Any]) -> RoleSpec:
    merged = dict(base_payload)
    update = {key: value for key, value in item.items() if key not in {"reason"}}
    prompt_append = update.pop("system_prompt_append", None)
    if isinstance(merged.get("metadata"), dict) and isinstance(update.get("metadata"), dict):
        update["metadata"] = {**merged["metadata"], **update["metadata"]}
    if isinstance(merged.get("memory_policy"), dict) and isinstance(update.get("memory_policy"), dict):
        update["memory_policy"] = {**merged["memory_policy"], **update["memory_policy"]}
    merged.update(update)
    if isinstance(prompt_append, str) and prompt_append.strip():
        base_prompt = str(merged.get("system_prompt") or "").rstrip()
        merged["system_prompt"] = f"{base_prompt}\n\n{prompt_append.strip()}" if base_prompt else prompt_append.strip()
    merged["name"] = role_name
    if isinstance(merged.get("llm_backend"), str):
        merged["llm_backend"] = {"backend_id": merged["llm_backend"]}
    if "llm_backend" not in merged:
        raise ValueError(f"role {role_name!r} requires llm_backend when created")
    return RoleSpec.model_validate(merged)


def _validate_role(role: RoleSpec, known_llm_backend_ids: set[str], allowed_tool_names: set[str]) -> None:
    if not role.system_prompt.strip():
        raise ValueError(f"role {role.name!r} requires non-empty system_prompt")
    if role.llm_backend.backend_id not in known_llm_backend_ids:
        raise ValueError(f"role {role.name!r} references unknown LLM backend {role.llm_backend.backend_id!r}")
    unknown_tools = sorted(set(role.allowed_tools) - allowed_tool_names) if allowed_tool_names else []
    if unknown_tools:
        raise ValueError(f"role {role.name!r} references unknown tool(s): {', '.join(unknown_tools)}")
    _reject_private_reasoning(role.metadata, path=f"role {role.name}.metadata")


def _reject_private_reasoning(value: Any, *, path: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) in PRIVATE_REASONING_KEYS:
                raise ValueError(f"{path} must not include private reasoning field {key!r}")
            _reject_private_reasoning(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_private_reasoning(item, path=f"{path}[{index}]")


def _append_history(
    agents_path: Path,
    task_id: str,
    run_ref: str,
    payload: dict[str, Any],
    result: RolePoolUpdateResult,
) -> None:
    history_path = agents_path.with_name(agents_path.name + ".updates.jsonl")
    record = {
        "schema_version": "v1",
        "task_id": task_id,
        "run_ref": run_ref,
        "update_hash": "role-pool-sha256-" + sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16],
        "update": _json_compatible(payload),
        "result": result.to_json(),
    }
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    object.__setattr__(result, "history_ref", str(history_path))


def _payload_from_markdown(text: str) -> dict[str, Any]:
    from evolab.config.agents import _load_agents_payload

    payload = _load_agents_payload(text, source="agents.md")
    if not isinstance(payload, dict):
        raise ValueError("agents.md payload must be an object")
    return payload


def _dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _json_compatible(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_compatible(item) for item in value]
    return value
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest tests/test_role_pool_runtime.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add evolab/runtime/role_pool.py tests/test_role_pool_runtime.py
git commit -m "feat: add role pool update runtime"
```

## Task 2: Dynamic Config Compilation And Seed agents.md

**Files:**
- Modify: `evolab/config/agents.py`
- Modify: `evolab/cli.py`
- Test: `tests/test_cli_clean_run.py`

- [ ] **Step 1: Add failing tests for seed role materialization and dynamic defaults**

Add tests to `tests/test_cli_clean_run.py`:

```python
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
    assert "subagents" not in compiled


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
            content: '{"route":"END","instruction":"No execution needed.","metadata":{"no_role_pool_update_reason":"seed is sufficient"}}'
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

    try:
        run_clean_demo(config_path, lab_root=tmp_path / "lab")
    except RuntimeError:
        pass

    agents_path = tmp_path / "lab" / "configs" / "agents.md"
    assert agents_path.exists()
    assert "GeneralistAgent" in agents_path.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_cli_clean_run.py::test_compile_experiment_config_defaults_to_dynamic_role_pool tests/test_cli_clean_run.py::test_clean_run_materializes_seed_agents_md -q
```

Expected: first test fails because `agents_ref` and dynamic defaults are absent, or second test fails because `agents.md` is not created.

- [ ] **Step 3: Add seed role helper**

Modify `evolab/config/agents.py`:

```python
def default_seed_roles(*, llm_backend_id: str, allowed_tools: list[str]) -> dict[str, RoleSpec]:
    return {
        "GeneralistAgent": RoleSpec(
            name="GeneralistAgent",
            system_prompt=(
                "You are a general EvoLab worker. Inspect the assigned task, use the prepared tools, "
                "produce traceable artifacts when requested, and report failures explicitly."
            ),
            llm_backend={"backend_id": llm_backend_id},
            allowed_tools=list(allowed_tools),
            required_skills=[],
            metadata={
                "role_pool_seed": True,
                "role_pool_generation": 0,
                "specialization": "general task execution",
            },
        )
    }
```

If Pydantic rejects the dict for `llm_backend`, use the existing `BackendBinding` import and construct `BackendBinding(backend_id=llm_backend_id)`.

- [ ] **Step 4: Compile short configs into role-pool dynamic configs**

Modify `_compile_experiment_config()` in `evolab/cli.py`:

```python
agents_ref = config.get("agents_ref")
if agents_ref is None:
    agents_ref = "agents.md"
elif not isinstance(agents_ref, str) or not agents_ref.strip():
    raise ValueError("agents_ref must be a non-empty string when provided")

subagents_payload = config.get("subagents")
if subagents_payload is not None:
    raise ValueError("subagents is no longer a supported execution entry; use agents_ref or seed_agents")

role_map: dict[str, dict[str, Any]] = {}
seed_agents_payload = config.get("seed_agents")
if isinstance(seed_agents_payload, dict):
    for name, payload in seed_agents_payload.items():
        role_map[name] = RoleSpec.model_validate(
            {
                "name": name,
                "system_prompt": payload["system_prompt"],
                "llm_backend": _backend_binding_from_config(
                    payload.get("llm_backend"),
                    default_backend_id=llm_backend_id,
                    field_name=f"seed_agents.{name}.llm_backend",
                ).model_dump(mode="json"),
                "agent_memory_backend": None,
                "allowed_tools": _string_list_or_default(payload.get("allowed_tools"), allowed_tools),
                "required_skills": _string_list_or_default(payload.get("required_skills", payload.get("skillset")), []),
                "memory_policy": dict(payload.get("memory_policy") or {}),
                "metadata": dict(payload.get("metadata") or {}),
            }
        ).model_dump(mode="json")
else:
    from evolab.config.agents import default_seed_roles

    role_map = {
        name: role.model_dump(mode="json")
        for name, role in default_seed_roles(
            llm_backend_id=llm_backend_id,
            allowed_tools=allowed_tools,
        ).items()
    }
```

Then construct the task config with the generated role pool and dynamic settings:

```python
dynamic_subagents = _dynamic_subagents_from_config(
    config,
    backends,
    default_llm_backend_id=llm_backend_id,
    allowed_tools=allowed_tools,
)
task_config = TaskConfig(
    task_id=task_id,
    goal=task,
    roles=role_map,
    agents_ref=agents_ref,
    meta_agent=meta_agent,
    task_memory_backend=BackendBinding(backend_id=task_memory_backend_id),
    dynamic_subagents=dynamic_subagents,
    runtime_policy=runtime_policy,
)
```

Make `_dynamic_subagents_from_config()` return an enabled default when config omits `dynamic_subagents`:

```python
if raw is None:
    return DynamicSubagentsConfig(
        enabled=True,
        mode="dynamic",
        scope="per_task",
        planner_backend={"backend_id": llm_backend_id},
        default_worker_backend={"backend_id": llm_backend_id},
        allowed_tool_names=_default_allowed_tools(config),
    )
```

- [ ] **Step 5: Materialize generated agents.md into Lab**

Modify `_copy_agents_config()` in `evolab/cli.py`:

```python
def _copy_agents_config(*, task_config: TaskConfig, config_dir: Path, lab_root: Path) -> TaskConfig:
    agents_ref = task_config.agents_ref or "agents.md"
    destination = lab_root / "configs" / Path(agents_ref).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = _resolve_config_ref(config_dir, agents_ref)
    if source.exists():
        shutil.copy2(source, destination)
    else:
        destination.write_text(
            render_agents_markdown(
                task_config.roles,
                note="Materialized seed role pool for automatic EvoLab role evolution.",
            ),
            encoding="utf-8",
        )
    return task_config.model_copy(update={"agents_ref": str(destination.relative_to(lab_root))})
```

Import `render_agents_markdown` at the top of `evolab/cli.py`.

- [ ] **Step 6: Run focused tests**

Run:

```bash
pytest tests/test_cli_clean_run.py::test_compile_experiment_config_defaults_to_dynamic_role_pool tests/test_cli_clean_run.py::test_clean_run_materializes_seed_agents_md -q
```

Expected: both tests pass.

- [ ] **Step 7: Commit**

```bash
git add evolab/config/agents.py evolab/cli.py tests/test_cli_clean_run.py
git commit -m "feat: compile configs to dynamic role pool"
```

## Task 3: Role-Pool Evolution Before Dynamic Planning

**Files:**
- Modify: `evolab/runtime/task_runtime.py`
- Test: `tests/test_dynamic_workflow_runtime.py`

- [ ] **Step 1: Write a failing runtime test showing MetaAgent updates role pool before planner**

Add to `tests/test_dynamic_workflow_runtime.py`:

```python
def test_dynamic_runtime_updates_agents_md_before_planner_reads_role_pool(tmp_path: Path):
    agents_path = tmp_path / "agents.md"
    agents_path.write_text(
        render_agents_markdown(
            {
                "GeneralistAgent": RoleSpec(
                    name="GeneralistAgent",
                    system_prompt="General work.",
                    llm_backend=BackendBinding(backend_id="worker-llm"),
                    allowed_tools=["write_report"],
                )
            }
        ),
        encoding="utf-8",
    )
    meta_response = LLMRuntimeResponse(
        action=SubAgentAction(
            action="final_answer",
            content=json.dumps(
                {
                    "route": "END",
                    "instruction": "Role pool updated.",
                    "metadata": {
                        "role_pool_update": {
                            "reason": "Need writer specialization.",
                            "roles": {
                                "WriterAgent": {
                                    "system_prompt": "Write final reports.",
                                    "llm_backend": {"backend_id": "worker-llm"},
                                    "allowed_tools": ["write_report"],
                                }
                            },
                        }
                    },
                }
            ),
        )
    )
    planner_response = LLMRuntimeResponse(
        action=SubAgentAction(
            action="final_answer",
            content=json.dumps(
                {
                    "workflow_id": "wf-1",
                    "task_summary": "write report",
                    "article_context_summary": None,
                    "dynamic_subagents": [
                        {
                            "subagent_id": "writer",
                            "role_name": "WriterAgent",
                            "goal": "Write final report.",
                            "system_prompt": "Use writer template.",
                            "allowed_tools": ["write_report"],
                            "output_schema": {"type": "object"},
                        }
                    ],
                    "workflow_nodes": [
                        {"node_id": "write", "subagent_id": "writer", "output_artifacts": ["report.md"]}
                    ],
                    "workflow_edges": [],
                    "artifact_contracts": {},
                    "validation_rules": [],
                    "planner_rationale_summary": "WriterAgent exists after role-pool update.",
                }
            ),
        )
    )

    runtime = _dynamic_runtime_fixture(
        tmp_path,
        agents_ref=str(agents_path),
        meta_responses=[meta_response],
        planner_responses=[planner_response],
        worker_responses=[LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="done"))],
    )

    result = runtime.run(TaskRequest(task_id="task-1", origin=TaskOrigin.HUMAN, purpose=TaskPurpose.SCIENCE, goal="write report"))

    assert result["status"] == "completed"
    assert "WriterAgent" in agents_path.read_text(encoding="utf-8")
```

Add this helper in `tests/test_dynamic_workflow_runtime.py` before the new test:

```python
def _dynamic_runtime_fixture(
    tmp_path: Path,
    *,
    agents_ref: str,
    meta_responses: list[LLMRuntimeResponse],
    planner_responses: list[LLMRuntimeResponse],
    worker_responses: list[LLMRuntimeResponse],
) -> TaskRuntime:
    registry = ToolRegistry()
    registry.register(write_report_tool_spec())
    task_config = TaskConfig(
        task_id="task-1",
        goal="write report",
        agents_ref=agents_ref,
        roles={},
        meta_agent=MetaAgentSpec(
            name="MetaAgent",
            system_prompt="Manage the role pool. Return JSON only.",
            llm_backend=BackendBinding(backend_id="meta-llm"),
            memory_backend=BackendBinding(backend_id="meta-memory"),
        ),
        task_memory_backend=BackendBinding(backend_id="task-memory"),
        dynamic_subagents=DynamicSubagentsConfig(
            enabled=True,
            planner_backend=DynamicBackendBinding(backend_id="planner-llm"),
            default_worker_backend=DynamicBackendBinding(backend_id="worker-llm"),
            allowed_tool_names=["write_report"],
        ),
    )
    return TaskRuntime(
        task_config=task_config,
        llm_runtimes={
            "meta-llm": FakeLLMRuntime(responses=meta_responses),
            "planner-llm": FakeLLMRuntime(responses=planner_responses),
            "worker-llm": FakeLLMRuntime(responses=worker_responses),
        },
        memory_runtimes={"meta-memory": NullMemoryBackend(), "task-memory": NullMemoryBackend()},
        skill_runtimes={"skill": FakeSkillBackend(skills=[])},
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(registry),
    )
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
pytest tests/test_dynamic_workflow_runtime.py::test_dynamic_runtime_updates_agents_md_before_planner_reads_role_pool -q
```

Expected: fail because role-pool update is gated behind optional preplanning metadata or because planner does not see the new role.

- [ ] **Step 3: Add `_maybe_evolve_role_pool()` to `TaskRuntime`**

Modify `evolab/runtime/task_runtime.py`:

```python
from evolab.runtime.role_pool import apply_role_pool_update, role_pool_update_payload
```

Add method inside `TaskRuntime`:

```python
def _maybe_evolve_role_pool(
    self,
    *,
    request: TaskRequest,
    reason: str,
    role_results: list[dict[str, Any]] | None = None,
    planning_feedback: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if self.task_config is None or self.task_config.meta_agent is None or self.task_config.agents_ref is None:
        return None
    meta_agent = self.task_config.meta_agent
    meta_llm = self._llm_runtime(meta_agent.llm_backend.backend_id)
    meta_memory = self._meta_memory_runtime(meta_agent)
    run_ref = f"role-pool-{uuid4()}"
    decision, llm_call_ref, meta_memory_request, meta_memory_bundle = self._next_dispatch_decision(
        request=request,
        meta_agent=meta_agent,
        meta_llm=meta_llm,
        run_ref=run_ref,
        step_index=-1,
        role_results=role_results or [],
        meta_memory=meta_memory,
        preplanning_context={
            "enabled": True,
            "stage": "role_pool_evolution",
            "reason": reason,
            "planning_feedback": _json_compatible(planning_feedback or {}),
            "required_response": (
                "Return JSON with route END and metadata.role_pool_update, "
                "or metadata.no_role_pool_update_reason."
            ),
        },
    )
    payload = role_pool_update_payload(decision.metadata)
    result_payload = None
    if payload is not None:
        result = apply_role_pool_update(
            agents_path=self._agents_config_path(),
            payload=payload,
            task_id=request.task_id,
            run_ref=run_ref,
            known_llm_backend_ids=set(self.llm_runtimes),
            allowed_tool_names=_runtime_allowed_tool_names(self.tool_runtime, self.task_config.runtime_policy.metadata),
        )
        result_payload = result.to_json()
        decision.metadata["role_pool_update_result"] = result_payload
        self.trajectory_collector.record_event(
            event_type="role_pool_update_applied" if result.status == "updated" else "role_pool_update_rejected",
            subject_type="agents_config",
            subject_ref=str(self._agents_config_path()),
            task_id=request.task_id,
            run_ref=run_ref,
            metadata=result_payload,
        )
    else:
        decision.metadata.setdefault("role_pool_update_result", {"status": "no_op"})
    meta_memory_update_result = self._update_meta_agent_memory(
        request=request,
        meta_agent=meta_agent,
        meta_memory=meta_memory,
        meta_memory_bundle=meta_memory_bundle,
        decision=decision,
        run_ref=run_ref,
        step_index=-1,
        role_results=role_results or [],
        llm_call_ref=llm_call_ref,
    )
    self._save_meta_agent_run(
        request=request,
        run_ref=run_ref,
        decision=decision,
        step_index=-1,
        role_results=role_results or [],
        llm_call_ref=llm_call_ref,
        meta_memory_request=meta_memory_request,
        meta_memory_bundle=meta_memory_bundle,
        meta_memory_update_result=meta_memory_update_result,
    )
    return result_payload
```

Add helper near other module-level helpers:

```python
def _runtime_allowed_tool_names(tool_runtime: ToolRuntime | None, metadata: dict[str, Any]) -> set[str]:
    configured = metadata.get("dynamic_allowed_tool_names")
    if isinstance(configured, list):
        return {item for item in configured if isinstance(item, str)}
    if tool_runtime is None:
        return set()
    registry = getattr(tool_runtime, "_registry", None)
    specs = getattr(registry, "_specs", {})
    return set(specs) if isinstance(specs, dict) else set()
```

- [ ] **Step 4: Invoke role-pool evolution before planner**

Modify `_maybe_run_dynamic_subagents()`:

```python
self._maybe_evolve_role_pool(
    request=request,
    reason="before_dynamic_workflow_planning",
    role_results=[],
)
```

Place this before creating `DynamicWorkflowPlanner` and before reading `static_roles = self._optional_roles()`.

- [ ] **Step 5: Tighten preplanning validation**

Modify `_validate_meta_preplanning_decision()`:

```python
if decision.action == DispatchAction.RUN_SUBAGENT:
    raise RuntimeError("role-pool preplanning must not route executable subagent work")
if require_feedback_decision and not _meta_preplanning_has_update_or_noop(decision.metadata):
    raise RuntimeError("role-pool feedback preplanning requires role_pool_update or no_role_pool_update_reason")
if not _meta_preplanning_has_update_or_noop(decision.metadata):
    raise RuntimeError("role-pool preplanning requires role_pool_update or no_role_pool_update_reason")
```

Update `_meta_preplanning_has_update_or_noop()` to call `role_pool_update_payload(metadata)` instead of only the legacy helper.

- [ ] **Step 6: Run focused runtime test**

Run:

```bash
pytest tests/test_dynamic_workflow_runtime.py::test_dynamic_runtime_updates_agents_md_before_planner_reads_role_pool -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add evolab/runtime/task_runtime.py tests/test_dynamic_workflow_runtime.py
git commit -m "feat: evolve role pool before dynamic planning"
```

## Task 4: Planner Role-Pool Template Semantics

**Files:**
- Modify: `evolab/runtime/dynamic_workflow.py`
- Modify: `evolab/runtime/task_runtime.py`
- Test: `tests/test_dynamic_workflow_planner.py`
- Test: `tests/test_dynamic_subagent_factory.py`

- [ ] **Step 1: Write failing planner prompt test**

Add to `tests/test_dynamic_workflow_planner.py`:

```python
def test_dynamic_planner_prompt_uses_role_pool_templates_name():
    runtime = FakeLLMRuntime(
        responses=[
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content=json.dumps(_valid_dynamic_workflow_payload()),
                )
            )
        ]
    )
    planner = DynamicWorkflowPlanner(
        planner_llm=runtime,
        config=DynamicSubagentsConfig(
            enabled=True,
            planner_backend=DynamicBackendBinding(backend_id="planner"),
            default_worker_backend=DynamicBackendBinding(backend_id="worker"),
            allowed_tool_names=[],
        ),
        tool_runtime=ToolRuntime(ToolRegistry()),
        skill_backend=FakeSkillBackend(skills=[]),
        available_llm_backend_ids={"planner", "worker"},
    )

    planner.plan(
        request=TaskRequest(task_id="task-1", origin=TaskOrigin.HUMAN, purpose=TaskPurpose.SCIENCE, goal="x"),
        role_pool_templates=[{"name": "GeneralistAgent", "system_prompt": "Work."}],
    )

    prompt_payload = json.loads(runtime.requests[0].messages[-1].content)
    assert "role_pool_templates" in prompt_payload
    assert "static_fallback_subagents" not in prompt_payload
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
pytest tests/test_dynamic_workflow_planner.py::test_dynamic_planner_prompt_uses_role_pool_templates_name -q
```

Expected: fail because `plan()` accepts `static_subagents` and prompt uses `static_fallback_subagents`.

- [ ] **Step 3: Rename planner parameter and prompt key**

Modify `DynamicWorkflowPlanner.plan()`:

```python
def plan(
    self,
    *,
    request: TaskRequest,
    work_item: dict[str, Any] | None = None,
    role_pool_templates: list[dict[str, Any]] | None = None,
) -> DynamicPlanningOutcome:
    messages = self._messages(
        request=request,
        work_item=work_item,
        role_pool_templates=role_pool_templates or [],
    )
```

Modify `_messages()`:

```python
def _messages(
    self,
    *,
    request: TaskRequest,
    work_item: dict[str, Any] | None,
    role_pool_templates: list[dict[str, Any]],
) -> list[Message]:
    payload = {
        "task_id": request.task_id,
        "task_goal": request.goal,
        "work_item": work_item or {},
        "scope": self.config.scope,
        "mode": self.config.mode,
        "allowed_tool_names": self.config.allowed_tool_names,
        "max_subagents": self.config.max_subagents,
        "max_subagents_per_work_item": self.config.max_subagents_per_work_item,
        "default_worker_backend_id": (
            self.config.default_worker_backend.backend_id if self.config.default_worker_backend else None
        ),
        "role_pool_templates": role_pool_templates,
        "required_response": {
            "format": "JSON only",
            "schema": "DynamicWorkflowSpec",
            "forbidden_top_level_keys": ["name", "version", "description", "tasks", "workflow", "steps"],
            "requirements": [
                "Define the minimal necessary runtime-only subagents.",
                "Use allowed tools only.",
                "Construct a valid DAG.",
                "Do not include chain-of-thought or hidden reasoning.",
                "Use planner_rationale_summary for a concise rationale only.",
            ],
        },
    }
```

Remove `static_fallback_subagents` from the prompt payload.

- [ ] **Step 4: Update call sites**

Modify `TaskRuntime._maybe_run_dynamic_subagents()`:

```python
role_pool_templates = [_role_prompt_payload(role) for role in self._optional_roles()]
outcome = planner.plan(
    request=request,
    work_item=work_item,
    role_pool_templates=role_pool_templates,
)
```

- [ ] **Step 5: Run dynamic planner and factory tests**

Run:

```bash
pytest tests/test_dynamic_workflow_planner.py tests/test_dynamic_subagent_factory.py -q
```

Expected: pass after updating any tests that still assert the old prompt key.

- [ ] **Step 6: Commit**

```bash
git add evolab/runtime/dynamic_workflow.py evolab/runtime/task_runtime.py tests/test_dynamic_workflow_planner.py tests/test_dynamic_subagent_factory.py
git commit -m "refactor: feed planner role pool templates"
```

## Task 5: Task-Memory-Only Worker Execution

**Files:**
- Modify: `evolab/runtime/task_runtime.py`
- Test: `tests/test_dynamic_workflow_runtime.py`
- Test: `tests/test_memory_replay.py`

- [ ] **Step 1: Write failing test that dynamic workers do not update agent memory**

Add to `tests/test_dynamic_workflow_runtime.py`:

```python
def test_dynamic_worker_uses_task_memory_only(tmp_path: Path):
    result, resolver = _run_minimal_dynamic_task_with_memory(tmp_path)

    assert result["status"] == "completed"
    subagent_run = resolver.trajectory_registry().list_subagent_runs()[-1]
    metadata = subagent_run.metadata
    assert metadata["memory_mode"] == "task_only"
    assert "agent_memory_update_result" not in metadata
    assert metadata["task_memory_update_result"]["metadata"]["memory_scope"] == "task"
    state_records = resolver.backend_state_registry().list_states()
    assert all(record.metadata.get("memory_scope") != "agent" for record in state_records if record.metadata.get("role") != "meta")
```

Add this helper in `tests/test_dynamic_workflow_runtime.py`:

```python
def _run_minimal_dynamic_task_with_memory(tmp_path: Path) -> tuple[dict[str, Any], LabResolver]:
    lab_root = tmp_path / "lab"
    resolver = LabResolver.from_root(lab_root)
    agents_path = tmp_path / "agents.md"
    agents_path.write_text(
        render_agents_markdown(
            {
                "WriterAgent": RoleSpec(
                    name="WriterAgent",
                    system_prompt="Write final reports.",
                    llm_backend=BackendBinding(backend_id="worker-llm"),
                    allowed_tools=["write_report"],
                )
            }
        ),
        encoding="utf-8",
    )
    planner_response = LLMRuntimeResponse(
        action=SubAgentAction(
            action="final_answer",
            content=json.dumps(
                {
                    "workflow_id": "wf-memory",
                    "task_summary": "memory check",
                    "article_context_summary": None,
                    "dynamic_subagents": [
                        {
                            "subagent_id": "writer",
                            "role_name": "WriterAgent",
                            "goal": "Write final report.",
                            "system_prompt": "Write final reports.",
                            "allowed_tools": ["write_report"],
                            "output_schema": {"type": "object"},
                        }
                    ],
                    "workflow_nodes": [
                        {"node_id": "write", "subagent_id": "writer", "output_artifacts": ["report.md"]}
                    ],
                    "workflow_edges": [],
                    "artifact_contracts": {},
                    "validation_rules": [],
                    "planner_rationale_summary": "single dynamic writer",
                }
            ),
        )
    )
    runtime = _dynamic_runtime_fixture(
        tmp_path,
        agents_ref=str(agents_path),
        meta_responses=[
            LLMRuntimeResponse(
                action=SubAgentAction(
                    action="final_answer",
                    content='{"route":"END","instruction":"No role update.","metadata":{"no_role_pool_update_reason":"stable"}}',
                )
            )
        ],
        planner_responses=[planner_response],
        worker_responses=[LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="done"))],
    )
    runtime.backend_state_registry = resolver.backend_state_registry()
    result = runtime.run(
        TaskRequest(task_id="task-1", origin=TaskOrigin.HUMAN, purpose=TaskPurpose.SCIENCE, goal="write report")
    )
    return result, resolver
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
pytest tests/test_dynamic_workflow_runtime.py::test_dynamic_worker_uses_task_memory_only -q
```

Expected: fail because `_run_role()` currently searches and updates both agent and task memory.

- [ ] **Step 3: Add memory mode helpers**

Modify `evolab/runtime/task_runtime.py`:

```python
def _worker_memory_mode(dispatch_metadata: dict[str, Any]) -> str:
    if dispatch_metadata.get("worker_memory_mode") in {"task_only", "agent_and_task"}:
        return str(dispatch_metadata["worker_memory_mode"])
    if _dispatch_is_dynamic(dispatch_metadata):
        return "task_only"
    return "agent_and_task"
```

- [ ] **Step 4: Refactor `_run_role()` retrieval**

Replace the direct `agent_memory, task_memory = self._memory_runtimes_for_scopes(role)` block with:

```python
memory_mode = _worker_memory_mode(dispatch_metadata)
task_memory = self._memory_runtime_for_binding(self.task_config.task_memory_backend)
agent_memory = None if memory_mode == "task_only" else self._memory_runtime_for_binding(role.agent_memory_backend)
```

Then construct task retrieval unconditionally and agent retrieval only when `agent_memory is not None`:

```python
task_memory_bundle = task_memory.search(task_retrieval_request)
if agent_memory is None:
    agent_memory_bundle = MemoryBundle(
        backend_id="task-only",
        items=[],
        state_ref=None,
        metadata={"memory_mode": "task_only", "memory_scope": "agent", "memory_scope_id": agent_memory_scope_id},
    )
    memory_bundle = task_memory_bundle
else:
    agent_memory_bundle = agent_memory.search(agent_retrieval_request)
    memory_bundle = _combined_memory_bundle(agent_memory_bundle, task_memory_bundle)
```

- [ ] **Step 5: Refactor `_run_role()` memory update**

Replace unconditional agent/task adds with:

```python
task_memory_update_result = task_memory.add(request.task_id, "task", memory_update_messages)
agent_memory_update_result = None
if agent_memory is not None:
    agent_memory_update_result = agent_memory.add(request.task_id, role.name, memory_update_messages)
```

Register backend state only for task memory in task-only mode:

```python
if agent_memory_update_result is not None:
    _register_memory_state_update(
        registry=self.backend_state_registry,
        task_id=request.task_id,
        run_ref=run_ref,
        role=role.name,
        memory_scope="agent",
        memory_scope_id=agent_memory_scope_id,
        memory_bundle=agent_memory_bundle,
        update_result=agent_memory_update_result,
    )
_register_memory_state_update(
    registry=self.backend_state_registry,
    task_id=request.task_id,
    run_ref=run_ref,
    role=role.name,
    memory_scope="task",
    memory_scope_id=task_memory_scope_id,
    memory_bundle=task_memory_bundle,
    update_result=task_memory_update_result,
)
```

In trajectory metadata, include:

```python
"memory_mode": memory_mode,
"task_memory_bundle": _json_compatible(task_memory_bundle),
"task_memory_update_result": _json_compatible(task_memory_update_result),
```

Only include `agent_memory_bundle` and `agent_memory_update_result` when `agent_memory_update_result is not None`.

- [ ] **Step 6: Keep MetaAgent memory unchanged**

Run:

```bash
pytest tests/test_cli_clean_run.py::test_compile_experiment_config_accepts_meta_agent_memory_backend -q
```

Expected: pass. MetaAgent memory still binds through `meta_agent.memory_backend`.

- [ ] **Step 7: Run memory focused tests**

Run:

```bash
pytest tests/test_dynamic_workflow_runtime.py::test_dynamic_worker_uses_task_memory_only tests/test_memory_replay.py -q
```

Expected: pass after updating memory replay expectations to accept missing worker agent-scope updates on dynamic runs.

- [ ] **Step 8: Commit**

```bash
git add evolab/runtime/task_runtime.py tests/test_dynamic_workflow_runtime.py tests/test_memory_replay.py
git commit -m "feat: use task memory for dynamic workers"
```

## Task 6: Remove Static Default Runtime Path

**Files:**
- Modify: `evolab/runtime/task_runtime.py`
- Modify: `tests/test_static_dynamic_mode_compatibility.py`
- Modify: `tests/test_task_runtime_workflow_plan.py`
- Modify: `tests/test_cli_clean_run.py`

- [ ] **Step 1: Write failing test for missing dynamic config**

Add to `tests/test_static_dynamic_mode_compatibility.py`:

```python
def test_task_runtime_rejects_default_static_role_order():
    runtime = TaskRuntime(
        task_config=TaskConfig(
            task_id="task-1",
            goal="Run static roles.",
            roles={
                "Solver": RoleSpec(
                    name="Solver",
                    system_prompt="Solve.",
                    llm_backend=BackendBinding(backend_id="fake-llm"),
                )
            },
        ),
        llm_runtimes={"fake-llm": FakeLLMRuntime(default_content="done")},
        memory_runtimes={"memory": NullMemoryBackend()},
        skill_runtimes={"skill": FakeSkillBackend(skills=[])},
        prompt_builder=PromptBuilder(),
        tool_runtime=ToolRuntime(ToolRegistry()),
    )

    with pytest.raises(RuntimeError, match="dynamic_subagents.enabled=true"):
        runtime.run(TaskRequest(task_id="task-1", origin=TaskOrigin.HUMAN, purpose=TaskPurpose.SCIENCE, goal="Run static roles."))
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
pytest tests/test_static_dynamic_mode_compatibility.py::test_task_runtime_rejects_default_static_role_order -q
```

Expected: fail because runtime still executes configured roles in order.

- [ ] **Step 3: Remove default static role-order branch**

Modify `_run_without_reflector()` in `evolab/runtime/task_runtime.py`:

```python
def _run_without_reflector(self, request: TaskRequest) -> dict[str, Any]:
    if self.dispatch_loop is not None:
        return self.dispatch_loop(request)
    dynamic_result = self._maybe_run_dynamic_subagents(request)
    if dynamic_result is not None:
        return dynamic_result
    raise RuntimeError("TaskRuntime requires dynamic_subagents.enabled=true for default execution")
```

Remove the old block that iterated `self._roles()` in stage order.

- [ ] **Step 4: Update old tests to use dynamic fixtures**

For tests that intentionally verify prompt building, tool execution, or workflow planning with static roles, convert them to either:

```python
runtime = TaskRuntime(dispatch_loop=lambda request: {"task_id": request.task_id, "status": "completed", "final_answer": "ok", "runs": [], "run_refs": []})
```

when the test is not about default execution. For tests that exercise execution, use the complete dynamic config block below and add a planner fake response that returns one `workflow_node` for the role under test.

Use this replacement pattern in affected tests:

```python
task_config = task_config.model_copy(
    update={
        "agents_ref": str(agents_path),
        "dynamic_subagents": DynamicSubagentsConfig(
            enabled=True,
            planner_backend=DynamicBackendBinding(backend_id="planner-llm"),
            default_worker_backend=DynamicBackendBinding(backend_id="worker-llm"),
            allowed_tool_names=["write_report"],
        ),
    }
)
```

- [ ] **Step 5: Run affected tests**

Run:

```bash
pytest tests/test_static_dynamic_mode_compatibility.py tests/test_task_runtime_workflow_plan.py tests/test_cli_clean_run.py -q
```

Expected: pass after test migration.

- [ ] **Step 6: Commit**

```bash
git add evolab/runtime/task_runtime.py tests/test_static_dynamic_mode_compatibility.py tests/test_task_runtime_workflow_plan.py tests/test_cli_clean_run.py
git commit -m "refactor: remove static default runtime path"
```

## Task 7: Migrate Demo Configs And Documentation

**Files:**
- Modify: `configs/demo_v1_ci.yaml`
- Modify: `configs/demo_v1.yaml`
- Modify: `configs/biology_component_extraction_v1_generic_subagents.yaml`
- Create: `configs/agents/scientific_ie_agents.md`
- Modify: `README.md`
- Modify: `docs/configuration.md`
- Modify: `docs/dynamic_subagent_workflows.md`
- Modify: `evolab/runtime/README.md`

- [ ] **Step 1: Create shared scientific IE role pool file**

Create `configs/agents/scientific_ie_agents.md` with rendered roles equivalent to current generic agents:

````markdown
# EvoLab Agents

This file is the active EvoLab role pool. MetaAgent may update it automatically during dynamic role-pool evolution.

```json
{
  "schema_version": "v1",
  "agents": [
    {
      "schema_version": "v1",
      "name": "SurveyAgent",
      "system_prompt": "Survey assigned context, files, resources, schemas, policies, prior artifacts, and source documents. Report coverage, skipped items, artifacts, and failures.",
      "llm_backend": {"schema_version": "v1", "backend_id": "aigocode-gpt", "config_ref": null, "state_ref": null},
      "agent_memory_backend": null,
      "allowed_tools": ["list_files", "read_text", "inspect_file_metadata", "extract_sections", "search_text", "build_document_inventory", "discover_candidate_source_files", "write_report"],
      "required_skills": [],
      "memory_policy": {},
      "metadata": {"role_pool_seed": true, "role_pool_generation": 0, "specialization": "survey"}
    },
    {
      "schema_version": "v1",
      "name": "ExecAgent",
      "system_prompt": "Execute concrete assigned operations using tools and skills. Read files, inspect documents and tables, extract candidate information, and produce traceable intermediate outputs.",
      "llm_backend": {"schema_version": "v1", "backend_id": "aigocode-gpt", "config_ref": null, "state_ref": null},
      "agent_memory_backend": null,
      "allowed_tools": ["list_files", "read_text", "inspect_table", "read_table_slice", "extract_candidate_rows", "build_candidate_records", "write_report"],
      "required_skills": [],
      "memory_policy": {},
      "metadata": {"role_pool_seed": true, "role_pool_generation": 0, "specialization": "execution"}
    },
    {
      "schema_version": "v1",
      "name": "WriteAgent",
      "system_prompt": "Write final structured artifacts, reports, summaries, and audit files from validated upstream content. Preserve traceability and do not invent records.",
      "llm_backend": {"schema_version": "v1", "backend_id": "aigocode-gpt", "config_ref": null, "state_ref": null},
      "agent_memory_backend": null,
      "allowed_tools": ["write_jsonl", "write_report", "json_schema_validate", "serialize_final_records"],
      "required_skills": [],
      "memory_policy": {},
      "metadata": {"role_pool_seed": true, "role_pool_generation": 0, "specialization": "final writing"}
    }
  ]
}
```
````

If backend ids differ in a config, let CLI materialize config-local seed roles instead of using this file.

- [ ] **Step 2: Update configs**

For each migrated YAML:

- remove top-level `subagents`;
- add `agents_ref: configs/agents/scientific_ie_agents.md` when backend id matches;
- add or keep `dynamic_subagents.enabled=true`;
- ensure `meta_agent.memory_backend` uses `mem0-meta-memory` when configured;
- ensure worker memory config contains only task memory for execution, plus meta memory for MetaAgent.

Use this config shape for CI demo:

```yaml
agents_ref: agents.md
dynamic_subagents:
  enabled: true
  mode: dynamic
  scope: per_task
  planner_backend:
    backend_id: fake-llm
  default_worker_backend:
    backend_id: fake-llm
  allowed_tool_names: []
```

- [ ] **Step 3: Update docs**

Update docs to state:

```text
EvoLab no longer treats top-level subagents as the default execution model.
The active role pool is agents.md. MetaAgent may update this file automatically,
and DynamicWorkflowPlanner consumes the latest role templates for each run.
Dynamic worker roles use task-level memory only; MetaAgent may keep stable
agent-scope memory.
```

- [ ] **Step 4: Run config and docs focused tests**

Run:

```bash
pytest tests/test_biology_generic_subagents_config.py tests/test_cli_clean_run.py::test_clean_run_demo_v1_records_mem0_memory_lineage -q
```

Expected: pass after updating assertions from `subagents` to `agents_ref` and dynamic role pool.

- [ ] **Step 5: Commit**

```bash
git add configs README.md docs evolab/runtime/README.md tests/test_biology_generic_subagents_config.py tests/test_cli_clean_run.py
git commit -m "docs: migrate configs to dynamic role pool"
```

## Task 8: Full Regression And Cleanup

**Files:**
- Modify: tests touched by full-suite failures.
- Modify: docs touched by stale static-mode references.

- [ ] **Step 1: Run full suite**

Run:

```bash
pytest -q
```

Expected: remaining failures identify stale static-mode assumptions or memory lineage expectations.

- [ ] **Step 2: Remove stale static-mode wording from code comments and docs**

Run:

```bash
rg -n "static_fallback_subagents|default static|subagents is|subagents:|agent_memory_backend" README.md docs evolab configs tests
```

For each match, keep it only when it documents legacy migration or MetaAgent memory. Replace runtime-path wording with:

```text
role pool templates
dynamic role pool
task-level worker memory
```

- [ ] **Step 3: Run targeted tests for changed areas**

Run:

```bash
pytest tests/test_role_pool_runtime.py tests/test_dynamic_workflow_planner.py tests/test_dynamic_workflow_runtime.py tests/test_memory_replay.py tests/test_cli_clean_run.py -q
```

Expected: pass.

- [ ] **Step 4: Run full suite again**

Run:

```bash
pytest -q
```

Expected: pass.

- [ ] **Step 5: Commit final cleanup**

```bash
git add .
git commit -m "test: complete dynamic role pool migration"
```

## Self-Review

Spec coverage:

- Dynamic role generation default: Tasks 2, 3, 4, 6, and 7.
- `agents.md` as active auto-evolving role pool: Tasks 1, 2, and 3.
- MetaAgent direct add/delete/modify without human approval: Tasks 1 and 3.
- Remove static default mode: Task 6.
- Task-level worker memory only: Task 5.
- Stable MetaAgent memory: Tasks 3 and 5.
- Auditable guarded writes: Task 1.
- Config and docs migration: Task 7.
- Full verification: Task 8.

Type consistency:

- Canonical update field is `metadata.role_pool_update`.
- Migration aliases are handled by `role_pool_update_payload()`.
- Planner input is `role_pool_templates`.
- Worker dynamic memory mode is `task_only`.
- MetaAgent memory remains under existing `meta_agent.memory_backend`.

Execution order:

- Task 1 must land before runtime integration because it provides the writer and validator.
- Task 2 must land before broad config migration because it materializes seed role pools.
- Task 5 should land before Task 6 so dynamic runtime remains functional while static fallback is removed.
