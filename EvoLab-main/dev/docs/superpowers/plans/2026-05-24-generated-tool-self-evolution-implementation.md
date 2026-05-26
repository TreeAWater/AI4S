# Generated Tool Self-Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add task-local, LLM-generated Python tools that EvoLab can validate, register, execute, trace, reset per task, expose to dynamic planners, and use during runtime repair.

**Architecture:** Built-in generic tools remain in `ToolRegistry`; generated tools live in a task-scoped registry managed by `ToolRuntime`. A new `evolab.runtime.generated_tools` service validates generated Python packages, persists source artifacts, wraps execution through a subprocess adapter, and registers passing tools. `TaskRuntime` runs tool-code preplanning before role-pool evolution and gives DynamicWorkflowPlanner an effective tool catalog that includes validated generated tools.

**Tech Stack:** Python 3.11, Pydantic contracts, existing `ToolRuntime`, existing `TaskRuntime`, existing dynamic workflow planner/factory, `pytest`, subprocess-based generated tool execution.

---

## File Structure

- Create `evolab/contracts/generated_tools.py`
  Defines typed generated-tool package, capability grants, validation result, registration record, and effective catalog contracts.
- Modify `evolab/contracts/common.py`
  Adds runtime policy budgets and generated tool capability switches.
- Modify `evolab/contracts/repair.py`
  Replaces untyped `RepairPlan.new_runtime_tool` with typed generated-tool packages.
- Create `evolab/runtime/generated_tools.py`
  Persists source files, validates package shape/code, runs smoke tests, builds subprocess handlers, registers task-local generated tools, and builds effective catalogs.
- Modify `evolab/tools/runtime.py`
  Adds task-scoped generated tool registry and effective lookup support without erasing generated tools during `prepare()`.
- Modify `evolab/runtime/dynamic_workflow.py`
  Adds `TaskEffectiveToolCatalog` to planner messages, templates, validation, and subagent factory validation.
- Modify `evolab/runtime/task_runtime.py`
  Resets generated tools per task, runs MetaAgent tool-code preplanning, builds `GeneratedToolBuilder`, passes effective catalog to planner/factory, and includes generated tool provenance in skill observations.
- Modify `evolab/runtime/capability_repair.py`
  Uses generated tool builder/registrar for `RepairPlan.new_runtime_tool`, rewrites retry plans to registered names, and emits `new_tool` promotion candidates.
- Add `tests/test_generated_tools.py`
  Contract, validation, subprocess execution, registration, reset, capability grant, and failure-mode tests.
- Modify `tests/test_tool_runtime.py`
  Adds generated registry persistence and prepared-tool enforcement regressions.
- Modify `tests/test_dynamic_workflow_planner.py` and `tests/test_dynamic_workflow_runtime.py`
  Adds effective catalog planner visibility and runtime preplanning integration tests.
- Modify `tests/test_capability_repair.py`
  Adds generated repair tool path tests.
- Update `docs/dynamic_subagent_workflows.md` and `README.md`
  Documents generated task-local Python tools and policy defaults.

---

### Task 1: Generated Tool Contracts And Runtime Policy

**Files:**
- Create: `evolab/contracts/generated_tools.py`
- Modify: `evolab/contracts/common.py`
- Modify: `evolab/contracts/repair.py`
- Test: `tests/test_generated_tools.py`

- [ ] **Step 1: Write failing contract tests**

Add `tests/test_generated_tools.py` with these initial tests:

```python
from __future__ import annotations

import pytest

from evolab.contracts.common import RuntimePolicy
from evolab.contracts.generated_tools import (
    GeneratedToolCapabilityGrant,
    GeneratedToolFile,
    GeneratedToolPackage,
    GeneratedToolRegistration,
    GeneratedToolSmokeTest,
    GeneratedToolValidationResult,
    TaskEffectiveToolCatalog,
)
from evolab.contracts.repair import RepairPlan
from evolab.contracts.tools import ToolSpec


def test_generated_tool_package_validates_primary_module_and_files():
    package = GeneratedToolPackage(
        tool_name="extract_rows",
        reason="Need task-specific row extraction.",
        manifest={
            "description": "Extract task rows.",
            "parameters_schema": {"type": "object"},
        },
        files=[
            GeneratedToolFile(
                path="tool.py",
                content="TOOL_SPEC = {'name': 'extract_rows', 'description': 'Extract.', 'parameters_schema': {'type': 'object'}}\n\ndef run(arguments, context):\n    return 'ok'\n",
            )
        ],
        smoke_tests=[GeneratedToolSmokeTest(name="basic", arguments={})],
    )

    assert package.primary_module == "tool.py"
    assert package.files[0].path == "tool.py"


def test_generated_tool_package_rejects_path_escape():
    with pytest.raises(ValueError, match="must be a relative path inside the package"):
        GeneratedToolPackage(
            tool_name="bad",
            reason="path escape",
            manifest={"description": "Bad.", "parameters_schema": {"type": "object"}},
            files=[GeneratedToolFile(path="../bad.py", content="")],
        )


def test_runtime_policy_exposes_generated_tool_limits():
    policy = RuntimePolicy()

    assert policy.allow_runtime_tool_creation is True
    assert policy.max_generated_tools_per_task == 8
    assert policy.max_generated_tool_files == 8
    assert policy.max_generated_tool_source_bytes == 64_000
    assert policy.generated_tool_validation_timeout_s == 5
    assert policy.generated_tool_execution_timeout_s == 10
    assert policy.generated_tool_max_output_bytes == 256_000
    assert policy.generated_tool_allow_network is False
    assert policy.generated_tool_allow_subprocess is False
    assert policy.generated_tool_allowed_env_keys == []


def test_repair_plan_accepts_typed_generated_runtime_tool():
    package = GeneratedToolPackage(
        tool_name="fix_rows",
        reason="The original call requested a missing task-specific tool.",
        manifest={"description": "Fix rows.", "parameters_schema": {"type": "object"}},
        files=[GeneratedToolFile(path="tool.py", content="TOOL_SPEC = {}\ndef run(arguments, context):\n    return 'ok'\n")],
    )

    plan = RepairPlan(
        repair_id="repair-1",
        failure_id="failure-1",
        diagnosis="missing_tool",
        repair_action="create_runtime_tool",
        rationale="Need generated tool.",
        new_runtime_tool=package,
    )

    assert plan.new_runtime_tool.tool_name == "fix_rows"


def test_effective_catalog_combines_builtin_and_generated_specs():
    builtin = ToolSpec(name="read_text", description="Read text.", parameters_schema={"type": "object"})
    generated = ToolSpec(
        name="generated_task_extract_rows",
        description="Generated rows.",
        parameters_schema={"type": "object"},
        metadata={"generated_tool": True},
    )

    catalog = TaskEffectiveToolCatalog(
        task_id="task-1",
        builtin_allowed_tool_names=["read_text"],
        generated_tool_names=["generated_task_extract_rows"],
        tool_specs_by_name={"read_text": builtin, "generated_task_extract_rows": generated},
        provenance_by_name={"generated_task_extract_rows": {"code_hash": "abc"}},
    )

    assert catalog.effective_allowed_tool_names == ["read_text", "generated_task_extract_rows"]
    assert catalog.tool_specs_by_name["generated_task_extract_rows"].metadata["generated_tool"] is True


def test_generated_tool_registration_records_validation_and_grants():
    grant = GeneratedToolCapabilityGrant(
        allowed_read_roots=["/tmp/input"],
        allowed_write_root="/tmp/lab/artifacts/generated",
    )
    registration = GeneratedToolRegistration(
        requested_tool_name="extract_rows",
        registered_tool_name="gt_task_extract_rows",
        task_id="task-1",
        run_ref="run-1",
        tool_spec=ToolSpec(name="gt_task_extract_rows", description="Generated.", parameters_schema={"type": "object"}),
        module_path="/tmp/lab/artifacts/generated/tool.py",
        code_hash="abc",
        validation=GeneratedToolValidationResult(valid=True, status="passed"),
        capability_grant=grant,
    )

    assert registration.validation.valid is True
    assert registration.capability_grant.allow_network is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest -q tests/test_generated_tools.py
```

Expected: import errors for `evolab.contracts.generated_tools` and missing `RuntimePolicy` fields.

- [ ] **Step 3: Implement generated tool contracts**

Create `evolab/contracts/generated_tools.py`:

```python
from __future__ import annotations

from hashlib import sha256
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import Field, model_validator

from evolab.contracts.common import StrictBaseModel
from evolab.contracts.tools import ToolSpec


class GeneratedToolFile(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    path: str
    content: str

    @model_validator(mode="after")
    def validate_relative_package_path(self) -> "GeneratedToolFile":
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts or not self.path.strip():
            raise ValueError("generated tool file path must be a relative path inside the package")
        return self


class GeneratedToolSmokeTest(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    expect_status: Literal["ok", "error"] | None = None


class GeneratedToolCapabilityGrant(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    allowed_read_roots: list[str] = Field(default_factory=list)
    allowed_write_root: str | None = None
    allowed_env_keys: list[str] = Field(default_factory=list)
    allow_network: bool = False
    allow_subprocess: bool = False
    allowed_imports: list[str] = Field(default_factory=list)


class GeneratedToolPackage(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    tool_name: str
    reason: str
    manifest: dict[str, Any] = Field(default_factory=dict)
    files: list[GeneratedToolFile] = Field(default_factory=list)
    primary_module: str = "tool.py"
    smoke_tests: list[GeneratedToolSmokeTest] = Field(default_factory=list)
    capability_grant: GeneratedToolCapabilityGrant | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_package(self) -> "GeneratedToolPackage":
        if not self.tool_name.strip():
            raise ValueError("generated tool package requires tool_name")
        if not self.reason.strip():
            raise ValueError("generated tool package requires reason")
        if not self.files:
            raise ValueError("generated tool package requires at least one file")
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("generated tool package contains duplicate file paths")
        if self.primary_module not in set(paths):
            raise ValueError("generated tool primary_module must match one package file")
        return self

    @property
    def source_bytes(self) -> int:
        return sum(len(item.content.encode("utf-8")) for item in self.files)

    @property
    def package_hash(self) -> str:
        digest = sha256()
        for item in sorted(self.files, key=lambda file: file.path):
            digest.update(item.path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(item.content.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()


class GeneratedToolValidationResult(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    valid: bool
    status: Literal["passed", "failed", "skipped"]
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    smoke_test_results: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GeneratedToolRegistration(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    requested_tool_name: str
    registered_tool_name: str
    task_id: str
    run_ref: str
    tool_spec: ToolSpec
    module_path: str
    code_hash: str
    validation: GeneratedToolValidationResult
    capability_grant: GeneratedToolCapabilityGrant = Field(default_factory=GeneratedToolCapabilityGrant)
    artifact_refs: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskEffectiveToolCatalog(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    task_id: str
    builtin_allowed_tool_names: list[str] = Field(default_factory=list)
    generated_tool_names: list[str] = Field(default_factory=list)
    tool_specs_by_name: dict[str, ToolSpec] = Field(default_factory=dict)
    provenance_by_name: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @property
    def effective_allowed_tool_names(self) -> list[str]:
        seen: set[str] = set()
        names: list[str] = []
        for name in [*self.builtin_allowed_tool_names, *self.generated_tool_names]:
            if name not in seen:
                names.append(name)
                seen.add(name)
        return names
```

- [ ] **Step 4: Add runtime policy fields**

Modify `RuntimePolicy` in `evolab/contracts/common.py`:

```python
    max_generated_tools_per_task: int = Field(default=8, ge=0)
    max_generated_tool_files: int = Field(default=8, ge=1)
    max_generated_tool_source_bytes: int = Field(default=64_000, ge=0)
    generated_tool_validation_timeout_s: int = Field(default=5, ge=1)
    generated_tool_execution_timeout_s: int = Field(default=10, ge=1)
    generated_tool_max_output_bytes: int = Field(default=256_000, ge=1)
    generated_tool_allowed_imports: list[str] = Field(default_factory=list)
    generated_tool_allow_network: bool = False
    generated_tool_allow_subprocess: bool = False
    generated_tool_allowed_env_keys: list[str] = Field(default_factory=list)
```

Place these after `allow_global_tool_mutation`.

- [ ] **Step 5: Type `RepairPlan.new_runtime_tool`**

Modify `evolab/contracts/repair.py`:

```python
from evolab.contracts.generated_tools import GeneratedToolPackage
```

Change the field:

```python
    new_runtime_tool: GeneratedToolPackage | None = None
```

- [ ] **Step 6: Run contract tests**

Run:

```bash
pytest -q tests/test_generated_tools.py
```

Expected: all tests in `tests/test_generated_tools.py` pass.

- [ ] **Step 7: Commit**

```bash
git add evolab/contracts/generated_tools.py evolab/contracts/common.py evolab/contracts/repair.py tests/test_generated_tools.py
git commit -m "feat: add generated tool contracts"
```

---

### Task 2: ToolRuntime Task-Local Generated Registry

**Files:**
- Modify: `evolab/tools/runtime.py`
- Modify: `tests/test_tool_runtime.py`
- Test: `tests/test_tool_runtime.py`

- [ ] **Step 1: Add failing ToolRuntime tests**

Append to `tests/test_tool_runtime.py`:

```python
def test_generated_tools_survive_prepare_until_task_reset():
    registry = ToolRegistry()
    registry.register(_spec("read_text"), lambda arguments: "read")
    runtime = ToolRuntime(registry)
    runtime.activate_generated_tool_scope("task-1")
    runtime.register_task_generated_tool(
        _spec("gt_extract", generated_tool=True, task_id="task-1"),
        lambda arguments: ToolResult(
            call_id="local",
            status="ok",
            content="generated",
            metadata={"source": "generated"},
        ),
        provenance={"code_hash": "abc"},
    )

    first = runtime.prepare(
        required_tools=["gt_extract"],
        allowed_tools=["gt_extract"],
        policy=RuntimePolicy(),
    )
    second = runtime.prepare(
        required_tools=["read_text", "gt_extract"],
        allowed_tools=["read_text", "gt_extract"],
        policy=RuntimePolicy(),
    )

    assert [spec.name for spec in first.tool_specs] == ["gt_extract"]
    assert [spec.name for spec in second.tool_specs] == ["read_text", "gt_extract"]

    result = runtime.execute_tool_name("call-1", "gt_extract", {})
    assert result.status == "ok"
    assert result.metadata["generated_tool"]["code_hash"] == "abc"

    runtime.reset_task_generated_tools("task-1")
    after_reset = runtime.prepare(
        required_tools=["gt_extract"],
        allowed_tools=["gt_extract"],
        policy=RuntimePolicy(),
    )
    assert after_reset.tool_specs == []


def test_generated_tool_scope_rejects_interleaved_task_switch():
    runtime = ToolRuntime(ToolRegistry())
    runtime.activate_generated_tool_scope("task-1")
    runtime.register_task_generated_tool(
        _spec("gt_extract", generated_tool=True),
        lambda arguments: "ok",
        provenance={},
    )

    with pytest.raises(ValueError, match="cannot switch generated tool scope"):
        runtime.activate_generated_tool_scope("task-2")


def test_unprepared_generated_tool_call_is_rejected():
    runtime = ToolRuntime(ToolRegistry())
    runtime.activate_generated_tool_scope("task-1")
    runtime.register_task_generated_tool(_spec("gt_extract", generated_tool=True), lambda arguments: "ok")
    runtime.prepare(required_tools=[], allowed_tools=["gt_extract"], policy=RuntimePolicy())

    result = runtime.execute_tool_name("call-1", "gt_extract", {})

    assert result.status == "error"
    assert result.metadata["error_type"] == "unprepared_tool"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest -q tests/test_tool_runtime.py::test_generated_tools_survive_prepare_until_task_reset tests/test_tool_runtime.py::test_generated_tool_scope_rejects_interleaved_task_switch tests/test_tool_runtime.py::test_unprepared_generated_tool_call_is_rejected
```

Expected: `ToolRuntime` has no generated tool scope methods.

- [ ] **Step 3: Add task-local generated registry fields and APIs**

Modify `ToolRuntime.__init__`:

```python
        self._generated_task_id: str | None = None
        self._generated_specs: dict[str, ToolSpec] = {}
        self._generated_handlers: dict[str, ToolHandler] = {}
        self._generated_provenance: dict[str, dict[str, Any]] = {}
```

Add methods to `ToolRuntime`:

```python
    def activate_generated_tool_scope(self, task_id: str) -> None:
        if not task_id:
            raise ValueError("generated tool scope requires task_id")
        if self._generated_task_id == task_id:
            return
        if self._generated_specs:
            raise ValueError("cannot switch generated tool scope while generated tools are active")
        self._generated_task_id = task_id

    def reset_task_generated_tools(self, task_id: str | None = None) -> None:
        if task_id is not None and self._generated_task_id not in {None, task_id}:
            raise ValueError("cannot reset generated tools for a different task scope")
        self._generated_task_id = task_id if task_id is not None else None
        self._generated_specs = {}
        self._generated_handlers = {}
        self._generated_provenance = {}
        if self._prepared_tool_names is not None:
            self._prepared_tool_names = {
                name for name in self._prepared_tool_names if name not in self._generated_specs
            }

    def register_task_generated_tool(
        self,
        spec: ToolSpec,
        handler: ToolHandler,
        *,
        provenance: dict[str, Any] | None = None,
        replace_existing: bool = False,
    ) -> None:
        if self._generated_task_id is None:
            raise ValueError("activate generated tool scope before registering generated tools")
        if self._registry.get_spec(spec.name) is not None and not replace_existing:
            raise ValueError(f"generated tool {spec.name!r} would replace a built-in tool")
        if spec.name in self._generated_specs and not replace_existing:
            raise ValueError(f"generated tool {spec.name!r} is already registered")
        self._generated_specs[spec.name] = spec
        self._generated_handlers[spec.name] = handler
        self._generated_provenance[spec.name] = dict(provenance or {})

    def generated_tool_names(self) -> list[str]:
        return list(self._generated_specs)

    def generated_tool_provenance(self, name: str) -> dict[str, Any]:
        return dict(self._generated_provenance.get(name) or {})
```

Fix `reset_task_generated_tools()` after implementation so it drops prepared generated names before clearing:

```python
        generated_names = set(self._generated_specs)
        self._generated_task_id = task_id if task_id is not None else None
        self._generated_specs = {}
        self._generated_handlers = {}
        self._generated_provenance = {}
        if self._prepared_tool_names is not None:
            self._prepared_tool_names = {name for name in self._prepared_tool_names if name not in generated_names}
```

- [ ] **Step 4: Update effective lookup and provenance wrapping**

Modify `_get_effective_spec()`:

```python
    def _get_effective_spec(self, name: str) -> ToolSpec | None:
        return self._runtime_specs.get(name) or self._generated_specs.get(name) or self._registry.get_spec(name)
```

Modify `_get_effective_handler()`:

```python
    def _get_effective_handler(self, name: str) -> ToolHandler:
        if name in self._runtime_handlers:
            return self._runtime_handlers[name]
        if name in self._generated_handlers:
            return self._generated_handlers[name]
        return self._registry.get_handler(name)
```

Modify `_execute_tool_name()` after handler output normalization:

```python
            if name in self._generated_specs:
                metadata = dict(result.metadata)
                metadata["generated_tool"] = {
                    "task_id": self._generated_task_id,
                    **self._generated_provenance.get(name, {}),
                }
                result = result.model_copy(update={"metadata": metadata})
            return result
```

Implement that by assigning `result` before return for both `ToolResult` and string outputs.

- [ ] **Step 5: Run ToolRuntime tests**

Run:

```bash
pytest -q tests/test_tool_runtime.py
```

Expected: all ToolRuntime tests pass.

- [ ] **Step 6: Commit**

```bash
git add evolab/tools/runtime.py tests/test_tool_runtime.py
git commit -m "feat: add task-local generated tool registry"
```

---

### Task 3: Generated Tool Runtime Service

**Files:**
- Create: `evolab/runtime/generated_tools.py`
- Modify: `tests/test_generated_tools.py`
- Test: `tests/test_generated_tools.py`

- [ ] **Step 1: Add failing generated runtime tests**

Append to `tests/test_generated_tools.py`:

```python
from pathlib import Path

from evolab.runtime.generated_tools import GeneratedToolRuntime
from evolab.tools.runtime import ToolRegistry, ToolRuntime


VALID_TOOL_CODE = """\
TOOL_SPEC = {
    "name": "extract_rows",
    "description": "Extract rows for this task.",
    "parameters_schema": {"type": "object", "properties": {"value": {"type": "string"}}},
    "metadata": {"generated_tool": True}
}

def run(arguments, context):
    return {
        "status": "ok",
        "content": "value=" + str(arguments.get("value")),
        "metadata": {"task_id": context.get("task_id")}
    }
"""


def _package(code: str = VALID_TOOL_CODE, *, tool_name: str = "extract_rows") -> GeneratedToolPackage:
    return GeneratedToolPackage(
        tool_name=tool_name,
        reason="Need generated behavior.",
        manifest={
            "description": "Extract rows for this task.",
            "parameters_schema": {"type": "object"},
        },
        files=[GeneratedToolFile(path="tool.py", content=code)],
        smoke_tests=[GeneratedToolSmokeTest(name="basic", arguments={"value": "alpha"})],
    )


def test_generated_runtime_persists_validates_registers_and_executes(tmp_path: Path):
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    generated = GeneratedToolRuntime(
        artifact_root=tmp_path,
        tool_runtime=tool_runtime,
        policy=RuntimePolicy(),
    )

    registration = generated.register_package(
        package=_package(),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert registration.validation.valid is True
    assert registration.registered_tool_name.startswith("gt_task_1_run_1_extract_rows")
    assert Path(registration.module_path).exists()
    assert registration.code_hash

    bundle = tool_runtime.prepare(
        required_tools=[registration.registered_tool_name],
        allowed_tools=[registration.registered_tool_name],
        policy=RuntimePolicy(),
    )
    assert [spec.name for spec in bundle.tool_specs] == [registration.registered_tool_name]

    result = tool_runtime.execute_tool_name(
        "call-1",
        registration.registered_tool_name,
        {"value": "beta"},
    )
    assert result.status == "ok"
    assert result.content == "value=beta"
    assert result.metadata["generated_tool"]["code_hash"] == registration.code_hash


def test_generated_runtime_rejects_syntax_error_without_registration(tmp_path: Path):
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    generated = GeneratedToolRuntime(tmp_path, tool_runtime=tool_runtime, policy=RuntimePolicy())

    registration = generated.register_package(
        package=_package("def broken(:\n    pass\n"),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert registration.validation.valid is False
    assert any("invalid Python syntax" in error for error in registration.validation.errors)
    assert tool_runtime.generated_tool_names() == []


def test_generated_runtime_rejects_forbidden_static_capabilities(tmp_path: Path):
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    generated = GeneratedToolRuntime(tmp_path, tool_runtime=tool_runtime, policy=RuntimePolicy())

    registration = generated.register_package(
        package=_package("import subprocess\nTOOL_SPEC = {'name': 'x', 'description': 'x', 'parameters_schema': {'type': 'object'}}\ndef run(arguments, context):\n    return 'x'\n"),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert registration.validation.valid is False
    assert any("subprocess" in error for error in registration.validation.errors)


def test_generated_runtime_smoke_test_failure_rejects_tool(tmp_path: Path):
    code = """\
TOOL_SPEC = {"name": "extract_rows", "description": "Extract.", "parameters_schema": {"type": "object"}}
def run(arguments, context):
    raise RuntimeError("boom")
"""
    tool_runtime = ToolRuntime(ToolRegistry())
    tool_runtime.activate_generated_tool_scope("task-1")
    generated = GeneratedToolRuntime(tmp_path, tool_runtime=tool_runtime, policy=RuntimePolicy())

    registration = generated.register_package(
        package=_package(code),
        task_id="task-1",
        run_ref="run-1",
        context={"task_id": "task-1"},
    )

    assert registration.validation.valid is False
    assert any("smoke test" in error for error in registration.validation.errors)
    assert tool_runtime.generated_tool_names() == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest -q tests/test_generated_tools.py
```

Expected: import error for `evolab.runtime.generated_tools`.

- [ ] **Step 3: Implement `GeneratedToolRuntime`**

Create `evolab/runtime/generated_tools.py` with these public pieces:

```python
from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
from uuid import uuid4

from evolab.contracts.common import RuntimePolicy
from evolab.contracts.generated_tools import (
    GeneratedToolCapabilityGrant,
    GeneratedToolPackage,
    GeneratedToolRegistration,
    GeneratedToolValidationResult,
    TaskEffectiveToolCatalog,
)
from evolab.contracts.tools import ToolResult, ToolSpec
from evolab.tools.runtime import ToolRuntime
```

Implement:

```python
class GeneratedToolRuntime:
    def __init__(
        self,
        artifact_root: str | Path,
        *,
        tool_runtime: ToolRuntime,
        policy: RuntimePolicy,
        trajectory_collector: Any | None = None,
    ) -> None:
        self.artifact_root = Path(artifact_root)
        self.tool_runtime = tool_runtime
        self.policy = policy
        self.trajectory_collector = trajectory_collector
        self._registered_count = 0

    def register_package(
        self,
        *,
        package: GeneratedToolPackage,
        task_id: str,
        run_ref: str,
        context: dict[str, Any] | None = None,
    ) -> GeneratedToolRegistration:
        registered_name = _registered_tool_name(task_id, run_ref, package.tool_name)
        package_dir = self.artifact_root / "generated_tools" / registered_name
        validation_errors = self._preflight(package)
        package_dir.mkdir(parents=True, exist_ok=True)
        for file in package.files:
            destination = package_dir / file.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(file.content, encoding="utf-8")
        primary_path = package_dir / package.primary_module
        grant = package.capability_grant or _default_grant(package_dir)
        tool_spec = _fallback_spec(package, registered_name)
        validation = GeneratedToolValidationResult(
            valid=False if validation_errors else True,
            status="failed" if validation_errors else "passed",
            errors=validation_errors,
        )
        if not validation_errors:
            validation, tool_spec = self._validate_load_and_smoke(
                package=package,
                registered_name=registered_name,
                module_path=primary_path,
                context={**(context or {}), "task_id": task_id, "capability_grant": grant.model_dump(mode="json")},
            )
        registration = GeneratedToolRegistration(
            requested_tool_name=package.tool_name,
            registered_tool_name=registered_name,
            task_id=task_id,
            run_ref=run_ref,
            tool_spec=tool_spec,
            module_path=str(primary_path),
            code_hash=package.package_hash,
            validation=validation,
            capability_grant=grant,
            metadata={"package_dir": str(package_dir), "reason": package.reason},
        )
        if validation.valid:
            self.tool_runtime.register_task_generated_tool(
                tool_spec,
                _subprocess_handler(
                    module_path=primary_path,
                    timeout_s=self.policy.generated_tool_execution_timeout_s,
                    max_output_bytes=self.policy.generated_tool_max_output_bytes,
                    context={**(context or {}), "task_id": task_id, "capability_grant": grant.model_dump(mode="json")},
                ),
                provenance={
                    "requested_tool_name": package.tool_name,
                    "registered_tool_name": registered_name,
                    "code_hash": package.package_hash,
                    "module_path": str(primary_path),
                    "validation": validation.model_dump(mode="json"),
                },
            )
            self._registered_count += 1
        return registration
```

Implement private helpers:

```python
def _registered_tool_name(task_id: str, run_ref: str, requested: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", requested).strip("_") or "tool"
    task_slug = re.sub(r"[^A-Za-z0-9_]+", "_", task_id).strip("_")[:24] or "task"
    run_slug = re.sub(r"[^A-Za-z0-9_]+", "_", run_ref).strip("_")[:24] or "run"
    return f"gt_{task_slug}_{run_slug}_{slug}"


def _default_grant(package_dir: Path) -> GeneratedToolCapabilityGrant:
    return GeneratedToolCapabilityGrant(allowed_write_root=str(package_dir))


def _fallback_spec(package: GeneratedToolPackage, registered_name: str) -> ToolSpec:
    manifest = dict(package.manifest)
    return ToolSpec(
        name=registered_name,
        description=str(manifest.get("description") or package.reason),
        parameters_schema=manifest.get("parameters_schema") if isinstance(manifest.get("parameters_schema"), dict) else {"type": "object"},
        metadata={"generated_tool": True, "requested_tool_name": package.tool_name, **(manifest.get("metadata") if isinstance(manifest.get("metadata"), dict) else {})},
    )
```

For validation, parse AST and reject forbidden imports:

```python
FORBIDDEN_IMPORTS = {"subprocess", "socket", "requests", "urllib", "httpx", "os"}

def _static_python_errors(source: str, *, policy: RuntimePolicy) -> list[str]:
    errors: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"invalid Python syntax: {exc}"]
    forbidden = set()
    if not policy.generated_tool_allow_subprocess:
        forbidden.add("subprocess")
    if not policy.generated_tool_allow_network:
        forbidden.update({"socket", "requests", "urllib", "httpx"})
    if not policy.generated_tool_allowed_env_keys:
        forbidden.add("os")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in forbidden:
                    errors.append(f"import {root!r} is not allowed by generated tool policy")
        if isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in forbidden:
                errors.append(f"import {root!r} is not allowed by generated tool policy")
    return errors
```

For subprocess execution, write a tiny runner file next to the tool module when executing. The runner loads `tool.py`, calls `run(arguments, context)`, and prints JSON with `status`, `content`, and `metadata`.

- [ ] **Step 4: Build effective catalog helper**

In `GeneratedToolRuntime`, add:

```python
def build_effective_tool_catalog(
    *,
    task_id: str,
    builtin_allowed_tool_names: list[str],
    tool_runtime: ToolRuntime,
) -> TaskEffectiveToolCatalog:
    tool_specs_by_name = {}
    for name in [*builtin_allowed_tool_names, *tool_runtime.generated_tool_names()]:
        spec = tool_runtime._get_effective_spec(name)
        if spec is not None:
            tool_specs_by_name[name] = spec
    provenance_by_name = {
        name: tool_runtime.generated_tool_provenance(name)
        for name in tool_runtime.generated_tool_names()
    }
    return TaskEffectiveToolCatalog(
        task_id=task_id,
        builtin_allowed_tool_names=builtin_allowed_tool_names,
        generated_tool_names=tool_runtime.generated_tool_names(),
        tool_specs_by_name=tool_specs_by_name,
        provenance_by_name=provenance_by_name,
    )
```

- [ ] **Step 5: Run generated runtime tests**

Run:

```bash
pytest -q tests/test_generated_tools.py tests/test_tool_runtime.py
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add evolab/runtime/generated_tools.py tests/test_generated_tools.py
git commit -m "feat: validate and register generated tools"
```

---

### Task 4: Dynamic Workflow Effective Tool Catalog

**Files:**
- Modify: `evolab/runtime/dynamic_workflow.py`
- Modify: `tests/test_dynamic_workflow_planner.py`
- Modify: `tests/test_dynamic_workflow_runtime.py`
- Test: `tests/test_dynamic_workflow_planner.py`, `tests/test_dynamic_workflow_runtime.py`

- [ ] **Step 1: Add failing planner prompt test**

Add to `tests/test_dynamic_workflow_planner.py`:

```python
def test_dynamic_planner_prompt_includes_effective_generated_tool_catalog():
    registry = ToolRegistry()
    registry.register(ToolSpec(name="read_text", description="Read text.", parameters_schema={"type": "object"}), lambda args: "ok")
    runtime = ToolRuntime(registry)
    runtime.activate_generated_tool_scope("task-1")
    runtime.register_task_generated_tool(
        ToolSpec(name="gt_task_extract", description="Generated extract.", parameters_schema={"type": "object"}, metadata={"generated_tool": True}),
        lambda args: "ok",
        provenance={"code_hash": "abc"},
    )
    catalog = TaskEffectiveToolCatalog(
        task_id="task-1",
        builtin_allowed_tool_names=["read_text"],
        generated_tool_names=["gt_task_extract"],
        tool_specs_by_name={
            "read_text": registry.get_spec("read_text"),
            "gt_task_extract": runtime._get_effective_spec("gt_task_extract"),
        },
        provenance_by_name={"gt_task_extract": {"code_hash": "abc"}},
    )
    planner = DynamicWorkflowPlanner(
        planner_llm=FakeLLMRuntime(default_content="{}"),
        config=DynamicSubagentsConfig(
            enabled=True,
            mode="dynamic",
            planner_backend={"backend_id": "planner"},
            default_worker_backend={"backend_id": "worker"},
            allowed_tool_names=["read_text"],
        ),
        tool_runtime=runtime,
        effective_tool_catalog=catalog,
    )

    messages = planner._messages(request=_request(), work_item=None, role_pool_templates=[])
    payload = json.loads(messages[1].content)

    assert payload["configured_builtin_allowed_tool_names"] == ["read_text"]
    assert payload["effective_allowed_tool_names"] == ["read_text", "gt_task_extract"]
    assert payload["effective_tool_catalog"]["gt_task_extract"]["metadata"]["generated_tool"] is True
```

Use existing helper names in the file. If `_request()` or imports are absent, add local imports for `TaskRequest`, `TaskOrigin`, `TaskPurpose`, `FakeLLMRuntime`, `ToolRuntime`, `ToolRegistry`, `ToolSpec`, `DynamicSubagentsConfig`, and `TaskEffectiveToolCatalog`.

- [ ] **Step 2: Add failing validation test for generated tools**

Add to `tests/test_dynamic_workflow_runtime.py` or `tests/test_dynamic_workflow_planner.py`:

```python
def test_dynamic_workflow_validation_accepts_generated_tool_from_effective_catalog():
    registry = ToolRegistry()
    runtime = ToolRuntime(registry)
    runtime.activate_generated_tool_scope("task-1")
    runtime.register_task_generated_tool(
        ToolSpec(name="gt_task_extract", description="Generated extract.", parameters_schema={"type": "object"}, metadata={"generated_tool": True}),
        lambda args: "ok",
        provenance={"code_hash": "abc"},
    )
    catalog = TaskEffectiveToolCatalog(
        task_id="task-1",
        builtin_allowed_tool_names=[],
        generated_tool_names=["gt_task_extract"],
        tool_specs_by_name={"gt_task_extract": runtime._get_effective_spec("gt_task_extract")},
        provenance_by_name={"gt_task_extract": {"code_hash": "abc"}},
    )
    spec = DynamicWorkflowSpec(
        workflow_id="wf-generated",
        task_summary="Use generated tool.",
        dynamic_subagents=[
            DynamicSubAgentSpec(
                subagent_id="worker",
                role_name="GeneratedWorker",
                goal="Use generated tool.",
                system_prompt="Use generated tool.",
                allowed_tools=["gt_task_extract"],
            )
        ],
        workflow_nodes=[DynamicWorkflowNodeSpec(node_id="node-worker", subagent_id="worker")],
        planner_rationale_summary="Generated tool is available.",
    )

    _prepared, report = validate_dynamic_workflow_spec(
        spec,
        config=DynamicSubagentsConfig(
            enabled=True,
            mode="dynamic",
            planner_backend={"backend_id": "planner"},
            default_worker_backend={"backend_id": "worker"},
            allowed_tool_names=[],
        ),
        available_llm_backend_ids={"worker"},
        tool_runtime=runtime,
        skill_backend=None,
        task_id="task-1",
        effective_tool_catalog=catalog,
    )

    assert report.valid is True
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
pytest -q tests/test_dynamic_workflow_planner.py::test_dynamic_planner_prompt_includes_effective_generated_tool_catalog tests/test_dynamic_workflow_runtime.py::test_dynamic_workflow_validation_accepts_generated_tool_from_effective_catalog
```

Expected: constructor/signature errors for `effective_tool_catalog`.

- [ ] **Step 4: Add effective catalog to planner and validation**

Modify `DynamicWorkflowPlanner.__init__` in `evolab/runtime/dynamic_workflow.py`:

```python
        effective_tool_catalog: TaskEffectiveToolCatalog | None = None,
```

Set:

```python
        self.effective_tool_catalog = effective_tool_catalog
```

Import:

```python
from evolab.contracts.generated_tools import TaskEffectiveToolCatalog
```

Modify `_messages()` payload:

```python
        effective_allowed_tool_names = (
            self.effective_tool_catalog.effective_allowed_tool_names
            if self.effective_tool_catalog is not None
            else list(self.config.allowed_tool_names)
        )
        effective_tool_catalog = (
            {
                name: spec.model_dump(mode="json")
                for name, spec in self.effective_tool_catalog.tool_specs_by_name.items()
            }
            if self.effective_tool_catalog is not None
            else {}
        )
```

Use those values in the payload:

```python
            "configured_builtin_allowed_tool_names": self.config.allowed_tool_names,
            "effective_allowed_tool_names": effective_allowed_tool_names,
            "effective_tool_catalog": effective_tool_catalog,
            "allowed_tool_names": effective_allowed_tool_names,
```

Pass `effective_allowed_tool_names` into `_dynamic_workflow_response_template()`.

- [ ] **Step 5: Update `validate_dynamic_workflow_spec()` and `_validate_tool_names()`**

Add parameter:

```python
    effective_tool_catalog: TaskEffectiveToolCatalog | None = None,
```

Thread it into every `_validate_tool_names()` call.

Change `_validate_tool_names()` signature:

```python
    effective_tool_catalog: TaskEffectiveToolCatalog | None = None,
```

Change allowed lookup:

```python
    allowed = set(
        effective_tool_catalog.effective_allowed_tool_names
        if effective_tool_catalog is not None
        else config.allowed_tool_names
    )
```

Change spec existence check:

```python
        elif tool_runtime._get_effective_spec(tool_name) is None:
            local_errors.append(f"{context} requests unknown tool {tool_name!r}")
```

Keep the error wording:

```python
local_errors.append(f"{context} requests tool {tool_name!r} outside effective allowed tool names")
```

- [ ] **Step 6: Update `DynamicSubAgentFactory`**

Add `effective_tool_catalog` to `DynamicSubAgentFactory.__init__`, store it, and pass it into `_validate_tool_names()`.

- [ ] **Step 7: Run dynamic tests**

Run:

```bash
pytest -q tests/test_dynamic_workflow_planner.py tests/test_dynamic_workflow_runtime.py::test_dynamic_workflow_validation_accepts_generated_tool_from_effective_catalog
```

Expected: selected tests pass.

- [ ] **Step 8: Commit**

```bash
git add evolab/runtime/dynamic_workflow.py tests/test_dynamic_workflow_planner.py tests/test_dynamic_workflow_runtime.py
git commit -m "feat: expose generated tools to dynamic planner"
```

---

### Task 5: MetaAgent Tool-Code Preplanning In TaskRuntime

**Files:**
- Modify: `evolab/runtime/task_runtime.py`
- Modify: `evolab/runtime/generated_tools.py`
- Modify: `tests/test_dynamic_workflow_runtime.py`
- Test: `tests/test_dynamic_workflow_runtime.py`

- [ ] **Step 1: Add failing runtime integration test**

Add to `tests/test_dynamic_workflow_runtime.py`:

```python
def test_dynamic_runtime_runs_tool_code_preplanning_before_planner(tmp_path: Path):
    generated_code = (
        "TOOL_SPEC = {"
        "\"name\": \"extract_rows\", "
        "\"description\": \"Generated extract.\", "
        "\"parameters_schema\": {\"type\": \"object\"}, "
        "\"metadata\": {\"generated_tool\": True}"
        "}\n"
        "def run(arguments, context):\n"
        "    return {\"status\": \"ok\", \"content\": \"generated ok\", \"metadata\": {\"task_id\": context.get(\"task_id\")}}\n"
    )
    meta_response = LLMRuntimeResponse(
        action=SubAgentAction(
            action="final_answer",
            content=json.dumps(
                {
                    "action": "finish_task",
                    "metadata": {
                        "generated_tool_package": {
                            "tool_name": "extract_rows",
                            "reason": "Need generated extractor.",
                            "manifest": {
                                "description": "Generated extract.",
                                "parameters_schema": {"type": "object"},
                            },
                            "files": [{"path": "tool.py", "content": generated_code}],
                            "smoke_tests": [{"name": "basic", "arguments": {}}],
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
                    "workflow_id": "wf-generated",
                    "task_summary": "Use generated tool.",
                    "article_context_summary": "test",
                    "dynamic_subagents": [
                        {
                            "subagent_id": "worker",
                            "role_name": "GeneratedWorker",
                            "goal": "Use generated tool.",
                            "system_prompt": "Use generated tool.",
                            "input_schema": {"type": "object"},
                            "output_schema": {"type": "object"},
                            "allowed_tools": ["gt_task_generated_run_extract_rows"],
                            "llm_backend_id": "worker",
                        }
                    ],
                    "workflow_nodes": [{"node_id": "node-worker", "subagent_id": "worker"}],
                    "workflow_edges": [],
                    "artifact_contracts": {},
                    "validation_rules": [],
                    "planner_rationale_summary": "Generated tool is available.",
                    "metadata": {},
                }
            ),
        )
    )
    worker_response = LLMRuntimeResponse(action=SubAgentAction(action="final_answer", content="done"))
    registry = ToolRegistry()
    tool_runtime = ToolRuntime(registry)
    runtime = TaskRuntime(
        task_config=TaskConfig(
            task_id="task-generated",
            goal="Use generated extractor.",
            meta_agent=MetaAgentSpec(
                llm_backend=BackendBinding(backend_id="meta"),
                system_prompt="Manage tools.",
            ),
            agents_ref=_write_agents_ref(tmp_path, {"GeneratedWorker": _role("GeneratedWorker", tools=[])}),
            dynamic_subagents=DynamicSubagentsConfig(
                enabled=True,
                mode="dynamic",
                planner_backend={"backend_id": "planner"},
                default_worker_backend={"backend_id": "worker"},
                allowed_tool_names=[],
                max_planner_retries=0,
                require_output_schema=False,
            ),
            runtime_policy=RuntimePolicy(),
        ),
        llm_runtimes={
            "meta": FakeLLMRuntime(responses=[meta_response]),
            "planner": FakeLLMRuntime(responses=[planner_response]),
            "worker": FakeLLMRuntime(responses=[worker_response]),
        },
        tool_runtime=tool_runtime,
        skill_runtimes={"skill": FakeSkillBackend()},
        lab_root=tmp_path,
    )

    result = runtime.run(_request("task-generated"))

    assert result["status"] == "completed"
    assert tool_runtime.generated_tool_names()
```

Adjust helper names to existing test helpers. If the generated registered name uses UUID suffixes, assert planner prompt contains a generated tool and use a planner runtime that reads the prompt and emits the actual generated name.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest -q tests/test_dynamic_workflow_runtime.py::test_dynamic_runtime_runs_tool_code_preplanning_before_planner
```

Expected: no preplanning hook exists or generated tool is not visible to planner.

- [ ] **Step 3: Add generated tool preplanning helpers**

In `evolab/runtime/task_runtime.py`, import:

```python
from evolab.contracts.generated_tools import GeneratedToolPackage, TaskEffectiveToolCatalog
from evolab.runtime.generated_tools import GeneratedToolRuntime, build_effective_tool_catalog
```

Add helpers near `_maybe_evolve_role_pool()`:

```python
def _generated_tool_package_payload(metadata: dict[str, Any] | None) -> Any | None:
    if not isinstance(metadata, dict):
        return None
    for key in ("generated_tool_package", "tool_code_update", "runtime_tool_package"):
        value = metadata.get(key)
        if isinstance(value, dict):
            return value
    return None


def _no_generated_tool_reason(metadata: dict[str, Any] | None) -> str | None:
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("no_generated_tool_reason")
    return value.strip() if isinstance(value, str) and value.strip() else None
```

Add method:

```python
    def _maybe_evolve_generated_tools(self, request: TaskRequest, *, dynamic_config: Any) -> TaskEffectiveToolCatalog | None:
        if self.tool_runtime is None:
            return None
        self.tool_runtime.reset_task_generated_tools(request.task_id)
        self.tool_runtime.activate_generated_tool_scope(request.task_id)
        generated_runtime = self._generated_tool_runtime()
        if self.task_config is None or self.task_config.meta_agent is None:
            return build_effective_tool_catalog(
                task_id=request.task_id,
                builtin_allowed_tool_names=list(getattr(dynamic_config, "allowed_tool_names", [])),
                tool_runtime=self.tool_runtime,
            )
        meta_agent = self.task_config.meta_agent
        run_ref = f"tool-code-{uuid4()}"
        meta_llm = self._llm_runtime(meta_agent.llm_backend.backend_id)
        decision, _llm_call_ref, _memory_request, _memory_bundle = self._next_dispatch_decision(
            request=request,
            meta_agent=meta_agent,
            meta_llm=meta_llm,
            run_ref=run_ref,
            step_index=-2,
            role_results=[],
            meta_memory=self._meta_memory_runtime(meta_agent),
            preplanning_context={
                "enabled": True,
                "stage": "tool_code_evolution",
                "purpose": "Before role-pool evolution and dynamic planning, generate task-local Python tools when the task needs specialized executable capability.",
                "allowed_actions": [
                    "Return END with metadata.generated_tool_package when a task-local Python tool should be created.",
                    "Return END with metadata.no_generated_tool_reason when built-in tools are sufficient.",
                    "Do not assign extraction, validation, writing, or other executable workflow work during this preplanning step.",
                ],
                "tool_code_package_contract": _generated_tool_package_contract(),
                "configured_builtin_allowed_tool_names": list(getattr(dynamic_config, "allowed_tool_names", [])),
            },
        )
        payload = _generated_tool_package_payload(decision.metadata)
        if payload is not None and self.task_config.runtime_policy.allow_runtime_tool_creation:
            package = GeneratedToolPackage.model_validate(payload)
            registration = generated_runtime.register_package(
                package=package,
                task_id=request.task_id,
                run_ref=run_ref,
                context={"task_id": request.task_id, "task_goal": request.goal},
            )
            self.trajectory_collector.record_event(
                event_type="generated_tool_registered" if registration.validation.valid else "generated_tool_rejected",
                subject_type="generated_tool",
                subject_ref=registration.registered_tool_name,
                task_id=request.task_id,
                run_ref=run_ref,
                metadata=registration.model_dump(mode="json"),
            )
        else:
            self.trajectory_collector.record_event(
                event_type="generated_tool_no_op",
                subject_type="generated_tool",
                subject_ref=request.task_id,
                task_id=request.task_id,
                run_ref=run_ref,
                metadata={"no_generated_tool_reason": _no_generated_tool_reason(decision.metadata) or "no package returned"},
            )
        return build_effective_tool_catalog(
            task_id=request.task_id,
            builtin_allowed_tool_names=list(getattr(dynamic_config, "allowed_tool_names", [])),
            tool_runtime=self.tool_runtime,
        )
```

Add `_generated_tool_runtime()`:

```python
    def _generated_tool_runtime(self) -> GeneratedToolRuntime:
        artifact_root = self.lab_root or Path.cwd()
        if self.tool_runtime is None or self.task_config is None:
            raise RuntimeError("generated tool runtime requires task_config and ToolRuntime")
        return GeneratedToolRuntime(
            artifact_root=artifact_root,
            tool_runtime=self.tool_runtime,
            policy=self.task_config.runtime_policy,
            trajectory_collector=self.trajectory_collector,
        )
```

- [ ] **Step 4: Insert generated tool evolution before role-pool evolution**

In `_maybe_run_dynamic_subagents()` before `_maybe_evolve_role_pool()`:

```python
        effective_tool_catalog = self._maybe_evolve_generated_tools(request, dynamic_config=dynamic_config)
        role_pool_evolved = self._maybe_evolve_role_pool(request, dynamic_config=dynamic_config)
```

Pass `effective_tool_catalog` into `DynamicWorkflowPlanner` and `_execute_dynamic_workflow_spec()`.

- [ ] **Step 5: Add package contract helper**

Add helper near `_role_pool_update_contract()`:

```python
def _generated_tool_package_contract() -> dict[str, Any]:
    return {
        "optional_location": "metadata.generated_tool_package",
        "no_op_location": "metadata.no_generated_tool_reason",
        "schema": {
            "tool_name": "semantic requested tool name",
            "reason": "short evidence-based reason",
            "manifest": {"description": "tool description", "parameters_schema": {"type": "object"}},
            "files": [{"path": "tool.py", "content": "complete Python source with TOOL_SPEC and run(arguments, context)"}],
            "primary_module": "optional primary module path; defaults to tool.py",
            "smoke_tests": [{"name": "basic", "arguments": {}}],
        },
        "requirements": [
            "Return complete Python source files.",
            "Expose TOOL_SPEC and run(arguments, context).",
            "Do not include hidden reasoning.",
            "Use task-local generated tools only.",
        ],
    }
```

- [ ] **Step 6: Run runtime integration tests**

Run:

```bash
pytest -q tests/test_dynamic_workflow_runtime.py::test_dynamic_runtime_runs_tool_code_preplanning_before_planner tests/test_generated_tools.py tests/test_tool_runtime.py
```

Expected: selected tests pass.

- [ ] **Step 7: Commit**

```bash
git add evolab/runtime/task_runtime.py evolab/runtime/generated_tools.py tests/test_dynamic_workflow_runtime.py
git commit -m "feat: run generated tool preplanning"
```

---

### Task 6: Capability Repair Generated Tools

**Files:**
- Modify: `evolab/runtime/capability_repair.py`
- Modify: `evolab/runtime/task_runtime.py`
- Modify: `tests/test_capability_repair.py`
- Test: `tests/test_capability_repair.py`

- [ ] **Step 1: Add failing repair test**

Add to `tests/test_capability_repair.py`:

```python
def test_capability_repair_generates_runtime_tool_for_missing_capability(tmp_path: Path):
    registry = ToolRegistry()
    runtime = ToolRuntime(registry)
    runtime.activate_generated_tool_scope("task-1")
    generated_code = (
        "TOOL_SPEC = {\"name\": \"missing_tool\", \"description\": \"Generated missing tool.\", \"parameters_schema\": {\"type\": \"object\"}, \"metadata\": {\"generated_tool\": True}}\n"
        "def run(arguments, context):\n"
        "    return {\"status\": \"ok\", \"content\": \"generated repair ok\", \"metadata\": {\"task_id\": context.get(\"task_id\")}}\n"
    )
    repair_runtime = CapabilityRepairRuntime()
    policy = RuntimePolicy(enable_runtime_capability_repair=True)
    tool_call = ToolCall(call_id="call-1", name="missing_tool", arguments={})
    tool_result = ToolResult(
        call_id="call-1",
        status="error",
        content="tool 'missing_tool' was not prepared for this run",
        metadata={"error_type": "unprepared_tool"},
    )

    outcome = repair_runtime.maybe_repair(
        task_id="task-1",
        run_ref="run-1",
        step_id="step-1",
        role="solver",
        task_goal="Need missing tool.",
        tool_call=tool_call,
        tool_result=tool_result,
        active_skill_bundle=_skill_bundle(),
        tool_runtime=runtime,
        trajectory_collector=_trajectory(tmp_path),
        runtime_policy=policy,
        repair_log=[],
        generated_tool_runtime=GeneratedToolRuntime(tmp_path, tool_runtime=runtime, policy=policy),
        generated_tool_package=GeneratedToolPackage(
            tool_name="missing_tool",
            reason="Repair missing tool.",
            manifest={"description": "Generated missing tool.", "parameters_schema": {"type": "object"}},
            files=[GeneratedToolFile(path="tool.py", content=generated_code)],
            smoke_tests=[GeneratedToolSmokeTest(name="basic", arguments={})],
        ),
    )

    assert outcome is not None
    assert outcome.retry_record is not None
    assert outcome.retry_record.result.status == "ok"
    assert outcome.promotion_candidates[0]["candidate_type"] == "new_tool"
```

Add helper `_trajectory(tmp_path)` if absent by constructing `FileTrajectoryRegistry(tmp_path / "trajectory")` and passing it through existing `TrajectoryCollector` helper used in this test file.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest -q tests/test_capability_repair.py::test_capability_repair_generates_runtime_tool_for_missing_capability
```

Expected: `maybe_repair()` does not accept generated tool runtime/package.

- [ ] **Step 3: Extend `CapabilityRepairRuntime.maybe_repair()` signature**

Add optional parameters:

```python
        generated_tool_runtime: Any | None = None,
        generated_tool_package: Any | None = None,
```

Keep defaults so existing tests still pass.

- [ ] **Step 4: Create generated tool for missing capability**

After planning and before `validate_and_retry()`, add:

```python
        generated_registration = None
        if (
            runtime_policy.allow_runtime_tool_creation
            and plan.new_runtime_tool is not None
            and generated_tool_runtime is not None
        ):
            package = generated_tool_package or plan.new_runtime_tool
            generated_registration = generated_tool_runtime.register_package(
                package=package,
                task_id=task_id,
                run_ref=run_ref,
                context={"task_id": task_id, "task_goal": task_goal, "role": role},
            )
            repair_entry["generated_tool_registration"] = generated_registration.model_dump(mode="json")
            trajectory_collector.record_event(
                event_type="generated_tool_registered" if generated_registration.validation.valid else "generated_tool_rejected",
                subject_type="generated_tool",
                subject_ref=generated_registration.registered_tool_name,
                task_id=task_id,
                run_ref=run_ref,
                parent_ref=plan.repair_id,
                metadata=generated_registration.model_dump(mode="json"),
            )
            if generated_registration.validation.valid:
                plan = plan.model_copy(
                    update={
                        "retry_plan": {
                            **plan.retry_plan,
                            "tool_name": generated_registration.registered_tool_name,
                        }
                    }
                )
```

In `CapabilityRepairPlanner.plan()`, update the missing capability branch to set a typed package shell only when enough package content exists. For now, leave `new_runtime_tool=None` in planner and let `TaskRuntime`/test pass `generated_tool_package`; the full LLM builder comes in Task 7.

- [ ] **Step 5: Emit `new_tool` promotion candidate**

Change candidate type logic:

```python
candidate_type = "new_tool" if generated_registration is not None else ("tool_patch" if plan.tool_overlay_patch is not None else "skill_patch")
target_id = (
    generated_registration.registered_tool_name
    if generated_registration is not None
    else (
        plan.tool_overlay_patch.base_tool_name
        if plan.tool_overlay_patch is not None
        else (plan.skill_overlay_patch.target_skill_id if plan.skill_overlay_patch is not None else None)
    )
)
affected_ids = (
    [generated_registration.registered_tool_name]
    if generated_registration is not None
    else [
        value
        for value in [
            plan.tool_overlay_patch.name if plan.tool_overlay_patch is not None else None,
            plan.skill_overlay_patch.target_skill_id if plan.skill_overlay_patch is not None else None,
        ]
        if isinstance(value, str)
    ]
)
```

- [ ] **Step 6: Thread generated runtime from `TaskRuntime._execute_tool_call()`**

In `_execute_tool_call()`, pass:

```python
            generated_tool_runtime=self._generated_tool_runtime() if self.tool_runtime is not None and self.task_config is not None else None,
```

Do not pass `generated_tool_package`; repair-time builder support is Task 7.

- [ ] **Step 7: Run repair tests**

Run:

```bash
pytest -q tests/test_capability_repair.py tests/test_generated_tools.py tests/test_tool_runtime.py
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit**

```bash
git add evolab/runtime/capability_repair.py evolab/runtime/task_runtime.py tests/test_capability_repair.py
git commit -m "feat: repair with generated runtime tools"
```

---

### Task 7: GeneratedToolBuilder LLM Integration

**Files:**
- Modify: `evolab/runtime/generated_tools.py`
- Modify: `evolab/runtime/capability_repair.py`
- Modify: `evolab/runtime/task_runtime.py`
- Modify: `tests/test_generated_tools.py`
- Modify: `tests/test_capability_repair.py`
- Test: `tests/test_generated_tools.py`, `tests/test_capability_repair.py`

- [ ] **Step 1: Add failing builder test**

Add to `tests/test_generated_tools.py`:

```python
def test_generated_tool_builder_parses_llm_package_response():
    code = "TOOL_SPEC = {'name': 'extract_rows', 'description': 'Extract.', 'parameters_schema': {'type': 'object'}}\ndef run(arguments, context):\n    return 'ok'\n"
    response = LLMRuntimeResponse(
        action=SubAgentAction(
            action="final_answer",
            content=json.dumps(
                {
                    "tool_name": "extract_rows",
                    "reason": "Need extractor.",
                    "manifest": {"description": "Extract.", "parameters_schema": {"type": "object"}},
                    "files": [{"path": "tool.py", "content": code}],
                }
            ),
        )
    )
    llm = FakeLLMRuntime(responses=[response])
    builder = GeneratedToolBuilder(llm_runtime=llm)

    package = builder.build(
        task_id="task-1",
        task_goal="Extract.",
        run_ref="run-1",
        built_in_tool_specs=[],
        generated_tool_specs=[],
        role_pool_templates=[],
        artifact_root="/tmp/artifacts",
        capability_grant=GeneratedToolCapabilityGrant(),
    )

    assert package.tool_name == "extract_rows"
    assert package.files[0].path == "tool.py"
    assert llm.requests[0].generation_config.metadata["runtime_stage"] == "generated_tool_builder"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest -q tests/test_generated_tools.py::test_generated_tool_builder_parses_llm_package_response
```

Expected: `GeneratedToolBuilder` is missing.

- [ ] **Step 3: Implement `GeneratedToolBuilder`**

In `evolab/runtime/generated_tools.py`, add:

```python
from evolab.contracts.common import Message
from evolab.contracts.llm import LLMGenerationConfig
```

Add class:

```python
class GeneratedToolBuilder:
    def __init__(self, *, llm_runtime: Any) -> None:
        self.llm_runtime = llm_runtime

    def build(
        self,
        *,
        task_id: str,
        task_goal: str,
        run_ref: str,
        built_in_tool_specs: list[dict[str, Any]],
        generated_tool_specs: list[dict[str, Any]],
        role_pool_templates: list[dict[str, Any]],
        artifact_root: str,
        capability_grant: GeneratedToolCapabilityGrant,
        failure_signal: dict[str, Any] | None = None,
        requested_tool_name: str | None = None,
    ) -> GeneratedToolPackage:
        payload = {
            "task_id": task_id,
            "task_goal": task_goal,
            "run_ref": run_ref,
            "failure_signal": failure_signal or {},
            "requested_tool_name": requested_tool_name,
            "built_in_tool_specs": built_in_tool_specs,
            "generated_tool_specs": generated_tool_specs,
            "role_pool_templates": role_pool_templates,
            "artifact_root": artifact_root,
            "capability_grant": capability_grant.model_dump(mode="json"),
            "required_response": {
                "format": "JSON only",
                "schema": "GeneratedToolPackage",
                "requirements": [
                    "Return complete Python source files.",
                    "The primary module must define TOOL_SPEC and run(arguments, context).",
                    "Do not include hidden reasoning.",
                ],
            },
        }
        response = self.llm_runtime.generate(
            [
                Message(role="system", content="You are EvoLab GeneratedToolBuilder. Return JSON only."),
                Message(role="user", content=json.dumps(payload, indent=2, sort_keys=True)),
            ],
            [],
            LLMGenerationConfig(model="", temperature=0, max_output_tokens=4096, metadata={"runtime_stage": "generated_tool_builder"}),
        )
        if response.action.action != "final_answer" or not response.action.content:
            raise RuntimeError("GeneratedToolBuilder expected a final_answer JSON response")
        return GeneratedToolPackage.model_validate(json.loads(response.action.content))
```

- [ ] **Step 4: Select builder backend in `TaskRuntime`**

Add `_generated_tool_builder()` to `TaskRuntime`:

```python
    def _generated_tool_builder(self, *, preferred_backend_id: str | None = None) -> GeneratedToolBuilder | None:
        if self.task_config is None:
            return None
        backend_id = (
            self.task_config.runtime_policy.metadata.get("generated_tool_builder_backend_id")
            if isinstance(self.task_config.runtime_policy.metadata, dict)
            else None
        )
        if not isinstance(backend_id, str) or not backend_id:
            backend_id = preferred_backend_id
        if not backend_id and self.task_config.dynamic_subagents is not None and self.task_config.dynamic_subagents.default_worker_backend is not None:
            backend_id = self.task_config.dynamic_subagents.default_worker_backend.backend_id
        if not backend_id and self.task_config.meta_agent is not None:
            backend_id = self.task_config.meta_agent.llm_backend.backend_id
        if not backend_id or backend_id not in self.llm_runtimes:
            return None
        return GeneratedToolBuilder(llm_runtime=self._llm_runtime(backend_id))
```

In `_maybe_evolve_generated_tools()`, if MetaAgent returns a request without files, use builder:

```python
if payload is not None:
    package = GeneratedToolPackage.model_validate(payload) if payload.get("files") else builder.build(...)
```

In `_execute_tool_call()`, pass `generated_tool_builder=self._generated_tool_builder(preferred_backend_id=role.llm_backend.backend_id if hasattr(role, "llm_backend") else None)` into repair runtime.

- [ ] **Step 5: Use builder inside `CapabilityRepairRuntime`**

Add `generated_tool_builder: Any | None = None` to `maybe_repair()`.

When `plan.new_runtime_tool is None` and `generated_tool_builder is not None` for `plan.repair_action == "create_runtime_tool"`, call builder:

```python
package = generated_tool_package
if package is None and generated_tool_builder is not None and plan.repair_action == "create_runtime_tool":
    package = generated_tool_builder.build(
        task_id=task_id,
        task_goal=task_goal,
        run_ref=run_ref,
        built_in_tool_specs=[],
        generated_tool_specs=[],
        role_pool_templates=[],
        artifact_root="",
        capability_grant=GeneratedToolCapabilityGrant(),
        failure_signal=signal.model_dump(mode="json"),
        requested_tool_name=tool_call.name,
    )
```

Then register `package`.

- [ ] **Step 6: Run builder and repair tests**

Run:

```bash
pytest -q tests/test_generated_tools.py tests/test_capability_repair.py
```

Expected: selected tests pass.

- [ ] **Step 7: Commit**

```bash
git add evolab/runtime/generated_tools.py evolab/runtime/task_runtime.py evolab/runtime/capability_repair.py tests/test_generated_tools.py tests/test_capability_repair.py
git commit -m "feat: build generated tools with llm"
```

---

### Task 8: Trace, Artifact, Observation, And Role-Pool Guardrails

**Files:**
- Modify: `evolab/runtime/task_runtime.py`
- Modify: `evolab/runtime/role_pool.py`
- Modify: `tests/test_role_pool_runtime.py`
- Modify: `tests/test_dynamic_workflow_runtime.py`
- Test: `tests/test_role_pool_runtime.py`, `tests/test_dynamic_workflow_runtime.py`

- [ ] **Step 1: Add role-pool guardrail test**

Add to `tests/test_role_pool_runtime.py`:

```python
def test_role_pool_update_rejects_task_local_generated_tool_reference(tmp_path: Path):
    agents_path = tmp_path / "agents.md"
    agents_path.write_text(render_agents_markdown({"SurveyAgent": _role("SurveyAgent")}), encoding="utf-8")

    result = apply_role_pool_update(
        agents_path=agents_path,
        payload={
            "reason": "Do not persist task-local generated tools.",
            "roles": {
                "SurveyAgent": {
                    "allowed_tools": ["gt_task_extract_rows"],
                }
            },
        },
        task_id="task-1",
        run_ref="run-1",
        known_llm_backend_ids={"llm-local"},
        allowed_tool_names=["read_text"],
    )

    assert result.status == "rejected"
    assert any("unknown allowed_tools" in error for error in result.errors)
```

This may already pass after Task 4; keep it as a regression.

- [ ] **Step 2: Add trajectory/observation test**

Extend the preplanning integration test to assert:

```python
event_types = [event.event_type for event in trajectory_registry.list_events()]
assert "generated_tool_registered" in event_types
saved_runs = task_registry.list_runs("task-generated")
assert any("generated_tool" in record.metadata for run in saved_runs for record in run.tool_trace.calls)
```

Use the actual registry helper methods available in the test file.

- [ ] **Step 3: Ensure generated metadata reaches tool results**

In `ToolRuntime` this was added in Task 2. In `TaskRuntime`, ensure `_perform_tool_call()` keeps `ToolResult.metadata` intact when managing artifacts and saving traces. Do not strip `generated_tool`.

- [ ] **Step 4: Include generated provenance in skill observations**

Where skill observations are built in `TaskRuntime`, include generated tool summary:

```python
"generated_tools": [
    {
        "name": name,
        **self.tool_runtime.generated_tool_provenance(name),
    }
    for name in self.tool_runtime.generated_tool_names()
] if self.tool_runtime is not None else []
```

Add this under observation metadata next to `repair_trajectory`.

- [ ] **Step 5: Run guardrail and integration tests**

Run:

```bash
pytest -q tests/test_role_pool_runtime.py tests/test_dynamic_workflow_runtime.py::test_dynamic_runtime_runs_tool_code_preplanning_before_planner
```

Expected: selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add evolab/runtime/task_runtime.py evolab/runtime/role_pool.py tests/test_role_pool_runtime.py tests/test_dynamic_workflow_runtime.py
git commit -m "feat: trace generated tool provenance"
```

---

### Task 9: Documentation And Final Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/dynamic_subagent_workflows.md`
- Modify: `docs/params_runtime.md`
- Test: full targeted and full suite

- [ ] **Step 1: Update docs**

Add to `README.md` near the dynamic role/tool runtime description:

```markdown
### Self-Evolving Task Tools

EvoLab has two tool layers. Built-in generic tools are registered by EvoLab
(`list_files`, `read_text`, table tools, schema tools, output tools, and domain
artifact tools). Task-specialized tools can be generated as Python modules
during a task. Generated tools are validated, stored as task artifacts,
registered task-locally, exposed to the dynamic planner through the effective
tool catalog, and cleared when the task changes. They do not mutate the global
tool registry or reusable `agents.md` role pool.
```

Add to `docs/dynamic_subagent_workflows.md`:

```markdown
Dynamic planners receive `effective_allowed_tool_names`, which is the configured
built-in allowlist plus validated task-local generated tools. Planner output
must still be structural JSON only; executable Python belongs to the generated
tool preplanning or repair stages, not workflow specs.
```

Add to `docs/params_runtime.md` under runtime policy:

```markdown
Generated tool policy fields include `max_generated_tools_per_task`,
`max_generated_tool_files`, `max_generated_tool_source_bytes`,
`generated_tool_validation_timeout_s`, `generated_tool_execution_timeout_s`,
`generated_tool_max_output_bytes`, `generated_tool_allow_network`,
`generated_tool_allow_subprocess`, `generated_tool_allowed_env_keys`, and
`generated_tool_allowed_imports`.
```

- [ ] **Step 2: Run targeted tests**

Run:

```bash
pytest -q \
  tests/test_generated_tools.py \
  tests/test_tool_runtime.py \
  tests/test_dynamic_workflow_planner.py \
  tests/test_dynamic_workflow_runtime.py \
  tests/test_capability_repair.py \
  tests/test_role_pool_runtime.py
```

Expected: all selected tests pass.

- [ ] **Step 3: Run full test suite**

Run:

```bash
pytest -q -rs
```

Expected: all tests pass, with the existing skip count only.

- [ ] **Step 4: Static cleanup**

Run:

```bash
git diff --check
rg -n "TBD|TODO|placeholder|handler-ref-only|tools.md registry" evolab tests docs README.md || true
```

Expected: `git diff --check` has no output. The `rg` command has no new placeholder matches introduced by this implementation.

- [ ] **Step 5: Commit docs and cleanup**

```bash
git add README.md docs/dynamic_subagent_workflows.md docs/params_runtime.md
git commit -m "docs: document generated task tools"
```

---

## Final Review Checklist

- [ ] `ToolRuntime.prepare()` does not erase task-local generated tools.
- [ ] New task reset clears task-local generated tools.
- [ ] Generated tools cannot replace built-ins by default.
- [ ] Generated Python source is persisted as a task artifact.
- [ ] Invalid generated code never becomes a prepared tool.
- [ ] DynamicWorkflowPlanner sees `effective_allowed_tool_names`.
- [ ] DynamicWorkflowPlanner still rejects executable code in workflow specs.
- [ ] Runtime repair can generate, validate, register, retry, and record a `new_tool` candidate.
- [ ] Reusable roles in `agents.md` cannot persist task-local generated tool names.
- [ ] Full `pytest -q -rs` passes.
