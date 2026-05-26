# SDK Session Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make EvoLab an installable SDK where users run sessions from Python scripts, `session.run()` returns `None`, all internal EvoLab state lives under `<lab_dir>/.evolab`, and user-visible outputs live outside `.evolab`.

**Architecture:** Add a public SDK layer that compiles Python `SessionConfig` into internal runtime objects, then refactor Lab layout and runtime path factories so queues, registries, roles, memory, skills, tools, trajectories, and generated tool state use `.evolab`. The old YAML task-config startup path is removed from the primary product path and any remaining CLI behavior becomes secondary to the SDK.

**Tech Stack:** Python 3.11, Pydantic v2, setuptools, pytest, existing EvoLab runtime/backends/tools.

---

## File Structure

- Create `evolab/sdk.py` for public `TaskSpec`, `SessionConfig`, and `EvoLabSession`.
- Create `evolab/session_runtime.py` for SDK-to-runtime compilation and synchronous task execution.
- Modify `evolab/__init__.py` to export the SDK API.
- Modify `evolab/lab/layout.py` so `LabLayout.root` is the user-visible Lab and all internal paths use `state_root = root / ".evolab"`.
- Modify `evolab/lab/resolver.py` only as needed to consume the new layout paths.
- Modify `evolab/runtime/task_worker.py` and `evolab/runtime/task_runtime.py` path factories so internal artifacts use `.evolab` and user-facing outputs use the Lab root/output directories.
- Modify `evolab/runtime/generated_tools.py` integration points so generated tool packages are stored under `.evolab`.
- Modify `evolab/cli.py` to stop being the primary config-file session path. Keep only non-session utilities or a small development compatibility hook if needed by tests during migration.
- Modify `pyproject.toml` for package metadata and eventual `src/` package layout.
- Create `examples/minimal_session.py` and `examples/scientific_ie_session.py`.
- Create `dev/` and move research artifacts/configs/reports there after runtime tests pass.
- Rewrite `README.md` around the SDK workflow.
- Add `tests/test_sdk_session_config.py`.
- Add `tests/test_sdk_lab_layout.py`.
- Add `tests/test_sdk_session_runtime.py`.
- Add `tests/test_package_install.py` or a packaging smoke test.

## Task 1: Public SDK Models

**Files:**
- Create: `evolab/sdk.py`
- Modify: `evolab/__init__.py`
- Test: `tests/test_sdk_session_config.py`

- [ ] **Step 1: Write failing tests for `TaskSpec` prompt rendering and `run()` return contract**

```python
from pathlib import Path

from evolab import EvoLabSession, SessionConfig, TaskSpec


def test_task_spec_renders_canonical_prompt():
    task = TaskSpec(
        goal="Extract facts.",
        resources="Use files in data/.",
        expected_outputs="Write report.md.",
        success_criteria="Every claim cites a source.",
        optional_context="Prefer concise output.",
    )

    assert task.to_prompt() == (
        "1. goal: Extract facts.\n"
        "2. resources: Use files in data/.\n"
        "3. expected_outputs: Write report.md.\n"
        "4. success_criteria: Every claim cites a source.\n"
        "5. optional_context: Prefer concise output."
    )


def test_session_run_returns_none(tmp_path: Path):
    session = EvoLabSession(
        SessionConfig(
            lab_dir=tmp_path / "lab",
            task=TaskSpec(
                goal="Write a report.",
                resources="No external files.",
                expected_outputs="report.md",
                success_criteria="report.md exists.",
            ),
            llm={"default": {"type": "fake", "responses": []}},
            memory={"task": {"type": "null"}},
            skills={"default": {"type": "fake", "skills": []}},
        )
    )

    assert session.run() is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sdk_session_config.py -q`

Expected: import errors for `TaskSpec`, `SessionConfig`, or `EvoLabSession`.

- [ ] **Step 3: Implement minimal SDK models**

Add `TaskSpec`, `SessionConfig`, and `EvoLabSession` in `evolab/sdk.py`. For this task, `EvoLabSession.run()` may initialize only enough to satisfy the return contract; real runtime execution is added later.

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from evolab.contracts.common import StrictBaseModel


class TaskSpec(StrictBaseModel):
    goal: str
    resources: str
    expected_outputs: str
    success_criteria: str
    optional_context: str | None = None

    def to_prompt(self) -> str:
        lines = [
            f"1. goal: {self.goal}",
            f"2. resources: {self.resources}",
            f"3. expected_outputs: {self.expected_outputs}",
            f"4. success_criteria: {self.success_criteria}",
        ]
        if self.optional_context is not None:
            lines.append(f"5. optional_context: {self.optional_context}")
        return "\n".join(lines)


class SessionConfig(StrictBaseModel):
    lab_dir: Path
    task: TaskSpec | str
    env_file: Path | None = None
    llm: dict[str, Any]
    memory: dict[str, Any] = Field(default_factory=dict)
    skills: dict[str, Any] = Field(default_factory=dict)
    tools: dict[str, Any] = Field(default_factory=dict)
    runtime: dict[str, Any] = Field(default_factory=dict)
    seed_roles: dict[str, Any] | None = None
    meta_agent: dict[str, Any] | None = None


class EvoLabSession:
    def __init__(self, config: SessionConfig) -> None:
        self.config = config

    @property
    def lab_dir(self) -> Path:
        return self.config.lab_dir

    @property
    def state_dir(self) -> Path:
        return self.config.lab_dir / ".evolab"

    def initialize(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        self.initialize()
        return None
```

Update `evolab/__init__.py`:

```python
__version__ = "0.1.0"

from evolab.sdk import EvoLabSession, SessionConfig, TaskSpec

__all__ = ["__version__", "EvoLabSession", "SessionConfig", "TaskSpec"]
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_sdk_session_config.py tests/test_imports.py -q`

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add evolab/sdk.py evolab/__init__.py tests/test_sdk_session_config.py
git commit -m "feat: add sdk session models"
```

## Task 2: `.evolab` Lab Layout Boundary

**Files:**
- Modify: `evolab/lab/layout.py`
- Modify: `evolab/lab/resolver.py`
- Test: `tests/test_sdk_lab_layout.py`

- [ ] **Step 1: Write failing tests for internal path isolation**

```python
from pathlib import Path

from evolab.lab.layout import LabLayout
from evolab.lab.resolver import LabResolver


def test_lab_layout_places_internal_state_under_dot_evolab(tmp_path: Path):
    layout = LabLayout(tmp_path / "lab")

    assert layout.root == tmp_path / "lab"
    assert layout.state_root == tmp_path / "lab" / ".evolab"
    assert layout.tasks_queue_dir == layout.state_root / "queues" / "tasks"
    assert layout.evolve_queue_dir == layout.state_root / "queues" / "evolve"
    assert layout.registries_dir == layout.state_root / "registries"
    assert layout.trajectory_dir == layout.state_root / "trajectories"
    assert layout.snapshots_dir == layout.state_root / "snapshots"
    assert layout.agents_path == layout.state_root / "AGENTS.md"
    assert layout.generated_tools_dir == layout.state_root / "generated_tools"


def test_lab_resolver_ensure_does_not_create_internal_state_at_lab_root(tmp_path: Path):
    lab = tmp_path / "lab"
    (lab / "input.txt").parent.mkdir(parents=True)
    (lab / "input.txt").write_text("user data", encoding="utf-8")

    LabResolver(LabLayout(lab)).ensure()

    assert (lab / "input.txt").read_text(encoding="utf-8") == "user data"
    assert (lab / ".evolab" / "queues" / "tasks").is_dir()
    assert not (lab / "queues").exists()
    assert not (lab / "registries").exists()
    assert not (lab / "trajectories").exists()
    assert not (lab / "snapshots").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sdk_lab_layout.py -q`

Expected: assertions fail because current layout creates root-level internal directories.

- [ ] **Step 3: Refactor `LabLayout`**

Change `LabLayout` so existing internal properties point under `state_root` and add explicit user-facing output/artifact paths.

```python
class LabLayout:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    @property
    def state_root(self) -> Path:
        return self.root / ".evolab"

    @property
    def agents_path(self) -> Path:
        return self.state_root / "AGENTS.md"

    @property
    def output_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def user_artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def tasks_queue_dir(self) -> Path:
        return self.state_root / "queues" / "tasks"

    @property
    def evolve_queue_dir(self) -> Path:
        return self.state_root / "queues" / "evolve"

    @property
    def trajectory_dir(self) -> Path:
        return self.state_root / "trajectories"

    @property
    def registries_dir(self) -> Path:
        return self.state_root / "registries"

    @property
    def snapshots_dir(self) -> Path:
        return self.state_root / "snapshots"

    @property
    def generated_tools_dir(self) -> Path:
        return self.state_root / "generated_tools"
```

Update `ensure()` to create `state_root`, all internal subdirectories, plus user-visible `outputs/` and `artifacts/` only if needed. Do not create root-level internal directories.

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_sdk_lab_layout.py -q`

Expected: pass.

- [ ] **Step 5: Run affected layout tests**

Run: `pytest tests/test_lab_queue.py tests/test_lab_state_registry.py tests/test_lab_state_builder.py tests/test_task_worker.py -q`

Expected: either pass or fail only where tests assert old root-level paths; update those tests to the new `.evolab` contract.

- [ ] **Step 6: Commit**

```bash
git add evolab/lab/layout.py evolab/lab/resolver.py tests/test_sdk_lab_layout.py tests/test_lab_queue.py tests/test_lab_state_registry.py tests/test_lab_state_builder.py tests/test_task_worker.py
git commit -m "refactor: place lab internals under dot evolab"
```

## Task 3: SDK Session Initialization

**Files:**
- Modify: `evolab/sdk.py`
- Create: `evolab/session_runtime.py`
- Test: `tests/test_sdk_session_runtime.py`

- [ ] **Step 1: Write failing initialization tests**

```python
from pathlib import Path

from evolab import EvoLabSession, SessionConfig, TaskSpec


def _config(lab: Path) -> SessionConfig:
    return SessionConfig(
        lab_dir=lab,
        task=TaskSpec(
            goal="Write a report.",
            resources="Use input.txt.",
            expected_outputs="report.md",
            success_criteria="report.md exists.",
        ),
        llm={"default": {"type": "fake", "responses": []}},
        memory={"task": {"type": "null"}},
        skills={"default": {"type": "fake", "skills": []}},
    )


def test_initialize_creates_agents_tools_skills_memory_dirs(tmp_path: Path):
    lab = tmp_path / "lab"

    EvoLabSession(_config(lab)).initialize()

    state = lab / ".evolab"
    assert (state / "AGENTS.md").is_file()
    assert (state / "tools").is_dir()
    assert (state / "skills").is_dir()
    assert (state / "memory").is_dir()
    assert (state / "configs").is_dir()
    assert not (lab / "AGENTS.md").exists()
    assert not (lab / "memory").exists()


def test_initialize_reuses_existing_lab_without_deleting_user_files(tmp_path: Path):
    lab = tmp_path / "lab"
    lab.mkdir()
    (lab / "input.txt").write_text("keep me", encoding="utf-8")

    EvoLabSession(_config(lab)).initialize()
    EvoLabSession(_config(lab)).initialize()

    assert (lab / "input.txt").read_text(encoding="utf-8") == "keep me"
```

- [ ] **Step 2: Run tests to verify fail**

Run: `pytest tests/test_sdk_session_runtime.py -q`

Expected: missing `AGENTS.md`, tools, skills, or memory initialization.

- [ ] **Step 3: Implement initialization service**

Create `evolab/session_runtime.py` with:

- `initialize_lab(config: SessionConfig) -> LabLayout`
- seed role materialization using `render_agents_markdown(default_seed_roles(...))`
- creation of `.evolab/tools`, `.evolab/skills`, `.evolab/memory`, `.evolab/configs`
- no deletion of user files

Wire `EvoLabSession.initialize()` to this service.

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_sdk_session_runtime.py tests/test_sdk_lab_layout.py -q`

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add evolab/sdk.py evolab/session_runtime.py tests/test_sdk_session_runtime.py
git commit -m "feat: initialize sdk lab state"
```

## Task 4: SDK Runtime Execution Without Task Config Files

**Files:**
- Modify: `evolab/session_runtime.py`
- Modify: `evolab/sdk.py`
- Modify: `evolab/runtime/task_worker.py`
- Modify: `evolab/runtime/task_runtime.py`
- Test: `tests/test_sdk_session_runtime.py`

- [ ] **Step 1: Write failing offline run test**

```python
import json
from pathlib import Path

from evolab import EvoLabSession, SessionConfig, TaskSpec


def test_sdk_run_writes_report_and_returns_none(tmp_path: Path):
    lab = tmp_path / "lab"
    response = {
        "workflow_id": "wf-report",
        "task_summary": "Write report.",
        "dynamic_subagents": [
            {
                "subagent_id": "writer",
                "role_name": "GeneralistAgent",
                "goal": "Write report.md",
                "system_prompt": "Write the final report.",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
                "allowed_tools": ["write_report"],
            }
        ],
        "workflow_nodes": [{"node_id": "node-writer", "subagent_id": "writer"}],
        "workflow_edges": [],
        "artifact_contracts": {},
        "validation_rules": [],
        "planner_rationale_summary": "One writer is enough.",
    }
    session = EvoLabSession(
        SessionConfig(
            lab_dir=lab,
            task=TaskSpec(
                goal="Write a report.",
                resources="No external files.",
                expected_outputs="report.md",
                success_criteria="report.md exists.",
            ),
            llm={
                "default": {
                    "type": "fake",
                    "responses": [
                        {"action": {"action": "final_answer", "content": "{\"route\":\"END\",\"instruction\":\"No generated tool needed.\",\"metadata\":{\"no_generated_tool_reason\":\"Built-ins are enough.\"}}"}},
                        {"action": {"action": "final_answer", "content": "{\"route\":\"END\",\"instruction\":\"Keep role pool.\",\"metadata\":{\"no_role_pool_update_reason\":\"Seed is enough.\"}}"}},
                        {"action": {"action": "final_answer", "content": json.dumps(response)}},
                        {"action": {"action": "tool_call", "tool_calls": [{"call_id": "write-1", "name": "write_report", "arguments": {"path": "report.md", "content": "Done."}}]}},
                        {"action": {"action": "final_answer", "content": "Report written."}},
                    ],
                }
            },
            memory={"task": {"type": "null"}},
            skills={"default": {"type": "fake", "skills": []}},
            tools={"builtin": True},
        )
    )

    assert session.run() is None
    assert (lab / "report.md").read_text(encoding="utf-8") == "Done."
    assert not (lab / "queues").exists()
    assert (lab / ".evolab" / "queues" / "tasks").is_dir()
```

- [ ] **Step 2: Run test to verify fail**

Run: `pytest tests/test_sdk_session_runtime.py::test_sdk_run_writes_report_and_returns_none -q`

Expected: execution wiring is missing.

- [ ] **Step 3: Implement SDK run compilation**

In `session_runtime.py`, build:

- `TaskRequest` from `SessionConfig.task`
- internal `TaskConfig` from SDK config
- LLM, memory, skill, evolution backend maps from Python dictionaries
- `ToolRegistry` with built-in tools
- `TaskWorker` or direct `TaskRuntime` execution with layout rooted at `.evolab`

The task request may be saved to `.evolab/registries/task`, but no user-authored
task config file is written.

- [ ] **Step 4: Update output tool path policy**

Ensure user-facing writing tools resolve relative paths against `lab_dir`, not
`.evolab`, while generated tool/runtime internals keep their artifacts under
`.evolab`.

- [ ] **Step 5: Run SDK runtime tests**

Run: `pytest tests/test_sdk_session_runtime.py tests/test_tool_runtime.py tests/test_dynamic_workflow_runtime.py -q`

Expected: pass after migrating old root-path assertions.

- [ ] **Step 6: Commit**

```bash
git add evolab/sdk.py evolab/session_runtime.py evolab/runtime/task_worker.py evolab/runtime/task_runtime.py tests/test_sdk_session_runtime.py
git commit -m "feat: run sdk sessions without task config files"
```

## Task 5: Role And Tool Evolution Path Migration

**Files:**
- Modify: `evolab/runtime/task_runtime.py`
- Modify: `evolab/runtime/generated_tools.py`
- Modify: `evolab/tools/runtime.py`
- Test: `tests/test_role_pool_runtime.py`
- Test: `tests/test_generated_tools.py`
- Test: `tests/test_dynamic_workflow_runtime.py`

- [ ] **Step 1: Write failing path tests**

Add tests asserting:

- role pool updates modify `.evolab/AGENTS.md`
- `AGENTS.md.updates.jsonl` lives under `.evolab`
- generated tool source and smoke logs live under `.evolab/generated_tools`
- generated tools do not create internal files outside `.evolab`

- [ ] **Step 2: Run targeted tests to verify fail**

Run:

```bash
pytest tests/test_role_pool_runtime.py tests/test_generated_tools.py tests/test_dynamic_workflow_runtime.py -q
```

Expected: old path assumptions fail.

- [ ] **Step 3: Move role pool path handling to layout**

Replace direct `configs/agents.md` assumptions with `LabLayout.agents_path` or
SDK-provided `task_config.agents_ref = str(layout.agents_path)`.

- [ ] **Step 4: Move generated tool artifact root**

Use `layout.generated_tools_dir / task_id / run_ref` for generated tool packages.

- [ ] **Step 5: Run targeted tests**

Run:

```bash
pytest tests/test_role_pool_runtime.py tests/test_generated_tools.py tests/test_dynamic_workflow_runtime.py tests/test_capability_repair.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add evolab/runtime/task_runtime.py evolab/runtime/generated_tools.py evolab/tools/runtime.py tests/test_role_pool_runtime.py tests/test_generated_tools.py tests/test_dynamic_workflow_runtime.py tests/test_capability_repair.py
git commit -m "refactor: store evolved roles and tools under dot evolab"
```

## Task 6: Remove Primary YAML Startup Path

**Files:**
- Modify: `evolab/cli.py`
- Modify/delete: YAML config tests that only validate old startup behavior
- Test: `tests/test_cli_clean_run.py`
- Test: `tests/test_imports.py`

- [ ] **Step 1: Decide retained CLI scope**

Keep CLI commands that operate on existing Lab state, such as exporters or
trajectory visualization. Remove `clean-run` as the primary task startup, or
mark it as a dev compatibility command backed by SDK config only if tests need a
transition.

- [ ] **Step 2: Write tests for no public YAML session startup**

Tests should assert the README/examples use SDK scripts and that public imports
do not require config file loaders.

- [ ] **Step 3: Remove or demote old config compiler**

Move old YAML config compiler helpers into `dev/` if still useful for
experiments, or delete them if no tests depend on them after migration.

- [ ] **Step 4: Run CLI/import tests**

Run: `pytest tests/test_imports.py tests/test_cli_clean_run.py -q`

Expected: updated tests pass under new CLI scope.

- [ ] **Step 5: Commit**

```bash
git add evolab/cli.py tests/test_cli_clean_run.py tests/test_imports.py
git commit -m "refactor: make sdk the primary session entrypoint"
```

## Task 7: Package Layout And Examples

**Files:**
- Modify: `pyproject.toml`
- Move: `evolab/` to `src/evolab/`
- Create: `examples/minimal_session.py`
- Create: `examples/scientific_ie_session.py`
- Test: `tests/test_package_install.py`

- [ ] **Step 1: Write packaging smoke test**

Create a test that builds a wheel, installs it into a temporary venv, imports
`evolab`, and runs `examples/minimal_session.py` in offline mode.

- [ ] **Step 2: Run packaging test to verify fail**

Run: `pytest tests/test_package_install.py -q`

Expected: fail before package metadata/layout updates.

- [ ] **Step 3: Move package to `src/evolab`**

Use `git mv evolab src/evolab`. Update `pyproject.toml`:

```toml
[tool.setuptools.packages.find]
where = ["src"]
include = ["evolab*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

Add project metadata: `readme`, authors, classifiers, URLs, and package data
for curated runtime resources.

- [ ] **Step 4: Add examples**

`examples/minimal_session.py` should run a fake/offline session and write a
report. `examples/scientific_ie_session.py` should show real API configuration
without embedding secrets.

- [ ] **Step 5: Run packaging tests**

Run:

```bash
pytest tests/test_package_install.py tests/test_imports.py -q
python -m build
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src examples tests/test_package_install.py
git commit -m "build: package evolab as sdk"
```

## Task 8: Repository Cleanup And README

**Files:**
- Modify: `README.md`
- Move: `configs/` to `dev/configs/` unless curated examples remain
- Move: `reports/` to `dev/reports/`
- Move: `artifacts/` to `dev/artifacts/`
- Move: `development_plan.md`, `V0_SKILL_TOOL_TEST_PLAN.md`, and old project docs to `dev/docs/` when not public
- Keep: `docs/superpowers/specs/2026-05-24-sdk-session-refactor-design.md`
- Test: docs/package smoke commands

- [ ] **Step 1: Move development artifacts**

Use `git mv` for tracked files and directories. Do not move tests or public SDK
examples.

- [ ] **Step 2: Rewrite README**

README must lead with:

- `pip install evolab`
- minimal SDK script
- `.evolab` internal-state contract
- where task outputs appear
- backend config examples
- development notes pointing to `dev/`

- [ ] **Step 3: Check docs for obsolete primary YAML instructions**

Run:

```bash
rg -n "clean-run|task config|configs/demo|agents_ref|subagents:" README.md docs examples src/evolab
```

Expected: no obsolete primary workflow references. Internal compatibility or
historical dev references must be clearly marked as dev/historical.

- [ ] **Step 4: Commit**

```bash
git add README.md dev docs examples
git commit -m "docs: document sdk workflow"
```

## Task 9: Full Verification And Cocodex Sync

**Files:**
- All changed files

- [ ] **Step 1: Run focused suites**

```bash
pytest tests/test_sdk_session_config.py tests/test_sdk_lab_layout.py tests/test_sdk_session_runtime.py -q
pytest tests/test_role_pool_runtime.py tests/test_generated_tools.py tests/test_dynamic_workflow_runtime.py tests/test_capability_repair.py -q
pytest tests/test_imports.py tests/test_package_install.py -q
```

- [ ] **Step 2: Run full suite**

```bash
pytest -q -rs
```

- [ ] **Step 3: Run packaging smoke**

```bash
python -m build
```

- [ ] **Step 4: Run diff checks**

```bash
git diff --check
git status --short
```

- [ ] **Step 5: Commit final fixes if needed**

Use focused commits for any test or packaging fixes.

- [ ] **Step 6: Sync**

After the worktree is clean:

```bash
cocodex sync
```

Expected: session branch publishes or returns a semantic merge task. If a task
file is produced, read it and perform the semantic merge before syncing again.
