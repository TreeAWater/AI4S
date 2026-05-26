# EvoLab SDK Session Refactor Design

## Purpose

EvoLab should become an installable Python SDK. A user should be able to
`pip install evolab`, write a small Python script, point it at a Lab directory,
describe a task in natural language, configure backends in Python, and run an
EvoLab session without writing a task config file.

This is a major runtime and repository refactor, not a wrapper around the
current YAML `clean-run` path. The new SDK path is the product path. The old
task config file startup model does not need to remain supported.

## Hard Requirements

1. `session.run()` has no business return value.
   It starts the session and blocks until the task finishes or fails. EvoLab's
   useful output is written into the Lab directory as reports, structured files,
   and artifacts.

2. `.evolab/` is the only EvoLab internal state boundary.
   Every file needed by EvoLab itself must live under `<lab_dir>/.evolab`.
   This includes role pools, tools, skills, memory stores, queues, registries,
   trajectories, generated tool code, config snapshots, runtime snapshots, and
   internal logs.

3. The Lab root outside `.evolab/` is for the task workspace only.
   It should contain user-provided files and EvoLab-produced user-facing output.
   Internal runtime state must not leak into the Lab root.

4. Users define task and configuration in Python.
   Task config files are not part of the new public workflow. The SDK may keep
   internal Pydantic models, but users should not author `TaskConfig` YAML.

5. This refactor must be covered by new tests.
   Tests must prove directory isolation, SDK initialization, task execution
   without config files, and package installability.

## Public SDK Shape

The user-facing API should be small and explicit:

```python
from evolab import EvoLabSession, SessionConfig, TaskSpec

session = EvoLabSession(
    SessionConfig(
        env_file="/path/to/.env",
        lab_dir="/path/to/my_lab",
        task=TaskSpec(
            goal="What I want EvoLab to do.",
            resources="What files, data, or context I am giving EvoLab.",
            expected_outputs="What files or reports I expect EvoLab to produce.",
            success_criteria="How the output should be evaluated.",
            optional_context="Examples, ground truth, expert notes, constraints.",
        ),
        llm={
            "default": {
                "type": "api",
                "api": "openai-responses",
                "model": "gpt-5.5",
                "api_key_env": "OPENAI_API_KEY",
                "extra_body": {"reasoning": {"effort": "high"}},
            }
        },
        memory={"task": {"type": "method", "method": "mem0"}},
        skills={"default": {"type": "graph"}},
        tools={"builtin": True, "self_evolving": True},
    )
)

session.run()
```

`run()` returns `None`. To inspect results, users read files such as
`<lab_dir>/report.md`, `<lab_dir>/outputs/*.jsonl`, or other paths declared in
the task's expected outputs.

### Public Models

- `TaskSpec`
  - `goal: str`
  - `resources: str`
  - `expected_outputs: str`
  - `success_criteria: str`
  - `optional_context: str | None`
  - `to_prompt() -> str` produces the canonical natural-language task body.

- `SessionConfig`
  - `lab_dir: Path | str`
  - `env_file: Path | str | None`
  - `task: TaskSpec | str`
  - `llm: dict[str, Any]`
  - `memory: dict[str, Any]`
  - `skills: dict[str, Any]`
  - `tools: dict[str, Any] | None`
  - `runtime: dict[str, Any] | None`
  - `seed_roles: dict[str, Any] | None`
  - `meta_agent: dict[str, Any] | None`

- `EvoLabSession`
  - `__init__(config: SessionConfig)`
  - `initialize() -> None`
  - `run() -> None`
  - optional inspection helpers may expose paths, but not business results:
    `lab_dir`, `state_dir`, `report_paths()`.

## Lab Layout

The SDK must initialize a Lab directory like this:

```text
my_lab/
  input.pdf
  notes.md
  data/
  report.md
  outputs/
  artifacts/

  .evolab/
    AGENTS.md
    tools/
      builtin/
      generated/
      task_local/
    skills/
      pool/
      graph/
      evolution/
    memory/
      task/
      meta/
      stores/
    queues/
      tasks/
      evolve/
    registries/
      task/
      trajectory/
      backend_state/
      lab_state/
      snapshots/
    trajectories/
      meta_agent/
      subagent/
      llm_calls/
      evolution/
    configs/
    generated_tools/
    snapshots/
    logs/
```

The layout object should distinguish:

- `lab_dir`: user-visible workspace root.
- `state_dir`: `<lab_dir>/.evolab`.
- `output_dir`: default user-visible output root, for example
  `<lab_dir>/outputs`.
- `artifact_dir`: default user-visible artifact root, for example
  `<lab_dir>/artifacts`.

All registries, queues, memory stores, role pools, tool pools, generated tools,
skill pools, and trajectories use `state_dir`. User-facing reports and task
deliverables use `lab_dir`, `output_dir`, or `artifact_dir`.

## Initialization Semantics

When `EvoLabSession.initialize()` runs:

1. If `lab_dir` does not exist, create it.
2. If `lab_dir` exists but `.evolab` does not, initialize `.evolab`.
3. If `.evolab` exists but is incomplete, repair missing required directories
   and preserve existing state.
4. If `.evolab` is already initialized, reuse it.
5. Load `env_file` first, then process environment variables.
6. Materialize or reuse `.evolab/AGENTS.md`.
7. Materialize or reuse `.evolab/tools`, `.evolab/skills`, and
   `.evolab/memory`.
8. Build in-memory internal `TaskRequest` and `TaskConfig` equivalents from
   `SessionConfig`; users never author these files.

Initialization must not delete the Lab directory. Destructive cleanup is not a
default SDK behavior.

## Runtime Architecture

### Task Description

The SDK compiles `TaskSpec` into a canonical task prompt:

```text
1. goal: ...
2. resources: ...
3. expected_outputs: ...
4. success_criteria: ...
5. optional_context: ...
```

This text is stored in internal task metadata and sent to the MetaAgent and
dynamic planner.

### Role Evolution

The active role pool is `.evolab/AGENTS.md`.

The MetaAgent may inspect and update this file directly. Supported updates
remain add, delete, and edit role definitions. There is no human review gate for
role-pool evolution during normal SDK execution.

Dynamic workers use task-level memory by default. Stable MetaAgent memory may
use `.evolab/memory/meta`.

### Tool Evolution

The tool set has two layers:

- Built-in generic tools, defined by EvoLab and registered by the SDK.
- Task-specialized tools, generated by the LLM as Python code.

Each new task activates a fresh task-local generated tool scope. Generated tool
source code, validation artifacts, smoke-test logs, and provenance live under
`.evolab/generated_tools` or `.evolab/tools/generated`. Generated tools may be
made available to the current task planner and workers, but they must not write
internal state outside `.evolab`.

User-facing files created by generated or built-in tools must be written outside
`.evolab`, under the Lab workspace paths allowed by the task.

### Memory And Skills

Memory stores live under `.evolab/memory`. The SDK default should provide a
task-level memory backend and a MetaAgent memory backend. A no-memory option may
exist for tests, but production SDK defaults should initialize real local stores
when configured.

Skill pools and skill evolution state live under `.evolab/skills`. Packaged seed
skills may be copied into the Lab state directory during initialization so the
Lab is self-contained.

### Queues And Registries

The existing file queue and registry architecture can be preserved, but all
paths move under `.evolab`:

- `.evolab/queues/tasks`
- `.evolab/queues/evolve`
- `.evolab/registries/task`
- `.evolab/registries/trajectory`
- `.evolab/registries/backend_state`
- `.evolab/registries/lab_state`
- `.evolab/registries/snapshots`

The SDK can run synchronously, but it should still use the same internal
registries so future async workers can reuse the state.

## Repository Layout

The repository should be organized for packaging:

```text
README.md
pyproject.toml
src/
  evolab/
examples/
  minimal_session.py
  scientific_ie_session.py
dev/
  docs/
  reports/
  artifacts/
  configs/
  scripts/
tests/
```

The public package should include only runtime code and curated package data.
Historical experiment configs, intermediate reports, and task-specific
development artifacts move to `dev/`.

The README should prioritize:

1. Installation with `pip install evolab`.
2. A minimal Python SDK script.
3. Lab directory semantics.
4. Backend configuration.
5. Where to find outputs.

It should not present YAML task config files as the primary workflow.

## Migration Strategy

This is not a compatibility-preserving refactor. The implementation should:

1. Introduce SDK config models and Lab layout tests.
2. Move internal Lab state paths under `.evolab`.
3. Replace config-file startup with in-memory SDK compilation.
4. Update runtime path factories and tool artifact roots.
5. Update role and tool evolution paths.
6. Add examples based on Python scripts.
7. Move development artifacts into `dev/`.
8. Rewrite README.
9. Remove or demote old YAML startup code and tests.

The old `clean-run` CLI may be removed or converted into a thin dev-only helper.
It should not constrain the SDK design.

## Testing Requirements

New tests should cover:

- `SessionConfig` accepts natural-language `TaskSpec` fields and renders the
  canonical prompt.
- `EvoLabSession.initialize()` creates Lab and `.evolab` for a missing path.
- Existing uninitialized Lab directories are initialized without deleting user
  files.
- Existing initialized Labs are reused.
- Internal directories are all under `.evolab`.
- No queues, registries, trajectories, memory stores, generated tools, or
  `AGENTS.md` are created outside `.evolab`.
- `session.run()` returns `None`.
- A fake/offline SDK session writes a user-visible report to the Lab root or
  configured output path.
- Role evolution updates `.evolab/AGENTS.md`.
- Generated tool evolution writes generated source and provenance under
  `.evolab`.
- Package smoke test builds and installs a wheel, imports `evolab`, and runs a
  minimal offline example.

Existing tests should be migrated away from root-level `configs/`, root-level
`skills/`, root-level `registries/`, and YAML task config assumptions.

## Risks

- Path migration touches nearly every runtime subsystem.
- Existing tests may pass while still leaking state into the Lab root unless
  explicit negative assertions are added.
- Packaged seed skills and domain resources need resource loading that works
  from an installed wheel.
- Generated tool file guards must distinguish user workspace writes from
  internal `.evolab` writes.
- Removing YAML startup will invalidate older demos and docs; examples must be
  updated in the same branch.

## Acceptance Criteria

- A user can install the package and run an EvoLab session from a Python script.
- The script does not write or reference a task config file.
- `session.run()` returns `None`.
- User-visible outputs appear outside `.evolab`.
- All EvoLab internal state appears inside `.evolab`.
- Full tests pass, including SDK layout tests and packaging smoke tests.
- README and examples describe the SDK workflow as the primary workflow.
