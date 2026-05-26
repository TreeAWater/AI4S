# EvoLab

EvoLab is an SDK for running self-evolving agent sessions in a user-owned Lab
folder. Users describe a task in Python, point EvoLab at a Lab directory, and
let the session write its working state and outputs to disk.

The current public interface is the Python SDK:

```python
from pathlib import Path

from evolab import EvoLabSession, SessionConfig, TaskSpec

session = EvoLabSession(
    SessionConfig(
        env_file=Path(".env"),
        lab_dir=Path("/tmp/my-evolab-lab"),
        task=TaskSpec(
            goal="Summarize the supplied papers into a concise report.",
            resources="The Lab folder contains papers/ with source documents.",
            expected_outputs="report.md in the Lab folder.",
            success_criteria="The report cites the supplied sources and covers the requested scope.",
            optional_context="Prefer short sections and explicit uncertainty.",
        ),
        llm={
            "default": {
                "type": "fake",
                "default_content": "offline response",
            }
        },
        memory={"task": {"type": "null"}},
        skills={"default": {"type": "fake", "skills": []}},
    )
)

session.run()
```

`session.run()` returns `None`. EvoLab results are files in the Lab folder, such
as `report.md`, JSONL records, tables, or other task outputs.

## Install

Python 3.11+ is required.

```bash
pip install evolab
```

For local development:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Lab Layout

Every Lab has one EvoLab state directory:

```text
<lab>/
  .evolab/
    AGENTS.md
    tools/
    generated_tools/
    skills/
    memory/
    queues/
    registries/
    trajectories/
    configs/
  report.md
  outputs/
  artifacts/
  user-provided-files...
```

All EvoLab internal state lives under `.evolab`: role pools, generated tools,
skill pools, memory stores, queues, registries, trajectories, and internal
configs. Files outside `.evolab` are the user's task workspace: user inputs and
user-visible outputs produced by the session.

## Task Description

`TaskSpec` keeps the user-facing task contract explicit:

- `goal`: what the session should do.
- `resources`: what files, data, APIs, examples, or other materials are
  available.
- `expected_outputs`: what files or artifacts should be produced.
- `success_criteria`: how the result should be judged.
- `optional_context`: examples, constraints, ground truth, expert notes, or
  other optional guidance.

These fields are natural language. The SDK compiles them into EvoLab's internal
runtime models; users do not write task config files.

## Self-Evolution

EvoLab keeps a task-level evolving role pool in `.evolab/AGENTS.md`. The
MetaAgent can inspect, add, remove, or update roles during a session.

Tools are split into two layers:

- Built-in general tools, such as file reading, text/table processing, schema
  validation, and output writing.
- Task-specialized tools generated as Python code for the current task. These
  are stored under `.evolab/generated_tools` and reset for each new task.

Task memory is stored under `.evolab/memory`. Because roles can evolve during a
run, task-level memory is the stable default.

## Examples

Run the offline minimal example:

```bash
PYTHONPATH=src python3 examples/minimal_session.py
```

See `examples/` for SDK entry scripts. Historical development docs, old
experiment configs, scientific IE seed packages, benchmark artifacts, and helper
scripts live under `dev/`.

## Repository Layout

```text
src/evolab/    package source
examples/      SDK entry scripts
tests/         regression tests
dev/           development-only docs, configs, skills, reports, scripts
```
