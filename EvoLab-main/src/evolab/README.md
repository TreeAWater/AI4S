# EvoLab Source Package

`src/evolab` contains the installable Python package. The public SDK surface is
small by design: `EvoLabSession`, `SessionConfig`, and `TaskSpec` are exported
from `evolab.__init__`.

## Main Entry Points

- `sdk.py`: public SDK models and `EvoLabSession`.
- `session_runtime.py`: bridges SDK configs into the internal runtime models.
- `cli.py`: legacy/dev command helpers and backend builder functions reused by
  the SDK runtime.

## Package Map

- `backends/`: provider implementations for LLMs, memory, skills, embeddings,
  rewards, evolution, and trainers.
- `config/`: configuration models and role-pool markdown parsing.
- `contracts/`: Pydantic contracts shared across runtime, tools, and backends.
- `lab/`: Lab folder layout, queues, and resolver helpers.
- `registries/`: file-backed registries for task, trajectory, state, and
  snapshots.
- `runtime/`: task execution orchestration, dynamic workflows, role/tool
  evolution, and export helpers.
- `tools/`: built-in task tools and tool runtime registry.

## Boundaries

SDK users should not need to import internal modules directly. Internal modules
should communicate through contracts from `contracts/` and should keep EvoLab
state inside `<lab>/.evolab`. User-visible outputs belong in the Lab root,
`outputs/`, or `artifacts/`.

