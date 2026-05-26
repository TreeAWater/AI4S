# Evolution Backends

This package contains backends that propose or produce LLM/runtime evolution
artifacts after task execution.

## Module Map

- `fake.py`: deterministic evolution backend for tests and offline demos.
- `agent0_sage.py`: fake Agent0/SAGE-style trainer integration used by legacy
  evolution paths.
- `prompt_overlay.py`: prompt overlay evolution backend.

## Contract

Evolution backends consume `LLMEvolutionRequest` and return
`LLMEvolutionResult`. Results may include promoted state refs, dry-run
artifacts, standard metrics, and metadata for registries.

## Boundaries

Evolution backends should not mutate task execution state directly. Runtime code
records evolution outputs through trajectory and backend state registries.

