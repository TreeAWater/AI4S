# Config

`evolab.config` contains internal configuration models and parsers. SDK users
normally write `SessionConfig`; this package holds the lower-level config shape
used after SDK compilation.

## Module Map

- `task_config.py`: `TaskConfig`, role specs, meta-agent specs, backend
  bindings, and runtime policy inputs.
- `agents.py`: `AGENTS.md` role-pool rendering and parsing helpers.
- `env.py`: `.env` parsing and environment reference helpers.

## Boundaries

Do not add task-specific defaults here. Config code should validate and parse
declared settings, while task behavior belongs in runtime, backends, tools, or
the evolving role pool.

