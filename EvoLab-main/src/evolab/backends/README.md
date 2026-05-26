# Backends

`evolab.backends` contains pluggable implementations behind stable runtime
contracts. Runtime code should depend on backend interfaces, not on provider
details.

## Subpackages

- `llm/`: chat/generation backends, including OpenAI-compatible APIs, fake
  runtimes, and local trainable rollout state.
- `embeddings/`: embedding backends used by native memory methods.
- `memory/`: memory backends and native memory methods.
- `skills/`: skill retrieval, package loading, graph search, and skill
  evolution support.
- `evolution/`: LLM evolution backends used after task runs.
- `trainers/`: trainable backend adapters such as SFT and OPSD.
- `rewards/`: reward calculators used for training/export workflows.

## Development Rules

- Keep provider-specific SDK objects inside backend implementations.
- Convert all external responses into EvoLab contracts before returning.
- Do not silently fall back from a real backend to a fake backend.
- Do not log secrets or inline credentials.
- Add new backend types through explicit builder validation.

