# LLM Backends

This package implements EvoLab LLM backend interfaces and provider adapters.

## Module Map

- `base.py`: `LLMBackend` and `LLMRuntime` abstractions.
- `api.py`: OpenAI Responses and Chat Completions compatible runtime.
- `fake.py`: deterministic scripted backend for tests and examples.
- `local.py`: local trainable rollout backend that reads promoted state refs.

## Runtime Contract

An `LLMBackend` creates an `LLMRuntime` with `instantiate(state_ref)`. A runtime
generates `LLMRuntimeResponse` objects from messages, tool specs, and generation
config. Tool call responses must preserve call ids and tool names so
`TaskRuntime` can continue the tool loop.

## Development Rules

- Keep provider SDK objects inside this package.
- Validate credentials at backend construction; missing credentials should fail
  clearly.
- Do not write API keys into trajectory, state refs, or metadata.
- Keep fake backends deterministic and explicit.

