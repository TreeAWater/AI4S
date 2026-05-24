# Lab Resolver, Registries, And Fake LLM Design

## Context

The framework already has filesystem-backed implementations for lab layout, queues, task registry, trajectory registry, and backend state registry. The current registry classes are concrete only, and callers manually construct registries from `LabLayout` paths. LLM tests also define local fakes inside test files, but there is no reusable fake LLM runtime shell for deterministic development and registry/runtime tests.

## Goal

Add a small resolver and shared interfaces that make the filesystem lab easier to wire, make registry contracts explicit, and provide a deterministic fake LLM runtime shell for tests and development.

## Design

Add `evolab/lab/resolver.py` with `LabResolver`. The resolver owns a `LabLayout` and exposes methods for standard lab components:

- `ensure()`
- `task_queue()`
- `evolve_queue()`
- `task_registry()`
- `trajectory_registry()`
- `backend_state_registry()`

Formalize registry interfaces without changing existing persistence behavior:

- `TrajectoryRegistry` ABC in `evolab/registries/trajectory.py`
- `BackendStateRegistry` ABC in `evolab/registries/backend_state.py`
- `FileTrajectoryRegistry(TrajectoryRegistry)`
- `FileBackendStateRegistry(BackendStateRegistry)`

Extend the file registries with read/list APIs needed for round-trip tests:

- trajectory reads by ref for meta-agent, subagent, LLM call, and evolution records;
- trajectory list/query support for persisted record types;
- backend state lookup by `state_ref`;
- backend state listing by optional `backend_id`.

Add `evolab/backends/llm/fake.py`:

- `FakeLLMRuntime` implements the `LLMRuntime.generate(...)` shape.
- `FakeLLMBackend` inherits `LLMBackend` and returns a fake runtime from `instantiate(...)`.
- fake runtime can return queued `LLMRuntimeResponse` objects or a default final-answer response.
- fake runtime records each request for assertions.

Export fake LLM classes from `evolab.backends.llm`.

## Behavior

No existing behavior changes:

- Existing `FileTrajectoryRegistry.save_*` methods continue appending JSONL records.
- Existing `FileBackendStateRegistry.register_candidate(...)`, `promote(...)`, and `resolve_active_state(...)` behavior stays compatible.
- `TaskWorker` startup remains compatible with direct constructor injection, but can use `LabResolver` internally to reduce duplicated path construction.
- The fake LLM shell is explicit support code; real API and local trainable backends keep their current behavior.

## Testing

Add tests for:

- `LabResolver` creates and returns the standard queues and registries for a layout.
- `FileTrajectoryRegistry` inherits `TrajectoryRegistry` and round-trips all shared run/call record types through JSONL persistence.
- `FileBackendStateRegistry` inherits `BackendStateRegistry` and round-trips backend state records through JSONL persistence.
- `FakeLLMBackend` inherits `LLMBackend`, records instantiate state refs, and returns a fake runtime.
- `FakeLLMRuntime` records generate requests and returns queued/default responses.

## Out Of Scope

This change does not implement a real local LLM, add training algorithms, change promotion policy, or change task dispatch semantics.
