# Backend and Trainer ABC Design

## Context

`evolab.backends.llm.LLMBackend` and `evolab.backends.trainers.LLMTrainer` currently use structural `Protocol` interfaces. `ApiLLMBackend`, `LocalTrainableLLMBackend`, and `BlankTrainer` satisfy those protocols by shape, but the code does not show an explicit parent-child relationship. This makes the implementation less direct than the intended backend and trainer architecture.

## Goal

Make the common LLM backend and trainer abstractions real runtime parents while preserving existing behavior.

## Design

Convert `LLMBackend` from a `Protocol` into an abstract base class. It will keep the `backend_id` attribute contract and declare `instantiate(state_ref)` as an abstract method returning an `LLMRuntime`.

Convert `LLMTrainer` from a `Protocol` into an abstract base class. It will keep the `trainer_id` attribute contract and declare `train(request)` as an abstract method returning an `LLMEvolutionResult`.

Update concrete implementations to inherit from those bases:

- `ApiLLMBackend(LLMBackend)`
- `LocalTrainableLLMBackend(LLMBackend)`
- `BlankTrainer(LLMTrainer)`

`LLMRuntime` should remain a `Protocol` because runtimes are execution adapters and structural compatibility is useful there. This change is only about the parent relationship for configured backend and trainer types.

## Behavior

No product behavior changes:

- `ApiLLMBackend` still requires credentials unless a client is injected.
- `ApiLLMBackend.instantiate(...)` still rejects non-null `state_ref`.
- `LocalTrainableLLMBackend.instantiate(...)` still raises `NotImplementedError`.
- `BlankTrainer.train(...)` still raises `NotImplementedError`.
- `EvolveWorker` still accepts trainer objects through the existing `dict[str, LLMTrainer]` constructor shape.

## Testing

Add focused tests that assert:

- `ApiLLMBackend` and `LocalTrainableLLMBackend` are subclasses of `LLMBackend`.
- `ApiLLMBackend` instances satisfy `isinstance(..., LLMBackend)`.
- `BlankTrainer` is a subclass and instance of `LLMTrainer`.
- Existing API backend and promotion tests continue to pass.

## Out of Scope

This change does not introduce concrete training algorithms, local inference implementation, new registries, or new worker dispatch behavior.
