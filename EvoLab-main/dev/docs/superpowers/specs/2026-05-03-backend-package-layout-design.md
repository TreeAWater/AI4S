# Backend Package Layout Design

## Context

`evolab.backends.memory` already uses a package layout with a shared `base.py`, provider-specific modules, and `__init__.py` public exports. LLM backends and trainers still live in flat modules:

- `evolab/backends/llm.py`
- `evolab/backends/trainers.py`

Those flat modules make it awkward to add multiple implementations that share the same abstract base class.

## Goal

Move LLM backends and trainers into package folders so future implementations can live in separate modules while existing imports continue to work.

## Design

Create `evolab/backends/llm/`:

- `base.py`: `LLMRuntime`, `LLMBackend`
- `api.py`: API-backed LLM implementation, OpenAI Responses runtime, config, and helper functions
- `local.py`: `LocalTrainableLLMBackend`
- `__init__.py`: re-export all current public names from the old `evolab.backends.llm` module

Create `evolab/backends/trainers/`:

- `base.py`: `LLMTrainer`
- `blank.py`: `BlankTrainer`
- `__init__.py`: re-export all current public names from the old `evolab.backends.trainers` module

Delete the old flat module files after the packages are in place. Python cannot safely have both `evolab/backends/llm.py` and `evolab/backends/llm/` as the same import target.

## Compatibility

Existing public imports must continue to work:

- `from evolab.backends.llm import ApiLLMBackend`
- `from evolab.backends.llm import LLMBackend`
- `from evolab.backends.llm import LocalTrainableLLMBackend`
- `from evolab.backends.trainers import BlankTrainer`
- `from evolab.backends.trainers import LLMTrainer`

New implementation-specific imports should also work:

- `from evolab.backends.llm.base import LLMBackend`
- `from evolab.backends.llm.api import ApiLLMBackend`
- `from evolab.backends.llm.local import LocalTrainableLLMBackend`
- `from evolab.backends.trainers.base import LLMTrainer`
- `from evolab.backends.trainers.blank import BlankTrainer`

## Behavior

No behavior changes:

- `LLMRuntime` remains a `Protocol`.
- `LLMBackend` and `LLMTrainer` remain ABCs.
- `ApiLLMBackend` keeps the same credential, client injection, OpenAI Responses, structured output, and tool-call behavior.
- `LocalTrainableLLMBackend.instantiate(...)` keeps raising `NotImplementedError`.
- `BlankTrainer.train(...)` keeps raising `NotImplementedError`.
- `EvolveWorker` keeps importing `LLMTrainer` from `evolab.backends.trainers`.

## Testing

Add import-compatibility tests that assert:

- package root exports point to the same classes as implementation modules;
- concrete classes still subclass the shared ABCs;
- existing API backend and promotion tests still pass.

## Out Of Scope

This change does not add new LLM providers, local inference behavior, trainer algorithms, registries, or worker dispatch behavior.
