# Backend and Trainer ABC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `LLMBackend` and `LLMTrainer` real abstract parent classes for their concrete implementations.

**Architecture:** Keep `LLMRuntime` as a structural `Protocol`, because runtimes are adapter-like execution objects. Convert the configured backend and trainer interfaces to `ABC` classes with abstract methods, then make the existing concrete classes inherit from them. Preserve all existing public behavior.

**Tech Stack:** Python, `abc.ABC`, `abc.abstractmethod`, pytest.

---

## File Structure

- Modify `evolab/backends/llm.py`: replace `LLMBackend(Protocol)` with `LLMBackend(ABC)`, add `abstractmethod`, and make `ApiLLMBackend` and `LocalTrainableLLMBackend` inherit from `LLMBackend`.
- Modify `evolab/backends/trainers.py`: replace `LLMTrainer(Protocol)` with `LLMTrainer(ABC)`, add `abstractmethod`, and make `BlankTrainer` inherit from `LLMTrainer`.
- Modify `tests/test_api_llm_backend.py`: add assertions that concrete LLM backend classes subclass and instantiate as `LLMBackend`.
- Modify `tests/test_promotion.py`: add assertions that `BlankTrainer` subclasses and instantiates as `LLMTrainer`.

### Task 1: LLM Backend ABC

**Files:**
- Modify: `tests/test_api_llm_backend.py`
- Modify: `evolab/backends/llm.py`

- [ ] **Step 1: Write the failing LLM backend inheritance test**

Change the import block in `tests/test_api_llm_backend.py` to include `LLMBackend` and `LocalTrainableLLMBackend`:

```python
from evolab.backends.llm import (
    ApiLLMBackend,
    ApiLLMBackendConfig,
    LLMBackend,
    LocalTrainableLLMBackend,
    serialize_message_for_responses,
)
```

Add this test after `_FakeClient`:

```python
def test_concrete_llm_backends_inherit_from_llm_backend(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    api_backend = ApiLLMBackend(
        ApiLLMBackendConfig(provider="openai", model="gpt-4.1-mini"),
        client=_FakeClient(),
    )

    assert issubclass(ApiLLMBackend, LLMBackend)
    assert issubclass(LocalTrainableLLMBackend, LLMBackend)
    assert isinstance(api_backend, LLMBackend)
    assert isinstance(LocalTrainableLLMBackend(), LLMBackend)
```

- [ ] **Step 2: Run the LLM inheritance test to verify it fails**

Run:

```bash
pytest tests/test_api_llm_backend.py::test_concrete_llm_backends_inherit_from_llm_backend -q
```

Expected: FAIL because `LLMBackend` is still a `Protocol` that is not marked `@runtime_checkable`, so `issubclass(..., LLMBackend)` raises a `TypeError`.

- [ ] **Step 3: Convert the LLM backend interface to an ABC**

In `evolab/backends/llm.py`, change the imports from:

```python
import json
import os
from typing import Any, Literal, Protocol
```

to:

```python
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Literal, Protocol
```

Change `LLMBackend` from:

```python
class LLMBackend(Protocol):
    backend_id: str

    def instantiate(self, state_ref: str | None) -> LLMRuntime:
        ...
```

to:

```python
class LLMBackend(ABC):
    backend_id: str

    @abstractmethod
    def instantiate(self, state_ref: str | None) -> LLMRuntime:
        raise NotImplementedError
```

Change the concrete class declarations from:

```python
class ApiLLMBackend:
```

and:

```python
class LocalTrainableLLMBackend:
```

to:

```python
class ApiLLMBackend(LLMBackend):
```

and:

```python
class LocalTrainableLLMBackend(LLMBackend):
```

- [ ] **Step 4: Run the LLM backend tests to verify they pass**

Run:

```bash
pytest tests/test_api_llm_backend.py -q
```

Expected: PASS for all tests in `tests/test_api_llm_backend.py`.

### Task 2: Trainer ABC

**Files:**
- Modify: `tests/test_promotion.py`
- Modify: `evolab/backends/trainers.py`

- [ ] **Step 1: Write the failing trainer inheritance test**

Change the import in `tests/test_promotion.py` from:

```python
from evolab.backends.trainers import BlankTrainer
```

to:

```python
from evolab.backends.trainers import BlankTrainer, LLMTrainer
```

Add this test after `FakeTrainer`:

```python
def test_blank_trainer_inherits_from_llm_trainer():
    trainer = BlankTrainer()

    assert issubclass(BlankTrainer, LLMTrainer)
    assert isinstance(trainer, LLMTrainer)
```

- [ ] **Step 2: Run the trainer inheritance test to verify it fails**

Run:

```bash
pytest tests/test_promotion.py::test_blank_trainer_inherits_from_llm_trainer -q
```

Expected: FAIL because `LLMTrainer` is still a `Protocol` that is not marked `@runtime_checkable`, so `issubclass(BlankTrainer, LLMTrainer)` raises a `TypeError`.

- [ ] **Step 3: Convert the trainer interface to an ABC**

In `evolab/backends/trainers.py`, change the imports from:

```python
from typing import Protocol
```

to:

```python
from abc import ABC, abstractmethod
```

Change `LLMTrainer` from:

```python
class LLMTrainer(Protocol):
    trainer_id: str

    def train(self, request: LLMEvolutionRequest) -> LLMEvolutionResult:
        ...
```

to:

```python
class LLMTrainer(ABC):
    trainer_id: str

    @abstractmethod
    def train(self, request: LLMEvolutionRequest) -> LLMEvolutionResult:
        raise NotImplementedError
```

Change:

```python
class BlankTrainer:
```

to:

```python
class BlankTrainer(LLMTrainer):
```

- [ ] **Step 4: Run the promotion tests to verify they pass**

Run:

```bash
pytest tests/test_promotion.py -q
```

Expected: PASS for all tests in `tests/test_promotion.py`.

### Task 3: Full Verification and Commit

**Files:**
- Verify: all modified files

- [ ] **Step 1: Run focused test files together**

Run:

```bash
pytest tests/test_api_llm_backend.py tests/test_promotion.py -q
```

Expected: PASS.

- [ ] **Step 2: Run the full test suite**

Run:

```bash
pytest -q
```

Expected: PASS.

- [ ] **Step 3: Inspect the final diff**

Run:

```bash
git diff -- evolab/backends/llm.py evolab/backends/trainers.py tests/test_api_llm_backend.py tests/test_promotion.py
```

Expected: Diff only includes the ABC conversion, concrete inheritance, and focused tests.

- [ ] **Step 4: Commit the implementation**

Run:

```bash
git add evolab/backends/llm.py evolab/backends/trainers.py tests/test_api_llm_backend.py tests/test_promotion.py docs/superpowers/specs/2026-05-02-backend-trainer-abc-design.md docs/superpowers/plans/2026-05-02-backend-trainer-abc-implementation.md
git commit -m "refactor: make backend trainer interfaces abstract"
```

Expected: Commit succeeds.
