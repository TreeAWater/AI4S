# Backend Package Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move LLM backends and trainers into package folders while keeping existing imports stable.

**Architecture:** Follow the existing `evolab.backends.memory` package pattern: shared ABCs in `base.py`, implementation-specific modules beside them, and root package `__init__.py` re-exports. Delete the old flat modules after creating package folders so Python imports resolve to the packages.

**Tech Stack:** Python packages, pytest, `abc.ABC`, `typing.Protocol`.

---

## File Structure

- Create: `evolab/backends/llm/__init__.py`
- Create: `evolab/backends/llm/base.py`
- Create: `evolab/backends/llm/api.py`
- Create: `evolab/backends/llm/local.py`
- Delete: `evolab/backends/llm.py`
- Create: `evolab/backends/trainers/__init__.py`
- Create: `evolab/backends/trainers/base.py`
- Create: `evolab/backends/trainers/blank.py`
- Delete: `evolab/backends/trainers.py`
- Modify: `tests/test_api_llm_backend.py`
- Modify: `tests/test_promotion.py`

### Task 1: Add Failing Import Compatibility Tests

- [ ] **Step 1: Extend LLM backend imports and test**

In `tests/test_api_llm_backend.py`, import package-module classes:

```python
from evolab.backends.llm.api import ApiLLMBackend as PackageApiLLMBackend
from evolab.backends.llm.base import LLMBackend as BaseLLMBackend
from evolab.backends.llm.local import LocalTrainableLLMBackend as PackageLocalTrainableLLMBackend
```

Extend `test_concrete_llm_backends_inherit_from_llm_backend`:

```python
    assert LLMBackend is BaseLLMBackend
    assert ApiLLMBackend is PackageApiLLMBackend
    assert LocalTrainableLLMBackend is PackageLocalTrainableLLMBackend
```

- [ ] **Step 2: Extend trainer imports and test**

In `tests/test_promotion.py`, import package-module classes:

```python
from evolab.backends.trainers.base import LLMTrainer as BaseLLMTrainer
from evolab.backends.trainers.blank import BlankTrainer as PackageBlankTrainer
```

Extend `test_blank_trainer_inherits_from_llm_trainer`:

```python
    assert LLMTrainer is BaseLLMTrainer
    assert BlankTrainer is PackageBlankTrainer
```

- [ ] **Step 3: Verify tests fail before the package move**

Run:

```bash
pytest tests/test_api_llm_backend.py::test_concrete_llm_backends_inherit_from_llm_backend tests/test_promotion.py::test_blank_trainer_inherits_from_llm_trainer -q
```

Expected: FAIL during import because `evolab.backends.llm` and `evolab.backends.trainers` are still flat modules, not packages.

### Task 2: Create LLM Backend Package

- [ ] **Step 1: Move shared LLM interfaces to `base.py`**

Create `evolab/backends/llm/base.py` with `LLMRuntime` and `LLMBackend` exactly preserving current signatures.

- [ ] **Step 2: Move API implementation to `api.py`**

Create `evolab/backends/llm/api.py` with `ApiLLMBackendConfig`, helper functions, `OpenAIResponsesRuntime`, and `ApiLLMBackend`. Import `LLMBackend` and `LLMRuntime` from `evolab.backends.llm.base`.

- [ ] **Step 3: Move local backend to `local.py`**

Create `evolab/backends/llm/local.py` with `LocalTrainableLLMBackend`.

- [ ] **Step 4: Add public exports**

Create `evolab/backends/llm/__init__.py`:

```python
from evolab.backends.llm.api import ApiLLMBackend, ApiLLMBackendConfig, serialize_message_for_responses
from evolab.backends.llm.base import LLMBackend, LLMRuntime
from evolab.backends.llm.local import LocalTrainableLLMBackend

__all__ = [
    "ApiLLMBackend",
    "ApiLLMBackendConfig",
    "LLMBackend",
    "LLMRuntime",
    "LocalTrainableLLMBackend",
    "serialize_message_for_responses",
]
```

- [ ] **Step 5: Delete the old flat file**

Delete `evolab/backends/llm.py`.

### Task 3: Create Trainer Package

- [ ] **Step 1: Move trainer ABC to `base.py`**

Create `evolab/backends/trainers/base.py` with `LLMTrainer` exactly preserving the current signature.

- [ ] **Step 2: Move blank trainer to `blank.py`**

Create `evolab/backends/trainers/blank.py` with `BlankTrainer`. Import `LLMTrainer` from `evolab.backends.trainers.base`.

- [ ] **Step 3: Add public exports**

Create `evolab/backends/trainers/__init__.py`:

```python
from evolab.backends.trainers.base import LLMTrainer
from evolab.backends.trainers.blank import BlankTrainer

__all__ = [
    "BlankTrainer",
    "LLMTrainer",
]
```

- [ ] **Step 4: Delete the old flat file**

Delete `evolab/backends/trainers.py`.

### Task 4: Verify And Commit

- [ ] **Step 1: Run focused tests**

Run:

```bash
pytest tests/test_api_llm_backend.py tests/test_promotion.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full suite**

Run:

```bash
pytest -q
```

Expected: PASS.

- [ ] **Step 3: Check for generated artifacts and inspect diff**

Run:

```bash
find evolab tests -type d -name __pycache__ -prune -exec rm -rf {} +
git status --short
git diff --stat
```

Expected: no generated cache files; diff only includes package layout, import tests, and docs.

- [ ] **Step 4: Commit**

Run:

```bash
git add evolab/backends/llm evolab/backends/trainers tests/test_api_llm_backend.py tests/test_promotion.py docs/superpowers/plans/2026-05-03-backend-package-layout-implementation.md
git add -u evolab/backends/llm.py evolab/backends/trainers.py
git commit -m "refactor: split backend implementations into packages"
```

Expected: Commit succeeds.
