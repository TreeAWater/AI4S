# Lab Resolver, Registries, And Fake LLM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Lab resolver, explicit registry ABCs/read APIs, a deterministic fake LLM shell, and registry-backed record round-trip tests.

**Architecture:** Extend the existing filesystem skeleton. Keep file registries append-only, add typed read/list helpers, and keep public imports stable. The fake LLM lives beside the real LLM implementations under `evolab.backends.llm`.

**Tech Stack:** Python, pytest, Pydantic models, filesystem JSONL registries.

---

## File Structure

- Create: `evolab/lab/resolver.py`
- Modify: `evolab/runtime/task_worker.py`
- Modify: `evolab/registries/trajectory.py`
- Modify: `evolab/registries/backend_state.py`
- Create: `evolab/backends/llm/fake.py`
- Modify: `evolab/backends/llm/__init__.py`
- Modify: `tests/test_lab_queue.py`
- Modify: `tests/test_registries.py`
- Create: `tests/test_fake_llm_backend.py`
- Modify: `tests/test_api_llm_backend.py`

### Task 1: Failing Tests

- [ ] **Step 1: Add resolver tests**

Add tests to `tests/test_lab_queue.py` that import `LabResolver`, construct it from a `LabLayout`, call `ensure()`, and assert it returns `FileWorkQueue`, `FileTaskRegistry`, `FileTrajectoryRegistry`, and `FileBackendStateRegistry` rooted under the layout paths.

- [ ] **Step 2: Add registry ABC and round-trip tests**

Add tests to `tests/test_registries.py` that assert file registries inherit new ABCs and can read/list round-tripped `MetaAgentRunRecord`, `SubagentRunRecord`, `LLMCallRecord`, `EvolutionRunRecord`, and `BackendStateRecord` records.

- [ ] **Step 3: Add fake LLM tests**

Create `tests/test_fake_llm_backend.py` that asserts `FakeLLMBackend` inherits `LLMBackend`, `FakeLLMRuntime` records `generate(...)` calls, queued responses are returned before default responses, and package root exports match `evolab.backends.llm.fake`.

- [ ] **Step 4: Run new focused tests and verify they fail**

Run:

```bash
pytest tests/test_lab_queue.py::test_lab_resolver_returns_standard_components tests/test_registries.py::test_trajectory_registry_round_trips_shared_records tests/test_registries.py::test_backend_state_registry_round_trips_records tests/test_fake_llm_backend.py -q
```

Expected: FAIL because resolver, registry ABC/read APIs, and fake LLM shell are not implemented yet.

### Task 2: Implement Resolver

- [ ] **Step 1: Create `LabResolver`**

Implement `evolab/lab/resolver.py` with:

- `__init__(self, layout: LabLayout | Path | str)`
- `ensure(self) -> None`
- `task_queue(self) -> FileWorkQueue`
- `evolve_queue(self) -> FileWorkQueue`
- `task_registry(self) -> FileTaskRegistry`
- `trajectory_registry(self) -> FileTrajectoryRegistry`
- `backend_state_registry(self) -> FileBackendStateRegistry`

- [ ] **Step 2: Use resolver in `TaskWorker.startup()`**

Modify `TaskWorker.startup()` to construct `LabResolver(self.layout)` after `self.layout.ensure()` and use it for default queue/registry construction. Preserve all existing dependency injection behavior.

### Task 3: Implement Registry ABCs And Reads

- [ ] **Step 1: Add `TrajectoryRegistry` ABC**

Add abstract methods matching current save APIs plus read/list/query methods:

- `save_meta_agent_run`
- `save_subagent_run`
- `save_llm_call`
- `save_evolution_run`
- `get_meta_agent_run`
- `get_subagent_run`
- `get_llm_call`
- `get_evolution_run`
- `list_meta_agent_runs`
- `list_subagent_runs`
- `list_llm_calls`
- `list_evolution_runs`
- `query_subagent_runs`

- [ ] **Step 2: Extend `FileTrajectoryRegistry`**

Make `FileTrajectoryRegistry(TrajectoryRegistry)`. Add generic JSONL loading helpers and implement the get/list methods by matching refs.

- [ ] **Step 3: Add `BackendStateRegistry` ABC**

Add abstract methods:

- `register_candidate`
- `get_state`
- `list_states`
- `promote`
- `resolve_active_state`

- [ ] **Step 4: Extend `FileBackendStateRegistry`**

Make `FileBackendStateRegistry(BackendStateRegistry)`. Implement `get_state(state_ref)` and `list_states(backend_id=None)` using existing JSONL data.

### Task 4: Implement Fake LLM Shell

- [ ] **Step 1: Create `evolab/backends/llm/fake.py`**

Implement:

- `FakeLLMRuntime`
- `FakeLLMBackend`
- a lightweight request record for runtime assertions using existing `LLMRuntimeRequest`

`FakeLLMRuntime.generate(...)` should append an `LLMRuntimeRequest` to `requests`, return queued responses FIFO when present, otherwise return a final-answer response with configurable default content.

- [ ] **Step 2: Export fake classes**

Update `evolab/backends/llm/__init__.py` to export `FakeLLMBackend` and `FakeLLMRuntime`.

### Task 5: Verify And Commit

- [ ] **Step 1: Run focused tests**

Run:

```bash
pytest tests/test_lab_queue.py tests/test_registries.py tests/test_fake_llm_backend.py tests/test_api_llm_backend.py tests/test_task_worker.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full suite**

Run:

```bash
pytest -q
```

Expected: PASS.

- [ ] **Step 3: Clean generated caches and inspect diff**

Run:

```bash
find evolab tests -type d -name __pycache__ -prune -exec rm -rf {} +
git status --short
git diff --stat
```

Expected: no generated cache files; diff only includes resolver, registry, fake LLM, tests, and this plan.

- [ ] **Step 4: Commit**

Run:

```bash
git add evolab/lab/resolver.py evolab/runtime/task_worker.py evolab/registries/trajectory.py evolab/registries/backend_state.py evolab/backends/llm/fake.py evolab/backends/llm/__init__.py tests/test_lab_queue.py tests/test_registries.py tests/test_fake_llm_backend.py tests/test_api_llm_backend.py docs/superpowers/plans/2026-05-03-lab-resolver-registries-fake-llm-implementation.md
git commit -m "feat: add lab resolver registries fake llm shell"
```

Expected: Commit succeeds.
