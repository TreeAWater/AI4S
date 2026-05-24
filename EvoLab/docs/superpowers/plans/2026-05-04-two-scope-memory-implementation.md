# Two-Scope Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make TaskRuntime read and write both agent-wise memory and task-wise memory for every subagent run.

**Architecture:** Extend config contracts with explicit memory bindings, add scope metadata to retrieval requests, combine two memory bundles for prompting, and persist detailed per-scope memory traceability in `SubagentRunRecord.metadata`. Keep `MemoryBackend.add(...)` unchanged and keep memory evolution inside TaskRuntime only.

**Tech Stack:** Python 3.11, Pydantic v2, pytest, existing EvoLab contracts and filesystem trajectory registry.

---

## File Structure

- Modify `evolab/config/task_config.py`: add `RoleSpec.agent_memory_backend` and `TaskConfig.task_memory_backend`.
- Modify `evolab/backends/memory/mem0.py`: allow `memory_scope` and `memory_scope_id` placeholders in `user_id_template`, and derive scoped user ids from retrieval filters.
- Modify `evolab/runtime/task_runtime.py`: resolve and use agent/task memory scopes, combine bundles for prompt input, and record per-scope metadata.
- Modify `tests/test_task_worker.py`: add the runtime behavior test for two memory scopes.
- Modify `tests/test_mem0_memory_backend.py`: add tests for scoped Mem0 user id templates.
- Modify `docs/superpowers/specs/2026-05-04-two-scope-memory-design.md`: keep as design record.

### Task 1: Config Contracts For Memory Scopes

**Files:**
- Modify: `evolab/config/task_config.py`
- Modify: `tests/test_task_worker.py`

- [ ] **Step 1: Write failing config/runtime test**

Add a test in `tests/test_task_worker.py` that constructs:

```python
task_config = TaskConfig(
    task_id=request.task_id,
    goal=request.goal,
    task_memory_backend=BackendBinding(backend_id="task-memory"),
    roles={
        "solver": RoleSpec(
            name="solver",
            system_prompt="You solve scientific tasks.",
            llm_backend=BackendBinding(backend_id="llm-local"),
            agent_memory_backend=BackendBinding(backend_id="agent-memory"),
        )
    },
)
```

Expected behavior after implementation:

- agent memory receives one `search` request with filters:
  - `memory_scope == "agent"`
  - `memory_scope_id == "agent:solver"`
- task memory receives one `search` request with filters:
  - `memory_scope == "task"`
  - `memory_scope_id == "task:task-1"`
- agent memory receives one `add(task_id, "solver", messages)`.
- task memory receives one `add(task_id, "task", messages)`.
- LLM prompt includes both `Agent Memory` and `Task Memory`.
- saved trajectory metadata includes:
  - `agent_memory_bundle`
  - `task_memory_bundle`
  - `agent_memory_update_result`
  - `task_memory_update_result`

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
pytest tests/test_task_worker.py::test_default_task_runtime_reads_and_writes_agent_and_task_memory_scopes -q
```

Expected: FAIL because `TaskConfig` and `RoleSpec` do not yet accept the new memory binding fields.

- [ ] **Step 3: Add config fields**

Add to `RoleSpec`:

```python
agent_memory_backend: BackendBinding | None = None
```

Add to `TaskConfig`:

```python
task_memory_backend: BackendBinding | None = None
```

- [ ] **Step 4: Run test to verify next failure**

Run:

```bash
pytest tests/test_task_worker.py::test_default_task_runtime_reads_and_writes_agent_and_task_memory_scopes -q
```

Expected: FAIL because `TaskRuntime` still uses only one memory runtime.

### Task 2: TaskRuntime Two-Scope Memory Flow

**Files:**
- Modify: `evolab/runtime/task_runtime.py`
- Modify: `tests/test_task_worker.py`

- [ ] **Step 1: Implement runtime behavior**

Update `TaskRuntime.run(...)` so the default path:

- resolves agent memory from `role.agent_memory_backend.backend_id`;
- resolves task memory from `task_config.task_memory_backend.backend_id`;
- falls back to the first memory runtime for both scopes only when explicit bindings are absent;
- raises `RuntimeError` when only one of `role.agent_memory_backend` and `task_config.task_memory_backend` is configured;
- builds separate `RetrievalRequest` objects for agent and task memory;
- calls both `search(...)` methods;
- combines both bundles for prompt construction with stable section labels;
- calls agent memory `add(task_id, role.name, messages)`;
- calls task memory `add(task_id, "task", messages)`;
- stores per-scope bundles and update results in trajectory metadata.

- [ ] **Step 2: Run focused tests**

Run:

```bash
pytest tests/test_task_worker.py::test_default_task_runtime_reads_and_writes_agent_and_task_memory_scopes tests/test_task_worker.py::test_default_task_runtime_runs_memory_skill_llm_and_records_updates -q
```

Expected: PASS. The older single-memory test should keep passing through the compatibility fallback.

### Task 3: Mem0 Scoped User IDs

**Files:**
- Modify: `evolab/backends/memory/mem0.py`
- Modify: `tests/test_mem0_memory_backend.py`

- [ ] **Step 1: Write failing Mem0 scope test**

Add a test showing that:

```python
backend = Mem0MemoryBackend(
    Mem0MemoryConfig(user_id_template="{memory_scope_id}"),
    client=client,
)
request = RetrievalRequest(
    task_id="task-1",
    role="solver",
    query="prior work",
    filters={"memory_scope": "agent", "memory_scope_id": "agent:solver"},
)
bundle = backend.search(request)
```

uses `user_id == "agent:solver"` in search filters.

Add a second assertion with `role="task"` and task filters to verify task scope.

- [ ] **Step 2: Run Mem0 test to verify failure**

Run:

```bash
pytest tests/test_mem0_memory_backend.py::test_scoped_user_id_template_uses_memory_scope_fields -q
```

Expected: FAIL because template validation only allows `task_id` and `role`.

- [ ] **Step 3: Implement scoped template support**

Allow placeholders:

```python
{"task_id", "role", "memory_scope", "memory_scope_id"}
```

For `search(...)`, derive `memory_scope` and `memory_scope_id` from `request.filters`, defaulting to:

- `memory_scope="agent"`
- `memory_scope_id=f"agent:{request.role}"` unless `request.role == "task"`, then `memory_scope="task"` and `memory_scope_id=f"task:{request.task_id}"`

For `add(...)`, derive the same values from `task_id` and `role`.

Pass the derived scope context as Mem0 write metadata:

```python
client.add(mem0_messages, user_id=user_id, metadata=scope_context)
```

This makes scoped reads and scoped writes round-trip through backends that treat search filters as metadata filters.

- [ ] **Step 4: Run Mem0 tests**

Run:

```bash
pytest tests/test_mem0_memory_backend.py -q
```

Expected: PASS.

### Task 4: Full Regression

**Files:**
- No additional file changes expected.

- [ ] **Step 1: Run full suite**

Run:

```bash
pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Check diff**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors and only intended files changed.

- [ ] **Step 3: Commit**

Run:

```bash
git add docs/superpowers/specs/2026-05-04-two-scope-memory-design.md docs/superpowers/plans/2026-05-04-two-scope-memory-implementation.md evolab/config/task_config.py evolab/backends/memory/mem0.py evolab/runtime/task_runtime.py tests/test_task_worker.py tests/test_mem0_memory_backend.py
git commit -m "feat: add agent and task memory scopes"
```

Expected: commit succeeds.
