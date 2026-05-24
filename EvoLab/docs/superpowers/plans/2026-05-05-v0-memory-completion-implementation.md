# V0 Memory Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete ziyi-owned V0 Memory work: deterministic fake memory backend, two-scope memory state lineage records, focused tests, and V0 Memory notes.

**Architecture:** Keep memory evolution inside `TaskRuntime` / `TaskWorker` through `MemoryBackend.add(...)`. Agent-wise and task-wise memory remain separate scopes; each subagent run reads and writes both scopes, then records any returned memory `state_ref` into `BackendStateRegistry` as memory lineage. Fake memory provides deterministic in-process scoped storage for V0 demos and CI.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest, filesystem Lab registries.

---

### Task 1: Fake Memory Backend And Contracts

**Files:**
- Modify: `evolab/contracts/retrieval.py`
- Modify: `evolab/backends/memory/base.py`
- Create: `evolab/backends/memory/fake.py`
- Modify: `evolab/backends/memory/__init__.py`
- Create: `tests/test_fake_memory_backend.py`

- [ ] **Step 1: Write failing tests for fake memory export and deterministic scope behavior**

Add tests that import `FakeMemoryBackend`, instantiate it with agent/task seed records, call `search(...)` for `agent:<role>` and `task:<task_id>`, call `add(...)`, and assert:
- scoped searches only return the matching scope;
- `MemoryBundle.state_ref` changes after an add;
- the update result includes `status="updated"`, `previous_state_ref`, `state_ref`, `memory_scope`, and `memory_scope_id`;
- `instantiate(state_ref)` records requested state refs and returns a runtime.

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/test_fake_memory_backend.py
```

Expected: FAIL because `FakeMemoryBackend` is not implemented.

- [ ] **Step 2: Add lightweight memory update contracts**

In `evolab/contracts/retrieval.py`, add:

```python
BackendScope = Literal["agent", "task"]

class MemoryAddRequest(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    task_id: str
    role: str
    messages: list[Message]
    filters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

class MemoryUpdateResult(StrictBaseModel):
    schema_version: Literal["v1"] = "v1"
    status: Literal["updated", "skipped", "failed", "degraded"]
    state_ref: str | None = None
    previous_state_ref: str | None = None
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

Import `Message` from `evolab.contracts.common`. Keep the existing `MemoryBackend.add(task_id, role, messages)` method stable so Mem0 and runtime callers do not need a disruptive interface migration before V0.

- [ ] **Step 3: Implement deterministic fake backend**

Create `evolab/backends/memory/fake.py` with a `FakeMemoryBackend(MemoryBackend)` class. It should:
- store records keyed by `memory_scope_id`;
- derive `memory_scope` and `memory_scope_id` from `RetrievalRequest.filters`, falling back to role conventions (`task` role maps to `task:<task_id>`, other roles map to `agent:<role>`);
- return `MemoryBundle(state_ref=...)` with deterministic refs like `fake-memory://<backend_id>/<scope_id>/v<N>`;
- append one deterministic `MemoryItem` on `add(...)`, using the last non-empty assistant/user/system/tool message content;
- return `MemoryUpdateResult(status="updated", previous_state_ref=..., state_ref=..., metadata={...})`;
- support `instantiate(state_ref)` and record `instantiated_state_refs`.

- [ ] **Step 4: Export fake backend**

Update `evolab/backends/memory/__init__.py` to export `FakeMemoryBackend`.

- [ ] **Step 5: Run focused and full tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/test_fake_memory_backend.py tests/test_mem0_memory_backend.py
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider
```

Expected: all tests pass.

---

### Task 2: Memory State Lineage In Task Runtime

**Files:**
- Modify: `evolab/runtime/task_runtime.py`
- Modify: `evolab/runtime/task_worker.py`
- Modify: `tests/test_task_worker.py`

- [ ] **Step 1: Write failing runtime lineage test**

Add a test using `FileBackendStateRegistry`, explicit `agent_memory_backend` and `task_memory_backend`, and existing `RecordingMemoryRuntime` update results. After `TaskRuntime.run(...)`, assert two memory `BackendStateRecord`s exist:
- one for agent memory with `backend_id="agent-memory"`, `state_ref="agent-state-after"`, `parent_state_refs=["agent-state-before"]`;
- one for task memory with `backend_id="task-memory"`, `state_ref="task-state-after"`, `parent_state_refs=["task-state-before"]`;
- both records have `backend_type="memory"`, `created_from_task_id`, `created_from_run_ref`, and scope metadata.

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/test_task_worker.py::test_default_task_runtime_records_memory_state_lineage
```

Expected: FAIL because `TaskRuntime` does not write memory state records yet.

- [ ] **Step 2: Pass backend state registry into default runtime**

Add optional `backend_state_registry` to `TaskRuntime.__init__`. In `TaskWorker.startup()`, pass `self.backend_state_registry` when constructing `TaskRuntime`.

- [ ] **Step 3: Record memory update lineage after add**

After both memory `add(...)` calls and before saving `SubagentRunRecord`, call a helper for each scope:

```python
_record_memory_state_update(
    registry=self.backend_state_registry,
    task_id=request.task_id,
    run_ref=run_ref,
    role=role.name,
    memory_scope="agent",
    memory_scope_id=f"agent:{role.name}",
    memory_bundle=agent_memory_bundle,
    update_result=agent_memory_update_result,
)
```

The helper should accept either Pydantic models or dict update results. If no `state_ref` is present, it should do nothing for V0. If `artifact_refs` are present, validate them into `ArtifactRef` objects. Register `BackendStateRecord(backend_type="memory", active=True, parent_state_refs=[previous state ref if present], metadata={...})`.

- [ ] **Step 4: Run focused and full tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/test_task_worker.py::test_default_task_runtime_records_memory_state_lineage
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider
```

Expected: all tests pass.

---

### Task 3: Memory V0 Notes

**Files:**
- Create: `docs/memory.md`

- [ ] **Step 1: Write concise V0 Memory documentation**

Create `docs/memory.md` covering:
- ownership boundary: memory evolves only through TaskWorker/TaskRuntime post-run `add(...)`; EvolveWorker is for parameter evolution;
- agent-wise vs task-wise scope semantics and default scope IDs;
- config fields `RoleSpec.agent_memory_backend` and `TaskConfig.task_memory_backend`;
- fake backend behavior and deterministic state refs;
- Mem0 adapter status and recommended `user_id_template="{memory_scope_id}"`;
- V0 known issues for V1: richer failure suite, replay checks, demo_v1 integration, and external Mem0 service verification.

- [ ] **Step 2: Run docs-adjacent smoke tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/test_fake_memory_backend.py tests/test_task_worker.py tests/test_mem0_memory_backend.py
```

Expected: all tests pass.

---

### Task 4: Final Review And Verification

**Files:**
- Review all files changed by Tasks 1-3.

- [ ] **Step 1: Run full validation**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider
git diff --check
```

Expected: full test suite passes and diff whitespace check is clean.

- [ ] **Step 2: Review scope**

Confirm:
- `EvolveWorker` memory behavior is unchanged;
- no real Mem0 network call is required by tests;
- `docs/implementation_plan.md` is not overwritten;
- unrelated user-created files are not removed.
