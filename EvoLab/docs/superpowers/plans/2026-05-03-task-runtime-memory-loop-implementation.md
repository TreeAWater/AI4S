# Task Runtime Memory Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal default TaskRuntime subagent path that retrieves memory/skills, runs an LLM, triggers TaskWorker-local memory update, records skill observation, and persists a subagent trajectory.

**Architecture:** Keep the existing injectable `dispatch_loop` behavior. Implement the default path inside `TaskRuntime` using current contracts and registries, with helper functions for backend selection, run refs, action validation, and JSON-safe metadata. Memory evolution stays in TaskRuntime via `memory.add(...)`; EvolveWorker remains limited to LLM parameter evolution.

**Tech Stack:** Python 3.11, Pydantic v2 models, pytest, existing filesystem trajectory registry, existing fake/test-local backends.

---

## File Structure

- Modify `tests/test_task_worker.py`: add a runtime integration-style unit test using test-local memory, skill, and LLM fakes.
- Modify `evolab/runtime/task_runtime.py`: implement the minimal default subagent path and small private helpers.
- No change to `evolab/runtime/evolve_worker.py`: memory evolution is not queued or handled there.

### Task 1: Default TaskRuntime Memory Loop

**Files:**
- Modify: `tests/test_task_worker.py`
- Modify: `evolab/runtime/task_runtime.py`

- [ ] **Step 1: Write the failing test**

Add a test that constructs a `TaskRuntime` with:

- one `TaskConfig` role bound to an LLM backend id,
- one test-local memory runtime that records `search` and `add`,
- one test-local skill runtime that records `get` and `look_at`,
- one test-local LLM runtime that records prompt messages and returns `final_answer`,
- a `FileTrajectoryRegistry`.

Assert that `runtime.run(request)` returns the final answer, both pre-run retrieval hooks run, both post-run hooks run, and the saved `SubagentRunRecord` includes memory/skill bundles and post-run update metadata.

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
pytest tests/test_task_worker.py::test_default_task_runtime_runs_memory_skill_llm_and_records_updates -q
```

Expected: FAIL with `NotImplementedError: task dispatch loop is not implemented`.

- [ ] **Step 3: Implement minimal runtime path**

In `evolab/runtime/task_runtime.py`, keep the current `dispatch_loop` branch. For the default branch:

- validate that `task_config` has roles;
- select the first role in insertion order;
- resolve the LLM runtime from `role.llm_backend.backend_id`;
- resolve the first configured memory and skill runtimes;
- build `RetrievalRequest`;
- call `memory.search(...)` and `skill.get(...)`;
- build prompt messages;
- call `llm.generate(prompt_messages, [], LLMGenerationConfig())`;
- require `response.action.action == "final_answer"`;
- build output `Message(role="assistant", content=...)`;
- call `memory.add(...)` and `skill.look_at(...)`;
- save `SubagentRunRecord` with update metadata;
- return `{"task_id": ..., "run_ref": ..., "role": ..., "final_answer": ...}`.

- [ ] **Step 4: Run the focused test to verify it passes**

Run:

```bash
pytest tests/test_task_worker.py::test_default_task_runtime_runs_memory_skill_llm_and_records_updates -q
```

Expected: PASS.

- [ ] **Step 5: Run existing task worker tests**

Run:

```bash
pytest tests/test_task_worker.py -q
```

Expected: PASS.

### Task 2: Full Regression

**Files:**
- No additional file changes expected.

- [ ] **Step 1: Run the full test suite**

Run:

```bash
pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Inspect git diff**

Run:

```bash
git diff -- docs/superpowers/specs/2026-05-03-task-runtime-memory-loop-design.md docs/superpowers/plans/2026-05-03-task-runtime-memory-loop-implementation.md tests/test_task_worker.py evolab/runtime/task_runtime.py
```

Expected: diff is limited to the approved design, plan, focused test, and runtime implementation.

- [ ] **Step 3: Commit**

Run:

```bash
git add docs/superpowers/specs/2026-05-03-task-runtime-memory-loop-design.md docs/superpowers/plans/2026-05-03-task-runtime-memory-loop-implementation.md tests/test_task_worker.py evolab/runtime/task_runtime.py
git commit -m "feat: add task runtime memory loop"
```

Expected: commit succeeds.
