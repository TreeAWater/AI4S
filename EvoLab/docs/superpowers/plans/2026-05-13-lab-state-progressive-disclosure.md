# Lab State Progressive Disclosure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace scattered Lab state consumption with a coherent Lab object model and progressive-disclosure input for MetaAgent routing.

**Architecture:** LabStateRegistry stores curated Lab objects: run ledgers, subagent reports, artifact indexes, training indexes, evolution products, state indexes, and prompt-safe digests. TaskWorker and TaskRuntime keep existing trajectory registries as the raw trace pool while also writing the higher-level Lab state objects. MetaAgent receives only an index/digest by default and can request detailed refs in later routing metadata.

**Tech Stack:** Python, Pydantic contracts, file-backed registries, pytest.

---

### Task 1: Wire LabState Registry

**Files:**
- Modify: `evolab/lab/layout.py`
- Modify: `evolab/lab/resolver.py`
- Test: `tests/test_lab_queue.py`

- [x] Add `registries/lab_state` to the standard Lab layout.
- [x] Add `LabResolver.lab_state_registry()`.
- [x] Verify layout/resolver tests pass.

### Task 2: Runtime Lab Object Writes

**Files:**
- Modify: `evolab/runtime/task_worker.py`
- Modify: `evolab/runtime/task_runtime.py`
- Test: `tests/test_task_worker.py`

- [ ] Inject `FileLabStateRegistry` into TaskWorker and TaskRuntime.
- [ ] Save a `RunLedgerRecord` when a queue task starts, completes, or fails.
- [ ] Save a `SubagentReportRecord` whenever a subagent returns a completed or failed result.
- [ ] Index produced artifacts with `ArtifactIndexRecord`.
- [ ] Index LLM traces with `TrainingIndexRecord`.

### Task 3: Lab Index And Digest Builder

**Files:**
- Create: `evolab/runtime/lab_state.py`
- Modify: `evolab/runtime/task_runtime.py`
- Test: `tests/test_task_worker.py`

- [ ] Build `LabStateIndex` from task registry, trajectory registry, backend states, LabStateRegistry, and recent queue status.
- [ ] Build `LabStateDigest` as a compact human-readable summary with detail refs.
- [ ] Persist index/digest snapshots before each MetaAgent dispatch.
- [ ] Replace count-only MetaAgent `lab_state` with `{index, digest, requested_details}`.

### Task 4: Progressive Detail Requests

**Files:**
- Modify: `evolab/runtime/task_runtime.py`
- Test: `tests/test_task_worker.py`

- [ ] Accept `lab_state_detail_requests` in MetaAgent decision metadata.
- [ ] Resolve requested refs from reports, artifacts, training samples, trajectories, and backend states.
- [ ] Include resolved details in the next MetaAgent input.
- [ ] Record requested refs in MetaAgent trajectory metadata.

### Task 5: Documentation And Verification

**Files:**
- Modify: `docs/configuration.md`
- Modify: `docs/implementation_plan.md`
- Possibly modify: `docs/v1_release_checklist.md`

- [ ] Document the new Lab object model.
- [ ] Document progressive disclosure for MetaAgent.
- [ ] Run focused tests and relevant broader runtime tests.
