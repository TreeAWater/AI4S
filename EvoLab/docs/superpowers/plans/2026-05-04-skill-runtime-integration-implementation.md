# Skill Runtime Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect skill graph retrieval with runtime tool preparation and prompt-injectable skill context.

**Architecture:** Add `evolab/runtime/skill_retrieval.py` as the integration helper. Keep `ToolRuntime.prepare()` filtering semantics unchanged and enforce missing required tool errors in the helper. Extend `PromptBuilder` with an optional `skill_context` parameter and wire the helper into `TaskRuntime`.

**Tech Stack:** Python 3.11-style type hints, Pydantic v2 contracts, pytest.

---

## File Structure

- Create: `evolab/runtime/skill_retrieval.py`
- Modify: `evolab/runtime/prompt_builder.py`
- Modify: `evolab/runtime/task_runtime.py`
- Add: `tests/test_skill_runtime_integration.py`
- Modify: `tests/test_task_worker.py`

## Task 1: Integration Helper Tests

- [x] Write a failing test for request-to-skill retrieval with a real `GraphSkillBackend`.
- [x] Assert required tools are aggregated and prepared into tool specs.
- [x] Assert missing required tools raise a clear `MissingRequiredToolError`.
- [x] Assert `skill_context` contains selected skills, tree paths, graph summary, and required tools.

## Task 2: Helper And Prompt Implementation

- [x] Add `PreparedSkillRuntimeContext`.
- [x] Add `prepare_skill_runtime_context(...)`.
- [x] Add `build_skill_context(...)`.
- [x] Extend `PromptBuilder.build(...)` with optional skill context JSON.

## Task 3: TaskRuntime Integration

- [x] Add a failing `TaskRuntime` test proving prepared tool specs reach the LLM and prompt context is recorded.
- [x] Wire `TaskRuntime.run()` to `prepare_skill_runtime_context(...)`.
- [x] Persist `tool_bundle` and `skill_context` in subagent run metadata.

## Task 4: Verification

- [x] Run targeted integration/runtime tests.
- [x] Run `pytest -q`.
- [x] Run `git diff --check`.
