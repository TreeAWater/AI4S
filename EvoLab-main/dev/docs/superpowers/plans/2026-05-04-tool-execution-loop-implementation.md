# Tool Execution Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal `TaskRuntime` tool-call loop using the existing `ToolRuntime.execute()` implementation.

**Architecture:** Keep tool execution in `ToolRuntime`; `TaskRuntime` only orchestrates LLM responses, tool observations, and trajectory metadata. Tool observations are stored as `Message(role="tool")`, and call/result metadata is stored as a `ToolTrace`.

**Tech Stack:** Python 3.11-style type hints, existing Pydantic contracts, pytest.

---

## File Structure

- Modify: `evolab/runtime/task_runtime.py`
- Modify: `tests/test_task_worker.py`
- Create: `docs/superpowers/specs/2026-05-04-tool-execution-loop-design.md`
- Create: `docs/superpowers/plans/2026-05-04-tool-execution-loop-implementation.md`

## Task 1: Red Tests

- [x] Add a failing test for one LLM tool call followed by a final answer.
- [x] Assert the tool handler is called through `ToolRuntime.execute()`.
- [x] Assert unregistered tools produce an error observation in the next LLM context.
- [x] Assert handler exceptions produce an error observation in the next LLM context.
- [x] Assert trajectory metadata contains the tool call/result trace.

## Task 2: Runtime Loop

- [x] Replace the single LLM generation branch with a bounded loop.
- [x] Execute `SubAgentAction.tool_call` through `ToolRuntime.execute()`.
- [x] Append `ToolResult` as a `role="tool"` observation for the next LLM call.
- [x] Preserve existing final-answer behavior.

## Task 3: Metadata And Artifact Hook

- [x] Store tool call/result records as `ToolTrace` in trajectory metadata.
- [x] Add a narrow optional `tool_artifact_registrar` callback for future `ToolResult.artifact_refs`.

## Task 4: Verification

- [x] Run targeted tool loop tests.
- [x] Run full targeted runtime/tool/skill tests.
- [x] Run `pytest -q`.
- [x] Run `git diff --check`.
