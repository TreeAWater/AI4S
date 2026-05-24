# Tool Execution Loop Design

Date: 2026-05-04

## Goal

Integrate `ToolRuntime.execute()` with `TaskRuntime` so subagent runs can handle LLM `tool_call` actions before producing a final answer.

## Architecture

`TaskRuntime.run()` remains the orchestration point. It still prepares skill-driven tool specs before the first LLM call. After that, it runs a bounded loop controlled by `TaskConfig.runtime_policy.max_tool_steps`.

Each loop iteration calls the configured LLM with the current messages and prepared tool specs. If the response action is `final_answer`, the runtime exits the loop and records the run as before. If the response action is `tool_call`, the runtime passes the returned `ToolCall` to the existing `ToolRuntime.execute()` method, appends the resulting `ToolResult` as a `role="tool"` observation message, and calls the LLM again with the expanded context.

## Error Handling

`ToolRuntime.execute()` already converts missing tools and handler exceptions into `ToolResult(status="error")`, so `TaskRuntime` treats those as normal observations. The LLM receives the error result in the next round and can decide how to recover or finalize.

If the LLM returns a non-tool, non-final action, `TaskRuntime` keeps the existing `NotImplementedError` behavior. If the LLM keeps returning tool calls beyond `max_tool_steps`, `TaskRuntime` raises a clear runtime error.

## Trajectory Metadata

The runtime records every tool call/result pair as a `ToolTrace` in `SubagentRunRecord.metadata["tool_trace"]`. This keeps the public run record shape stable while making tool execution inspectable.

## Artifact Hooks

`TaskRuntime` accepts an optional `tool_artifact_registrar` callback. The callback is invoked when a `ToolResult` contains `artifact_refs`, preserving a narrow extension point for future artifact registration without changing `ToolRuntime.execute()`.
