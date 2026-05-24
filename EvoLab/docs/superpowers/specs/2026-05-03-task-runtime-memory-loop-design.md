# Task Runtime Memory Loop Design

Date: 2026-05-03

## Goal

Implement the first real TaskRuntime subagent execution path for EvoLab V1, focused on Memory participation in task execution. This path should make memory retrieval and post-run memory update observable in normal TaskWorker execution while preserving the architecture boundary that Memory evolves inside TaskWorker, not EvolveWorker.

## Scope

This feature adds a minimal single-subagent execution loop to the existing `TaskRuntime`. It does not implement the full meta-agent dispatch loop, tool-call loop, LLM training, memory consolidation algorithms, or skill graph evolution algorithms.

## Architecture

`TaskRuntime.run(...)` keeps its injectable `dispatch_loop` escape hatch for tests and custom orchestration. When no dispatch loop is injected, the default runtime executes one configured role from `TaskConfig.roles`.

The default path:

1. Selects the first configured role unless a future caller provides richer dispatch context.
2. Builds a `RetrievalRequest` from the task id, role name, task goal, task origin, and task purpose.
3. Calls the configured Memory backend runtime with `memory.search(request)`.
4. Calls the configured Skill backend runtime with `skill.get(request)`.
5. Builds canonical prompt messages through `PromptBuilder`.
6. Calls the role's LLM runtime and accepts only `final_answer` in this minimal implementation.
7. Saves a `SubagentRunRecord` to `TrajectoryRegistry`.
8. Calls `memory.add(task_id, role, prompt_messages + output_messages)` after the run. This is the Memory update/evolving hook for V1.
9. Calls `skill.look_at(...)` after the run with compact run context.
10. Returns a small result dictionary with `task_id`, `run_ref`, `role`, and `final_answer`.

## Memory Boundary

Memory evolution is in-place backend behavior triggered by `MemoryBackend.add(...)` from TaskWorker/TaskRuntime after a subagent run. The EvolveWorker remains responsible only for parameter/LLM state evolution and promotion. No memory evolve jobs are enqueued.

## Traceability

The `MemoryBundle.state_ref` field remains the pre-run memory state/version reference surfaced by the memory backend. The post-run update result is stored in `SubagentRunRecord.metadata["memory_update_result"]` as JSON-compatible data. This keeps the public record extensible without introducing a larger `MemoryUpdateResult` contract before the backend semantics are stable.

Skill post-run observation results are stored similarly in `SubagentRunRecord.metadata["skill_update_result"]`.

## Error Handling

The default runtime fails fast when:

- `TaskConfig` is missing or has no roles.
- The selected role's LLM backend runtime is missing.
- The selected memory or skill backend runtime is missing.
- The LLM returns an action other than `final_answer`.

`TaskWorker.run_once()` already catches runtime exceptions and marks the task job failed.

## Testing

Tests should verify that the default runtime:

- Calls memory search before prompt construction.
- Calls skill retrieval before prompt construction.
- Calls LLM generation with prompt content containing retrieved memory and skills.
- Calls memory add after the LLM returns.
- Calls skill look-at after the LLM returns.
- Saves a `SubagentRunRecord` containing task provenance, retrieval request, memory bundle, skill bundle, prompt messages, output messages, backend ids, state refs, and update metadata.
- Does not touch `EvolveWorker` or enqueue memory evolution jobs.
