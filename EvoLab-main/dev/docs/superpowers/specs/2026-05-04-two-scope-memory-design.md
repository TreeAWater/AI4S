# Two-Scope Memory Design

Date: 2026-05-04

## Goal

Separate EvoLab task execution memory into two explicit scopes:

- **Agent-wise memory**: persistent memory for one configured subagent role.
- **Task-wise memory**: shared memory for one task execution.

Every subagent run should read both scopes before generation and write both scopes after generation. Memory evolution remains in TaskWorker / TaskRuntime through `MemoryBackend.add(...)`; EvolveWorker remains responsible only for parameter / LLM state evolution.

## Current State

The current `TaskRuntime` uses a single memory runtime per subagent run. `Mem0MemoryBackend` defaults to `user_id_template="{task_id}:{role}"`, which produces a mixed task-role scope. That makes memory persist across repeated calls to the same role within one task, but it does not model:

- agent role memory whose lifecycle matches the role lifecycle;
- task memory shared across all roles in the task;
- simultaneous read/write of both memories per subagent run.

## Scope Semantics

### Agent-Wise Memory

Agent-wise memory belongs to a role such as `solver`, `surveyor`, or `writer`. Its lifecycle matches the configured role lifecycle. When the meta-agent routes work to the same role multiple times, that role should continue from its prior agent memory instead of receiving a fresh backend state for each route.

The default agent memory scope id is:

```text
agent:<role>
```

### Task-Wise Memory

Task-wise memory belongs to a task id. It is shared by every role executing inside that task. It stores task context, intermediate findings, cross-agent coordination state, and task-local summaries.

The default task memory scope id is:

```text
task:<task_id>
```

## Configuration

Extend config contracts without removing existing behavior:

- `RoleSpec.agent_memory_backend: BackendBinding | None`
- `TaskConfig.task_memory_backend: BackendBinding | None`

The runtime should use these explicit bindings when provided. For backward compatibility, if no explicit memory bindings exist, the runtime may fall back to the first configured memory runtime for both scopes only in the existing minimal default path.

Partial explicit configuration is invalid. If only `agent_memory_backend` or only `task_memory_backend` is configured, the runtime should fail fast instead of routing the missing side to an arbitrary memory backend.

## Runtime Flow

For each subagent run:

1. Resolve role and LLM runtime as today.
2. Resolve agent memory runtime from `role.agent_memory_backend`.
3. Resolve task memory runtime from `task_config.task_memory_backend`.
4. Build two retrieval requests:
   - agent request: `role=<role>`, `filters["memory_scope"]="agent"`, `filters["memory_scope_id"]="agent:<role>"`
   - task request: `role="task"`, `filters["memory_scope"]="task"`, `filters["memory_scope_id"]="task:<task_id>"`
5. Search both memories.
6. Build prompt with clearly separated agent memory and task memory content.
7. Run the LLM.
8. Write the run messages to agent memory using `agent_memory.add(task_id, role, messages)`.
9. Write the run messages to task memory using `task_memory.add(task_id, "task", messages)`.
10. Save both memory bundles and both update results into trajectory metadata.

If agent and task memory bindings point to the same backend object, the runtime still makes two separate calls with different scope roles / filters so the backend can isolate storage.

## Mem0 Scope Mapping

`Mem0MemoryBackend` should support a `user_id_template` with `memory_scope` and `memory_scope_id` placeholders in addition to `task_id` and `role`.

Recommended templates:

```text
{memory_scope_id}
```

or:

```text
{memory_scope}:{memory_scope_id}
```

The runtime supplies scope values through `RetrievalRequest.filters`. `Mem0MemoryBackend.add(...)` currently receives only `task_id`, `role`, and messages, so it should derive scope from the role convention used by the runtime:

- role equal to the subagent role writes agent memory;
- role equal to `"task"` writes task memory.

`Mem0MemoryBackend.add(...)` should pass scope metadata to Mem0 writes so later searches with `memory_scope` and `memory_scope_id` filters can retrieve the same records. This keeps the public `MemoryBackend` interface stable for this step.

## Trajectory Records

Keep the existing `SubagentRunRecord.memory_bundle` field as the combined memory visible to prompt construction. Store detailed two-scope traceability in metadata:

- `agent_memory_bundle`
- `task_memory_bundle`
- `agent_memory_update_result`
- `task_memory_update_result`

The combined `memory_bundle` should preserve both sets of items and include metadata describing the contributing scope bundle state refs.

## Out Of Scope

- Full meta-agent dispatch loop.
- Memory summarization or consolidation algorithms.
- New EvolveWorker behavior.
- Backend-private memory deletion, compaction, or promotion policies.

## Testing

Tests should verify:

- Config contracts accept `agent_memory_backend` and `task_memory_backend`.
- `TaskRuntime` searches agent and task memory before LLM generation.
- Prompts contain separate agent and task memory sections.
- `TaskRuntime` writes both memory scopes after LLM generation.
- `SubagentRunRecord.metadata` records both bundles and both update results.
- `EvolveWorker` remains untouched by memory behavior.
