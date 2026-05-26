# Dynamic Role Self-Evolution Design

## Summary

EvoLab will move from static configured subagents to a dynamic, self-evolving
role pool. The active role pool lives in `agents.md`. MetaAgent owns that pool:
before dynamic planning, it reads the current role pool, task state, LabState,
trajectory feedback, reflector feedback, and stable MetaAgent memory. If it
decides the pool needs a role addition, deletion, or prompt/tool/skill update,
the runtime applies the update directly to `agents.md` without human approval.

Dynamic workflow planning then consumes the updated role pool and creates the
task-specific runtime subagents and workflow DAG. Worker roles no longer have
agent-level memory because role identities are expected to change over time.
Worker execution uses task-level memory only. MetaAgent keeps stable memory for
long-term role-pool evolution.

## Goals

- Make dynamic role generation and workflow planning the default execution path.
- Treat `agents.md` as the active, automatically evolving role pool.
- Allow MetaAgent to directly add, delete, and modify roles in `agents.md`.
- Remove the default static `subagents` execution mode.
- Use task-level memory for all worker subagents.
- Preserve stable MetaAgent memory for long-term routing and role evolution.
- Keep role-pool updates auditable, reversible by history, and protected by
  structural validation.

## Non-Goals

- Human review gates for role updates.
- Per-worker agent memory keyed by dynamic role names.
- Keeping the old V0 behavior where `TaskRuntime` runs static roles in order.
- Promoting biology-specific role identities into stable generic architecture.
- Merging MetaAgent and DynamicWorkflowPlanner into one large planner.

## Architecture

The new main path is:

```text
TaskRequest
-> MetaAgentRolePoolRuntime
   -> read current agents.md
   -> read MetaAgent memory
   -> read LabState, trajectory, reflector feedback, active backend feedback
   -> decide role_pool_update or no_role_pool_update_reason
   -> validate and write agents.md immediately
-> DynamicWorkflowPlanner
   -> read updated role pool templates
   -> produce DynamicWorkflowSpec for task or work item
-> DynamicSubAgentFactory
   -> instantiate runtime worker roles from workflow spec and role templates
-> TaskRuntime dynamic workflow execution
   -> worker roles use task memory only
   -> persist trajectories, artifacts, LabState, and training index
-> Post-run feedback
   -> next MetaAgent role-pool evolution step can consume it
```

Responsibilities:

- MetaAgent manages the long-lived role pool.
- DynamicWorkflowPlanner creates per-task or per-work-item workflows.
- DynamicSubAgentFactory creates runtime-only worker roles.
- Worker roles execute work and do not own long-term identity memory.
- MetaAgent memory stores durable role-pool evolution lessons.
- Task memory stores shared context for worker execution inside one task.

## Configuration

New experiment configs are role-pool centered:

```yaml
task: |
  Natural-language task.

meta_agent:
  prompt_ref: configs/prompts/meta_role_pool_manager.md
  memory_backend: mem0-meta-memory
  llm_backend: aigocode-gpt

agents_ref: configs/agents/scientific_ie_agents.md

dynamic_subagents:
  enabled: true
  mode: dynamic
  scope: per_work_item
  planner_backend:
    backend_id: aigocode-gpt
  default_worker_backend:
    backend_id: aigocode-gpt
  allowed_tool_names:
    - list_files
    - read_text
    - inspect_table
    - write_report

backends:
  llm: {}
  memory:
    mem0-meta-memory: {}
    mem0-task-memory: {}
  skill: {}
```

Config rules:

- `subagents` is removed as the default user-facing execution entry.
- `agents_ref` is the active role-pool reference.
- If a short config omits `agents_ref`, the CLI materializes a seed
  `agents.md` into the Lab config directory.
- `dynamic_subagents.enabled=true` is required for task execution.
- Internal `TaskConfig.roles` may exist only as seed materialization input or
  narrow test fixture data during migration.
- The runtime no longer falls back to "run configured roles in order".

## agents.md Semantics

`agents.md` is the active role pool. It remains Markdown with a fenced JSON
payload parsed by existing agents config helpers. Each role is a reusable
template:

```json
{
  "name": "TableEvidenceTriageAgent",
  "system_prompt": "Inspect table evidence and identify extractable records.",
  "llm_backend": {"backend_id": "aigocode-gpt"},
  "allowed_tools": ["inspect_table", "read_table_slice", "write_report"],
  "required_skills": ["scientific_table_structure_understanding"],
  "metadata": {
    "role_pool_generation": 4,
    "created_by": "meta_agent",
    "created_from_task_id": "task-123",
    "created_from_run_ref": "meta-abc",
    "status": "active",
    "specialization": "table evidence triage"
  }
}
```

Role metadata is for audit, planner selection, and rollback support. It is not
used as a memory scope. Dynamic worker memory does not follow role names.

Every successful write appends `agents.md.updates.jsonl` with:

- task id
- MetaAgent run ref
- before revision
- after revision
- role names added, removed, and modified
- update reason
- validation warnings
- compact update payload

## Role-Pool Evolution Contract

MetaAgent receives a role-pool update contract in its prompt. It may return a
normal routing or planning response with:

```json
{
  "metadata": {
    "role_pool_update": {
      "reason": "Recent runs repeatedly needed table triage before extraction.",
      "roles": {
        "TableEvidenceTriageAgent": {
          "system_prompt": "Inspect table evidence before extraction.",
          "llm_backend": {"backend_id": "aigocode-gpt"},
          "allowed_tools": ["inspect_table", "read_table_slice", "write_report"],
          "required_skills": ["scientific_table_structure_understanding"],
          "metadata": {"specialization": "table evidence triage"}
        },
        "ExecAgent": {
          "system_prompt_append": "When table-heavy inputs are assigned, inspect table metadata before extracting records."
        }
      },
      "remove_roles": ["ObsoleteArticleSurveyAgent"]
    }
  }
}
```

Supported aliases can preserve compatibility during migration:

- `role_pool_update`
- `agent_config_update`
- `agents_update`

The canonical field after migration is `metadata.role_pool_update`.

## Runtime Flow

`TaskRuntime` gains a clear role-pool evolution stage:

```text
_maybe_evolve_role_pool(request, context)
```

This stage runs:

- before dynamic planning starts;
- before recovery replanning when prior dynamic planning or execution produced
  useful feedback;
- after reflector feedback becomes visible to a later task or later planning
  cycle.

The stage:

1. Loads current `agents.md`.
2. Builds a MetaAgent input with task text, LabState, trajectory summaries,
   recent reflector feedback, active evolved backend feedback, MetaAgent
   memory, and current role templates.
3. Asks for `role_pool_update` or `no_role_pool_update_reason`.
4. Validates the proposed role pool.
5. Writes `agents.md` atomically when valid.
6. Appends update history.
7. Records a trajectory event.
8. Updates MetaAgent memory with the decision summary.

The dynamic planner then receives the updated role pool as
`role_pool_templates`. Existing names such as `static_fallback_subagents` should
be renamed because the role pool is no longer a static fallback path.

## Memory Model

Memory has two active layers:

```text
MetaAgent memory:
  scope: agent
  scope_id: agent:<meta_agent.name>
  purpose: durable role-pool evolution lessons

Task memory:
  scope: task
  scope_id: task:<task_id>
  purpose: shared context for all worker roles in one task
```

Worker subagents:

- retrieve only task memory;
- write only task memory;
- do not retrieve or update `agent:<worker-role>`;
- may still see LabState, upstream artifacts, and skill context.

`MethodMemoryBackend` still supports both agent and task scopes because
MetaAgent uses agent scope and because old tests will be migrated gradually.
The main dynamic runtime path no longer binds worker roles to agent memory.

## Validation And Recovery

Role-pool updates are automatic, but writes are guarded:

- MetaAgent output must parse as JSON.
- The proposed `agents.md` must parse through `parse_agents_markdown()`.
- At least one active role must remain.
- Every active role requires `name`, non-empty `system_prompt`, and
  `llm_backend.backend_id`.
- LLM backend ids must exist in configured LLM backends.
- Allowed tools must be registered tools or allowed by dynamic config.
- Required skills may be missing, but missing skills produce warnings.
- Private reasoning fields such as `chain_of_thought`, `reasoning`, and
  `hidden_reasoning` are rejected.
- Failed updates do not overwrite `agents.md`.
- Rejected updates are recorded in trajectory and update history.
- After a bounded number of consecutive invalid update attempts, the task fails
  clearly instead of looping.

Dynamic planning recovery:

- If role-pool update fails but a prior valid pool exists, planning continues
  with the prior pool until the failure limit is reached.
- If dynamic workflow validation fails, the runtime can invoke role-pool
  evolution again with validation errors included in context.
- There is no fallback to static role-order execution.

## Migration Plan

1. Add role-pool evolution runtime around current agents config helpers.
2. Rename dynamic planner prompt input from static fallback subagents to role
   pool templates.
3. Make short config compilation materialize `agents.md` when needed.
4. Require dynamic execution for new clean-run configs.
5. Remove default V0 static role-order branch from `TaskRuntime`.
6. Change dynamic worker execution to task-memory-only.
7. Migrate demo configs from `subagents` to `agents_ref`.
8. Migrate tests from static runtime expectations to dynamic role-pool
   expectations.
9. Keep low-level contract tests for `RoleSpec`, parser helpers, and memory
   backend scopes, but remove them as user-facing static-mode coverage.

## Test Plan

Focused tests:

- Short config without `agents_ref` materializes a seed `agents.md`.
- Short config with `agents_ref` reads the active role pool.
- MetaAgent adds a role and the next planner call sees it.
- MetaAgent edits a role prompt and history records before and after revisions.
- MetaAgent deletes a role, but deleting every role is rejected.
- Invalid backend ids reject the update without overwriting `agents.md`.
- Invalid tool names reject or warn according to dynamic config policy.
- Private reasoning fields reject the update.
- Worker roles retrieve and write only task memory.
- MetaAgent memory still uses `agent:<meta_agent.name>`.
- Dynamic planner consumes `role_pool_templates`.
- Runtime no longer falls back to static role-order execution.
- Bad role-pool update followed by valid existing role pool continues planning
  until the configured failure limit.

Integration tests:

- `clean-run` dynamic CI demo completes from a fresh Lab.
- Biology config runs through dynamic role pool and per-work-item planning.
- Role-pool updates are visible in trajectory and `agents.md.updates.jsonl`.
- Memory replay sees task memory updates without worker agent-scope updates.
- Full `pytest -q` passes after migration.

## Compatibility Notes

Existing `subagents` configs will not remain a supported default execution
mode. During migration, internal helpers may translate old inline roles into a
seed `agents.md` so tests can be converted incrementally. That translation is
not a stable public contract.

The memory backend contracts remain broader than the new main path. Agent scope
continues to exist for MetaAgent memory, while dynamic worker agent memory is
removed from the default execution path.
