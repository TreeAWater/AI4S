# Dynamic Subagent Workflows

Dynamic workflow planning is EvoLab's default execution model for new configs.
EvoLab no longer treats top-level `subagents` as the user-facing entry point.
The active role pool is `agents.md`; MetaAgent may update this file
automatically, and DynamicWorkflowPlanner consumes the latest role templates for
each run.

Dynamic worker roles use task-level memory only. MetaAgent may keep stable
agent-scope memory through `meta_agent.memory_backend`, normally
`mem0-meta-memory`.

## Config Shape

Configs point to a role pool and enable dynamic planning:

```yaml
task: Process the lab inputs and write final artifacts.
agents_ref: agents/scientific_ie_agents.md
meta_agent:
  memory_backend: mem0-meta-memory
  system_prompt: Return route JSON only.
dynamic_subagents:
  enabled: true
  mode: dynamic
  scope: per_task
  planner_backend:
    backend_id: aigocode-gpt
  default_worker_backend:
    backend_id: aigocode-gpt
  allowed_tool_names:
    - read_text
    - json_schema_validate
    - write_jsonl
    - write_report
```

For configs inside `configs/`, `agents_ref` is resolved relative to the config
file directory. Use `agents/scientific_ie_agents.md`, not
`configs/agents/scientific_ie_agents.md`.

## Planner And Workers

`planner_backend` produces a validated `DynamicWorkflowSpec` JSON document.
`default_worker_backend` runs dynamic workers when a generated worker does not
name its own backend. Backend IDs must be configured under `backends.llm`.

The planner receives the latest role-pool templates from `agents.md`. Generated
workers are runtime-only instances derived from those templates; they are not
written back as top-level YAML `subagents`. Reusable role improvements should be
applied to `agents.md` by MetaAgent role-pool evolution.

Dynamic planners receive `effective_allowed_tool_names`, which is the configured
built-in allowlist plus validated task-local generated tools. Planner output
must still be structural JSON only; executable Python belongs to the generated
tool preplanning or repair stages, not workflow specs.

## Scope

`scope: per_task` asks the planner for one workflow for the whole task.

`scope: per_work_item` asks the planner for a separate workflow per work item.
Work items can come from task metadata or work-item routing metadata.

## Safety

Planner output is JSON only and is validated against `DynamicWorkflowSpec`.

Validation checks include:

- backend IDs must exist in configured LLM runtimes
- tools must be inside `dynamic_subagents.allowed_tool_names`
- dynamic worker `llm_backend_id` defaults to `default_worker_backend`
- output schemas are required when `require_output_schema=true`
- DAG node and edge references must be valid
- cyclic DAGs are rejected
- skill retrieval goes through `SkillBackend`
- tool execution goes through `ToolRuntime`
- dynamic workers do not execute arbitrary code
- chain-of-thought fields are rejected

If validation fails, EvoLab records the dynamic planning failure and the task
continues to report a dynamic failed result. It does not switch to the old
non-dynamic role-order runtime.

## Observability

Dynamic workflow records are written under the lab root:

```text
dynamic_workflows/<task_id>/<workflow_id>/
  dynamic_workflow_spec.json
  dynamic_subagents.json
  dynamic_workflow_trace.json
  dynamic_subagent_records.jsonl
  planner_validation_report.json
  fallback_reason.json
```

Trajectory registries also receive `SubagentRunRecord`, LLM call records, tool
calls, memory lineage, and task events. This keeps dynamic runs consumable by
post-run skill evolution, memory replay, and Trace2Skill.
