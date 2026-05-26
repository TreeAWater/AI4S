# Workflow Planning Runtime

The v1 execution chain has two levels.

At the MetaAgent level:

```text
Natural-language task config
-> agents_ref role pool
-> DynamicWorkflowPlanner
-> runtime dynamic workers
-> final dynamic workflow result
```

At the selected SubAgent level:

```text
RetrievalRequest
-> SkillBundle
-> WorkflowPlan
-> ToolBundle
-> NodeExecutionRecord / ToolTrace / Artifacts
-> PlanExecutionTrace
-> skill.look_at(...)
```

## Contracts

Workflow contracts live in `evolab/contracts/workflow.py`:

- `WorkflowPlan`
- `WorkflowNode`
- `WorkflowEdge`
- `NodeExecutionRecord`
- `PlanExecutionTrace`

These contracts are lightweight and JSON-serializable. They describe a
skill-level DAG inside one runtime worker. The dynamic workflow level is
planned from the active role pool in `agents_ref`; generated workers are
runtime instances, not top-level config entries.

## MetaAgent Routing

New experiment configs point `agents_ref` at reusable role-pool templates and
enable `dynamic_subagents`. MetaAgent receives the natural-language task,
available role-pool templates, current Lab state, completed run summaries, and
trajectory/failure context.

It returns JSON only:

```json
{"route":"END","instruction":"Role pool reviewed.","metadata":{"no_role_pool_update_reason":"No reusable update needed."}}
```

or:

```json
{"route":"END","instruction":"Final artifacts are complete.","metadata":{"final_answer":"..."}}
```

The runtime validates dynamic workflow planner output against the active role
pool, configured LLM backends, and `dynamic_subagents.allowed_tool_names`. It
does not force a fixed Survey -> Design -> Exec -> Critic -> Write chain. The
planner may select the subset of role-pool templates needed for the current task
and may generate recovery workers when coverage or artifacts are incomplete.

## Planner

`SkillWorkflowPlanner` builds one `WorkflowNode` per selected skill in a `SkillBundle`.

Planning uses, in order:

1. retrieval relationship metadata such as `depends_on`, `requires`, `consumes`, `prerequisite`, `validates`, `produces`, `related_to`, and `complements`
2. simple required-input / expected-output overlap
3. deterministic scientific IE phase order

Direction rules:

- `A depends_on B`, `A requires B`, `A consumes B`, or `A prerequisite B` becomes `B -> A`.
- `A validates B` becomes `B -> A`.
- `A produces B` becomes `A -> B`.
- weak relationship edges are oriented by phase order when possible.

Cycle handling removes optional, inferred, or phase-order edges before mandatory edges. Cycle warnings are recorded in `WorkflowPlan.metadata.planning_warnings`.

## Runtime

Workflow execution is controlled by `RuntimePolicy.enable_workflow_planning`.

When disabled, the existing flat runtime path is unchanged.

When enabled, `TaskRuntime`:

1. retrieves skills with the existing `prepare_skill_runtime_context(...)`
2. builds a `WorkflowPlan`
3. prepares tools for each workflow node or selected skill
4. adds the plan to skill context
5. executes nodes in topological order
6. uses the existing `ToolRuntime.execute(...)` for all tool calls
7. records node outputs, tool traces, artifacts, and plan trace metadata
8. sends the full observation to `skill.look_at(...)`

The runtime expects required tools from the selected `SkillBundle` to be
registered before execution. Tool preparation is scoped to the current role and,
when workflow planning is enabled, to the current internal workflow node. For
scientific IE tasks, use `register_scientific_ie_tools(...)` to register the v1
generic local implementations.

Limits:

- `RuntimePolicy.max_workflow_nodes`
- `RuntimePolicy.max_tool_steps_per_node`
- `RuntimePolicy.max_tool_steps`

## Required Tools

`GraphSkillBackend` does not execute tools. It returns selected skills and their `required_tools`.

`prepare_skill_runtime_context(...)` aggregates required tools from the bundle and selected skills, then calls `ToolRuntime.prepare(...)`. Missing required tools still raise `MissingRequiredToolError`.

Scientific IE v1 required tools are implemented as simple local handlers under `evolab/tools/`:

- file and document tools
- table and spreadsheet tools
- schema and output tools
- optional human tools

These handlers are deterministic test/demo implementations. Production sandboxing and remote artifact services are future work.
