# Human Tools Runtime

EvoLab v1 represents human participation as normal tools exposed to agents. Human review is not a mandatory workflow node.

## Tools

`evolab/tools/human.py` provides:

- `ask_human`
- `request_human_review`
- `notify_human`

Register them with:

```python
from evolab.tools.human import register_human_tools

register_human_tools(registry)
```

Each tool has `metadata.requires_human = true`.

## Runtime Policy

Human tools are policy-controlled:

- `RuntimePolicy.allow_human_tools`
- `RuntimePolicy.allowed_human_tools`
- `RuntimePolicy.max_human_requests_per_run`
- `RuntimePolicy.human_tool_mock_mode`

`prepare_skill_runtime_context(...)` passes allowed human tools as optional tools when human tools are enabled. They are not added to `SkillBundle.required_tools`, and missing optional human tools do not fail skill preparation.

`ToolRuntime.prepare(...)` exposes human tools only when they are registered, role-allowed, and policy-enabled.

`ToolRuntime.execute(...)` records human tool calls through the same `ToolCallRecord` and `ToolTrace` path as every other tool call.

## Mock Adapter

`MockHumanToolAdapter` provides deterministic local behavior:

- `ask_human` returns `MOCK_HUMAN_RESPONSE: proceed with conservative extraction.`
- `request_human_review` returns `approved`, or `needs_revision` when instructions mention conflict
- `notify_human` returns `delivered=true`

This is intended for tests and local demos. Production human routing can provide a different adapter while keeping the same tool specs and trace path.

Human tools are not required by stable scientific IE skills. `prepare_skill_runtime_context(...)` exposes them as optional tools when allowed by policy, so an agent can ask for human input during a workflow node without turning human review into a mandatory DAG node.

