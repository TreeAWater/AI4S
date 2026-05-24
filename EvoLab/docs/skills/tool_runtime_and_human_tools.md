# Tool Runtime And Human Tools

EvoLab v1 uses one tool execution system:

- `ToolRegistry` stores `ToolSpec` entries and handlers.
- `ToolRuntime.prepare(...)` exposes registered, allowed tools for a run.
- `ToolRuntime.execute(...)` executes prepared tool calls and returns `ToolResult`.
- `TaskRuntime` records `ToolCallRecord`, `ToolTrace`, and artifact refs.

There is no parallel scientific IE tool runtime.

## Required And Optional Tools

Reusable skills declare required tools in their skill package metadata. `prepare_skill_runtime_context(...)` aggregates:

- `SkillBundle.required_tools`
- each selected skill's `required_tools`

Missing required tools raise `MissingRequiredToolError`.

Optional tools can be exposed through `ToolRuntime.prepare(..., optional_tools=...)`. Missing optional tools do not fail preparation.

## Human Tools

Human participation is tool-mediated:

- `ask_human`
- `request_human_review`
- `notify_human`

Each has `metadata.requires_human = true`. They are hidden when `RuntimePolicy.allow_human_tools = false` and exposed only when registered, role-allowed, and policy-enabled.

The mock adapter is deterministic and intended for tests/demos. Production adapters can route to real collaboration systems without changing the tool trace contract.

## Scientific IE Tool Registration

Use:

```python
from evolab.tools.scientific_ie import register_scientific_ie_tools

register_scientific_ie_tools(registry, artifact_root="artifacts", base_dir="lab/demo_v1")
```

This registers the generic v1 tools required by the scientific IE reusable skills. The tools operate on local files and simple fixtures. They do not run shell commands and do not implement domain-specific extraction logic.

`TaskRuntime` sends the LLM a tool observation whose content includes the
human-readable result plus JSON-formatted metadata for successful tools. This
keeps retrieved text, table previews, validation reports, paths, and artifact
refs visible to the next LLM turn while preserving the full `ToolResult` in
trajectory metadata.
