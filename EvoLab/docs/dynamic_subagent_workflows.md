# Dynamic Subagent Workflows

EvoLab now supports an optional `dynamic_subagents` workflow mode. This feature does not replace the existing static subagent system.

## Static Mode

Static mode is the existing behavior:

- subagents are defined in YAML
- `MetaAgent` routes to configured roles such as `SurveyAgent`, `ExecAgent`, `CriticAgent`, and `WriteAgent`
- existing static workflow execution, traces, artifacts, registries, and tests remain unchanged

If `dynamic_subagents` is absent, EvoLab behaves as before. If `dynamic_subagents.enabled=false`, EvoLab also behaves as before.

```yaml
task: Process the lab inputs.
meta_agent:
  system_prompt: Return route JSON only.
subagents:
  SurveyAgent:
    system_prompt: Survey available inputs.
  WriteAgent:
    system_prompt: Write final artifacts.
```

Use static mode for stable workflows, regression tests, reproducible dispatch chains, and tasks where the agent roles are already known.

## Dynamic Mode

Dynamic mode is config-gated. A planner LLM generates a validated JSON workflow spec. EvoLab validates the spec, instantiates ephemeral runtime-only subagents, executes the DAG through the existing runtime, and records traces and artifacts.

Dynamic subagents are not written back into the static YAML and are not permanent skill-library entries.

Use dynamic mode when each task or work item may need a different workflow shape, for example a text-only document versus a document with supplementary workbooks.

## Planner And Worker Backends

Dynamic mode uses separate backend roles:

- `planner_backend`: used only to produce `DynamicWorkflowSpec` JSON
- `default_worker_backend`: used by dynamic subagents when `llm_backend_id` is omitted

The planner and worker backend IDs are configured in YAML. They are not hardcoded in EvoLab code.

Recommended cheap planner example: `gpt-4.1-nano`.

Worker example: OpenRouter Qwen30B.

```yaml
backends:
  llm:
    gpt-planner-nano:
      type: api
      api: openai-chat-completions
      api_key_env: OPENAI_API_KEY
      base_url: https://api.openai.com/v1
      model: gpt-4.1-nano
      max_output_tokens: 2048

    openrouter-qwen30b:
      type: api
      api: openai-chat-completions
      api_key_env: OPENROUTER_API_KEY
      base_url: https://openrouter.ai/api/v1
      model: qwen/qwen3-30b-a3b-instruct-2507
      max_output_tokens: 2048

dynamic_subagents:
  enabled: true
  mode: dynamic
  scope: per_work_item
  planner_backend:
    backend_id: gpt-planner-nano
  default_worker_backend:
    backend_id: openrouter-qwen30b
  fallback_to_static: true
  max_subagents_per_work_item: 6
  max_planner_retries: 2
  allow_skill_required_tools: true
  require_output_schema: true
  allowed_tool_names:
    - list_files
    - read_text
    - search_text
    - inspect_file_metadata
    - extract_sections
    - inspect_excel_workbook
    - read_excel_sheet
    - inspect_table
    - read_table_slice
    - detect_table_header
    - normalize_table
    - profile_table
    - json_schema_validate
    - write_jsonl
    - write_report
```

Do not put API keys in YAML. Use environment variables such as `OPENAI_API_KEY` and `OPENROUTER_API_KEY`.

## Scope

`scope: per_task` asks the planner for one workflow for the whole task.

`scope: per_work_item` asks the planner for a separate workflow per work item. Work items can come from task metadata or work-item routing metadata.

## Safety

Planner output is JSON only and is validated against `DynamicWorkflowSpec`.

Validation checks include:

- backend IDs must exist in configured LLM runtimes
- unknown tools are rejected
- tools must be inside `dynamic_subagents.allowed_tool_names`
- dynamic subagent `llm_backend_id` defaults to `default_worker_backend`
- output schemas are required when `require_output_schema=true`
- DAG node and edge references must be valid
- cyclic DAGs are rejected
- skill retrieval goes through `SkillBackend`
- tool execution goes through `ToolRuntime`
- dynamic subagents do not execute arbitrary code
- chain-of-thought fields are rejected

If validation fails and `fallback_to_static=true`, EvoLab records the fallback reason and uses the existing static workflow path. If `fallback_to_static=false`, the task fails clearly with the validation errors.

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

Trajectory registries also continue to receive `SubagentRunRecord`, LLM call records, tool calls, and task events. This keeps dynamic runs consumable by post-run skill evolution and Trace2Skill.

## Example Workflow Shapes

Text-only item:

- `TextContextAgent`
- `EvidenceWriterAgent`

Workbook-style item:

- `TextContextAgent`
- `WorkbookInspectorAgent`
- `TableTriageAgent`
- `EvidenceValidatorAgent`
- `SchemaWriterAgent`

These are examples only. The implementation is domain-generic and does not hardcode article names, biology prompts, promoter rules, or expected outputs.
