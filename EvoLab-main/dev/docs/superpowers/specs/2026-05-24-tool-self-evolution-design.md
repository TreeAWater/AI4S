# Tool Self-Evolution Design

## Summary

EvoLab will support self-evolving tools by letting an LLM generate executable
Python tool code for a specific task. The tool system is split into two layers:

- Built-in generic tools are defined by EvoLab and registered through the normal
  `ToolRegistry`. Examples include file, text, table, schema, output, human, and
  scientific artifact tools.
- Task-specialized tools are generated as Python packages during a task. They
  are loaded, validated, registered, traced, and reset by the runtime. They are
  allowed to overfit to the current task.

The generated tool layer is not a Markdown-only registry. A generated tool is a
real Python module with a `ToolSpec`, handler entrypoint, validation report, and
provenance. It is task-local by default and does not mutate the global built-in
tool registry. Global promotion requires a separate explicit promotion design.

## Goals

- Let MetaAgent and runtime repair create new Python tools directly through an
  LLM.
- Reset the specialized tool set at the start of every new task.
- Preserve generated tools across dynamic workflow nodes inside the same task.
- Make generated tools available to DynamicWorkflowPlanner after validation.
- Keep all tool calls going through `ToolRuntime.prepare()` and
  `ToolRuntime.execute()`.
- Record generated source code, validation outcomes, code hashes, tool calls,
  failures, retries, and promotion candidates in task-level artifacts and
  trajectory events.
- Keep worker memory task-level; generated tools do not create tool-level or
  agent-level memory scopes.

## Non-Goals

- A Markdown-only `tools.md` registry as the main mechanism.
- Human review before task-local generated tools can be used.
- Persisting task-generated tools into reusable roles in `agents.md`.
- Letting DynamicWorkflowPlanner return executable code inside workflow specs.
- Treating process-level execution as a complete adversarial sandbox. The first
  implementation is for autonomous trusted research runs with validation,
  budgets, provenance, and task locality.

## Architecture

The main flow is:

```text
TaskRequest
-> reset task-generated tool registry
-> register built-in generic tools
-> MetaAgent tool-code evolution preplanning
   -> inspect task, built-in tool catalog, role pool, LabState, feedback
   -> return generated Python tool package or no-op reason
   -> persist source under task artifacts
   -> validate manifest, code shape, importability, smoke tests
   -> register generated tools in task-local generated registry
-> MetaAgent role-pool evolution
   -> update agents.md for reusable roles only
   -> reject persistent roles that depend on task-local generated tools
-> DynamicWorkflowPlanner
   -> consume updated role pool and effective tool catalog
   -> assign built-in and generated tools to runtime subagents
-> Dynamic workflow execution
   -> workers call tools through ToolRuntime
   -> failure repair can generate more task-local Python tools
   -> generated tools become available to retries or replanning in this task
-> task closes
   -> task-generated registry is discarded
   -> source/provenance remain as task artifacts
```

`ToolRuntime` gains a distinct task-local generated registry. The current
`_runtime_specs` and `_runtime_handlers` are reset by `prepare()`, so they are
not suitable for task-wide generated tools. The generated registry is keyed by
`task_id`; `TaskRuntime` activates exactly one task scope before preplanning,
planning, repair, and worker execution. If a caller tries to switch task scopes
on a shared `ToolRuntime` while generated tools are active, the runtime must
either reset the old task scope first or reject the switch. Parallel or
interleaved task execution must use separate `ToolRuntime` instances until a
multi-task keyed execution context is implemented end to end.

The new effective lookup order inside the active task scope is:

```text
per-prepare runtime overlays
-> task-local generated tools
-> built-in registry tools
```

`prepare()` continues to reset per-run prepared state, but it does not erase
task-local generated tools. New tasks call an explicit reset method before any
MetaAgent or workflow execution starts.

The effective tool catalog is a runtime object, not a persistent config file:

```text
TaskEffectiveToolCatalog
  task_id
  builtin_allowed_tool_names
  generated_tool_names
  effective_allowed_tool_names
  tool_specs_by_name
  provenance_by_name
```

`dynamic_subagents.allowed_tool_names` remains the configured built-in seed
allowlist. After generated tools validate, `TaskRuntime` builds
`effective_allowed_tool_names = builtin_allowed_tool_names +
generated_tool_names` and passes that catalog to the planner and workflow
validator. The config object does not need to be mutated on disk.

## Generated Tool Package

The LLM returns a structured generated tool package. The runtime accepts either
a single-file or multi-file package, but every package must include one primary
module. The primary module exposes this interface:

```python
TOOL_SPEC = {
    "name": "extract_promoter_variant_rows_for_task_abc",
    "description": "Extract promoter variant rows from the current task's supplementary workbook.",
    "parameters_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "sheet_name": {"type": "string"}
        },
        "required": ["path"]
    },
    "metadata": {
        "generated_tool": True,
        "scope": "task"
    }
}

def run(arguments: dict, context: dict) -> dict | str:
    path = arguments.get("path")
    sheet_name = arguments.get("sheet_name")
    return {
        "status": "ok",
        "content": f"prepared extraction for {path}",
        "metadata": {
            "path": path,
            "sheet_name": sheet_name,
            "task_id": context.get("task_id")
        }
    }
```

The returned value can be a string or a `ToolResult`-compatible mapping. The
generated subprocess adapter normalizes it to `ToolResult` before returning to
`ToolRuntime`. The core `ToolRuntime` handler contract remains `str |
ToolResult`; generated tool wrappers are responsible for converting dict output
to `ToolResult`.

The package record stored in artifacts contains:

- manifest and `ToolSpec`
- Python source files
- task id and run ref
- source LLM call ref when available
- code hash
- validation report
- smoke test inputs and outputs
- generated tool name after namespacing
- effective capability grants

Capability grants are explicit:

- `allowed_read_roots`: task input roots and Lab artifact roots available to the
  generated tool.
- `allowed_write_root`: the generated tool artifact directory for outputs.
- `allowed_env_keys`: environment variables copied into the subprocess.
- `allow_network`: whether network libraries and outbound access are allowed.
- `allow_subprocess`: whether the generated tool may spawn child processes.
- `allowed_imports`: optional import allowlist for strict runs.

Default grants allow read access to task inputs and Lab artifacts, write access
only to the generated tool artifact directory, no copied secrets, no subprocess
spawning, and no network. The subprocess runner uses a sanitized environment,
sets its working directory to the generated tool directory, and records grant
metadata in every result. Static checks reject direct use of `subprocess`,
network libraries, and environment reads unless the relevant grant is enabled.
These checks are a runtime policy boundary for trusted autonomous runs, not a
complete hostile-code sandbox.

Generated tool names are collision-safe. A generated tool cannot replace a
built-in tool unless a runtime policy explicitly allows replacement. The default
generated name uses a deterministic task/run prefix plus the requested semantic
name.

## Tool-Code Evolution Contract

MetaAgent receives a tool-code evolution contract during preplanning. It may
return:

```json
{
  "metadata": {
    "generated_tool_package": {
      "reason": "The task repeatedly needs a workbook-specific promoter row extractor.",
      "tool_name": "promoter_variant_row_extractor",
      "manifest": {
        "description": "Extract promoter variant rows from this task's workbook.",
        "parameters_schema": {
          "type": "object",
          "properties": {
            "path": {"type": "string"},
            "sheet_name": {"type": "string"}
          },
          "required": ["path"]
        }
      },
      "files": [
        {
          "path": "tool.py",
          "content": "TOOL_SPEC = {\"name\": \"promoter_variant_row_extractor\", \"description\": \"Extract promoter variant rows from this task's workbook.\", \"parameters_schema\": {\"type\": \"object\", \"properties\": {\"path\": {\"type\": \"string\"}}, \"required\": [\"path\"]}, \"metadata\": {\"generated_tool\": true, \"scope\": \"task\"}}\n\ndef run(arguments, context):\n    return {\"status\": \"ok\", \"content\": \"generated tool loaded\", \"metadata\": {\"path\": arguments.get(\"path\"), \"task_id\": context.get(\"task_id\")}}\n"
        }
      ],
      "smoke_tests": [
        {
          "name": "missing_file_returns_error",
          "arguments": {"path": "missing.xlsx"}
        }
      ]
    }
  }
}
```

or:

```json
{
  "metadata": {
    "no_generated_tool_reason": "The built-in file, table, and output tools cover the task."
  }
}
```

Aliases can be accepted during migration, but the canonical fields are
`metadata.generated_tool_package` and `metadata.no_generated_tool_reason`.

The same package shape is used by capability repair when a worker asks for an
unprepared tool or a failure detector identifies a missing capability. Existing
`RepairPlan.new_runtime_tool` becomes a typed generated-tool package rather than
an unstructured dictionary.

## Validation And Loading

Generated tool validation is automatic and blocks exposure on failure. It runs
before registration and records all outcomes.

Required gates:

- Validate package shape and paths. Files must stay inside the generated tool
  artifact directory.
- Validate `ToolSpec` with the existing `ToolSpec` contract.
- Validate `parameters_schema` is a JSON-schema object.
- Reject duplicate tool names unless replacement is explicitly allowed.
- Reject replacement of built-in tools by default.
- Parse Python with `ast.parse()` before writing importable code.
- Reject modules without `TOOL_SPEC` and `run`.
- Reject obvious private reasoning fields in manifest metadata.
- Import or execute the module in an isolated module namespace or subprocess.
- Run declared smoke tests with a timeout and max output size.
- Register only after validation passes.

The first implementation should execute generated tools through a subprocess
adapter by default. The adapter passes JSON arguments and a task context to the
generated module and expects JSON or text output. This gives timeout control,
crash isolation, and cleaner provenance. It is not a complete OS security
sandbox, so policy and provenance remain visible in traces.

## Runtime Policy

Existing policy flags remain meaningful:

- `allow_runtime_tool_creation` controls task-local generated tool creation.
- `allow_runtime_tool_patch` controls wrapper-style patches.
- `allow_global_tool_mutation` stays false by default and prevents generated
  tools from becoming global built-ins.

New policy fields:

- `max_generated_tools_per_task`
- `max_generated_tool_files`
- `max_generated_tool_source_bytes`
- `generated_tool_validation_timeout_s`
- `generated_tool_execution_timeout_s`
- `generated_tool_max_output_bytes`
- `generated_tool_allowed_imports`
- `generated_tool_allow_network`
- `generated_tool_allow_subprocess`
- `generated_tool_allowed_env_keys`

If creation is disabled or a budget is exhausted, the runtime records a rejected
tool evolution event and proceeds without exposing the generated tool.

## Generated Tool Builder

Generated code is produced by a `GeneratedToolBuilder` runtime service. It is
not part of DynamicWorkflowPlanner. `TaskRuntime` constructs the builder and
passes it to tool-code preplanning and capability repair.

Backend selection is deterministic:

1. `runtime_policy.metadata.generated_tool_builder_backend_id`
2. the current worker role backend during repair
3. `dynamic_subagents.default_worker_backend.backend_id`
4. `meta_agent.llm_backend.backend_id`

If none of those backends exists, generated tool creation is rejected with a
recorded `generated_tool_rejected` event.

The builder prompt receives:

- task id, task goal, and current work item
- failure signal and failed tool call for repair-triggered generation
- built-in tool specs and current generated tool specs
- role pool summaries
- artifact root and capability grants
- required package response schema
- instruction to return complete Python source files and no hidden reasoning

The builder writes nothing directly. It returns a generated-tool package to the
runtime. The runtime persists files, validates them, computes the registered
namespaced tool name, and returns a `GeneratedToolRegistration` record. When
repair requested a semantic tool name, the retry plan is rewritten to the final
registered generated tool name after validation. The original requested name is
kept in metadata.

## Dynamic Workflow Integration

DynamicWorkflowPlanner remains a structural planner. Its prompt continues to
forbid executable code. Instead, it receives an effective tool catalog built
after tool-code evolution:

```text
effective_tool_catalog = built-in allowed tools + validated task-generated tools
```

Planner output may reference generated tool names in
`dynamic_subagents[].allowed_tools`. Validation changes from checking only
`dynamic_subagents.allowed_tool_names` against the base registry to checking
`TaskEffectiveToolCatalog.effective_allowed_tool_names` against the effective
`ToolRuntime`.

The planner prompt includes both lists:

- `configured_builtin_allowed_tool_names`: the original config allowlist.
- `effective_allowed_tool_names`: built-ins plus validated generated tools.

The response template tells the planner to use only
`effective_allowed_tool_names`. Role-pool update validation still receives only
the persistent built-in allowlist, so reusable roles cannot depend on
task-local generated tools.

`DynamicSubAgentFactory` prepares the merged tool set exactly as before. The
only change is that generated tool specs are visible through effective tool
lookup and can be prepared when the planner selected them.

## Repair Integration

Runtime repair gets a real generated-tool path:

1. A tool call fails with `unprepared_tool`, schema mismatch, repeated empty
   output, or another missing-capability signal.
2. `CapabilityRepairPlanner` returns a `RepairPlan` with `new_runtime_tool`.
3. `CapabilityRepairRuntime` asks `GeneratedToolBuilder` for a package unless
   the plan already contains complete package files.
4. The generated-tool runtime validates and registers it task-locally.
5. The repair retry plan is rewritten to the validated registered tool name.
6. The failed call is retried with the new tool when the retry plan names it.
7. A `new_tool` promotion candidate is recorded, but global promotion is not
   performed by this design.

Generated repair tools are usable by subsequent nodes in the same task after a
retry or replanning step. They should not leak into the next task.

## Artifact And Trace Semantics

Every generated tool attempt records:

- `generated_tool_requested`
- `generated_tool_validated`
- `generated_tool_registered`
- `generated_tool_rejected`
- `generated_tool_executed`

Events include task id, run ref, code hash, generated tool name, source artifact
refs, validation result, and rejection errors. Tool call traces include generated
tool provenance in `ToolResult.metadata` so downstream skill observation and SFT
export can distinguish built-in and generated tools.

Source code is stored as a task artifact. It is not stored in agent memory.
Task-level memory may summarize that a generated tool worked or failed, but the
source of truth is the artifact and trajectory record.

## Error Handling

Tool-code evolution fails closed:

- Invalid package shape: reject and record errors.
- Invalid Python syntax: reject before registration.
- Import failure: reject and keep source artifact for debugging.
- Smoke test failure: reject by default.
- Runtime timeout: return a `ToolResult` error with `error_type` set to
  `generated_tool_timeout`.
- Oversized output: truncate or reject according to policy and include
  diagnostics.
- Duplicate generated name: namespace automatically when safe; reject built-in
  replacement by default.
- Planner references rejected tool: dynamic workflow validation fails for that
  plan attempt.

Rejected generated tools never become prepared tools.

## Implementation Scope

The first implementation should include:

- Typed generated-tool contracts.
- A generated-tool runtime that persists, validates, loads, registers, and
  executes task-local Python tools.
- `ToolRuntime` support for a task-local generated registry separate from
  per-prepare runtime overlays.
- MetaAgent preplanning stage for task-start tool-code evolution.
- Capability repair support for `new_runtime_tool`.
- Dynamic planner effective tool catalog support.
- Trajectory and artifact records for generated code and validation.
- Tests for reset, validation, execution, repair, and dynamic planner use.

Global promotion of generated tools into built-in packages is outside this
design and requires a separate explicit promotion path.

## Testing

Unit tests:

- Generated package with valid Python registers and executes.
- Invalid manifest, invalid schema, missing `run`, syntax error, import error,
  bad return type, and smoke test failure all fail closed.
- Generated tools obey prepared-tool enforcement.
- Generated tools persist across `prepare()` calls within one task.
- Generated tools are cleared at new task reset.
- Built-in tool replacement is rejected by default.
- Generated tool provenance appears in `ToolResult.metadata`.

Integration tests:

- MetaAgent emits `generated_tool_package`; runtime validates and planner sees
  the generated tool in the effective catalog.
- Dynamic workflow uses a generated tool from planner output.
- Capability repair turns a missing capability into `new_runtime_tool`,
  registers it, retries, and records a `new_tool` promotion candidate.
- Rejected generated tools are not exposed to planner or workers.
- Task-level trajectory records source artifacts, code hash, validation result,
  and execution events.

Regression tests:

- `ToolRuntime.prepare()` no longer erases task-local generated tools.
- Generated tools do not leak across tasks.
- Role-pool updates reject reusable roles that reference task-local generated
  tools.
- Dynamic workflow planner still rejects executable code in workflow specs.
