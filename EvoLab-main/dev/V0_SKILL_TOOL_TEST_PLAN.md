# V0 Skill Tool Test Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development when implementing these tests. Write each missing test first, run it to confirm the expected failure, then adjust only the minimum production behavior if a real defect is exposed.

**Goal:** Audit current v0 skill/tool-use test coverage and define the focused tests needed to prove the skill retrieval to tool execution to skill observation loop.

**Architecture:** Keep the scope at the runtime contract boundary: `SkillBackend`, `SkillBundle`, `ToolRuntime`, `TaskRuntime`, and `SubagentRunRecord` trace/artifact persistence. Use small fake LLM/memory dependencies only to drive `TaskRuntime`; do not test LLM parameter training, memory backend semantics, or full self-evolution.

**Tech Stack:** Python 3.11+, pytest, Pydantic contract models, file-backed temporary test registries.

---

## Scope

In scope:
- Skill retrieval before a subagent run.
- `SkillBundle` content, required tool aggregation, graph context metadata, and graph version refs.
- Controlled tool exposure from skill requirements, role allowed tools, and runtime policy.
- Tool execution, errors for unprepared tools, `ToolResult` artifact handling, `ToolTrace`, and persisted run records.
- Skill observation/update after the subagent run, including update summary and graph version lineage.

Out of scope:
- LLM parameter training.
- Memory backend correctness beyond minimal stubs needed by `TaskRuntime`.
- Full self-evolution or trainer workflows.
- Promotion logic.

## Audit Method

Searched current tests with:

```bash
rg -n "SkillBackend|SkillBundle|ToolRuntime|ToolTrace|ToolResult|ToolSpec|required_tools|look_at|graph_context_summary|artifact|allowed_tools|runtime_policy|prepare_skill_runtime_context" tests
```

Reviewed these files in detail:
- `tests/test_graph_skill_backend.py`
- `tests/test_skill_runtime_integration.py`
- `tests/test_tool_runtime.py`
- `tests/test_task_worker.py`
- `tests/test_skill_contracts_and_fake_backend.py`

## A. Existing Coverage By Acceptance Criteria

| # | Acceptance criterion | Existing coverage | Current strength |
|---|---|---|---|
| 1 | Subagent run calls `SkillBackend.get(...)` before run | `tests/test_task_worker.py::test_default_task_runtime_runs_memory_skill_llm_and_records_updates` asserts one `skill.get` call and request equality. | Covered at `TaskRuntime` level with fake skill backend. |
| 2 | `SkillBackend.get(...)` returns `SkillBundle` | `tests/test_graph_skill_backend.py::test_get_returns_matching_candidate_skill_from_json_graph`; `tests/test_skill_contracts_and_fake_backend.py::test_fake_skill_backend_returns_deterministic_bundle_and_update_refs`. | Covered for Graph and fake backends. |
| 3 | `SkillBundle` includes skills, `required_tools`, `graph_context_summary`, graph version metadata | `tests/test_graph_skill_backend.py::test_get_returns_matching_candidate_skill_from_json_graph`; `test_graph_context_summary_contains_three_layer_trace`; `test_get_renders_candidate_skill_operational_contract`; `tests/test_skill_runtime_integration.py::test_prepare_skill_runtime_context_retrieves_skills_tools_and_prompt_context`. | Covered as unit/integration fragments. |
| 4 | `ToolRuntime.prepare(...)` creates `ToolBundle` from `SkillBundle.required_tools`, `RoleSpec.allowed_tools`, and `runtime_policy` | `tests/test_skill_runtime_integration.py::test_prepare_skill_runtime_context_retrieves_skills_tools_and_prompt_context`; `tests/test_tool_runtime.py::test_prepare_filters_allowed_and_registered_tools`; `test_prepare_validates_runtime_policy`. | Covered, but aggregation from bundle-level plus per-skill tools needs a dedicated test. |
| 5 | Subagent `tool_call` is validated and executed by `ToolRuntime.execute(...)` | `tests/test_task_worker.py::test_default_task_runtime_executes_single_tool_call_and_records_trace`; `tests/test_tool_runtime.py::test_runtime_rejects_missing_registry_and_unprepared_tool_calls`. | Covered for happy path and direct unprepared path. |
| 6 | `ToolRuntime.execute(...)` returns `ToolResult`, registers artifacts, writes `ToolTrace` | `tests/test_tool_runtime.py::test_execute_preserves_tool_result_handler_artifacts_and_metadata`; `tests/test_task_worker.py::test_default_task_runtime_registers_tool_artifacts_and_records_them_in_trace`; `test_default_task_runtime_records_valid_and_invalid_tool_traces_and_lab_artifacts`. | Covered, including managed local artifacts and trace persistence. |
| 7 | Subagent run calls `SkillBackend.look_at(...)` after run | `tests/test_task_worker.py::test_default_task_runtime_runs_memory_skill_llm_and_records_updates`; `test_default_task_runtime_saves_skill_update_result_and_backend_state_lineage`. | Covered, but observation payload content needs stronger assertions. |
| 8 | `SkillBackend.look_at(...)` records observation, update summary, graph versions, affected skills | `tests/test_graph_skill_backend.py::test_look_at_writes_update_summary_jsonl_with_versions`; `tests/test_skill_contracts_and_fake_backend.py::test_fake_skill_backend_returns_deterministic_bundle_and_update_refs`. | Partially covered. Direct GraphSkillBackend test covers affected skills; runtime-driven observation does not. |
| 9 | One integration test proves retrieval -> aggregation -> controlled exposure -> execution -> trace/artifact logging -> observation/update | No single test currently uses retrieved skill context to drive the entire chain. Existing tests cover pieces across `test_skill_runtime_integration.py`, `test_tool_runtime.py`, and `test_task_worker.py`. | Missing. This is the main v0 acceptance gap. |

## B. Missing Tests

1. A single end-to-end v0 chain test.
   - Current tests do not prove that a real skill retrieval result controls the tools exposed to a subagent and then flows into `ToolTrace`, artifact logging, and `SkillBackend.look_at`.

2. Required tool aggregation from both bundle-level and per-skill tool lists.
   - `_aggregate_required_tools()` merges `SkillBundle.required_tools` and each skill's `required_tools`, but current tests mainly cover GraphSkillBackend sorted/dedup output or a single matching tool list.

3. Runtime-level denial when a skill requires a registered tool that the role did not allow.
   - `prepare_skill_runtime_context()` has a missing-tool test, but `TaskRuntime` should have a regression test proving the LLM is not called when skill-required tools are not allowed.

4. Skill observation payload completeness.
   - Runtime tests assert `graph_version_ref` and `skill_state_ref`, but not that `look_at` receives `skill_bundle`, selected skills, final answer, full `tool_trace`, and managed artifact refs.

5. Skill context includes optional refs.
   - Contracts cover `script_refs` and `resource_refs`, but no runtime context test proves `build_skill_context()` includes `artifact_refs`, `script_refs`, `resource_refs`, and `skill_state_ref`.

## C. Proposed Test Files

### Create `tests/test_v0_skill_tool_chain.py`

Responsibility:
- Own the v0 acceptance chain tests.
- Keep fixtures local so the file reads as an executable spec for the skill/tool-use loop.
- Use minimal fake memory and scripted LLM only as runtime dependencies.

Recommended tests:
- `test_v0_skill_tool_chain_runs_retrieval_tools_trace_artifacts_and_observation`
- `test_v0_skill_required_tool_denied_by_role_stops_before_llm_call`
- `test_v0_skill_observation_contains_tool_trace_and_selected_skill_metadata`

### Modify `tests/test_skill_runtime_integration.py`

Responsibility:
- Keep preparation/context unit integration coverage close to `prepare_skill_runtime_context()`.

Recommended tests:
- `test_prepare_skill_runtime_context_aggregates_bundle_and_skill_required_tools`
- `test_build_skill_context_includes_skill_artifact_script_resource_refs`

### Optional modification to `tests/test_graph_skill_backend.py`

Responsibility:
- Strengthen `GraphSkillBackend.look_at()` coverage only if the implementation should derive `affected_skill_ids` from structured runtime observations in the future.

Recommended test:
- `test_look_at_accepts_runtime_observation_payload_and_records_graph_update`

This optional test is valuable if `GraphSkillBackend.look_at()` is expected to consume `SkillObservationRequest` payloads directly rather than only direct update-summary event dictionaries.

## D. Test Designs

### 1. `test_v0_skill_tool_chain_runs_retrieval_tools_trace_artifacts_and_observation`

File:
- Create in `tests/test_v0_skill_tool_chain.py`

Purpose:
- Prove the full v0 acceptance chain in one test: skill retrieval, required tool aggregation, controlled tool exposure, valid tool execution, invalid unprepared tool rejection, trace/artifact persistence, and skill observation/update.

Input:
- A temporary JSON skill graph with version `v0-skill-graph-1`.
- One candidate skill `skill-write-report` whose `required_tools` include `write_report`.
- A `RecordingGraphSkillBackend` test subclass that delegates `get()` to `GraphSkillBackend.get()`, records `get_requests` and `look_at_events`, and calls `GraphSkillBackend.look_at()` with `source_run_id`, `candidate_skill_id`, `affected_skill_ids`, `update_type`, and provenance derived from the runtime observation.
- A `ToolRegistry` with one registered `write_report` handler that writes `report.txt` and returns `ToolResult(status="ok", artifact_refs=[ArtifactRef(...)]).`
- A scripted LLM that first calls `write_report`, then calls an unprepared `delete_file`, then returns final answer `done`.
- A `TaskRuntime` with `RoleSpec.allowed_tools=["write_report"]`, `tool_artifact_root_factory`, `trajectory_registry`, and minimal fake memory.

Assertions:
- `skill_backend.get_requests` has exactly one retrieval request with task id, role, query, task origin, and task purpose.
- First LLM call receives exactly one tool spec: `write_report`.
- The registered handler receives only the valid `write_report` call; no handler runs for `delete_file`.
- Persisted `SubagentRunRecord.skill_bundle.graph_version_ref == "v0-skill-graph-1"`.
- Persisted `SubagentRunRecord.skill_bundle.skills[0].skill_id == "skill-write-report"`.
- Persisted `metadata["skill_context"]["graph_context_summary"]["graph_version"] == "v0-skill-graph-1"`.
- Persisted `tool_calls` contains two records: `write_report` with `status == "ok"` and `delete_file` with `metadata["error_type"] == "unprepared_tool"`.
- Persisted `artifact_refs[0]` is lab-managed, exists on disk, and contains the report content.
- Persisted `metadata["tool_trace"]["calls"]` includes the managed artifact URI.
- `skill_backend.look_at_events[0]` includes `skill_bundle`, `tool_trace`, `final_answer == "done"`, and the managed artifact ref inside the first tool result.
- The GraphSkillBackend update log contains `source_run_id == result["run_ref"]`, `candidate_skill_id == "skill-write-report"`, `affected_skill_ids == ["skill-write-report"]`, `graph_version_before == "v0-skill-graph-1"`, and `graph_version_after == "v0-skill-graph-1"`.

Expected failure scenarios:
- `SkillBackend.get()` is skipped or called after LLM generation.
- Required tools are not exposed to the LLM.
- Role tool allow-list is bypassed and `delete_file` executes.
- Tool artifacts are not registered or copied into the managed artifact root.
- `ToolTrace` is missing from persisted run metadata or from the skill observation.
- `SkillBackend.look_at()` is skipped or receives an observation without skill/tool trace context.
- Graph update logging loses graph version or affected skill ids.

### 2. `test_v0_skill_required_tool_denied_by_role_stops_before_llm_call`

File:
- Create in `tests/test_v0_skill_tool_chain.py`

Purpose:
- Prove controlled exposure at the `TaskRuntime` boundary: if a retrieved skill requires a tool that is registered but not allowed by the role, the run fails before any LLM call.

Input:
- A fake skill backend returning `SkillBundle(required_tools=["write_report"], skills=[SkillItem(... required_tools=["write_report"])])`.
- A `ToolRegistry` with `write_report` registered.
- `RoleSpec.allowed_tools=[]`.
- A recording LLM that increments a call counter if invoked.

Assertions:
- `runtime.run(request)` raises `MissingRequiredToolError` with `write_report`.
- LLM call counter remains zero.
- Tool handler call counter remains zero.
- No `SubagentRunRecord` is persisted.
- `SkillBackend.look_at()` is not called, because the run never started tool-capable generation.

Expected failure scenarios:
- Required-but-disallowed tools are silently dropped.
- LLM receives no tools but the run continues, hiding the skill contract violation.
- Tool execution can occur despite role policy.

### 3. `test_v0_skill_observation_contains_tool_trace_and_selected_skill_metadata`

File:
- Create in `tests/test_v0_skill_tool_chain.py`

Purpose:
- Focus on the post-run observation contract sent to `SkillBackend.look_at(...)`.

Input:
- A recording skill backend returning a `SkillBundle` with:
  - `backend_id="skill-local"`
  - `graph_version_ref="graph-v1"`
  - `skill_state_ref="skill-state-before"`
  - one `SkillItem` with `artifact_refs`, `script_refs`, `resource_refs`, and `required_tools=["lookup"]`
  - `metadata["graph_context_summary"]` with one retrieval path
- A `lookup` tool returning `ToolResult(status="ok", metadata={"rows": 1})`.
- A scripted LLM calling `lookup` once and returning final answer.

Assertions:
- `look_at_events[0]["schema_version"] == "v1"`.
- Observation includes `retrieval_request` with role and query.
- Observation includes full `skill_bundle`, including selected skill id, `graph_version_ref`, `skill_state_ref`, and graph context summary.
- Observation includes `tool_trace.run_ref` matching the runtime result.
- Observation includes one tool trace call with tool name, arguments, status, content, artifact refs, and metadata.
- Observation includes `final_answer`.
- Persisted `metadata["skill_observation_request"]` equals the recorded observation.

Expected failure scenarios:
- Runtime sends the older ad hoc dict without `skill_bundle` or `tool_trace`.
- Tool trace exists in trajectory metadata but not in the skill observation.
- Artifact refs are present in `ToolResult` but absent from observation.
- Skill state refs are lost before observation.

### 4. `test_prepare_skill_runtime_context_aggregates_bundle_and_skill_required_tools`

File:
- Modify `tests/test_skill_runtime_integration.py`

Purpose:
- Prove required tool aggregation across both `SkillBundle.required_tools` and nested skill items before `ToolRuntime.prepare(...)`.

Input:
- A minimal fake skill backend whose `get()` returns:
  - `SkillBundle.required_tools == ["lookup"]`
  - one `SkillItem.required_tools == ["write_file", "lookup"]`
- A `ToolRuntime` with registered `lookup` and `write_file`.
- `allowed_tools=["lookup", "write_file"]`.
- `RuntimePolicy()`.

Assertions:
- Prepared `skill_bundle.required_tools == ["lookup", "write_file"]`.
- Prepared `tool_bundle.tool_specs` names are `["lookup", "write_file"]`.
- Prepared `skill_context["required_tools"] == ["lookup", "write_file"]`.

Expected failure scenarios:
- Runtime uses only bundle-level tools and misses per-skill tools.
- Runtime uses only per-skill tools and drops bundle-level tools.
- Runtime deduplicates incorrectly and exposes duplicate tool specs.

### 5. `test_build_skill_context_includes_skill_artifact_script_resource_refs`

File:
- Modify `tests/test_skill_runtime_integration.py`

Purpose:
- Prove the prompt/runtime skill context preserves operational references that a tool-capable subagent may need.

Input:
- Direct `SkillBundle` with:
  - `skill_state_ref="skill-state-v1"`
  - one `SkillItem` with `artifact_refs`, `script_refs`, `resource_refs`
  - graph context summary metadata
- Call `build_skill_context(skill_bundle)`.

Assertions:
- `context["skill_state_ref"] == "skill-state-v1"`.
- `context["selected_skills"][0]["artifact_refs"]` contains the serialized artifact ref.
- `context["selected_skills"][0]["script_refs"]` contains the serialized script ref.
- `context["selected_skills"][0]["resource_refs"]` contains the serialized resource ref.
- Existing fields remain present: `skill_id`, `name`, `required_tools`, `retrieval`, `graph_context_summary`.

Expected failure scenarios:
- Skill refs exist in contracts but are dropped from runtime context.
- Skill state refs are available in `SkillBundle` but not in prompt context.

### 6. `test_look_at_accepts_runtime_observation_payload_and_records_graph_update`

File:
- Optional modification in `tests/test_graph_skill_backend.py`

Purpose:
- Decide and lock the intended GraphSkillBackend behavior for structured runtime observations. Add this only if GraphSkillBackend should natively derive update summary fields from `SkillObservationRequest` payloads.

Input:
- A graph file with version `graph-v1`.
- A runtime-shaped observation dict:
  - `run_ref="run-1"`
  - `skill_bundle.skills[0].skill_id == "skill-1"`
  - `tool_trace.calls[0].result.status == "ok"`
  - metadata containing an explicit or derivable affected skill list
- Call `GraphSkillBackend.look_at(observation)`.

Assertions:
- Update log has one row.
- Logged `source_run_id == "run-1"`.
- Logged `candidate_skill_id == "skill-1"` if the implementation is expected to derive it.
- Logged `affected_skill_ids == ["skill-1"]` if the implementation is expected to derive it.
- Logged `graph_version_before == "graph-v1"` and `graph_version_after == "graph-v1"`.

Expected failure scenarios:
- `GraphSkillBackend.look_at()` only supports direct update events and cannot consume runtime observations.
- Runtime observations are recorded but lose affected skill ids.

## Implementation Order

1. Add `tests/test_v0_skill_tool_chain.py::test_v0_skill_tool_chain_runs_retrieval_tools_trace_artifacts_and_observation`.
2. Add `tests/test_v0_skill_tool_chain.py::test_v0_skill_required_tool_denied_by_role_stops_before_llm_call`.
3. Add `tests/test_v0_skill_tool_chain.py::test_v0_skill_observation_contains_tool_trace_and_selected_skill_metadata`.
4. Add `tests/test_skill_runtime_integration.py::test_prepare_skill_runtime_context_aggregates_bundle_and_skill_required_tools`.
5. Add `tests/test_skill_runtime_integration.py::test_build_skill_context_includes_skill_artifact_script_resource_refs`.
6. Add the optional `GraphSkillBackend.look_at()` runtime-observation test only after deciding that native behavior is part of v0.

## Validation Commands For The Test Work

Targeted command:

```bash
uv run pytest tests/test_v0_skill_tool_chain.py tests/test_skill_runtime_integration.py tests/test_graph_skill_backend.py::test_look_at_writes_update_summary_jsonl_with_versions -q
```

Full command:

```bash
uv run pytest -q
```

Whitespace check:

```bash
git diff --check
```

## Audit Conclusion

Current tests cover most v0 skill/tool-use pieces, but they are not yet tied together by one acceptance-level integration test. The highest-value next test is `test_v0_skill_tool_chain_runs_retrieval_tools_trace_artifacts_and_observation`, because it will fail if any part of the v0 loop is disconnected: skill retrieval, required tool exposure, tool execution control, artifact/trace persistence, or post-run skill observation/update logging.
