# Trace2Skill Evolver

EvoLab implements a domain-generic Trace2Skill-style evolver under
`evolab/backends/skills/trace2skill/`. It distills compact execution traces
into policy-gated skill update proposals without making `SkillBackend`
execute tools and without adding human-in-the-loop review to this phase.

## Pipeline

The implemented flow is:

1. `TracePoolBuilder` normalizes `SkillObservationRequest`, run records,
   `ToolTrace`, workflow metadata, and backend-state observations into
   `TraceRecord` objects.
2. Deterministic outcome classification separates success, failure, partial,
   and unknown traces.
3. `ParallelAnalystRunner` runs `ErrorAnalyst`, `SuccessAnalyst`,
   `CoverageAnalyst`, and `PatchProposalAnalyst` sequentially by default or in
   a bounded thread pool when `analyst_execution_mode="thread"`.
4. `Trace2SkillLLMExtractor` can replace deterministic analyst extraction when
   `enable_llm_analysts=True` and an EvoLab LLM runtime/backend is supplied.
5. `HierarchicalPatchConsolidator` groups, deduplicates, merges, and defers or
   rejects weak patches.
6. `ConflictChecker` and `SkillPatchValidator` gate missing tools, candidate ID
   collisions, invalid relations, duplicate relations, oversized examples, and
   missing target skills.
7. `Trace2SkillSkillBackendAdapter` converts valid patches into EvoLab
   `SkillUpdateProposal` objects.
8. `SkillEvolutionPolicy` decides whether each proposal is applied, staged, or
   rejected.
9. `SkillEvolutionRegressionGate` optionally compares before/after benchmark
   metrics before mutation.
10. `SkillLibraryUpdateTransaction` records applied, staged, and rejected work,
   before/after graph hashes, changed skills, and rollback metadata.
11. `Trace2SkillReportWriter` writes JSON, JSONL, and Markdown audit artifacts.

## LLM Extraction

Trace2Skill reuses EvoLab's existing LLM abstraction:

- `evolab.backends.llm.base.LLMRuntime.generate(...)`
- `evolab.backends.llm.api.ApiLLMBackend`
- `evolab.contracts.llm.LLMGenerationConfig`
- `evolab.contracts.common.Message`

`ApiLLMBackend` already owns OpenAI Responses-compatible configuration,
`api_key_env`, `base_url`, local OpenAI-compatible endpoints, and structured
JSON schema output. Trace2Skill does not duplicate that API configuration. A
caller passes an existing backend or runtime as `llm_client`:

```python
evolver = Trace2SkillEvolver(
    graph_backend=graph_backend,
    tool_registry=tool_registry,
    llm_client=api_llm_backend,  # or an already-instantiated LLMRuntime
)
result = evolver.run(
    Trace2SkillRunConfig(
        mode="combined",
        enable_llm_analysts=True,
        llm_config_ref="trace2skill-model",
        max_llm_retries=1,
        llm_temperature=0,
        dry_run=True,
    ),
    observations=[skill_observation],
)
```

The extractor asks for JSON only and passes a strict `response_json_schema`.
Parsed output is validated into `TrajectoryLesson` and `SkillPatchProposal`
models. Stored fields are limited to lesson type, target skill ID, evidence
summary, reusable principle, proposed delta, patch type, confidence, risk, and
source trace IDs. Private chain-of-thought fields are rejected.

If no LLM runtime is available, the model call fails, JSON parsing fails, schema
validation fails, or an unsupported patch type appears, the extractor records an
audit event and uses deterministic fallback when
`enable_deterministic_fallback=True`. Tests use `FakeLLMRuntime`; they never
call external APIs.

## Parallel Analysts

`ParallelAnalystRunner` supports:

- `analyst_execution_mode="sequential"` for deterministic default runs.
- `analyst_execution_mode="thread"` with `analyst_max_workers`.
- Optional per-analyst timeout through `analyst_timeout_seconds`.
- Stable sorting of lessons and patches after execution.
- Failure isolation: a failed analyst/trace is reported in metadata and does
  not abort the whole evolution run.

## Stable Mutation

Conservative policy remains safe by default. Stable library mutation happens
only after validation and policy approval. Supported policy-approved patch
types include:

- `required_tools_patch`
- `example_memory_patch`
- `procedure_step_patch`
- `precondition_patch`
- `failure_case_patch`
- `validation_rule_patch`
- `metadata_patch`

Embedded graph skills are updated in graph JSON. Package-backed skills mutate
their package metadata atomically where possible. Stable fields are used when
they exist (`required_tools`, `examples`, `procedure`). Extra evolution state is
stored under:

```yaml
metadata:
  evolution:
    examples: []
    procedure_notes: []
    preconditions: []
    failure_cases: []
    validation_rules: []
    usage_stats: {}
    provenance: []
```

Bounded storage limits are configured on `Trace2SkillRunConfig`, including
`max_examples_per_skill`, `max_failure_cases_per_skill`,
`max_procedure_notes_per_skill`, `max_preconditions_per_skill`,
`max_validation_rules_per_skill`, and `max_evolution_text_chars`.

## Candidate Promotion

Candidate promotion is disabled by default. It becomes available only when
`SkillEvolutionPolicy(auto_apply_valid_candidates=True)` is supplied. A valid
candidate patch is converted into a deterministic package-backed stable skill
under `skills/trace2skill/generated/`, inserted into the graph as a lightweight
skill node, and validated before commit. Candidate IDs are generated with a
collision-safe `skill.trace2skill.*.v1` prefix. Minimal safe category relations
are added only when the referenced category already exists.

## Relationship Updates

Relationship updates are staged by default. They apply only when
`SkillEvolutionPolicy(auto_apply_relationship_updates=True)` is supplied. The
evolver accepts only supported relation types, requires both endpoints to exist,
deduplicates existing edges, validates the graph, and records the transaction.
It does not perform broad automatic graph rewiring.

## Regression Gate

`SkillEvolutionRegressionGate` provides a generic benchmark interface:

- `BenchmarkTask`
- `BenchmarkRunResult`
- `BenchmarkRunner`
- `ReplayBenchmarkRunner`

When `enable_regression_gate=True`, the gate compares configured metrics such as
`accuracy`, `recall`, `precision`, `f1`, `task_success_rate`,
`retrieval_hit_rate`, or `tool_coverage`. If no benchmark runner is supplied,
the gate reports `skipped_no_benchmark` and does not claim improvement. If a
runner returns invalid or incomplete results, the gate reports `inconclusive`.
If after metrics regress beyond `regression_no_regression_threshold`, the gate
returns `fail_regression` and policy-approved mutations are blocked.

## Reports

Each run writes:

- `trace2skill_run_summary.json`
- `trace_pool_stats.json`
- `lessons.jsonl`
- `local_patch_proposals.jsonl`
- `consolidated_patches.json`
- `conflict_report.json`
- `validation_report.json`
- `converted_skill_update_proposals.jsonl`
- `policy_decisions.jsonl`
- `applied_transactions.jsonl`
- `regression_gate_report.json`
- `before_after_metrics.json`
- `trace2skill_audit_report.md`

Reports include trace pool stats, success/failure split, lesson and patch
counts, validation/conflict summaries, policy decisions, transactions, stable
skills updated, candidates promoted, relations added, before/after graph hashes,
retrieval impact summary, skipped or inconclusive items, and future-work
warnings.

## Future Work

Future versions can add stronger multi-agent LLM analyst prompts, large-scale
trace clustering, richer graph ontology insertion, a human review workflow, and
cross-project collective skill evolution. Those are not implemented in this
phase.
