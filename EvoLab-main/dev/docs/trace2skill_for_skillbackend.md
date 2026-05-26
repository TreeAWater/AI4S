# Trace2Skill for SkillBackend

Trace2Skill is integrated as an evolution layer around `GraphSkillBackend`, not
as a replacement for retrieval or runtime execution.

## Boundaries

- `GraphSkillBackend.get(...)` continues to retrieve `SkillBundle` objects.
- `GraphSkillBackend.look_at(...)` handles single-run post-run observations
  through the v1 proposal and policy path.
- `Trace2SkillEvolver` consumes batches of observations or run records and
  produces consolidated, policy-gated skill patch proposals.
- Tool execution remains in `ToolRuntime`; `SkillBackend` never executes tools.
- Workflow execution remains in `TaskRuntime`.
- Stable mutation is audited through `SkillLibraryUpdateTransaction` with
  before/after graph hashes.
- This phase does not add `ask_human`, `request_human_review`, or any other
  human-review integration.

## Inputs

The trace pool builder accepts:

- `SkillObservationRequest`
- run records with embedded retrieval, LLM, and tool metadata
- `ToolTrace` embedded in observations or run records
- workflow metadata such as `PlanExecutionTrace`
- backend-state observation JSONL rows

It records selected skills, retrieved skills, tools used, missing tools,
artifacts, compact execution summaries, evaluation metrics, and failure
summaries. It does not store private chain-of-thought.

## LLM Configuration

LLM-backed lesson extraction reuses EvoLab's existing LLM backend/runtime API.
Pass an `ApiLLMBackend`, `FakeLLMBackend`/runtime, or another object matching
`LLMRuntime.generate(...)` to `Trace2SkillEvolver(llm_client=...)`.

`ApiLLMBackend` already supports OpenAI Responses-compatible APIs, `base_url`,
`api_key_env`, environment references, and local OpenAI-compatible endpoints.
Trace2Skill supplies the prompt, `LLMGenerationConfig`, temperature, retry
count, and JSON schema. It does not create a separate API client or hardcode
model credentials.

Important config fields:

- `enable_llm_analysts=True`
- `llm_config_ref`
- `max_llm_retries`
- `llm_temperature`
- `max_trace_summary_chars`
- `enable_deterministic_fallback=True`

Tests should use fake runtimes or fake backends. Real external calls are not
required for deterministic tests.

## Skill Deepening

For existing skills, Trace2Skill can propose:

- example memory patches
- failure case notes
- procedure step patches
- precondition patches
- validation rule patches
- required tool patches
- metadata patches

Default policy stages stable mutations. Policy-approved patches can update
embedded skills or package-backed `metadata.json`/`metadata.yaml` files. Stable
fields are preferred when present; otherwise bounded evolution metadata is
stored under `metadata.evolution`.

## Skill Creation

Low-coverage traces or traces with no selected skills can produce
`candidate_skill_creation` proposals. Conservative policy stages them. With
`SkillEvolutionPolicy(auto_apply_valid_candidates=True)`, valid candidates are
promoted into generated package-backed stable skills under
`skills/trace2skill/generated/`. Promotion validates the package metadata, graph
node, optional category relation, and before/after hashes before commit.

## Relationship Updates

Trace2Skill can propose relationship patches. Conservative policy stages them.
With `SkillEvolutionPolicy(auto_apply_relationship_updates=True)`, valid
relationship patches add deduplicated graph edges only when both endpoint skills
exist and the relation type is supported. The implementation does not infer
aggressive dependency structure or perform broad graph rewiring.

## Regression Gate

`SkillEvolutionRegressionGate` can run before mutation when
`enable_regression_gate=True`. A caller may supply a benchmark runner that
returns `BenchmarkRunResult` for before and after snapshots. The gate reports:

- `pass`
- `fail_regression`
- `inconclusive`
- `skipped_no_benchmark`

No benchmark means no claimed improvement. A failed regression blocks
policy-approved mutation and leaves the graph unchanged.

## Running

```python
from evolab.backends.skills.evolution import SkillEvolutionPolicy
from evolab.backends.skills.trace2skill import Trace2SkillEvolver, Trace2SkillRunConfig

result = Trace2SkillEvolver(
    graph_backend=graph_backend,
    tool_registry=tool_registry,
    llm_client=api_llm_backend,
    policy=SkillEvolutionPolicy(
        auto_apply_proposal_types=["required_tools_update_proposal"],
        auto_apply_valid_candidates=False,
        auto_apply_relationship_updates=False,
    ),
).run(
    Trace2SkillRunConfig(
        mode="combined",
        dry_run=True,
        enable_llm_analysts=False,
        analyst_execution_mode="sequential",
        output_dir="backend_state/trace2skill",
    ),
    observations=[skill_observation],
)
```

Use `dry_run=True` to generate lessons, patches, validation output, regression
gate reports, and audit artifacts without mutating the stable library.

## Current Limits

- LLM extraction is implemented as a JSON-schema constrained single-runtime
  bridge; stronger multi-agent prompts remain future work.
- Benchmark quality depends on the caller-supplied benchmark runner.
- Candidate promotion is intentionally conservative and deterministic.
- Relationship insertion is limited to safe one-edge updates.
- Human review is intentionally not implemented in this phase.
