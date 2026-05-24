# Agent0 SAGE Evolver Design

## Context

EvoLab already has task provenance contracts, evolution request/result contracts, trainer abstractions, queue workers, trajectory registries, and promotion validation. The current framework does not implement real proposer task generation or a solver training algorithm. Existing specs keep `TaskWorker` and `EvolveWorker` independent and preserve `TaskRequest` as the shared interface for human, benchmark, scheduler, and proposer tasks.

Agent0 and SAGE both describe closed-loop self-evolution, but at this stage EvoLab needs a bounded trainer implementation rather than a broad runtime rewrite. Agent0 contributes the curriculum/executor co-evolution pattern with tool-aware frontier tasks. SAGE contributes the Challenger, Planner, Solver, and Critic roles for generating, planning, solving, and filtering verifiable reasoning tasks.

## Goal

Add one concrete evolver implementation that coordinates proposer-generated training tasks, injected solver rollouts, critic filtering, and delegated solver evolution.

The evolver must:

- generate candidate tasks through the same `TaskRequest` contract as human and benchmark tasks;
- keep proposer tasks distinguishable through `origin`, `purpose`, relation, producer, round, and metadata fields;
- use an injected rollout boundary for solver attempts;
- delegate actual solver evolution to another `LLMTrainer`;
- write reproducible co-evolution artifacts under `artifact_root_uri`;
- avoid changing `TaskWorker`, queue ownership, or the default `TaskRuntime` behavior.

## Design

Add `evolab/backends/trainers/agent0_sage.py` with `Agent0SAGETrainer(LLMTrainer)`.

`Agent0SAGETrainer` is an orchestration trainer. It does not implement model fine-tuning directly. It runs a bounded proposer-solver co-evolution loop, builds accepted training samples, writes artifacts, then calls a nested `solver_trainer`.

The trainer owns these collaborator interfaces:

- proposer/challenger runner: produces candidate task data for a round;
- planner runner: produces a plan for a valid proposed task;
- rollout runner: accepts a `TaskRequest`, plan, and context, then returns the solver attempt result or trajectory reference;
- critic runner: scores task quality, plan quality, solver outcome, verifier signal, difficulty, and tool use;
- nested solver trainer: an `LLMTrainer` that evolves the solver using the accepted sample manifest.

The collaborator interfaces may be callables or lightweight protocols. They are injected into the trainer constructor so tests can use deterministic runners and production code can later bind LLM/tool-based implementations.

## Candidate Task Contract

Each accepted proposer candidate is converted into a normal `TaskRequest`.

Generated tasks use:

- `origin=TaskOrigin.PROPOSER`
- `purpose=TaskPurpose.TRAINING_ROLLOUT`
- `producer_ref="agent0_sage:<trainer_id>"`
- `round_id` set to a stable co-evolution round identifier
- `proposed_task_relation` populated with at least one human anchor task or trajectory
- `metadata` containing `candidate_id`, proposer role traces, tool requirements, difficulty estimates, source seed refs, and trainer configuration refs

This keeps generated tasks compatible with human and benchmark tasks while making their provenance explicit. Human and benchmark tasks remain distinguishable through `origin=HUMAN` and `origin=BENCHMARK`.

Invalid candidate task payloads are rejected and recorded in the summary. They must not call the rollout runner.

## Co-Evolution Flow

For each bounded round:

1. The proposer/challenger produces a candidate task proposal from the evolution request, previous accepted samples, lab signals, and proposer input refs.
2. The trainer validates and normalizes the proposal into a `TaskRequest`.
3. The planner produces a structured plan for the task.
4. The injected rollout runner executes the solver attempt for the task and plan.
5. The critic scores the task, plan, rollout, verifier result, tool-use signal, and estimated difficulty.
6. If the critic score and verifier status pass configured thresholds, the trainer appends the sample to the accepted manifest.
7. Rejected candidates are recorded with rejection reasons.

After the round budget is exhausted, or after enough accepted samples are collected, the trainer writes artifacts and calls the nested solver trainer.

## Solver Evolution

The solver evolves through delegation. `Agent0SAGETrainer` constructs a nested `LLMEvolutionRequest` for `solver_trainer.train(...)`.

The nested request points to a materialized sample artifact containing:

- generated `TaskRequest` JSON objects;
- planner outputs;
- rollout results or trajectory refs;
- critic scores;
- verifier outcomes;
- accepted and rejected sample metadata;
- source evolution request refs and round ids.

This artifact is not the source of truth for dynamic curriculum runs. For V1 dynamic Agent0/SAGE runs, the source of truth is an append-only run ledger under the evolution artifact root:

```text
evolution/llm/<evolution_run_ref>/
  run_spec.json
  events.jsonl
  manifest.latest.json
```

`run_spec.json` is the immutable starting intent: target backend, budget, initial proposer input refs, initial snapshot refs, allowed reward calculator refs, and stop criteria. `events.jsonl` is the authoritative append-only ledger. `manifest.latest.json` is a derived view for trainers that need a conventional dataset manifest.

Ledger event types include:

- `run_started`
- `snapshot_captured`
- `task_proposed`
- `task_enqueued`
- `rollout_completed`
- `reward_policy_updated`
- `reward_calculated`
- `sample_accepted`
- `sample_rejected`
- `curriculum_updated`
- `trainer_invoked`
- `trainer_completed`
- `candidate_created`
- `evolution_record_saved`
- `promotion_decided`
- `run_finished`
- `run_failed`
- `run_skipped`

The nested solver trainer returns the actual solver evolution result. `Agent0SAGETrainer` preserves that outcome where possible and adds Agent0/SAGE metadata and artifacts to the returned `LLMEvolutionResult`.

If the nested solver trainer returns `promoted_candidate`, the outer result returns `promoted_candidate` with the nested `new_state_ref`, `lora_role`, metrics, and artifacts plus the outer manifest artifacts. If the nested trainer returns `not_recommended`, `skipped`, or `failed`, the outer result follows that status and includes the co-evolution summary.

## Snapshots And Reward Policy

SAGE reward is allowed to depend on tool and skill deltas, including tool use, newly created skills/tools, and whether those new capabilities are useful in later rollouts. Reward calculators therefore cannot implicitly read the current mutable registry. They must consume frozen snapshot refs.

V1 adds snapshot contracts in `evolab/contracts/snapshots.py`:

- `ToolsetSnapshot`: tool specs plus implementation refs and parent snapshot refs.
- `SkillSnapshot`: skill backend id, skill state ref, graph version ref, skill items, and required tools.
- `RewardPolicySnapshot`: versioned reward calculator components, weights, config refs, and combination mode.
- `EnvironmentSnapshot`: task config ref, toolset snapshot ref, skill snapshot ref, reward policy snapshot ref, memory state refs, LLM state refs, and backend state refs.

`SnapshotRegistry` persists typed snapshots under the lab registry area and returns stable snapshot refs. `LLMEvolutionRequest.instance_snapshots` carries input snapshot refs for backwards-compatible evolution requests. `EvolutionRunRecord.input_snapshot_refs` and `EvolutionRunRecord.output_snapshot_refs` record before/after snapshot refs for the completed run.

Reward calculators live under `evolab/backends/rewards`. The shared `RewardCalculator` ABC accepts a `RewardCalculationRequest` containing examples, reward policy snapshot refs, before/after snapshot refs, curriculum state refs, and advantage calculation controls. Verifiers are modeled as `VerifierRewardCalculator` subclasses. `CompositeRewardCalculator` combines calculator outputs with `sum`, `mean`, `weighted_sum`, `max`, or `min`.

Because reward policy can evolve during curriculum learning, each scored sample must record the reward policy snapshot ref and enough calculator metadata to replay the score. Advantage calculation records the baseline implicitly through the `RewardCalculationRequest` fields until a dedicated baseline snapshot/ref is introduced.

## Artifacts

Artifacts are written under `LLMEvolutionRequest.artifact_root_uri` when the URI is local or file-based.

The synchronous V1 trainer artifact set is:

- `agent0_sage_samples.jsonl`: accepted samples, one JSON object per line;
- `agent0_sage_rejections.jsonl`: rejected candidates and reasons;
- `agent0_sage_summary.json`: counts, thresholds, nested trainer result summary, and metadata.

Dynamic V1 runs additionally write the run ledger files described above. Synchronous trainers may still produce accepted/rejected sample files as derived artifacts for nested solver trainers.

The returned `LLMEvolutionResult.artifact_refs` includes these artifacts using existing `ArtifactRef` contracts. Remote artifact roots are not uploaded by V1; remote roots produce a clear skipped or failed result depending on configuration.

## Error Handling

Invalid proposer output is recorded as a rejection and does not stop the whole run. Planner errors, rollout exceptions, and critic validation errors reject that candidate with a reason. If no valid candidate can be evaluated, the trainer returns `skipped`.

If accepted sample count is below the configured minimum, the trainer returns `not_recommended` without calling the nested solver trainer unless configured to allow low-sample training.

Nested solver trainer exceptions return `failed` with the exception message in metadata. Nested solver trainer `failed` results propagate as `failed`.

## Testing

Add TDD coverage for:

- `Agent0SAGETrainer` inherits `LLMTrainer` and is exported from `evolab.backends.trainers`;
- proposer-generated tasks validate as `TaskRequest` with `origin=PROPOSER`, `purpose=TRAINING_ROLLOUT`, and `proposed_task_relation`;
- generated tasks are distinguishable from human and benchmark tasks through provenance fields;
- invalid proposer output is rejected and does not call rollout;
- accepted candidate tasks call the injected rollout runner;
- critic thresholds control accepted versus rejected samples;
- accepted and rejected sample artifacts are written under `artifact_root_uri`;
- ledger events are append-only and can materialize the current sample manifest;
- snapshot refs for skill/tool/reward-policy state are available to reward calculators;
- nested `solver_trainer.train(...)` is called with a request pointing at the accepted sample artifact or derived manifest view;
- final result preserves nested solver evolution status, state ref, role, metrics, and adds Agent0/SAGE metadata.

## Out Of Scope

This change does not implement real model fine-tuning, queue-based asynchronous rollout waiting, a default product fake subagent loop, or changes to `TaskWorker` / `TaskRuntime` ownership. It does not require Agent0 or SAGE paper-faithful reward optimization; it creates the EvoLab trainer boundary needed to host those algorithms incrementally.
