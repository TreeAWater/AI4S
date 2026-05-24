# EvoLab Framework Skeleton Design

Date: 2026-05-02

## Goal

Build the first framework skeleton for EvoLab: Self-Evolving Agents for Scientific Research. This milestone is interface-first. It should give multiple contributors stable contracts and ownership boundaries without implementing backend algorithms such as memory retrieval, skill graph evolution, reward design, SFT/RL, LoRA training, or PEFT internals.

The first implementation should define a minimal but coherent runtime spine:

- Pydantic v2 data contracts for external records, jobs, requests, results, and configs.
- `Protocol` / ABC interfaces for backends, trainers, queues, registries, and runtimes.
- Thin TaskWorker and EvolveWorker skeletons.
- Filesystem Lab, queue, and registry implementations sufficient for tests.
- Real API-backed LLM and Mem0-backed memory adapters.
- Blank extension points for algorithms that are intentionally not implemented yet.

## Architecture

EvoLab V1 has two worker classes:

- `TaskWorker`: a rollout worker.
- `EvolveWorker`: a training worker.

There is no separate `ToolsWorker` in V1. Tool calls and LLM inference happen synchronously inside the TaskWorker subagent loop through an in-process `ToolRuntime`.

`LLMBackend` is an inference/runtime construction abstraction. It loads API models, local models, tokenizers, adapters, and active state for a subagent run. It does not own training. Parameter updates happen in EvolveWorker through trainer / training runtime interfaces.

## Package Layout

```text
evolab/
  contracts/
    common.py
    task.py
    dispatch.py
    retrieval.py
    tools.py
    llm.py
    records.py
    state.py
    evolution.py
  backends/
    llm.py
    memory.py
    skills.py
    trainers.py
  runtime/
    task_worker.py
    task_runtime.py
    subagent_loop.py
    evolve_worker.py
    promotion.py
  registries/
    trajectory.py
    backend_state.py
    task.py
  lab/
    layout.py
    queue.py
  config/
    task_config.py
```

`contracts/` is the shared dependency for other teams. Runtime modules may import contracts, but contracts should not import runtime implementation.

## TaskWorker

TaskWorker is a general rollout worker, not only a test-time worker. It consumes `TaskJob`s from `lab/queues/tasks/` and runs the same TaskRuntime for:

- Human scientific tasks.
- Proposer-generated training rollouts.
- Benchmark tasks.
- Scheduler-created evaluation or regression tasks.

TaskWorker startup initializes:

- Lab layout.
- Task and evolve queues.
- Trajectory, task, and backend state registries.
- Backend registry.
- All configured `MemoryBackend`, `SkillBackend`, and `LLMBackend` instances.
- `ToolRegistry` and in-process `ToolRuntime`.
- Prompt builder.

In the current skeleton, TaskWorker startup initializes backend runtimes from explicit backend bindings and `BackendBinding.state_ref`. Per-subagent active-state resolution from `BackendStateRegistry` is a later TaskRuntime/subagent-loop integration point; the skeleton does not call `resolve_active_state(...)` during task execution.

TaskWorker does not train, call parameter update functions, wait for evolve jobs, register candidate states, or make promotion decisions. After task close, it may write `LLMEvolutionRequest` payloads and enqueue evolve jobs for configured trainable backends.

## EvolveWorker

EvolveWorker consumes `EvolveJob`s from `lab/queues/evolve/`. It runs trainer / training runtime logic, records queue outcomes, validates promotion, registers candidate backend states, and applies the V1 promotion gate.

When EvolveWorker's proposer creates new tasks, EvolveWorker does not execute them. It serializes them as `TaskRequest(origin="proposer", purpose="training_rollout", ...)` into `lab/queues/tasks/`. TaskWorker runs those rollouts, and EvolveWorker later queries their trajectories for training.

The framework registers candidate states and promotes active state only through `FileBackendStateRegistry`. TaskWorker does not import EvolveWorker and never claims evolve jobs; its only evolution responsibility is queue handoff by writing an evolution request payload and enqueueing it.

V1 consolidation is represented in schema and promotion role validation. The current EvolveWorker skeleton does not reject consolidation jobs by mode; a future enablement policy may fail fast or skip before trainer dispatch.

## Task Provenance

Human scientific tasks and proposer-generated tasks share one request/job interface. They must remain distinguishable through provenance fields:

```python
origin: human | proposer | benchmark | scheduler
purpose: science | training_rollout | evaluation | regression
producer_ref: str | None
round_id: str | None
```

For proposer-generated tasks, the schema must also explain how the task relates to one or more human problems. `origin=proposer` requires `proposed_task_relation`.

`ProposedTaskRelation` records:

- Human anchor task refs.
- Human anchor trajectory refs.
- Proposer input refs.
- Relation type.
- Rationale for the relation.
- Target capabilities.
- Expected transfer back to the human anchor.
- Evaluation target task refs.

This lets training and promotion evaluate whether proposed rollouts actually improve performance on the human problems that motivated them.

## ToolRuntime

V1 uses in-process synchronous tool execution:

```text
LLM generate -> tool call -> ToolRuntime.execute -> ToolResult -> append observation -> LLM generate
```

`ToolRuntime.prepare(...)` builds a `ToolBundle` from:

- `SkillBundle.required_tools`
- `RoleSpec.allowed_tools`
- `RuntimePolicy`

`ToolRuntime.execute(...)` validates tool calls, executes the tool, records traces, registers artifacts, and returns `ToolResult`.

The interface should remain replaceable. Future async or remote tool execution can live behind the same `ToolRuntime` contract.

## Lab, Queues, And Registries

V1 uses filesystem Lab as the shared source of truth:

```text
lab/
  configs/
  queues/
    tasks/
    evolve/
  tasks/
    <task_id>/
  trajectories/
    meta_agent/
    subagent/
    llm_calls/
    evolution/
  evolution/
    llm/
      <evolution_run_ref>/
        request.json
        events.jsonl
  snapshots/
  registries/
    trajectory/
    backend_state/
    snapshots/
    task/
```

Queues carry small payloads or URIs:

- `TaskJob.request_payload_uri -> TaskRequest.json`
- `EvolveJob.request_payload_uri -> LLMEvolutionRequest.json`

Filesystem queue states are `queued | claimed | done | failed | skipped`. Claiming should use atomic rename or a lock file. Crash recovery can reclaim stale claimed jobs using `claimed_at + timeout`.

Core registries:

- `TrajectoryRegistry`: meta-agent runs, subagent runs, LLM calls, evolution runs, and query support.
- `BackendStateRegistry`: active state resolution, candidate state registration, promotion, and lineage.
- `SnapshotRegistry`: typed toolset, skill, environment, reward-policy, and miscellaneous snapshots.
- `TaskRegistry`: task requests, task state, origin queries, and human-anchor queries.

`LLMCallRecord` is the canonical call-level record for SFT / distillation export. Each runtime LLM call records the exact input messages, output messages, generation config, tool specs, action, raw response metadata, backend id, and resolved model name. Tool-use calls record the assistant tool-call message as the output; the following call's input includes the tool-result message. `SubagentRunRecord.llm_call_refs` and meta-agent run metadata point back to these records so both solver/reviewer and meta-agent teacher traces are reconstructable.

`BackendStateRegistry` is the registry contract for active state resolution, candidate state registration, promotion, and lineage. The current TaskWorker skeleton does not yet read active state from it during subagent execution, so promoted state is available to later integration rather than automatically picked up by this skeleton.

## First Implementation Scope

Implement:

- Pydantic contracts with JSON serialization and validation.
- Proposer task validation: proposer tasks require human anchor relation.
- Backend and trainer protocols.
- `ApiLLMBackend` with real provider calls. The first provider is an OpenAI Responses adapter with credentials and model parameters loaded from config or environment, strict structured-output schema validation, and function tool-call parsing. Lab initialization config may point at an untracked JSON secret source through `api_env.json_path`; LLM backend entries use `type: api`, `env_ref`, and a model name to bind one named secret entry without committing the key. Inline API keys in tracked backend config are rejected.
- `Mem0MemoryBackend` using the Mem0 Python SDK, adapting EvoLab scopes into Mem0 user IDs / filters.
- `GraphSkillBackend` as a real filesystem-backed graph store sufficient for get / look_at / version metadata. It uses canonical `skill_id`, UTF-8 JSON graph files, UTF-8 JSONL update logs, and explicit blank extension points for mining / rewiring.
- `LocalTrainableLLMBackend` inference/state-loading interface shape. Actual vLLM, adapter loading, and local model execution remain a `NotImplementedError` extension point in the skeleton if the environment dependency is not available.
- Trainer protocols and EvolveWorker wiring. `SFTTrainer` may provide a dry-run smoke-test path plus an optional dependency-gated `transformers` backend; production LoRA/PEFT remains out of scope.
- Filesystem work queue.
- Filesystem JSON/JSONL registries.
- TaskWorker skeleton that claims a task and runs thin TaskRuntime. The default TaskRuntime dispatch loop raises `NotImplementedError`; tests inject test-local runtimes / dispatch loops instead of product fake completion.
- EvolveWorker skeleton that claims an evolve job, appends run ledger events, calls a configured trainer if present, records skipped / failed / evolution results, validates promotion, and registers / promotes backend state through `FileBackendStateRegistry` when a real result exists.
- Unit tests for contracts, queue claim, proposer relation validation, tool runtime behavior, backend initialization, registry behavior, and promotion validation.

Do not implement:

- vLLM or LoRA training.
- Real proposer task generation.
- Human intervention runtime.
- Distributed queues.
- A productized CLI.

Do not add product mock implementations. If a component is not implemented, leave it as an explicit interface, blank extension point, or `NotImplementedError` path. Tests may use small test-local fakes only when they are not shipped as runtime backends.

## Promotion

V1 promotion is `accept_all` plus mandatory validation:

- `new_state_ref` must be non-empty.
- At least one artifact must be under `artifact_root_uri`.
- For non-cold-start training, `eval_score_after` must be present.
- `lora_role` must match the expected role for the evolution mode.

TaskWorker does not participate in promotion. In the current skeleton, promoted active state is recorded in the registry for later TaskRuntime/subagent-loop integration; TaskWorker does not yet read it on later subagent launches.

## Acceptance Criteria

- A static config can start a meta-agent and multiple dispatchable subagent roles.
- Human tasks and proposer tasks use one `TaskRequest` / `TaskJob` interface.
- Proposer tasks preserve human anchor relation and expected transfer metadata.
- TaskWorker initializes configured memory, skill, and LLM backend instances at startup.
- API-backed LLM calls go through a real provider adapter, not a product mock.
- API-backed LLM config can resolve an `env_ref` from an untracked `api_env.json_path` secret file and instantiate the provider client without exposing the key in tracked config. Relative secret-file resolution is limited to the config directory and its parent instead of walking arbitrary ancestors.
- The OpenAI Responses adapter validates strict structured-output schemas and parses function tool calls.
- Memory search/add goes through Mem0 SDK when `Mem0MemoryBackend` is configured.
- Tool calls execute synchronously through in-process `ToolRuntime`.
- Runtime LLM calls are saved as `LLMCallRecord` entries and are linked from subagent runs and meta-agent run metadata.
- Subagent trajectory contracts can record backend state refs; the current skeleton initializes those refs from explicit backend bindings, with registry active-state resolution left for follow-up integration.
- EvolveWorker can consume an existing trajectory through an evolve job, record queue outcome, and update backend state through the registry when promotion is valid.
- Promotion updates `BackendStateRegistry`; wiring later TaskWorker subagent launches to consume promoted active state is a follow-up integration point.
- Product code contains no mock backend or trainer; tests may use test-local fakes only.
