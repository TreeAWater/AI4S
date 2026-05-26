## **Goal**

在两周内按每周 6 个工作日(一天buffer)交付 EvoLab 的两个版本：

- **V0, Days 1-6:** 可运行骨架。证明 backend-centric dynamic workflow 可以端到端跑通，并且所有关键运行记录、artifact、state lineage 都能落到 Lab。

- **V1, Days 7-12:** MVP backend path。把 V0 的 fake backends 替换或扩展为 native mem0 memory method、GraphSkillBackend、ApiLLMBackend、local trainable mock，并完成失败路径、replay、docs 和 release sign-off。

V0 优先证明系统形状正确；V1 优先证明 MVP backend 可以接入同一组 contracts。真实 LoRA/SFT、federated consolidation、public skill governance、full resource mining 和复杂 HITL 审批流全部明确属于 V2 或后续版本。

## **Release Definitions**

### **V0 Release**

V0 是一个可运行、可测试、可复现的 skeleton：

- 静态 task config 可以启动一个 meta-agent 和多个 schedulable subagent roles。
- Main Runtime 支持动态 dispatch，不依赖固定 DAG。
- 每个 subagent run 都执行 memory retrieval、skill retrieval、tool preparation、LLM loop、trajectory save、memory update、skill update。
- Lab filesystem layout、TrajectoryRegistry、BackendStateRegistry 可用。
- FakeMemoryBackend、FakeSkillBackend、FakeLLMBackend、FakeEvolutionBackend 支持 deterministic tests。
- Mock offline evolution 可以生成 candidate state ref，运行 promotion gate，并记录 lineage。
- 一个 demo task 可以从 fresh checkout 运行完成，产出 state snapshots、trajectories、tool traces、artifacts 和 backend state records。
### **V1 Release**

V1 在 V0 skeleton 上接入 MVP backend 并完成 hardening：

- Native mem0 `MemoryMethod` 通过 `MethodMemoryBackend` 可用，CI 默认使用 fake LLM/embedding 驱动本地 SQLite mem0。
- GraphSkillBackend 支持 seed graph、skill retrieval、required tools、version refs 和 update summaries。
- ApiLLMBackend 支持 structured dispatch output，`evolve()` 返回 skipped。
- LocalTrainableLLMBackend mock 支持有效 `new_state_ref`、metrics、artifact refs 和 promotion/rejection scenarios。
- User-facing experiment config 使用短格式：自然语言 task、MetaAgent prompt、subagent name/prompt、Lab 路径和 backend 配置。CLI 负责编译成内部 typed `TaskRequest`/`TaskConfig`。
- MetaAgent route contract 使用 `{"route":"<subagent>","instruction":"..."}` 或 `{"route":"END", ...}`；route 候选来自 config 中的 subagent pool。
- Runtime 覆盖 4 条 promotion guard 负测。
- LLMCallRecord、ToolTrace、clean-lab replay test 可用。
- Memory、Skill、Params/LLM README 和 demo config 完成。
- Acceptance criteria 1-17 有 traceability；未进 V1 的项目显式标为 V2。
## **Architecture Decisions Frozen For This Plan**

- **V0 execution model:** in-process `inline_sync`。Task close 前完成 mock offline evolution，`TaskState.status="evolving"` 期间不进入终态。

- **V1 queue model:** 不引入独立 EvolveWorker 进程；可以保留 queue schema 作为 V2 extension point，但 V1 runtime 仍同步执行 evolution hook。

- **Backend boundary:** Main Runtime 只编排 lifecycle calls，不实现 memory algorithm、skill graph algorithm、reward design、SFT/RL 或 PEFT details。

- **Skill updates:** V1 是 in-place graph update，没有 human approval gate 或 promotion policy。

- **Memory updates:** V1 是 in-place update，没有 promotion policy。

- **LLM state promotion:** V1 使用 accept-all-after-backend-recommendation，并强制 4 条 runtime guard。

- **Consolidation:** `LLMEvolutionMode.CONSOLIDATION` 在 V1 fail fast；`InstanceSnapshot` 只保留 schema compatibility，不实现 privacy barrier。

## **Ownership**

- **ziyi owns Memory:** Memory contracts, fake memory backend, Mem0 adapter, scope mapping, memory update records, memory state refs, memory tests and docs.

- **jiayu owns Skills:** Skill contracts, fake skill backend, GraphSkillBackend, graph schema/versioning, required tool aggregation, update summaries, skill tests and docs.

- **chang owns Params/Runtime:** LLM contracts, LLMRuntime, fake/API/local trainable backends, LLMEvolutionRequest/Result, promotion gate, registries, runtime loop, replay and release tests.

- **All owners review shared contracts:** Message, ArtifactRef, GitRef, TaskConfig, TaskState, DispatchDecision, Lab layout, TrajectoryRegistry, BackendStateRegistry, run records.

## **Daily Owner Assignment**

Day

ziyi - Memory

jiayu - Skill

chang - Params/Runtime

Daily integration checkpoint

Day 1

Freeze memory contracts: `RetrievalRequest`, `BackendScope`, `MemoryItem`, `MemoryBundle`, `MemoryAddRequest`, `MemoryUpdateResult`, Mem0 config shape.

Freeze skill contracts: `SkillItem`, `SkillBundle`, `SkillObservationRequest`, `SkillUpdateResult`, graph node/edge metadata, `required_tools` semantics.

Freeze shared runtime and params contracts: schema conventions, `TaskConfig`, `TaskState`, `DispatchDecision`, `LLMBackend`, `LLMEvolutionRequest/Result`, repo skeleton.

All owners approve shared schema names and one minimal contract import test.

Day 2

Implement `FakeMemoryBackend.search/add`, memory version refs, memory contract round-trip tests.

Implement `FakeSkillBackend.get/look_at`, graph version refs, required tool output, skill contract round-trip tests.

Implement Lab resolver, `TrajectoryRegistry`, `BackendStateRegistry`, fake LLM runtime shell, shared record round-trip tests.

Registry save/load tests pass for memory, skill, LLM, and evolution records.

Day 3

Wire memory pre-run search and post-run add into runtime; verify empty memory bundle is valid and recorded.

Wire skill pre-run get and post-run look_at; verify skill bundle required tools are passed to tool preparation.

Implement config loader, meta-agent route/dispatch parse/validate/retry, fake subagent loop, no-fixed-DAG integration test.

Demo task dispatches two subagents and finishes with saved trajectories.

Day 4

Define memory bundle formatting for prompt context and memory failure behavior expected by runtime.

Co-own ToolRuntime required-tools behavior; test missing/invalid tools and skill context in prompts.

Implement `ToolRegistry`, `ToolRuntime.prepare/execute`, `PromptBuilder`, artifact registration, tool trace logging.

Valid tool calls produce artifacts/traces; invalid tools are rejected.

Day 5

Record memory update artifacts and memory state refs under Lab paths; add memory lineage assertions.

Record skill update artifacts, graph version refs, and skill state refs; add skill lineage assertions.

Implement task-close mock evolution, `FakeEvolutionBackend`, promotion guard, `EvolutionRunRecord`, state promotion tests.

Promotion, rejection, skipped, failed, and guard-failure tests pass.

Day 6

Sign off V0 memory path and add memory known-issues notes for V1.

Sign off V0 skill path and add skill known-issues notes for V1.

Create `configs/demo_v0.yaml`, clean-run command, V0 release notes, full V0 integration test.

Fresh checkout test suite and V0 demo pass without external services.

Day 7

Implement native mem0 method, scoped method backend mapping, SQLite persistence, embedding-backed retrieval, memory update summaries.

## **Week 1: V0, Runnable Skeleton**

### **Day 1 — Contract Freeze And Repo Skeleton**

Implement GraphSkillBackend seed graph, node/edge schema, graph version refs, retrieval by tags/text, update summary writes.

Add backend factory/config wiring for MVP backends while preserving fake backend CI defaults.

Mem0 adapter and graph backend tests pass through the same contracts used by fakes.

Day 8

Verify V1 memory records integrate with generated trajectories and backend state refs.

Verify V1 skill records include graph context, scripts/resources refs, and required tools.

Implement `ApiLLMBackend`, skipped evolution, local trainable mock, generation metadata and evolution scenarios.

API skipped evolution and local mock promotion/rejection tests pass.

Day 9

Replace fake memory path in `demo_v1` config; verify memory version refs in every subagent run.

Replace fake skill path in `demo_v1` config; verify graph context and required tools in every subagent run.

Ensure runtime instantiates configured backends without code changes; run full V1 integration task.

V1 demo completes with MVP backend paths and offline evolution records.

Day 10

Add memory replay checks and docs for reproducing search/add from saved refs.

Add skill replay checks, tool trace validation, and docs for reproducing skill retrieval/update from graph refs.

Implement clean-lab replay test, finalize `LLMCallRecord`, finalize `ToolTrace`, add minimal HITL event logging hooks.

Replay test passes without hidden process memory.

Day 11

Add memory failure suite: empty results, malformed filters, add failure, missing version ref, degraded update record.

Add skill failure suite: duplicate candidate, missing `required_tools`, graph update failure, version mismatch, invalid tool policy.

Add LLM/evolution failure suite: invalid structured output, instantiate failure, budget exceeded, empty trajectories, invalid artifact root, consolidation fail-fast.

Negative suite passes and failed candidates remain inspectable in Lab.

Day 12

Write Memory README, close memory acceptance items, sign off memory V1.

Write Skill README, close skill acceptance items, sign off skill V1.

Write Params/Runtime README, create V1 release checklist, complete acceptance traceability, mark V2 items, coordinate release candidate.

Full suite and V1 demo pass in a fresh Lab with all owner sign-offs.

## **Current Configuration Direction**

New experiment configs should not hand-write the full internal `task_config`.
Humans provide:

- natural-language `task` description;
- `meta_agent.system_prompt`;
- reusable `subagents` mapping from role name to `system_prompt`;
- `lab_root`;
- backend selections for LLM, memory, skill, and evolve-worker settings.

The CLI compiles this short format into internal `TaskRequest`/`TaskConfig`.
MetaAgent receives the task text, subagent list, Lab state, completed runs, and
trajectory/failure summaries. It routes to one configured subagent or to `END`.

The old long biology extraction config has been removed. The active biology
component extraction config is:

```text
configs/biology_component_extraction_v1_generic_subagents.yaml
```

- [ ] Freeze shared serialization conventions: JSON format, datetime format, URI/path rules, enum values, schema version field, error status field.
- [ ] Freeze task/runtime contracts: short experiment config, internal `TaskConfig`, `TaskState`, route decision, `RoleSpec`, `RuntimePolicy`, `PromptBuildRequest`.
- [ ] Freeze backend-facing contracts for V0: `RetrievalRequest`, `MemoryBundle`, `MemoryAddRequest`, `SkillBundle`, `SkillObservationRequest`, `LLMInstantiateRequest`, `LLMEvolutionRequest`, `LLMEvolutionResult`.
- [ ] Freeze shared records: `MetaAgentRunRecord`, `SubagentRunRecord`, `LLMCallRecord`, `EvolutionRunRecord`, `ToolTrace`, `BackendStateRecord`, `ArtifactRef`, `GitRef`.
- [ ] Resolve naming mismatches before implementation: `parent_state_refs` is a list; `ToolSpec` spelling is fixed; config uses `system_prompt`; `report_ref` and `metrics_ref` are `ArtifactRef | None`.
- [ ] Create package skeleton: `src/evolab/`, `tests/`, `configs/`, sample `lab/` fixture, and importable modules.
- [ ] Checkpoint: one contract import test validates a minimal config with one meta-agent and two roles.
### **Day 2 — Lab, Registries, And Fake Backends**

- [ ] Implement Lab path resolver and expected V0 layout under `lab/tasks/<task_id>/`, `lab/registries/`, and `lab/workspaces/<task_id>/files/`.
- [ ] Implement filesystem `TrajectoryRegistry.save_run/get/query()` for meta-agent, subagent, LLM call, evolution run, and tool traces.
- [ ] Implement `BackendStateRegistry` active state read/write and lineage append.
- [ ] Implement deterministic `FakeMemoryBackend.search/add` with version refs.
- [ ] Implement deterministic `FakeSkillBackend.get/look_at` with graph version refs and required tools.
- [ ] Implement deterministic `FakeLLMBackend.instantiate` and fake LLM runtime actions.
- [ ] Checkpoint: schema round-trip and registry save/load tests pass.
### **Day 3 — Dynamic Runtime Loop With Fakes**

- [ ] Implement config loader for short experiment configs, backend bindings, prompt builders, tool policy, and max dispatch/tool step limits.
- [ ] Implement MetaAgent route parsing, schema validation, invalid-route rejection, retry recording, and terminal `END` action.
- [ ] Wire subagent pre-run lifecycle: build retrieval request, call memory search, call skill get, prepare tools, build prompt messages.
- [ ] Implement fake subagent loop for `tool_call`, `final_answer`, `ask_human`, `abort`, and max tool step stop.
- [ ] Save meta-agent and subagent trajectories after every run.
- [ ] Checkpoint: integration test runs a no-fixed-DAG task where meta-agent dispatches two subagents and then finishes.
### **Day 4 — ToolRuntime, PromptBuilder, And Artifact Flow**

- [ ] Implement `ToolRegistry`, `ToolRuntime.prepare`, `ToolRuntime.execute`, and `ToolTraceLogger`.
- [ ] Add V0 tool set: `read_file`, `write_file`, and deterministic fake execution tool. `run_python` can be included if sandbox policy is simple and tested.
- [ ] Enforce role `allowed_tools` and runtime policy during tool preparation and execution.
- [ ] Implement default PromptBuilder that composes task state, instruction, memory bundle, skill bundle, available tools, and artifact refs into canonical messages.
- [ ] Register produced artifacts through Lab-managed paths.
- [ ] Checkpoint: tests prove invalid tools are rejected and valid tool calls produce trace records and artifacts.
### **Day 5 — Mock Offline Evolution And Promotion Gate**

- [ ] Implement task-close evolution phase with `TaskState.status="evolving"` until all evolution records are saved.
- [ ] Build `LLMEvolutionRequest(mode=BASICS)` for each completed subagent run whose backend supports evolution.
- [ ] Implement `FakeEvolutionBackend` scenarios: skipped, failed, not recommended, promoted candidate.
- [ ] Implement promotion guard: non-empty `new_state_ref`, artifact under `artifact_root_uri`, `eval_score_after` for non-cold-start, and valid `lora_role`.
- [ ] Save `EvolutionRunRecord` and update `BackendStateRegistry` only after guard pass.
- [ ] Checkpoint: tests pass for promoted, not recommended, skipped, failed, and each guard failure.
### **Day 6 — V0 Demo, Negative Tests, And Release Notes**

- [ ] Create `configs/demo_v0.yaml` and one demo task that exercises two subagent roles, memory/skill retrieval, tool execution, post-run updates, and mock evolution.
- [ ] Add clean-run command that initializes a fresh Lab and runs the V0 demo from config.
- [ ] Add negative tests for malformed dispatch JSON, unknown role, missing required fields, backend failure handling, Lab write failure, and max dispatch/tool limits.
- [ ] Add V0 release notes with exact test command, demo command, generated Lab paths, and known V1 work.
- [ ] Checkpoint: fresh checkout test suite and V0 demo pass without external services.
## **Week 2: V1, MVP Backends And Hardening**

### **Day 7 — Native Mem0 Method And GraphSkillBackend MVP**

- [ ] Implement native mem0 `MemoryMethod` behind `MethodMemoryBackend` with scoped search/add, local SQLite persistence, LLM extraction, and embedding-backed retrieval.
- [ ] Map runtime memory scope metadata to method requests and record memory update summaries in Lab.
- [ ] Implement persistent `GraphSkillBackend` seed graph storage, node schema, edge schema, graph version refs, and retrieval by tags/text.
- [ ] Return `SkillBundle.required_tools`, scripts/resources refs, and graph context summary.
- [ ] Keep fake backends as deterministic CI default.
- [ ] Checkpoint: native mem0 method/backend tests and graph backend retrieval/update tests pass.
### **Day 8 — API LLM Backend And Local Trainable Mock**

- [ ] Implement `ApiLLMBackend.instantiate()` with provider/model/generation config and structured output configuration.
- [ ] Implement `ApiLLMBackend.evolve()` returning skipped with a saved evolution record.
- [ ] Implement `LocalTrainableLLMBackend` mock that writes adapter-like artifacts, metrics, and valid/invalid state refs without real model training.
- [ ] Add generation metadata capture for model ID, parameters, token usage if provided, cost if provided, and latency.
- [ ] Checkpoint: tests cover API skipped evolution and local mock promoted/not-recommended/failed scenarios.
### **Day 9 — Runtime Integration With MVP Backends**

- [ ] Replace fake memory/skill/API paths in `configs/demo_v1.yaml` while preserving fake-only config for CI.
- [ ] Ensure Main Runtime can instantiate backends from config without code changes.
- [ ] Verify every subagent run records memory version ref, skill graph version ref, prompt messages, LLM call refs, tool traces, output messages, artifact refs, and backend config/state refs.
- [ ] Add role-specific prompt builder override support where needed for demo roles.
- [ ] Checkpoint: V1 integration test runs one full task with MVP backend paths and completes offline evolution.
### **Day 10 — Replay, Observability, And HITL Minimal Hooks**

- [ ] Implement clean-lab replay test that loads saved registries and verifies task state, backend state lineage, artifact refs, memory refs, and skill refs without hidden process memory.
- [ ] Finalize `LLMCallRecord` capture for prompts, raw response, tool calls, generation params, token usage, cost, latency, and timestamps.
- [ ] Finalize `ToolTrace` capture and artifact registration from tools.
- [ ] Add minimal HITL event records for `ask_human` and human intervention safe-point polling hooks without external Feishu integration.
- [ ] Checkpoint: replay test and HITL event logging tests pass.
### **Day 11 — Failure Handling And Runtime Guards**

- [ ] Add memory failure tests: empty search result, malformed filters, add failure, missing version ref, and degraded memory update recording.
- [ ] Add skill failure tests: duplicate candidate, missing `required_tools`, graph update failure, version mismatch, invalid tool policy, and update summary persistence.
- [ ] Add LLM/evolution failure tests: invalid structured output, backend instantiate failure, budget exceeded, empty training trajectories, invalid artifact root, and consolidation fail-fast.
- [ ] Verify failed runs preserve enough records for debugging and do not corrupt active backend state.
- [ ] Checkpoint: negative suite passes and rejected candidates remain inspectable in Lab.
### **Day 12 — Docs, Acceptance Traceability, And Release Candidate**

- [ ] Write Memory README: config fields, scope mapping, Mem0 setup, fake backend usage, failure behavior, and test commands.
- [ ] Write Skill README: graph schema, seed graph, retrieval behavior, update behavior, required tool policy, versioning, and test commands.
- [ ] Write Params/Runtime README: API backend config, local trainable mock, evolution artifacts, metrics contract, promotion gate, replay, and test commands.
- [ ] Create V1 demo task and release checklist output with paths to generated trajectories, artifacts, state snapshots, and backend state records.
- [ ] Complete acceptance criteria traceability table for items 1-17.
- [ ] Mark V2 items explicitly: real LoRA/SFT, async EvolveWorker, federated consolidation, public skill governance, full resource mining, external HITL integrations.
- [ ] Checkpoint: full suite and V1 demo pass in a fresh Lab; release candidate is marked only with owner sign-off.
## **Daily Integration Rhythm**

- Start of day: each owner states the concrete contract or module they will change and the test that will prove it.
- Midday: cross-owner review for any schema/config/registry change.
- End of day: merge checkpoint with test command, result, generated Lab path, and any acceptance criteria advanced.
- Any shared contract change after Day 2 requires all owners to approve the schema migration and update tests in the same day.
## **Acceptance Criteria Traceability**

- AC 1-4: V0 Days 1-3, static config, structured dispatch, role validation, no-fixed-DAG workflow.
- AC 5-7: V0 Days 2-4, Lab root, trajectories, logs, artifacts, state snapshots, subagent lifecycle records.
- AC 8-9: V0 Days 3-4 and V1 Day 7, memory/skill retrieval and update before/after every subagent run.
- AC 10: V1 Day 7, native mem0 `MethodMemoryBackend` and GraphSkillBackend.
- AC 11: V1 Day 8, ApiLLMBackend instantiate and skipped evolution.
- AC 12-13: V0 Day 5 and V1 Day 8, local trainable mock state refs and promotion rule.
- AC 14: V0 Days 2 and 5, BackendStateRegistry lineage and rollback-ready records.
- AC 15: V0 Day 1 and V1 Day 9, backend replacement through config without Main Runtime changes.
- AC 16: V0 Day 1 and V0 Day 5, LLMEvolutionRequest/LabSignals/InstanceSnapshot/EvolutionBudget/StandardEvolutionMetrics contracts.
- AC 17: V0 Day 5 and V1 Day 11, artifact root creation, budget failure, evolution lineage, and 4 promotion guards.
## **Release Risks And Mitigations**

- **Schema churn risk:** freeze shared contracts on Day 1; any later change must include migration and tests.

- **Backend integration risk:** fake backends remain CI default; MVP backends are tested through adapters and mocked external SDKs.

- **Skill scope creep:** V1 graph backend only needs seed graph, retrieval, required tools, update summary, and version refs. Full mining stays V2.

- **Memory scope creep:** V1 native mem0 method only needs scoped search/add, local persistence, retrieval, and update summaries. Public/private memory governance stays V2.

- **Training cost risk:** V1 local trainable backend is a mock writer behind the real contract. Real LoRA/SFT stays V2.

- **Promotion safety risk:** runtime guard failures never promote state, and candidate artifacts remain available for debugging.
