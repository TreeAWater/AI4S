# V1 Release Checklist

## Release Definition

V1 is complete when the same runtime contracts can run with MVP backends:

- short human-authored experiment config compiled to internal runtime contracts
- MetaAgent route JSON over configured reusable subagents
- Native mem0 memory method through `MethodMemoryBackend`
- GraphSkillBackend seed graph
- ApiLLMBackend
- SFT dry-run evolution produces local-trainable rollout states
- clean Lab replay and traceability
- Lab state index/digest snapshots for MetaAgent progressive disclosure
- V1 docs and explicit V2 boundary

## Demo Commands

Real backend demo:

```bash
python3 -m evolab.cli clean-run configs/demo_v1.yaml --lab-root /tmp/evolab-demo-v1-real-v7
```

CI/offline demo:

```bash
python3 -m evolab.cli clean-run configs/demo_v1_ci.yaml --lab-root /tmp/evolab-demo-v1-ci
```

Biology generic-subagent experiment config:

```bash
python3 -m evolab.cli clean-run \
  configs/biology_component_extraction_v1_generic_subagents.yaml \
  --lab-root /tmp/evolab-biology-component-run
```

Focused validation used during V1 sign-off:

```bash
pytest tests/test_task_worker.py tests/test_api_llm_backend.py \
  tests/test_scientific_ie_tool_coverage.py tests/test_cli_clean_run.py \
  tests/test_memory_replay.py -q
```

Observed result: `77 passed in 2.29s`.

Full-suite release gate:

```bash
pytest -q
```

Observed result: `321 passed in 3.66s`.

## Real Demo Evidence

Latest real V1 run path:

```text
/tmp/evolab-demo-v1-real-v7
```

Key generated files:

- `/tmp/evolab-demo-v1-real-v7/registries/trajectory/subagent.jsonl`
- `/tmp/evolab-demo-v1-real-v7/registries/trajectory/llm_calls.jsonl`
- `/tmp/evolab-demo-v1-real-v7/registries/trajectory/evolution.jsonl`
- `/tmp/evolab-demo-v1-real-v7/registries/backend_state/states.jsonl`
- `/tmp/evolab-demo-v1-real-v7/configs/skills/graphs/scientific_ie_seed_graph_v1.updates.jsonl`
- `/tmp/evolab-demo-v1-real-v7/evolution/llm/evo-09f6d9c2-4260-4dc5-8291-b0715562bed5/adapter.json`
- `/tmp/evolab-demo-v1-real-v7/evolution/llm/evo-09f6d9c2-4260-4dc5-8291-b0715562bed5/metrics.json`

Observed result:

- task completed in `queues/tasks/done`
- real API LLM produced 4 `LLMCallRecord`s
- GraphSkillBackend selected scientific IE skills and required tools
- scientific IE `read_text` tools read the seeded schema and article from Lab root
- `json_schema_validate` validated the extracted records with `status="ok"`
- final answer extracted `pTet`, `B0034`, `sfGFP`, and `B0015`
- Native mem0 agent/task memory wrote V1 state refs
- SFT dry-run produced a promoted candidate evolution record and local-trainable state manifest
- GraphSkillBackend update summary was written inside the Lab copy, not next to the repo source graph
- memory replay for `/tmp/evolab-demo-v1-real-v7` returned `ok: true`

## Acceptance Traceability

| AC | V1 Evidence |
| --- | --- |
| 1 | Short experiment config compiles to internal `TaskConfig`/`RoleSpec`; legacy internal configs remain supported for deterministic demos. |
| 2 | Meta-agent route/dispatch contracts, route validation, malformed JSON retry, and `END` handling are covered by runtime tests. |
| 3 | Role validation and backend binding are covered in `tests/test_task_worker.py` and CLI config builder tests. |
| 4 | Runtime supports flat and workflow-plan paths; workflow planning is covered by `tests/test_task_runtime_workflow_plan.py` and scientific IE DAG tests. |
| 5 | Lab layout writes task queues, registries, configs, inputs, and evolution artifacts under the selected lab root. |
| 6 | `TrajectoryRegistry` persists raw subagent, LLM call, tool call, event, and evolution records in JSONL. `LabStateRegistry` persists curated run ledgers, subagent reports, artifact indexes, training indexes, and MetaAgent-visible index/digest snapshots. |
| 7 | Tool traces and artifact refs are captured in subagent records; artifact refs are also indexed by producer run and role/status; evolution artifacts are registered in backend state. |
| 8 | Every subagent run performs agent and task memory retrieval before prompt construction. |
| 9 | Every subagent run writes agent and task memory after completion and records update lineage. |
| 10 | `MethodMemoryBackend` with native mem0 method and `GraphSkillBackend` are configured by `configs/demo_v1.yaml` and covered by targeted tests. |
| 11 | `ApiLLMBackend.instantiate()` runs the real demo; `ApiLLMBackend.evolve()` returns skipped for non-trainable API state. |
| 12 | SFT/OPSD dry-run trainers write local-trainable state manifests; `LocalTrainableLLMBackend` loads promoted states for rollout. |
| 13 | Promotion guard tests cover promoted, not recommended, failed, skipped, and invalid promotion cases. |
| 14 | `BackendStateRegistry` records memory and LLM state lineage with parent refs and active-state metadata. |
| 15 | Backend replacement is config-driven: `demo_v1.yaml` uses MVP backends, `demo_v1_ci.yaml` preserves deterministic fake paths, and the biology config declares only backend choices plus short task/subagent prompts. |
| 16 | Evolution contracts, budget model, instance snapshot compatibility, and standard metrics remain covered by contract and promotion tests. |
| 17 | Evolution artifact root, invalid artifact root failures, budget failures, lineage, and four promotion guards are covered by evolution tests. |

## V2 Boundary

Not part of V1:

- real LoRA/SFT training inside clean-run
- async standalone EvolveWorker process
- federated consolidation
- public skill governance
- full resource mining
- production HITL integrations
- production sandboxing and remote artifact stores
