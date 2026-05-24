# EvoLab Configuration

EvoLab now treats the user-facing experiment config as a short routing contract.
Humans write the task in natural language and name the reusable subagents that
MetaAgent may route to. The CLI compiles that short config into the internal
`TaskRequest` and `TaskConfig` models used by `TaskRuntime`.

## User-Facing Shape

The canonical biology extraction config is:

```bash
configs/biology_component_extraction_v1_generic_subagents.yaml
```

It contains only:

- `task`: natural-language task description, including success criteria and
  any files, datasets, schemas, policies, or output requirements the task needs.
- `meta_agent.system_prompt`: the routing policy and JSON-only response
  contract.
- `meta_agent.memory_backend`: optional MetaAgent memory backend binding. When
  set, MetaAgent retrieves and updates its own agent-scope memory before and
  after each routing decision.
- `subagents`: reusable subagent names and system prompts. The MetaAgent may
  route only to these names or to `END`.
- `lab_root`: default Lab output root.
- `backends`: LLM, memory, and skill backend choices.
- `evolve_worker`: current evolution-worker settings or placeholders.

Do not put task-specific executable roles, fixed DAGs, `task_config.roles`, or
structured task fields into this config. Domain details belong in natural
language task text or in domain package resources.

## MetaAgent Route Contract

MetaAgent returns exactly one JSON object per dispatch decision.

Dispatch a subagent:

```json
{"route":"ExecAgent","instruction":"Inspect assigned article packages and extract candidates.","metadata":{}}
```

Finish the task:

```json
{"route":"END","instruction":"Final artifacts were written.","metadata":{"final_answer":"..."}}
```

The runtime validates the route against the configured subagent names. It also
supports the older internal `DispatchDecision` shape for legacy demos and tests,
but new experiment configs should use `route`.

## Lab State And Progressive Disclosure

MetaAgent does not receive the full Lab filesystem by default. Before each
route decision, the runtime builds a prompt-safe Lab state payload:

- `index`: machine-readable refs and compact summaries for the current task,
  recent subagent reports, artifacts, backend states, training samples, and
  trajectory counts.
- `digest`: short natural-language summary plus the most relevant recent
  report/artifact snippets.
- `requested_details`: details explicitly requested by the previous MetaAgent
  decision.

The Lab persists higher-level state under:

```text
registries/lab_state/
  run_ledgers/<task_id>.json
  subagent_reports.jsonl
  artifact_index.jsonl
  training_index.jsonl
  indexes.jsonl
  digests.jsonl
```

Raw trajectories remain in `registries/trajectory/` and are the training trace
pool. Lab state records are curated pointers over that pool:

- run ledgers track task status, final answer, failure reason, and run refs;
- subagent reports capture the returned report/final answer, coverage,
  failures, skipped items, artifact refs, and dispatch metadata;
- artifact indexes identify producer runs and artifact roles/statuses;
- training indexes point to LLM call refs suitable for future SFT/RL data
  export;
- index/digest snapshots record what MetaAgent saw at each decision point.

To request more detail, MetaAgent may include refs in dispatch metadata:

```json
{
  "route": "ExecAgent",
  "instruction": "Continue from the failed table extraction report.",
  "metadata": {
    "lab_state_detail_requests": {
      "subagent_reports": ["report-subagent-123"],
      "artifacts": ["artifact-subagent-123-001"]
    }
  }
}
```

The next MetaAgent input will expand only those requested refs under
`lab_state.requested_details`.

## Internal Compilation

`evolab.cli._compile_experiment_config(...)` expands a short config into:

- `TaskRequest(origin=human, purpose=science)`
- `TaskConfig`
- one `RoleSpec` per configured subagent
- `RuntimePolicy` with workflow planning enabled
- route-contract metadata
- default scientific IE tools unless the config overrides `allowed_tools`

This compiled form is an implementation detail. It is what keeps the runtime
strictly typed without forcing humans to hand-write verbose internal schemas.

## Biology Component Extraction

The biology extraction config uses only generic reusable subagents:

- `SurveyAgent`
- `DesignAgent`
- `ExecAgent`
- `CriticAgent`
- `WriteAgent`

These names are reusable role identities, not biology-specific agents. Biology
specificity appears in the `task` text and in resources under:

```bash
domain_packages/biology_component_extraction_v1/
```

The old long biology config has been removed. Use
`configs/biology_component_extraction_v1_generic_subagents.yaml` for new biology
component extraction experiments.

## Running

Use `clean-run` with an explicit Lab root when testing:

```bash
python3 -m evolab.cli clean-run \
  configs/biology_component_extraction_v1_generic_subagents.yaml \
  --lab-root /tmp/evolab-biology-component-run
```

For a two-paper real-service verification of native mem0 with OpenRouter
DeepSeek, use:

```bash
python3 -m evolab.cli clean-run \
  configs/biology_component_extraction_v1_two_paper_mem0_deepseek.yaml \
  --lab-root /root/bio/evolab_component_ie_v1_two_paper_mem0_deepseek
```

That config reads `OPENAI_API_KEY`, calls `deepseek/deepseek-v4-flash` through
OpenRouter chat completions, and uses `openai/text-embedding-3-small` through
OpenRouter embeddings for native mem0 memory. It configures separate
`mem0-meta-memory`, `mem0-agent-memory`, and `mem0-task-memory` backends.

Native EverOS memory uses the same backend binding fields, with
`method: everos`. It is implemented inside EvoLab with local SQLite stores,
not by calling an EverOS/EverCore service:

```yaml
backends:
  memory:
    everos-agent-memory:
      type: method
      method: everos
      store_path: registries/memory/everos-agent.sqlite
      llm_backend: memory-extractor
      embedding_backend: memory-embedding
      scene_similarity_threshold: 0.78
      recollection_mode: scene
```

`clean-run` clears and recreates the Lab root. Do not point it at a directory
that contains outputs you need to preserve.
