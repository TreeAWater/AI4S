# Memory V1 README

This document is the V1 sign-off reference for EvoLab memory. It covers the
runtime boundary, scope model, native method backends, state refs, failure
behavior, replay checks, and test commands.

## Ownership Boundary

Memory updates happen only inside task execution. A subagent run reads memory
before prompt construction and writes memory after the run through
`TaskRuntime` / `TaskWorker` calling `MemoryBackend.add(...)`. MetaAgent may
also use memory when `MetaAgentSpec.memory_backend` is configured; it reads its
own memory before each dispatch decision and writes a compact decision summary
after a valid decision.

`EvolveWorker` does not own memory updates. Its boundary is LLM/parameter
evolution: training/evaluation candidates, promotion checks, and LLM state
lineage. Do not enqueue memory evolution jobs or mutate memory from
`EvolveWorker`.

## Scope Model

Every subagent run uses two memory scopes:

- Agent-wise memory: role-level memory whose lifecycle matches the configured
  subagent role. The default scope id is `agent:<role>`.
- Task-wise memory: shared task-local memory for all subagents in one task
  execution. The default scope id is `task:<task_id>`.

For each subagent run, runtime:

1. Builds an agent memory `RetrievalRequest` with
   `filters.memory_scope="agent"` and
   `filters.memory_scope_id="agent:<role>"`.
2. Builds a task memory `RetrievalRequest` with
   `filters.memory_scope="task"` and
   `filters.memory_scope_id="task:<task_id>"`.
3. Searches both backends before prompt construction.
4. Merges prompt-visible memory into `SubagentRunRecord.memory_bundle`.
5. Writes the memory-compatible run messages to agent memory.
6. Writes the same run messages to task memory.
7. Records both bundles and both update results in trajectory metadata.

Backends must treat `memory_scope_id` as an isolation boundary. If the same
physical backend instance backs both scopes, it must still keep
`agent:solver`, `agent:reviewer`, and `task:<task_id>` separate.

MetaAgent memory uses the same agent scope model. Its default scope id is
`agent:<meta_agent.name>`, usually `agent:meta`. This scope is separate from
subagent role scopes such as `agent:ExecAgent` and from task-wise memory.

## Memory Methods

EvoLab memory is configured through runtime `MemoryBackend`s. A method-backed
memory backend uses:

```json
{"type": "method", "method": "mem0"}
```

or:

```json
{"type": "method", "method": "everos"}
```

`mem0` selects the native `Mem0MemoryMethod` implementation behind
`MethodMemoryBackend`. It does not call the removed external Mem0 adapter path.

`everos` selects the native `EverOSMemoryMethod`. It implements EverOS-style
MemCell construction, MemScene consolidation, and scene-first reconstructive
recollection inside EvoLab. It does not call EverOS/EverCore HTTP services and
does not require MongoDB, Elasticsearch, Milvus, or Redis.

Tests and offline demos should use native methods with fake extraction and fake
embedding backends when they need deterministic behavior without external
services.

## Configuration

Memory bindings are explicit:

- `RoleSpec.agent_memory_backend` selects the role-level memory backend.
- `TaskConfig.task_memory_backend` selects the task-level memory backend.
- `MetaAgentSpec.memory_backend` optionally selects the MetaAgent memory backend.

Partial explicit subagent/task configuration is invalid. If a task explicitly
binds one side, it must bind both. MetaAgent memory is independent and optional;
when absent, MetaAgent uses only LabState, trajectory summaries, completed
runs, and requested details.

Native mem0 memory config:

```json
{
  "task_config": {
    "task_memory_backend": {"backend_id": "mem0-task-memory"},
    "meta_agent": {
      "memory_backend": {"backend_id": "mem0-meta-memory"}
    },
    "roles": {
      "solver": {
        "agent_memory_backend": {"backend_id": "mem0-agent-memory"}
      }
    }
  },
  "backends": {
    "llm": {
      "fake-memory-llm": {
        "type": "fake",
        "default_content": "{\"memory\":[]}"
      }
    },
    "embedding": {
      "fake-memory-embedding": {
        "type": "fake",
        "dimensions": 8
      }
    },
    "memory": {
      "mem0-meta-memory": {
        "type": "method",
        "method": "mem0",
        "store_path": "registries/memory/mem0-meta.sqlite",
        "llm_backend": "fake-memory-llm",
        "embedding_backend": "fake-memory-embedding"
      },
      "mem0-agent-memory": {
        "type": "method",
        "method": "mem0",
        "store_path": "registries/memory/mem0-agent.sqlite",
        "llm_backend": "fake-memory-llm",
        "embedding_backend": "fake-memory-embedding"
      },
      "mem0-task-memory": {
        "type": "method",
        "method": "mem0",
        "store_path": "registries/memory/mem0-task.sqlite",
        "llm_backend": "fake-memory-llm",
        "embedding_backend": "fake-memory-embedding"
      }
    }
  }
}
```

Native EverOS memory config:

```json
{
  "task_config": {
    "task_memory_backend": {"backend_id": "everos-task-memory"},
    "meta_agent": {
      "memory_backend": {"backend_id": "everos-meta-memory"}
    },
    "roles": {
      "solver": {
        "agent_memory_backend": {"backend_id": "everos-agent-memory"}
      }
    }
  },
  "backends": {
    "llm": {
      "memory-llm": {
        "type": "fake",
        "default_content": "{\"memcells\":[]}"
      }
    },
    "embedding": {
      "memory-embedding": {
        "type": "fake",
        "dimensions": 8
      }
    },
    "memory": {
      "everos-meta-memory": {
        "type": "method",
        "method": "everos",
        "store_path": "registries/memory/everos-meta.sqlite",
        "llm_backend": "memory-llm",
        "embedding_backend": "memory-embedding",
        "recollection_mode": "scene"
      },
      "everos-agent-memory": {
        "type": "method",
        "method": "everos",
        "store_path": "registries/memory/everos-agent.sqlite",
        "llm_backend": "memory-llm",
        "embedding_backend": "memory-embedding",
        "scene_similarity_threshold": 0.78,
        "default_search_top_k": 8
      },
      "everos-task-memory": {
        "type": "method",
        "method": "everos",
        "store_path": "registries/memory/everos-task.sqlite",
        "llm_backend": "memory-llm",
        "embedding_backend": "memory-embedding"
      }
    }
  }
}
```

Memory-off config for ablations:

```json
{
  "task_config": {
    "task_memory_backend": {"backend_id": "null-task-memory"},
    "roles": {
      "solver": {
        "agent_memory_backend": {"backend_id": "null-agent-memory"}
      }
    }
  },
  "backends": {
    "memory": {
      "null-agent-memory": {"type": "null"},
      "null-task-memory": {"type": "null"}
    }
  }
}
```

`FakeMemoryBackend` remains available only for unit tests that are not
validating mem0 behavior:

```json
{
  "type": "fake"
}
```

Use `NullMemoryBackend` when memory is intentionally disabled. Use
`FakeMemoryBackend` for narrow deterministic unit tests. Use native mem0 or
native EverOS with a fake memory-extraction LLM and fake embedding backend for
integration tests that must exercise method behavior, SQLite persistence,
scoped search/add, and method state refs.

## Backends

`MemoryBackend` is the runtime-facing interface used by `TaskRuntime` and
`TaskWorker`.

`MethodMemoryBackend` adapts EvoLab runtime contracts to algorithm-specific
memory methods. Native mem0 is implemented by `Mem0MemoryMethod`, which:

- persists scoped memories in a SQLite store under Lab registries
- requires explicit `llm_backend` and `embedding_backend` bindings
- uses the LLM runtime for add-time memory extraction
- uses the embedding runtime for semantic retrieval
- returns `MemoryBundle` and `MemoryUpdateResult` objects with logical state refs

Native EverOS is implemented by `EverOSMemoryMethod`, which:

- calls the configured LLM backend to construct MemCells and consolidate
  MemScenes
- calls the configured embedding backend for MemCell, record, and query vectors
- stores messages, MemCells, MemScenes, scene membership, searchable records,
  scope versions, and optional audit events in local SQLite
- returns scene-grounded reconstructed `MemoryItem`s from `search()`
- optionally uses `recollection_mode="agentic"` to call the LLM during search
  for necessary/sufficient context selection

`NullMemoryBackend` is the configured off switch for ablations and workflows
where memory must be disabled.

`FakeMemoryBackend` stores deterministic in-process `MemoryItem`s by
`memory_scope_id`. Use it for focused unit tests that do not validate native
mem0 behavior.

The fake extraction LLM used by native mem0 tests is not a memory backend. It
only returns deterministic extraction JSON to `Mem0MemoryMethod.add(...)`. The
fake embedding backend likewise supplies deterministic vectors for search/add
ranking. Both let the runtime exercise native mem0 end to end while keeping CI
offline.

External memory service availability, latency, retention policy, and
operational SLA are deployment concerns outside this V1 native method path.

## State Refs

Fake memory state refs use:

```text
fake-memory://<backend_id>/<len>:<encoded_scope_id>/v<N>
```

Native mem0 method state refs use:

```text
method://mem0/<len>:<backend_id>/<len>:<memory_scope>/<len>:<memory_scope_id>/v<N>
```

Native EverOS method state refs use:

```text
method://everos/<len>:<backend_id>/<len>:<memory_scope>/<len>:<memory_scope_id>/v<N>
```

These are EvoLab logical state refs. They provide stable lineage for runtime
records and include the backend identity plus memory scope so replay can
distinguish two backends using the same logical scope id.

For an updated memory result, `previous_state_ref` links to the pre-run bundle
state and `state_ref` identifies the post-update logical state. Runtime stores
updated states in `BackendStateRegistry` with:

- `backend_type="memory"`
- `created_from_task_id`
- `created_from_run_ref`
- `parent_state_refs`
- scope metadata
- original `MemoryUpdateResult`

Non-updated statuses do not register active backend state.

## Runtime Records

`SubagentRunRecord.memory_bundle` stores the combined prompt-visible memory.
Per-scope details are in `SubagentRunRecord.metadata`:

- `agent_memory_bundle`
- `task_memory_bundle`
- `agent_memory_update_result`
- `task_memory_update_result`
- `memory_update_result` for backward-compatible agent-side summary

Tool result messages are converted into assistant-compatible memory messages
before memory update. This keeps memory methods from receiving unsupported
`role="tool"` messages while preserving tool result content.

`MetaAgentRunRecord.metadata` stores MetaAgent memory details when configured:

- `meta_memory_retrieval_request`
- `meta_memory_bundle`
- `meta_memory_update_result`

## Replay

`evolab.runtime.memory_replay.replay_memory_trace(lab_root, task_id=None)` reads
only Lab registries and reconstructs memory traceability without process-local
backend state.

It validates:

- every subagent run has agent and task memory bundle metadata
- updated memory results have matching `BackendStateRecord`s
- backend ids and scope metadata match
- `parent_state_refs` include the expected previous state
- repeated scope ids continue from the prior update state

Replay does not instantiate memory backends or read process-local memory. For
native mem0 refs it keys lineage by backend id, memory scope, and scope id, then
checks registry parent refs and scope metadata against the saved trajectory.

Example:

```python
from evolab.runtime.memory_replay import replay_memory_trace

report = replay_memory_trace("lab/demo_v1", task_id="demo-v1")
assert report.ok, report.issues
```

## Failure Behavior

V1 memory failure behavior is explicit:

- Empty search results are valid and produce an empty `MemoryBundle`.
- Malformed native method scope filters fail fast with `ValueError`.
- Native mem0 search failures propagate because prompt construction cannot use
  a missing memory bundle.
- Native mem0 add failures return `MemoryUpdateResult(status="failed")` with
  error metadata. Failed updates do not register a new active backend state;
  `state_ref` may remain the previous logical state ref.
- Native EverOS extraction parse failures return `MemoryUpdateResult(status="failed")`.
- Native EverOS scene-consolidation failures use deterministic fallback scene
  metadata and return `status="degraded"` so the MemCell is still durable.
- `status="degraded"` update results remain in trajectory metadata but do not
  register active backend state.
- `status="updated"` without a non-empty `state_ref` remains in trajectory
  metadata but does not register active backend state.
- Invalid artifact refs on memory update records are preserved in registry
  metadata under `invalid_artifact_refs` and do not fail a completed memory
  mutation.

## V1 Demo

`configs/demo_v1.yaml` is the real V1 backend demo. It uses native mem0 method
memory for agent and task scopes, backed by local SQLite stores under the Lab
registry paths, a real API LLM configured through `.env`, `GraphSkillBackend`,
and SFT dry-run evolution that produces a local-trainable rollout state:

```bash
python3 -m evolab.cli clean-run configs/demo_v1.yaml --lab-root /tmp/evolab-demo-v1
```

The generated Lab contains subagent trajectories and memory backend state
records for both agent-wise and task-wise scopes.

For CI or offline local checks that should not call an external LLM, use
`configs/demo_v1_ci.yaml`. It preserves the deterministic fake LLM and fake
skill path while continuing to exercise native mem0 method memory with fake
memory LLM and embedding backends.

## Real API Native Mem0 Smoke

Use this smoke only when `.env` has both a chat LLM endpoint for native mem0
extraction and an embedding endpoint for semantic memory search/add. The smoke
does not use the removed Mem0 adapter and does not fake the memory LLM or
embedding service.

Example `.env` shape:

```dotenv
AIGOCODE_GPT_API=openai-responses
AIGOCODE_GPT_BASE_URL=https://api.example.test
AIGOCODE_GPT_API_KEY=...
AIGOCODE_GPT_MODEL=gpt-5.1-codex

MEMORY_EMBEDDING_API=openai-embeddings
MEMORY_EMBEDDING_BASE_URL=https://api.example.test
MEMORY_EMBEDDING_API_KEY=...
MEMORY_EMBEDDING_MODEL=text-embedding-3-small
```

For providers that use the OpenAI default base URL, the `*_BASE_URL` entries
may be omitted. The `*_MODEL` and `*_API_KEY` entries are still required unless
the model is supplied on the command line.

Command:

```bash
python3 scripts/smoke_native_mem0_api.py \
  --dotenv /root/EvoLab/.env \
  --lab-root /tmp/evolab-native-mem0-api-smoke
```

Use `--chat-env-ref` and `--embedding-env-ref` if the `.env` prefixes differ
from `aigocode-gpt` and `memory-embedding`. Use `--chat-model` or
`--embedding-model` to supply a model without adding `*_MODEL` to `.env`.

The script first inspects `.env` with `evolab.config.env.parse_dotenv`. If the
required chat or embedding entries are missing, it prints
`"status": "skipped"` with the missing keys and exits successfully by default.
Use `--require-credentials` when CI should fail instead of skipping.

When credentials are present, the script generates a temporary clean-run config
that uses:

- fake task LLM only to keep the subagent deterministic
- real API chat LLM for native mem0 extraction
- real API embedding backend for native mem0 search/add
- SQLite stores under `registries/memory`

Expected checks after a run:

- `registries/memory/*.sqlite` exists under the smoke Lab
- both `mem0-agent.sqlite` and `mem0-task.sqlite` contain at least one active
  `memory_records` row, which means the run exercised both extraction and
  embedding-backed storage
- subagent trajectory memory update metadata includes
  `memory_method="mem0"`
- agent and task memory update statuses are `updated` or `skipped`, and each
  update has diagnostics metadata; `failed` means the real API smoke failed
- copied Lab configs do not contain old `client: in_memory`

Ablation configs should use `NullMemoryBackend` for the no-memory condition.
Do not use fake memory as a no-memory ablation. Native mem0 ablations require
an explicit `embedding_backend`; leave native mem0 disabled until a real or
test-intended embedding backend is configured.

## Test Commands

Memory-specific V1 suite:

```bash
pytest -q \
  tests/test_embedding_backends.py \
  tests/test_memory_method_backend.py \
  tests/test_memory_method_store.py \
  tests/test_everos_memory_method.py \
  tests/test_native_mem0_method.py \
  tests/test_native_mem0_retrieval.py \
  tests/test_memory_failure_suite.py \
  tests/test_memory_replay.py \
  tests/test_task_worker.py \
  tests/test_cli_clean_run.py
```

Broader runtime regression suite:

```bash
pytest -q \
  tests/test_api_llm_backend.py \
  tests/test_runtime_contracts.py \
  tests/test_registries.py \
  tests/test_imports.py
```

If the full suite is run outside the project dependency environment, ensure
`PyYAML` is installed because the scientific IE tests import `yaml`.

## Acceptance Traceability

- AC 8: memory retrieval happens before every subagent prompt.
- AC 9: memory update happens after every subagent run for both scopes.
- AC 10: native mem0 method memory is available and tested with fake LLM /
  embedding runtimes.
- AC 14: memory state lineage is registered in `BackendStateRegistry`.
- AC 15: memory backend replacement is available through config without
  runtime code changes.

## V2 Boundary

The following are intentionally outside V1:

- memory deletion, compaction, distillation, and promotion policy
- external Mem0 service SLA and operational monitoring
- cross-task privacy and access-control governance
- federated memory consolidation
- public memory/skill governance workflows
