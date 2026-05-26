# Memory Methods And Native Mem0 Design

## Context

EvoLab currently exposes memory through `MemoryBackend.search(...)` and
`MemoryBackend.add(...)`. The current `Mem0MemoryBackend` is a Mem0-compatible
adapter: tests and demos normally use `InMemoryMem0Client`, while the external
`mem0ai/mem0` package path depends on Mem0's own OpenAI and vector-store
configuration. This is misleading for experiments because `type: mem0` can mean
either a local deterministic shim or an external library adapter, neither of
which gives EvoLab a first-class, inspectable memory method.

Going forward, EvoLab should treat each memory algorithm as a method implemented
behind a shared internal abstraction. `mem0` is the first method, not a special
external dependency. The implementation must be production-grade: local
persistent state, real LLM extraction when the method requires it, real
embedding-backed retrieval, scope isolation, lineage, and tests that exercise
the actual behavior.

The native mem0 method follows the current Mem0 OSS v3 design from official
documentation and installed source:

- `add(infer=True)` performs LLM-based structured memory extraction, not raw
  transcript storage.
- The current algorithm is single-pass ADD-only extraction with existing memory
  context, hash deduplication, embeddings, history, and entity linking.
- `search(...)` uses hybrid retrieval: semantic candidates, keyword/BM25 signal,
  entity boost, score fusion, thresholding, and `top_k`.

References:

- https://docs.mem0.ai/core-concepts/memory-operations/add
- https://docs.mem0.ai/core-concepts/memory-operations/search
- https://docs.mem0.ai/migration/oss-v2-to-v3

## Goals

1. Add a shared memory-method architecture so future memory algorithms can be
   implemented consistently as `MemoryBackend`s.
2. Implement `mem0` as an EvoLab-native local method after that abstraction,
   without depending on external `mem0ai/mem0` runtime behavior.
3. Use EvoLab-configured LLM services for memory extraction and an explicit
   EvoLab embedding backend for semantic retrieval.
4. Persist memory locally in a durable, inspectable format with registry
   lineage and replay support.
5. Remove or quarantine unused old Mem0 adapter code so `type: mem0` has one
   clear meaning.

## Non-Goals

- Do not build a toy keyword-only memory backend.
- Do not silently fall back to raw transcript storage when mem0 extraction or
  embeddings are required.
- Do not keep external Mem0 service behavior as the default implementation.
- Do not add graph traversal APIs beyond what native mem0 retrieval needs for
  entity boosting and linking.

## Architecture

The public runtime boundary remains `MemoryBackend`. Task runtime should not
need to know whether a backend uses mem0, episodic summaries, reflection, or a
future graph method.

Internally, memory backends delegate algorithmic behavior to a `MemoryMethod`.
The first production method is `Mem0MemoryMethod`.

Proposed module layout:

```text
evolab/backends/embeddings/
  __init__.py
  base.py
  api.py
  fake.py

evolab/backends/memory/
  base.py
  null.py
  fake.py
  method_backend.py
  methods/
    __init__.py
    base.py
    mem0.py
    stores.py
    retrieval.py
```

`MemoryBackend` remains the task-facing adapter. `MethodMemoryBackend` owns
state refs, scope mapping, and conversion between EvoLab contracts and method
contracts. `MemoryMethod` owns the actual algorithm.

## Contracts

`MemoryMethod`:

```python
class MemoryMethod(Protocol):
    method_name: str

    def search(self, request: MemorySearchRequest) -> MemorySearchResult: ...
    def add(self, request: MemoryIngestRequest) -> MemoryIngestResult: ...
    def instantiate(self, state_ref: str | None) -> MemoryMethod: ...
```

`MemorySearchRequest` includes:

- `task_id`
- `role`
- `query`
- `scope`
- `scope_id`
- `filters`
- `top_k`
- `threshold`
- `metadata`

`MemoryIngestRequest` includes:

- `task_id`
- `role`
- `scope`
- `scope_id`
- `messages`
- `observation_time`
- `metadata`

`MemoryIngestResult` returns:

- `status`: `updated | skipped | failed | degraded`
- `added_memory_ids`
- `skipped_memory_ids`
- `linked_memory_ids`
- `extractor_call_metadata`
- `storage_metadata`
- `diagnostics`

The backend maps these to the existing `MemoryBundle` and `MemoryUpdateResult`
contracts, preserving current runtime and registry behavior.

## Embedding Backend

Native mem0 requires semantic retrieval. EvoLab needs a first-class embedding
backend rather than reusing chat LLM APIs implicitly.

`EmbeddingBackend.instantiate(...)` returns an `EmbeddingRuntime`:

```python
class EmbeddingRuntime(Protocol):
    def embed(self, texts: list[str], *, purpose: str) -> EmbeddingResponse: ...
```

`ApiEmbeddingBackend` should support OpenAI-compatible embeddings endpoints:

```yaml
backends:
  embedding:
    text-embedding:
      type: api
      api: openai-embeddings
      model: text-embedding-3-small
      env_ref: openai
      base_url: https://api.openai.com/v1
```

For OpenRouter or other providers, the config must point to an endpoint that
actually supports embeddings. If a native mem0 memory config lacks an embedding
backend, startup fails fast.

Tests can use `FakeEmbeddingBackend` with deterministic vectors. Fake embeddings
are only for tests and CI, not for production configs.

## Store

Use SQLite as the primary local store. JSONL audit logs are optional but useful
for reproducibility and should be emitted by default.

SQLite tables:

- `memory_records`
  - `memory_id`
  - `scope`
  - `scope_id`
  - `content`
  - `content_hash`
  - `embedding_json`
  - `text_lemmatized`
  - `attributed_to`
  - `created_at`
  - `updated_at`
  - `metadata_json`
  - `deleted_at`
- `message_history`
  - `message_id`
  - `scope`
  - `scope_id`
  - `role`
  - `content`
  - `created_at`
  - `run_ref`
  - `metadata_json`
- `memory_history`
  - `history_id`
  - `memory_id`
  - `event`
  - `old_content`
  - `new_content`
  - `created_at`
  - `metadata_json`
- `memory_links`
  - `source_memory_id`
  - `target_memory_id`
  - `link_type`
  - `score`
  - `created_at`
- `memory_entities`
  - `entity_id`
  - `scope`
  - `scope_id`
  - `entity_text`
  - `entity_type`
  - `embedding_json`
  - `metadata_json`
- `entity_memory_links`
  - `entity_id`
  - `memory_id`

Scope isolation is mandatory. Every query and mutation filters by `scope_id`;
`agent:DesignAgent` and `task:<task_id>` must never share records unless a
future cross-scope method explicitly requests that behavior.

## Native Mem0 Add Flow

`Mem0MemoryMethod.add(...)`:

1. Validate scope and required dependencies.
2. Normalize EvoLab messages into `system/user/assistant` records. Tool outputs
   are included only after TaskRuntime has converted them to supported memory
   messages.
3. Load recent message history for the scope.
4. Embed the new-message text and retrieve top related existing memories from
   the same scope for deduplication context.
5. Call the configured EvoLab LLM runtime with the mem0-style ADD-only
   extraction prompt and a strict JSON schema.
6. Parse and validate `{"memory": [{"id", "text", "attributed_to",
   "linked_memory_ids"}]}`.
7. Hash each extracted memory and skip exact duplicates already present in the
   same scope or current batch.
8. Embed extracted memory texts in batch.
9. Insert memory records and memory history.
10. Extract/link entities for entity boosting.
11. Save raw message history even when extraction returns no memories.
12. Return update diagnostics and added IDs.

Extraction failures return `MemoryUpdateResult(status="failed")`. Empty
extractions are `skipped` only if no memory was created; they still record
message history when storage succeeded.

## Native Mem0 Search Flow

`Mem0MemoryMethod.search(...)`:

1. Validate filters and scope.
2. Embed the query.
3. Semantic search over the same scope, over-fetching candidates.
4. Compute keyword/BM25-like score from lemmatized text. If SQLite FTS5 is
   available, use it; otherwise use an in-process BM25 implementation over the
   scoped candidate pool.
5. Extract query entities and compute entity boosts from `memory_entities`.
6. Fuse semantic, keyword, and entity scores using mem0-style additive scoring.
7. Apply threshold and `top_k`.
8. Return `MemoryItem`s with scores and metadata.

Search failures propagate because prompt construction cannot safely continue
with an unavailable required memory bundle.

## Configuration

Recommended full config:

```yaml
backends:
  llm:
    openrouter-chat:
      type: api
      api: openai-chat-completions
      model: deepseek/deepseek-v4-flash
      env_ref: openrouter
      base_url: https://openrouter.ai/api/v1

  embedding:
    memory-embedding:
      type: api
      api: openai-embeddings
      model: text-embedding-3-small
      env_ref: openai

  memory:
    mem0-agent-memory:
      type: method
      method: mem0
      store_path: registries/memory/mem0-agent.sqlite
      audit_log_path: registries/memory/mem0-agent.audit.jsonl
      llm_backend: openrouter-chat
      embedding_backend: memory-embedding
      user_id_template: "{memory_scope_id}"
      default_search_top_k: 20
      default_search_threshold: 0.1

    mem0-task-memory:
      type: method
      method: mem0
      store_path: registries/memory/mem0-task.sqlite
      audit_log_path: registries/memory/mem0-task.audit.jsonl
      llm_backend: openrouter-chat
      embedding_backend: memory-embedding
      user_id_template: "{memory_scope_id}"
```

Compatibility option:

```yaml
type: mem0
implementation: native
```

The implementation may accept this as a shorthand, but internally it should
instantiate `MethodMemoryBackend(method=Mem0MemoryMethod(...))`.

External `mem0ai/mem0` support, if retained, must be renamed to an explicit
external adapter such as:

```yaml
type: mem0_external
```

or:

```yaml
type: mem0
implementation: external
```

The default `type: mem0` must not mean "external package with hidden OpenAI
configuration."

## Cleanup

Clean up old implementation after the new method abstraction lands:

- Remove `InMemoryMem0Client` from production exports.
- Move old adapter tests to an external-adapter test file if the adapter is
  retained.
- Update docs that currently describe `client: in_memory` as the normal mem0
  demo path.
- Update biology extraction configs to use native method config or explicit
  `null` memory for ablations.
- Keep `NullMemoryBackend` for ablation experiments.
- Keep `FakeMemoryBackend` only for deterministic unit tests and minimal CI
  demos that are explicitly not testing mem0 method quality.

## Error Behavior

- Missing LLM backend for native mem0: fail at backend construction.
- Missing embedding backend: fail at backend construction.
- Invalid scope or empty scope id: fail fast with `ValueError`.
- LLM extraction exception: update result `failed`; no active memory state
  registered.
- Invalid extractor JSON: update result `failed` with raw parse diagnostics.
- Empty but valid extraction: `skipped` with message history persisted.
- Storage commit failure: update result `failed`; partial writes must be rolled
  back by SQLite transaction.
- Search dependency failure: propagate exception.

## Testing

Focused unit tests:

- method backend maps EvoLab `RetrievalRequest` and `Message` inputs into method
  requests with correct scopes.
- native mem0 add calls the configured LLM with JSON schema and existing memory
  context.
- invalid extractor JSON fails without mutating store.
- empty extraction persists message history and returns `skipped`.
- duplicate extracted memory is skipped by hash in the same scope.
- same text in a different scope is allowed.
- search combines semantic score, keyword score, and entity boost.
- threshold and `top_k` are enforced.
- SQLite store survives process restart and `instantiate(state_ref)`.
- backend state registry records updated native mem0 state refs.

Integration tests:

- `clean-run` config builds embedding, LLM, method memory, skill, and task
  runtime together.
- one small real API smoke can run with user-provided credentials, but CI should
  rely on fake LLM and fake embedding fixtures.
- memory replay works against native mem0 registry and SQLite state.

## Migration Plan

1. Add embedding backend package and tests.
2. Add memory method contracts and `MethodMemoryBackend`.
3. Add SQLite store and audit logging.
4. Implement native mem0 extraction, ingestion, and retrieval.
5. Wire CLI config for `backends.embedding` and method-backed memory.
6. Migrate configs and docs.
7. Remove or quarantine obsolete external/in-memory Mem0 adapter paths.
8. Re-run the memory ablation using native mem0 and null memory.

## Decisions

- Production native mem0 configs must bind an explicit embedding backend. The
  abstraction will not hardcode a provider. Project demo configs may use
  OpenAI-compatible `text-embedding-3-small` when an OpenAI embedding key is
  available, but any config that lacks an embedding backend is invalid.
- Remove the old external Mem0 package adapter and `InMemoryMem0Client` from
  the default production path. If a concrete workflow later needs external
  `mem0ai/mem0`, it should be reintroduced under a separate explicit backend
  type, not behind default `type: mem0`.
- Native mem0 includes entity extraction and lemmatization as part of the
  method. The default implementation should port the relevant local Mem0 OSS
  extraction/lemmatization behavior into EvoLab-owned code where licensing
  permits, with tests against expected entity linking and BM25 text fields.
  Optional LLM-assisted entity extraction may be a config extension, but the
  native method must work without adding an LLM call to every search.
