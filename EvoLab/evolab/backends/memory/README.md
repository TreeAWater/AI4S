# Memory Backend Development

本模块定义 EvoLab runtime 面向 memory 的稳定边界。Runtime 只依赖
`MemoryBackend` 的 `search()`、`add()` 和 `instantiate()`；具体 memory
算法应实现为 `MemoryMethod`，再由 `MethodMemoryBackend` 适配到 runtime。
这样后续新增方法时，不需要改 `TaskRuntime` 的主流程。

## Module Map

- `base.py`：`MemoryBackend` 抽象，定义 runtime 可调用接口。
- `null.py`：`NullMemoryBackend`，用于 no-memory baseline 和消融实验。
- `fake.py`：`FakeMemoryBackend`，只用于窄单测和 deterministic fixture。
- `method_backend.py`：`MethodMemoryBackend`，把通用 `MemoryMethod` 包装成 runtime backend。
- `methods/base.py`：`MemoryMethod`、`MemorySearchRequest`、`MemoryIngestRequest` 等 method-level contract。
- `methods/mem0.py`：native mem0 method。负责 LLM extraction、dedupe、embedding、search fusion 和 state refs。
- `methods/everos.py`：native EverOS/EverMemOS method。负责 MemCell construction、MemScene consolidation、scene-first recollection 和 state refs。
- `methods/store.py`：SQLite store，维护 memories、messages、entities、links、scope versions 和 audit log。
- `methods/retrieval.py`：cosine、BM25-like keyword scoring 和 fusion utilities。

## Backend Contract

`MemoryBackend.search(request: RetrievalRequest)` 返回 `MemoryBundle`。
`MethodMemoryBackend` 会把 `RetrievalRequest.task_id`、`role` 和 filters 规范化为
method scope：

- `role == "task"` 默认使用 `memory_scope="task"`、`memory_scope_id="task:<task_id>"`。
- 其他 role 默认使用 `memory_scope="agent"`、`memory_scope_id="agent:<role>"`。
- 调用方可以用 filters 显式覆盖 `memory_scope` 和 `memory_scope_id`。

`MemoryBackend.add(task_id, role, messages)` 返回 `MemoryUpdateResult`。
adapter 会把 method 返回的 metadata 统一补充为：

- `memory_scope`
- `memory_scope_id`
- `memory_method`
- `added_memory_ids`
- `skipped_memory_ids`
- `linked_memory_ids`

method metadata 不应覆盖这些 canonical 字段。

## Native mem0 Method

`Mem0MemoryMethod` 是 EvoLab 内置的第一个 memory method。它不是外部 mem0
service client，也不依赖 `mem0ai` Python package。它在本地维护 SQLite memory
store，并显式使用 EvoLab 配置中的 LLM 和 embedding backend。

Add flow:

1. 从当前 `(scope, scope_id)` 读取已有 memories 和 recent messages。
2. 调用 `llm_backend_id` 对应的 LLM runtime，使用 JSON schema 约束抽取 durable facts。
3. 按 content hash 做 scope-local dedupe，保留 LLM 返回的 `linked_memory_ids`。
4. 对新 memory text 调用 `embedding_backend_id` 对应的 embedding runtime，`purpose="add"`。
5. 用 SQLite transaction 写入 messages、memory records、entity rows、links 和 scope version。
6. 返回 length-prefixed `method://mem0/.../v<version>` state ref，例如
   `method://mem0/12:memory-local/5:agent/12:agent:solver/v1`。

Search flow:

1. 对 query 调用 embedding runtime，`purpose="search"`。
2. 只在当前 `(scope, scope_id)` 内取候选 memory。
3. 计算 semantic cosine、BM25-like keyword score 和 entity boost。
4. 用 fusion score 排序，应用 `default_search_threshold` 和 `default_search_top_k`。
5. 返回 `MemoryItem`，metadata 中包含 component scores 和 method identity。

## Native EverOS Method

`EverOSMemoryMethod` 是 EvoLab 内置的 EverOS/EverMemOS-style memory method。
它借鉴 EverMemOS 的 MemCell、MemScene 和 reconstructive recollection 方法，
但不调用 EverOS/EverCore HTTP service，也不依赖 EverOS 的 MongoDB、Elasticsearch、
Milvus 或 Redis。所有持久化都由 EvoLab 本地 SQLite store 完成，LLM 和 embedding
调用走 EvoLab 已配置的 backends。

Add flow:

1. 从当前 `(scope, scope_id)` 读取 recent messages 和已有 MemScenes。
2. 调用 `llm_backend_id` 对新消息做 MemCell construction，抽取 episode、atomic
   facts、foresights、agent case 和 agent skills。
3. 对 MemCell 文本调用 embedding runtime，按 cosine similarity 归入已有
   MemScene，或创建新的 MemScene。
4. 调用 LLM 做 MemScene consolidation，更新 scene title、summary 和 tags。
5. 对 searchable records 做 embedding，并在 scope-local content hash 下 dedupe。
6. 在一个 SQLite transaction 中写入 messages、MemCells、MemScenes、scene
   members、searchable records 和 scope version。
7. 返回 `method://everos/.../v<version>` state ref。

Search flow:

1. 对 query 调用 embedding runtime。
2. 在当前 `(scope, scope_id)` 内读取 MemScenes 和 searchable records。
3. 对 records 计算 semantic score、BM25-like keyword score 和 entity overlap
   boost。
4. 聚合到 scene score，优先返回 scene-grounded reconstructed context。
5. `recollection_mode: scene` 走 deterministic scene-first recollection；
   `recollection_mode: agentic` 会额外调用 LLM，从候选 scenes 中选择必要且充分的
   memories 并压缩上下文。

EverOS config must be native. The backend builder rejects `base_url` and endpoint
fields so it cannot silently become a service wrapper.

## SQLite Store Invariants

- 所有 read/write 都必须按 `(scope, scope_id)` 隔离，不能跨 agent/task scope 泄漏。
- 每条 active memory 必须有 non-empty content、content hash 和 finite numeric embedding。
- write path 必须通过 store transaction 更新 records、links、entities、messages 和 scope version。
- `state_ref` version 来自 scope-local version，不是全局计数器。
- `audit_log_path` 可选；启用后应在成功写入后记录可追踪的 store mutation。

## Configuration

推荐配置：

```yaml
backends:
  llm:
    memory-extractor:
      type: api
      api: openai-chat-completions
      api_key_env: OPENAI_API_KEY
      model: deepseek/deepseek-v4-flash
      base_url: https://openrouter.ai/api/v1
  embedding:
    memory-embedding:
      type: api
      model: text-embedding-3-small
      api_key_env: OPENAI_API_KEY
  memory:
    mem0-meta-memory:
      type: method
      method: mem0
      store_path: registries/memory/mem0-meta.sqlite
      audit_log_path: registries/memory/mem0-meta.audit.jsonl
      llm_backend: memory-extractor
      embedding_backend: memory-embedding
    mem0-agent-memory:
      type: method
      method: mem0
      store_path: registries/memory/mem0-agent.sqlite
      audit_log_path: registries/memory/mem0-agent.audit.jsonl
      llm_backend: memory-extractor
      embedding_backend: memory-embedding
      top_k_existing: 10
      default_search_top_k: 8
      default_search_threshold: 0.15
    everos-agent-memory:
      type: method
      method: everos
      store_path: registries/memory/everos-agent.sqlite
      audit_log_path: registries/memory/everos-agent.audit.jsonl
      llm_backend: memory-extractor
      embedding_backend: memory-embedding
      scene_similarity_threshold: 0.78
      recollection_mode: scene
      default_search_top_k: 8
      default_search_threshold: 0.05
```

`type: mem0` 是兼容入口，但仍会构造 native `MethodMemoryBackend`。
新配置应优先使用 `type: method`、`method: mem0` 或
`type: method`、`method: everos`。

已移除的配置：

- `client`
- `client_type`
- inline `api_key` 或 `apiKey`
- 外部 mem0 service client
- `mem0ai` dependency

No-memory baseline 使用：

```yaml
backends:
  memory:
    no-memory:
      type: null
```

MetaAgent memory is configured on `meta_agent.memory_backend`, for example:

```yaml
meta_agent:
  memory_backend: mem0-meta-memory
  system_prompt: |
    Return route JSON only.
```

Runtime stores MetaAgent memory under agent scope `agent:<meta_agent.name>`,
usually `agent:meta`. This is separate from subagent role memory such as
`agent:ExecAgent` and from task-scope memory such as `task:<task_id>`.

## Runtime Binding

如果 method 声明 `llm_backend_id`，`TaskWorker.startup()` 会确保该 LLM runtime
被初始化，即使它没有绑定到任何 executable role。Embedding backends 会在 memory
backend 实例化前统一初始化，然后 `MethodMemoryBackend.bind_runtimes()` 把
`llm_runtimes` 和 `embedding_runtimes` 交给 method。

这意味着 memory extraction 可以使用与 subagent 不同的 LLM，也可以在实验中单独替换。
缺失的 LLM 或 embedding backend 应在 startup/binding 阶段失败，不能 silent fallback。

## Adding A Memory Method

新增 memory 方法时：

1. 在 `methods/base.py` contract 下实现 `MemoryMethod`。
2. 如果需要依赖 runtime，实现 `bind_runtimes(llm_runtimes=..., embedding_runtimes=...)`。
3. 如果 state ref 需要 backend identity，实现 `bind_backend_id(backend_id)`。
4. 让 `search()` 和 `add()` 只接收 method-level request，不直接依赖 `TaskRuntime`。
5. 返回稳定的 `state_ref` 和 method metadata，避免把 provider secret 写入 metadata。
6. 在 CLI backend builder 中增加 method config parsing 和 validation。
7. 添加 focused tests，覆盖 scope isolation、state refs、dependency binding、failure mode 和 replay。

不要把新方法直接写进 `TaskRuntime`。Runtime 的职责是编排生命周期，不是承载 memory
algorithm。

## Tests

相关测试入口：

```bash
pytest tests/test_memory_method_backend.py tests/test_native_mem0_method.py \
  tests/test_native_mem0_retrieval.py tests/test_memory_method_store.py \
  tests/test_everos_memory_method.py tests/test_memory_failure_suite.py \
  tests/test_memory_replay.py -q
```

真实 API 冒烟脚本在 `scripts/smoke_native_mem0_api.py`。它需要可用的 chat LLM 和
embedding credentials；缺失时测试会 skip，而不是退回 fake backend。
