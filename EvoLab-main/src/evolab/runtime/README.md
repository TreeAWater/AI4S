# Runtime Development

`evolab/runtime` 负责把 task config、backend runtimes、tools、trajectory 和 backend
state registry 编排成一次可复现的 task run。Runtime 不实现具体 memory algorithm、
skill retrieval algorithm 或 LLM provider；这些能力通过 backend contract 注入。

## Module Map

- `task_worker.py`：worker startup、backend instantiation、queue lifecycle、final artifact index。
- `task_runtime.py`：MetaAgent role-pool evolution、dynamic workflow execution、memory/skill retrieval、tool loop、memory update。
- `memory_replay.py`：基于 trajectory 和 backend state registry replay memory trace。
- `prompt_builder.py`：构建 MetaAgent/subagent prompt。
- `trajectory_collector.py`：记录 runtime events 和 run records。
- `task_close_evolution.py`：任务结束后的 evolution scheduling。

## Startup Order

`TaskWorker.startup()` 的 backend 初始化顺序是 memory 正确工作的关键：

1. 初始化 task/evolve queue、registries、tools 和 prompt builder。
2. 按 role/meta-agent binding 初始化 executable LLM runtimes。
3. 额外初始化 memory method 声明的 `llm_backend_id`，即使该 LLM 不属于任何 role。
4. 初始化全部 embedding backends。
5. 对支持 `bind_runtimes()` 的 memory backend 注入 `llm_runtimes` 和 `embedding_runtimes`。
6. 按 `memory_backend_bindings` 实例化 memory runtimes，包括 task 和可选的 MetaAgent memory。
7. 初始化 skill runtimes。
8. 构造 `TaskRuntime`。

新增 backend 依赖时，应优先放在对应 backend 的 `bind_runtimes()` 或 `instantiate()`
边界上，不要让 `TaskRuntime` 直接知道 provider-specific details。

## Memory Lifecycle In TaskRuntime

动态 worker run 的主路径：

1. 根据 task 的 `task_memory_backend` 构造 retrieval request；动态 worker roles 使用 task-level memory only。
2. 调用 memory runtime `search()`，把返回的 `MemoryBundle` 放入 prompt context 和 trajectory。
3. 执行 LLM/tool loop。
4. 将本轮 messages 写回 task memory backend 的 `add()`。
5. 把 `MemoryUpdateResult.state_ref` 写入 backend state registry 和 trajectory。

EvoLab no longer treats top-level `subagents` as the default execution model.
The active role pool is `agents.md`. MetaAgent may update this file
automatically, and DynamicWorkflowPlanner consumes the latest role templates for
each run. Role-pool worker entries should not declare worker agent-memory
backends.

Replay 依赖这些记录保持一致：

- retrieval bundle metadata 中的 `backend_id`、`memory_scope`、`memory_scope_id`、`memory_method`。
- update result 中的 `state_ref` 和 `previous_state_ref`。
- backend state registry 中按 backend id 维护的 active state lineage。

## MetaAgent Memory

MetaAgent semantic memory is optional and configured with
`meta_agent.memory_backend`. When present, each dispatch step:

1. Searches the configured memory backend with role `meta_agent.name` and
   scope `agent:<meta_agent.name>`.
2. Adds the retrieved memory payload to the MetaAgent routing prompt as
   `meta_memory`.
3. After a valid dispatch decision, writes a compact decision summary back to
   the same backend.
4. Records `meta_memory_bundle`, `meta_memory_update_result`, and the retrieval
   request in the MetaAgent trajectory metadata.

If `meta_agent.memory_backend` is absent, MetaAgent continues to rely only on
LabState, trajectory summaries, completed run payloads, and explicit requested
details.

## State Refs

Runtime 不解析 provider-specific state ref 内容，只把它们作为 opaque references
记录和传递。Native methods 当前使用：

```text
method://mem0/<backend-id-len>:<backend-id>/<scope-len>:<scope>/<scope-id-len>:<scope-id>/v<version>
method://everos/<backend-id-len>:<backend-id>/<scope-len>:<scope>/<scope-id-len>:<scope-id>/v<version>
```

解析和版本维护属于 memory method/store 的职责。Runtime 只负责：

- 在 startup 时把 active state ref 传给 backend `instantiate()`。
- 在 run 结束或 memory update 后记录新的 state ref。
- 在 replay 时比较 trajectory 与 registry 是否一致。

## Development Rules

- 不要在 `TaskRuntime` 中分支处理某个具体 memory method 或 embedding provider。
- 新 provider 应通过 backend builder、backend class 和 runtime binding 接入。
- 需要额外依赖的 backend 应显式声明 backend id；startup 阶段缺失配置时直接失败。
- Trajectory records 要保留足够信息用于 replay，但不要写入 secret。
- SDK session 应可在 fresh Lab root 上初始化 `.evolab` state、inputs、registries 和 artifacts。

## Tests

Runtime 和 memory binding 相关测试入口：

```bash
pytest tests/test_task_worker.py tests/test_task_runtime.py \
  tests/test_memory_replay.py tests/test_cli_clean_run.py -q
```

如果改动 memory backend binding，还应运行：

```bash
pytest tests/test_memory_method_backend.py tests/test_native_mem0_method.py \
  tests/test_everos_memory_method.py tests/test_native_mem0_api_smoke_script.py -q
```
