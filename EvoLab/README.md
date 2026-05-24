# EvoLab

EvoLab 是一个面向科学研究任务的 self-evolving agents 原型系统。它把任务执行拆成可替换的 backend lifecycle：memory retrieval/update、skill retrieval/update、tool execution、LLM runtime、trajectory logging 和 LLM evolution。当前仓库状态对应 `development_plan.md` 中的 V1 MVP backend path。

## 当前状态

V1 已完成的主路径：

- `MethodMemoryBackend + Mem0MemoryMethod/EverOSMemoryMethod`：native memory method path，支持 agent/task/meta scope、SQLite Lab registry state、logical state refs 和 replay。
- `GraphSkillBackend`：从 seed graph 检索 scientific IE skills，聚合 `required_tools`，记录 graph update summaries。
- `ApiLLMBackend`：接入 OpenAI Responses-compatible API，支持 EvoLab `ToolSpec` 转 function tools、多轮 tool call continuation、structured output config。
- `LocalTrainableLLMBackend`：V1 rollout-only local LLM backend，可加载 SFT/OPSD trainer 产出的 local-trainable state manifest。
- `TaskRuntime`：执行 MetaAgent route、memory -> skill -> tool preparation -> LLM/tool loop -> trajectory -> memory update -> skill observation。
- 短 experiment config：人类只写自然语言 task、MetaAgent prompt、subagent name/prompt、Lab 路径和 backend 配置；CLI 编译成内部 `TaskRequest`/`TaskConfig`。
- `clean-run` demo：可以在 fresh Lab 中跑出真实 scientific IE 任务结果、trajectory、backend state lineage 和 evolution artifacts。

V1 不包含真实 LoRA/SFT clean-run 训练、异步 standalone EvolveWorker、federated consolidation、public skill governance、full resource mining 或生产级 HITL 集成。这些在文档中明确归入 V2。

## 安装

要求 Python 3.11+。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

可选 SFT/Transformers 依赖：

```bash
pip install -e '.[sft]'
```

安装后可以使用 console script：

```bash
evolab --help
```

也可以直接用模块入口：

```bash
python3 -m evolab.cli --help
```

## 快速运行

### CI/offline V1 demo

这个路径不调用外部 LLM，适合本地冒烟和 CI。它保留 fake LLM/fake skill，同时仍走 native mem0 method memory，并使用 fake memory LLM/embedding backend 与本地 SQLite store。

```bash
python3 -m evolab.cli clean-run configs/demo_v1_ci.yaml --lab-root /tmp/evolab-demo-v1-ci
```

### 真实 V1 backend demo

真实 demo 使用：

- API LLM：从 `.env` 读取
- GraphSkillBackend：`configs/skills/graphs/scientific_ie_seed_graph_v1.json`
- native mem0 method memory with local SQLite stores
- scientific IE tools
- SFT dry-run evolution producing a local-trainable rollout state

`.env` 是团队共享的环境变量文件。`env_ref: "aigocode-gpt"` 会映射到
`AIGOCODE_GPT_*` 变量：

```dotenv
AIGOCODE_GPT_API=openai-responses
AIGOCODE_GPT_BASE_URL=https://api.example.com
AIGOCODE_GPT_API_KEY=your-api-key
```

运行：

```bash
python3 -m evolab.cli clean-run configs/demo_v1.yaml --lab-root /tmp/evolab-demo-v1-real
```

当前 V1 demo 的真实任务是从 seeded article 中抽取 biological component records，并用 seeded JSON schema 验证。期望输出包含 `pTet`、`B0034`、`sfGFP`、`B0015` 四条记录。

### Native memory method 配置

EvoLab 现在把 memory algorithm 作为 `MemoryMethod` 实现，并通过
`MethodMemoryBackend` 接入 runtime。`mem0` 和 `everos` 都是 native method
backend：本地 SQLite 负责持久化，配置中的 LLM backend 负责 memory
construction，embedding backend 负责 add/search 的向量化。

推荐写法：

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
    mem0-agent-memory:
      type: method
      method: mem0
      store_path: registries/memory/mem0-agent.sqlite
      llm_backend: memory-extractor
      embedding_backend: memory-embedding
      default_search_top_k: 8
      default_search_threshold: 0.15
    mem0-meta-memory:
      type: method
      method: mem0
      store_path: registries/memory/mem0-meta.sqlite
      llm_backend: memory-extractor
      embedding_backend: memory-embedding
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

兼容写法 `type: mem0` 仍会走 native implementation；旧的
`client`、`client_type`、inline API key 和外部 mem0 service client 已移除。
EverOS 的实现同样是 native implementation，不调用 EverOS/EverCore HTTP
service，不需要 MongoDB/Elasticsearch/Milvus/Redis；`base_url` 或 endpoint
配置会被拒绝。
如果要做 no-memory ablation，请配置 `type: null` 的 memory backend。
MetaAgent 如需自己的 semantic memory，请在短 config 中设置
`meta_agent.memory_backend: mem0-meta-memory`；未设置时 MetaAgent 仍只使用
LabState/trajectory 上下文。

### Biology component extraction config

当前组件信息提取实验使用短 config：

```bash
python3 -m evolab.cli clean-run \
  configs/biology_component_extraction_v1_generic_subagents.yaml \
  --lab-root /tmp/evolab-biology-component-run
```

两篇 paper 的真实 native mem0/deepseek 验证 config：

```bash
python3 -m evolab.cli clean-run \
  configs/biology_component_extraction_v1_two_paper_mem0_deepseek.yaml \
  --lab-root /root/bio/evolab_component_ie_v1_two_paper_mem0_deepseek
```

该 config 使用 `OPENAI_API_KEY` 访问 OpenRouter 的
`deepseek/deepseek-v4-flash` chat completions，并用 OpenRouter embeddings
endpoint 的 `openai/text-embedding-3-small` 支撑 native mem0 add/search。

这个 config 不再手写内部 `task_config` 或 biology-specific executable roles。它只定义：

- 自然语言任务描述，包括数据集、资源、输出和验收标准；
- MetaAgent 的 JSON-only route prompt；
- 五个通用 subagent：`SurveyAgent`、`DesignAgent`、`ExecAgent`、`CriticAgent`、`WriteAgent`；
- Lab 路径和 LLM/memory/skill backend。

MetaAgent 每轮输出：

```json
{"route":"ExecAgent","instruction":"...","metadata":{}}
```

或结束：

```json
{"route":"END","instruction":"...","metadata":{"final_answer":"..."}}
```

旧的 biology 长 config 已删除；新实验请使用 generic-subagent config。

## Lab 输出

`clean-run` 会清理并重建指定 Lab root。典型输出包括：

```text
<lab-root>/
  configs/
  inputs/
  queues/
    tasks/
    evolve/
  registries/
    task/
    trajectory/
      subagent.jsonl
      llm_calls.jsonl
      evolution.jsonl
    backend_state/
      states.jsonl
      active.json
  evolution/
    llm/
```

关键记录：

- `subagent.jsonl`：subagent run、prompt messages、memory/skill bundles、tool trace、final answer。
- `llm_calls.jsonl`：每次 LLM 调用的输入、输出、raw response、tool specs、generation config。
- `evolution.jsonl`：LLM evolution run record。
- `backend_state/states.jsonl`：memory/skill/LLM backend state lineage。
- `evolution/llm/<run-ref>/state/local_trainable_state.json`：SFT/OPSD dry-run 产出的 local-trainable rollout state manifest。

## 常用命令

运行 V0 demo：

```bash
python3 -m evolab.cli clean-run configs/demo_v0.yaml --lab-root /tmp/evolab-demo-v0
```

导出 SFT 数据集：

```bash
python3 -m evolab.cli export-sft \
  --lab-root /tmp/evolab-demo-v0 \
  --output-dir /tmp/evolab-demo-v0/artifacts/sft \
  --teacher-backend-id fake-llm
```

运行 SFT dry-run promotion：

```bash
python3 -m evolab.cli train-sft \
  --lab-root /tmp/evolab-demo-v0 \
  --backend-id fake-llm \
  --artifact-root /tmp/evolab-demo-v0/artifacts/sft-train \
  --training-backend dry_run \
  --promote-dry-run
```

## 验证

完整测试：

```bash
pytest -q
```

V1 focused validation：

```bash
pytest tests/test_task_worker.py tests/test_api_llm_backend.py \
  tests/test_scientific_ie_tool_coverage.py tests/test_cli_clean_run.py \
  tests/test_memory_replay.py -q
```

Memory replay 示例：

```bash
python3 - <<'PY'
from evolab.runtime.memory_replay import replay_memory_trace

report = replay_memory_trace("/tmp/evolab-demo-v1-real", task_id="demo-v1")
print(report.model_dump_json(indent=2))
assert report.ok, report.issues
PY
```

## 代码结构

```text
evolab/
  backends/        backend implementations: LLM, memory, skills, trainers, rewards
  config/          TaskConfig and backend binding models
  contracts/       shared Pydantic contracts and records
  lab/             Lab layout, resolver, queues
  registries/      filesystem registries for task, trajectory, backend state
  runtime/         TaskRuntime, TaskWorker, evolution, replay, SFT export
  tools/           ToolRegistry, ToolRuntime, scientific IE tools
configs/           demo configs and seed skill graphs
skills/            reusable scientific IE skill packages
domain_packages/   task/domain-specific schemas, policies, ontologies
docs/              V1 subsystem docs and release checklist
tests/             contract, runtime, backend, CLI, and integration tests
```

## 关键文档

- [development_plan.md](development_plan.md)：V0/V1 目标和 acceptance criteria。
- [docs/configuration.md](docs/configuration.md)：短 experiment config、MetaAgent route contract 和 biology extraction 配置。
- [docs/v1_release_checklist.md](docs/v1_release_checklist.md)：V1 release evidence 和 AC 1-17 traceability。
- [docs/params_runtime.md](docs/params_runtime.md)：API LLM、runtime、local-trainable rollout state、tool observation。
- [docs/memory.md](docs/memory.md)：memory scope、native mem0/EverOS methods、state refs、replay。
- [evolab/backends/memory/README.md](evolab/backends/memory/README.md)：memory backend/method 开发契约和 native method 实现说明。
- [evolab/backends/embeddings/README.md](evolab/backends/embeddings/README.md)：embedding backend contract、API/fake backend 和测试要求。
- [evolab/runtime/README.md](evolab/runtime/README.md)：TaskWorker/TaskRuntime 如何绑定和调用 memory、embedding、LLM runtime。
- [docs/skills/scientific_ie_tool_coverage.md](docs/skills/scientific_ie_tool_coverage.md)：scientific IE required tools。
- [docs/sft.md](docs/sft.md)：trajectory SFT export 和 dry-run/transformers training path。

## 配置原则

EvoLab 的核心边界是 config-driven backend replacement 和 MetaAgent-driven routing。Runtime 编排生命周期，但不把具体 memory algorithm、skill retrieval algorithm、LLM provider、training implementation 或 task-specific workflow 写死在主流程里。

当前配置原则：

- 新实验 config 优先使用短格式：`task` 自然语言、`meta_agent.system_prompt`、`subagents` name/prompt、`lab_root`、`backends`。
- CLI 会把短格式编译成内部 typed `TaskRequest`/`TaskConfig`。
- MetaAgent 只能 route 到配置中的 subagent 名称或 `END`。
- Biology task details 留在 task text 和 `domain_packages/biology_component_extraction_v1/` 资源中，不成为 executable role identity。
- `configs/demo_v1.yaml` 是真实 V1 path。
- `configs/demo_v1_ci.yaml` 是 deterministic CI/offline path。
- `.env` 提供 LLM 和后续外部服务的共享环境变量；当前 demo 的 native mem0 method 使用配置中的 fake memory LLM/embedding backend 和 Lab-local SQLite。
- scientific IE tools 默认不支持 shell execution。

## 当前限制

- V1 的 local trainable backend 只负责 rollout state loading；默认 SFT/OPSD dry-run 训练产出 manifest，不是真实模型训练。
- native mem0 demo 默认使用 Lab-local SQLite store 和 fake memory LLM/embedding backend，不验证外部 memory 服务 SLA。
- GraphSkillBackend V1 只覆盖 seed graph retrieval、required tools 和 update summaries。
- HITL 工具存在 mock/runtime hooks，但没有外部协作系统集成。
- production sandbox、remote artifact store、public skill governance 和 federated consolidation 均属于后续版本。
