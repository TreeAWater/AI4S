# Embedding Backend Development

本模块提供 EvoLab 的 embedding runtime 抽象。Embedding backend 是独立于 chat
LLM 的一等 backend，当前主要服务 native mem0 memory method 的 add/search
向量化，也可被后续 retrieval 或 evaluator backend 复用。

## Module Map

- `base.py`：`EmbeddingBackend` 和 `EmbeddingRuntime` contract。
- `api.py`：OpenAI-compatible embeddings API backend。
- `fake.py`：deterministic fake embedding backend，用于 CI/offline demo 和单测。
- `evolab/contracts/embeddings.py`：`EmbeddingResponse` shared contract。

## Contract

`EmbeddingBackend.instantiate(state_ref)` 返回一个 runtime。runtime 必须实现：

```python
embed(texts: list[str], *, purpose: str) -> EmbeddingResponse
```

约束：

- `vectors` 顺序必须与输入 `texts` 一一对应。
- 返回 vector 数量必须等于输入 text 数量。
- 每个 vector 应为 finite numeric list；store 层会再次校验。
- `EmbeddingResponse.backend_id` 必须是配置中的 backend id。
- `EmbeddingResponse.model` 必须标识实际使用的模型或 deterministic backend 名称。
- `metadata["purpose"]` 应保留调用方传入的 purpose，便于排查 add/search 行为。

不允许在 embedding backend 内 silent fallback 到 fake implementation。真实 API
缺少 credential、base URL 配错或 provider 报错时，应直接暴露错误。

## API Backend

`ApiEmbeddingBackend` 使用 OpenAI-compatible `client.embeddings.create(...)`。
配置入口：

```yaml
backends:
  embedding:
    memory-embedding:
      type: api
      api: openai-embeddings
      model: text-embedding-3-small
      api_key_env: OPENAI_API_KEY
      base_url: https://api.openai.com/v1
      timeout_seconds: 30
```

也可以通过 `.env` 和 `env_ref` 注入共享配置：

```dotenv
MEMORY_EMBEDDING_API=openai-embeddings
MEMORY_EMBEDDING_BASE_URL=https://api.openai.com/v1
MEMORY_EMBEDDING_API_KEY=your-api-key
MEMORY_EMBEDDING_MODEL=text-embedding-3-small
```

```yaml
backends:
  embedding:
    memory-embedding:
      type: api
      env_ref: memory-embedding
```

Inline `api_key` 或 `apiKey` 不允许写入 config；使用 `.env` 或 `api_key_env`。

## Fake Backend

`FakeEmbeddingBackend` 用 SHA-256 派生 deterministic normalized vectors：

```yaml
backends:
  embedding:
    fake-memory-embedding:
      type: fake
      dimensions: 8
```

它只适合 deterministic tests、offline demo 和 smoke path。不要用它评估真实 memory
retrieval 质量。

## Implementing A New Provider

新增 embedding provider 时：

1. 实现 `EmbeddingBackend.instantiate()` 和 runtime `embed()`。
2. 保持 input/output cardinality，不要重排 provider 返回的 vectors。
3. 把 provider response 解码为 plain `list[list[float]]`，不要把 SDK object 泄漏到 contract。
4. 不记录 secret、API key 或完整 credential path。
5. 在 CLI `_build_embedding_backends()` 中增加 config validation。
6. 添加单测覆盖成功调用、credential 缺失、invalid vector/cardinality 和 metadata。

## Tests

相关测试入口：

```bash
pytest tests/test_embedding_backends.py tests/test_native_mem0_method.py \
  tests/test_native_mem0_api_smoke_script.py -q
```

native mem0 的真实 API smoke 会同时验证 embedding backend 被实际调用，并检查生成的
SQLite store 中存在 active memory records。
