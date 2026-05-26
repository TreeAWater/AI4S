#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B}"

curl -fsS --max-time 10 "$BASE_URL/models" >/dev/null

python3 - <<'PY'
import os

from evolab.backends.llm import ApiLLMBackend, ApiLLMBackendConfig
from evolab.contracts.common import Message
from evolab.contracts.llm import LLMGenerationConfig

base_url = os.environ.get("BASE_URL", "http://127.0.0.1:8000/v1")
model = os.environ.get("MODEL", "Qwen/Qwen3.6-35B-A3B")

backend = ApiLLMBackend(
    ApiLLMBackendConfig(
        provider="openai",
        api="openai-chat-completions",
        hosting="local",
        api_key_env="LOCAL_QWEN_API_KEY",
        base_url=base_url,
        model=model,
        max_retries=0,
        timeout_seconds=120.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    ),
    backend_id="local-qwen36-vllm",
)
runtime = backend.instantiate(None)
response = runtime.generate(
    messages=[Message(role="user", content='Return JSON only: {"ok": true}')],
    tool_specs=[],
    generation_config=LLMGenerationConfig(model="", max_output_tokens=64),
)
print(response.action.action)
print((response.action.content or "").strip())
print(response.raw_response.get("model"))
PY
