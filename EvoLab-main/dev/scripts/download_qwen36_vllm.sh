#!/usr/bin/env bash
set -euo pipefail

VLLM_VENV="${VLLM_VENV:-/root/evolab-vllm}"
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B}"
MAX_WORKERS="${MAX_WORKERS:-4}"

if [ ! -x "$VLLM_VENV/bin/hf" ]; then
  echo "Hugging Face CLI not found: $VLLM_VENV/bin/hf" >&2
  echo "Install vLLM first with: uv venv $VLLM_VENV --python python3.10 && uv pip install --python $VLLM_VENV/bin/python vllm --torch-backend=auto" >&2
  exit 2
fi

exec "$VLLM_VENV/bin/hf" download "$MODEL" \
  --include '*.safetensors' \
  --include '*.json' \
  --include '*.txt' \
  --include '*.model' \
  --include '*.jinja' \
  --max-workers "$MAX_WORKERS"
