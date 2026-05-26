#!/usr/bin/env bash
set -euo pipefail

VLLM_VENV="${VLLM_VENV:-/root/evolab-vllm}"
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
DTYPE="${DTYPE:-bfloat16}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"

if [ ! -x "$VLLM_VENV/bin/vllm" ]; then
  echo "vLLM executable not found: $VLLM_VENV/bin/vllm" >&2
  echo "Install it with: uv venv $VLLM_VENV --python python3.10 && uv pip install --python $VLLM_VENV/bin/python vllm --torch-backend=auto" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES
export PATH="$VLLM_VENV/bin:$PATH"

extra_args=()
raw_extra_args="${VLLM_EXTRA_ARGS:-}"
unset VLLM_EXTRA_ARGS
if [ -n "$raw_extra_args" ]; then
  # shellcheck disable=SC2206
  extra_args=($raw_extra_args)
fi

exec "$VLLM_VENV/bin/vllm" serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --dtype "$DTYPE" \
  --reasoning-parser "$REASONING_PARSER" \
  --language-model-only \
  "${extra_args[@]}"
