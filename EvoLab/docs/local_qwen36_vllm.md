# Local Qwen3.6 vLLM

This setup serves `Qwen/Qwen3.6-35B-A3B` through a local OpenAI-compatible vLLM endpoint and routes EvoLab to it with `hosting: local`.

## Environment

The prepared vLLM environment lives outside the repo:

```bash
/root/evolab-vllm/bin/python
/root/evolab-vllm/bin/vllm
```

It was installed with:

```bash
uv venv /root/evolab-vllm --python python3.10
uv pip install --python /root/evolab-vllm/bin/python vllm --torch-backend=auto
uv pip install --python /root/evolab-vllm/bin/python socksio
```

## Start Server

Optionally pre-download model weights:

```bash
scripts/download_qwen36_vllm.sh
```

```bash
scripts/serve_qwen36_vllm.sh
```

Defaults:

- `MODEL=Qwen/Qwen3.6-35B-A3B`
- `HOST=127.0.0.1`
- `PORT=8000`
- `CUDA_VISIBLE_DEVICES=4,5,6,7`
- `TENSOR_PARALLEL_SIZE=4`
- `MAX_MODEL_LEN=32768`
- `GPU_MEMORY_UTILIZATION=0.85`
- `REASONING_PARSER=qwen3`

Override any of these with environment variables. Extra vLLM flags can be passed through `VLLM_EXTRA_ARGS`.

Qwen's model card recommends vLLM with `--reasoning-parser qwen3` for Qwen3.6. The script also passes `--language-model-only` because EvoLab's current biology extraction task is text-only and the smaller memory footprint is useful on shared GPUs.

The script prepends `/root/evolab-vllm/bin` to `PATH`; vLLM worker processes need that so tools such as `ninja` are visible during `torch.compile`.

For startup diagnosis on shared A800 GPUs, this reduced configuration has been verified:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
TENSOR_PARALLEL_SIZE=4 \
MAX_MODEL_LEN=8192 \
GPU_MEMORY_UTILIZATION=0.75 \
VLLM_EXTRA_ARGS="--enforce-eager" \
scripts/serve_qwen36_vllm.sh
```

The EvoLab config sends `extra_body.chat_template_kwargs.enable_thinking=false`. Without that request option, vLLM separates Qwen's thinking trace into the `reasoning` field and can return an empty assistant `content`, which EvoLab treats as `empty_model_response`.

## Smoke Test

After the server is ready:

```bash
scripts/smoke_local_qwen36_vllm.sh
```

For the biology task, use:

```bash
python3 -m evolab.cli clean-run \
  configs/tasks/biology_component_extraction_v1_28_article_work_items_local_qwen36.yaml \
  --lab-root /tmp/evolab-local-qwen36-vllm
```
