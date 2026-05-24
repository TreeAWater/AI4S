# SFT From Trajectories

EvoLab SFT uses `LLMCallRecord` trajectories as the teacher trace. It exports selected calls into JSONL samples and can train through the `SFTTrainer`.

## Export

Run a task first:

```bash
evolab clean-run configs/demo_v0.yaml --lab-root /tmp/evolab-demo
```

Export solver/reviewer final-answer calls:

```bash
evolab export-sft \
  --lab-root /tmp/evolab-demo \
  --output-dir /tmp/evolab-demo/artifacts/sft \
  --teacher-backend-id fake-llm
```

The exporter writes:

- `train.jsonl`
- `val.jsonl`
- `manifest.json`

Each JSONL row contains the reconstructed chat messages, source LLM call ref, source run ref, role, action, teacher backend id, and teacher model.

## Dry Run

The dry-run backend is for CI and pipeline checks. It writes adapter-like metadata and returns `not_recommended` unless promotion is explicitly enabled.

```bash
evolab train-sft \
  --lab-root /tmp/evolab-demo \
  --backend-id fake-llm \
  --artifact-root /tmp/evolab-demo/artifacts/sft-train \
  --training-backend dry_run
```

## Transformers

Install optional dependencies before real local training:

```bash
pip install -e '.[sft]'
```

Then train from an API-backed trajectory:

```bash
evolab clean-run configs/demo_api_init.yaml --lab-root /tmp/evolab-api

evolab train-sft \
  --lab-root /tmp/evolab-api \
  --backend-id aigocode-gpt \
  --training-backend transformers \
  --base-model-ref Qwen/Qwen2.5-0.5B-Instruct \
  --training-arg max_steps=5 \
  --training-arg learning_rate=2e-5
```

`training_backend=transformers` renders one full transcript, masks the rendered prompt prefix with `-100`, trains on the assistant completion, saves the model/tokenizer under the artifact root, and returns a promoted SFT state ref when training succeeds.

`train-sft` uses the normal promotion executor. A promoted result is registered in the lab backend-state registry, and an `EvolutionRunRecord` is saved to the trajectory registry.
