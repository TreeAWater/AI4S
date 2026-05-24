# SFT Dataset And Trainer Design

## Goal

Use recorded API or fake LLM trajectories as supervised fine-tuning data without introducing a separate LocalSFT concept. The trainer name and config type are `SFT`.

## Scope

- `LLMCallRecord` remains the source of truth for SFT / distillation data.
- `SFTDatasetSample` stores one selected call as a chat transcript plus source refs, teacher backend id, teacher model, role, action, and metadata.
- `SFTDatasetManifest` stores train/validation paths, sample counts, source refs, teacher ids/models, and selection config.
- `export_sft_dataset(...)` writes JSONL train/validation files and a manifest from a `TrajectoryRegistry`.
- `SFTTrainer` is an `LLMTrainer` that exports samples and then runs one configured training backend.
- CLI exposes `export-sft` for dataset creation and `train-sft` for training from an existing lab trajectory through the normal promotion executor.

## Selection Rules

The exporter filters by teacher backend id, runtime stage, action, source run ref, and source LLM call ref.

Defaults:

- runtime stages: `subagent_flat`, `workflow_node`
- actions: `final_answer`
- meta-agent calls excluded
- tool-call samples excluded

Tool-use final-answer samples reconstruct the assistant tool-call message before the corresponding tool result when the prior call exists in the same run.

## Training Backends

`training_backend="dry_run"` is the CI and smoke-test path. It writes adapter-like metadata and returns `not_recommended` by default. It can only return a promoted candidate when `promote_dry_run=true` is explicitly set.

`training_backend="transformers"` is the minimal real SFT path. It lazily imports optional dependencies, requires `base_model_ref`, renders one full transcript per sample, masks the rendered prompt prefix with `-100`, runs `transformers.Trainer`, saves the resulting model and tokenizer under the evolution artifact root, and returns a promoted candidate with the saved model artifact.

`LLMEvolutionRequest.budget.max_train_samples` caps the exported train samples. If both the SFT export config and the request budget set a max, the stricter cap wins.

Production LoRA/PEFT, model serving, remote training orchestration, and adapter composition remain out of scope for this spec.

## Validation

- Exported samples preserve full input/output messages needed for SFT.
- Tool-use transcripts include prior assistant tool-call context.
- `SFTTrainer` outputs valid `LLMEvolutionResult` objects.
- Dry-run promotion passes the promotion guard only when explicitly enabled.
- `train-sft` registers promoted state and saves an `EvolutionRunRecord`.
- `clean-run` can instantiate an SFT evolution backend from config without runtime code changes.
