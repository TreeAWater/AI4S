# Local Trainable Rollout State Design

## Goal

Make `LocalTrainableLLMBackend` a rollout-only backend that can instantiate promoted local LLM states produced by `EvolveWorker`-managed trainers.

## Scope

- `LocalTrainableLLMBackend` implements `LLMBackend` only. It does not schedule, train, evolve, or promote.
- `EvolveWorker` remains the training control plane. It consumes `LLMEvolutionRequest`, calls a trainer, saves the evolution record, then promotes the trainer result through `EvolutionExecutor`.
- SFT, OPSD, and Agent0SAGE trainers remain `LLMTrainer` implementations. They produce promoted candidate states that the rollout backend can instantiate later.
- The first implementation uses artifact-backed dry-run state manifests, not GPU model loading. This keeps CI deterministic while preserving the state handoff semantics needed for real model loading later.

## State Model

Trainer promotion results use a local-trainable state ref:

```text
local-trainable://<backend_id>/state/<uuid>
```

Each promoted state has a `local_trainable_state.json` manifest artifact with:

- `backend_id`
- `state_ref`
- `parent_state_ref`
- `created_by_trainer`
- `adapter_uri`
- `dataset_manifest_uri`
- `default_content`
- `metadata`

The manifest is an artifact attached to `LLMEvolutionResult`, so `EvolutionExecutor` stores it in `BackendStateRecord.artifact_refs`. `LocalTrainableLLMBackend.instantiate(state_ref)` resolves the manifest from an explicit registry-backed catalog or a direct local artifact ref.

## Runtime Behavior

Without a promoted state, `LocalTrainableLLMBackend.instantiate(None)` returns a base local runtime. With a promoted state, it loads the state manifest and returns a runtime whose responses include the promoted state's configured/default content and state metadata.

The runtime is intentionally deterministic in this phase. Real transformers/LoRA loading can replace the runtime internals later without changing the EvolveWorker/trainer/state contract.

## Trainer Interop

SFT and OPSD dry-run promotion should publish `local-trainable://...` states and attach the manifest artifact. Agent0SAGE already delegates solver updates to a nested trainer; if that nested trainer is SFT or OPSD, the wrapped result carries the same local-trainable state through promotion.

## Validation

- LocalTrainable is no longer an `LLMTrainer` and has no `train`/`evolve` API.
- SFT and OPSD promoted dry-run results return local-trainable state refs with state manifests.
- `EvolveWorker` promotion registers the trainer-produced state, and a later LocalTrainable instantiate can load it.
- Agent0SAGE wrapping preserves nested local-trainable state refs.
