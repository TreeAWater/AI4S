# Local Trainable Rollout State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make LocalTrainable an inference-only LLM backend that instantiates states produced by EvolveWorker-managed SFT, OPSD, and Agent0SAGE trainers.

**Architecture:** Add a small local-trainable state manifest contract shared by rollout and trainers. SFT/OPSD trainers publish `local-trainable://...` state refs and attach manifest artifacts; `LocalTrainableLLMBackend` resolves those manifests during `instantiate(state_ref)`. `EvolveWorker` remains the only training control path.

**Tech Stack:** Python 3.11, Pydantic v2, pytest, existing `LLMBackend`, `LLMTrainer`, `LLMEvolutionRequest`, `LLMEvolutionResult`, and `BackendStateRegistry`.

---

## File Structure

- Create `evolab/contracts/local_trainable.py`: state manifest and helper URI contract.
- Modify `evolab/backends/llm/local.py`: rollout-only backend and deterministic local runtime.
- Modify `evolab/backends/trainers/sft.py`: dry-run promotion writes local-trainable state manifest.
- Modify `evolab/backends/trainers/opsd.py`: dry-run promotion writes local-trainable state manifest.
- Modify `evolab/cli.py`: allow `backends.llm.<id>.type = local_trainable`; keep evolution trainer wiring in `evolution.backends`.
- Modify tests covering LocalTrainable, SFT, OPSD, Agent0SAGE, and EvolveWorker handoff.

## Tasks

- [ ] Write failing tests that assert LocalTrainable is rollout-only, not an `LLMTrainer`.
- [ ] Write failing tests that SFT and OPSD dry-run promotion return `local-trainable://` state refs and attach `local_trainable_state` manifest artifacts.
- [ ] Write failing integration test: EvolveWorker promotes SFT state, then LocalTrainable instantiates that promoted state through a registry-backed catalog.
- [ ] Implement the local-trainable state manifest contract.
- [ ] Refactor LocalTrainable runtime/backend to load manifests and remove trainer behavior.
- [ ] Update SFT/OPSD dry-run promotion to publish local-trainable state manifests.
- [ ] Add CLI builder support for rollout `local_trainable` LLM backends.
- [ ] Verify Agent0SAGE preserves nested local-trainable state refs.
- [ ] Run focused tests, then full `pytest -q`.
