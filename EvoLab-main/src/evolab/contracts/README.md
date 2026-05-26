# Contracts

`evolab.contracts` defines the Pydantic models shared across the SDK, runtime,
tools, registries, and backends. These models are the stable language between
subsystems.

## Groups

- `common.py`: strict base model, messages, artifacts, and runtime policy.
- `task.py`, `dispatch.py`, `workflow.py`, `dynamic_workflow.py`: task and
  dispatch orchestration contracts.
- `llm.py`, `embeddings.py`, `retrieval.py`, `tools.py`: backend/tool IO
  contracts.
- `records.py`, `state.py`, `lab_state.py`, `snapshots.py`: registry and
  trajectory records.
- `evolution.py`, `local_trainable.py`, `sft.py`, `opsd.py`, `repair.py`,
  `generated_tools.py`: evolution, training, repair, and generated tool
  contracts.

## Development Rules

Contracts should be provider-neutral and serializable with `model_dump`. Avoid
putting filesystem side effects, runtime calls, or provider SDK objects in this
package.

