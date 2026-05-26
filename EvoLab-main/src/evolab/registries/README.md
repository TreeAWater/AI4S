# Registries

Registries persist runtime records as local files inside the Lab's `.evolab`
state tree. They provide replayable state without requiring a database service.

## Module Map

- `task.py`: task request registry.
- `trajectory.py`: meta-agent, subagent, LLM call, and evolution run records.
- `backend_state.py`: backend state refs and lineage.
- `lab_state.py`: indexed Lab state snapshots and warnings.
- `snapshots.py`: environment, toolset, skill, and reward snapshots.

## Development Rules

Registry records should be append-friendly, JSON-serializable, and stable enough
for replay/debugging. Do not store API keys or transient provider SDK objects.

