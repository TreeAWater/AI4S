# Lab

`evolab.lab` defines the on-disk Lab workspace layout and queue/resolver
helpers.

## Module Map

- `layout.py`: canonical path mapping for user-visible workspace files and
  internal `.evolab` state.
- `queue.py`: file-backed queue implementation.
- `resolver.py`: factory for queues and registries tied to a `LabLayout`.

## Layout Rule

All EvoLab internal state must live under `<lab>/.evolab`: role pools, tools,
skills, memory, queues, registries, snapshots, trajectories, generated tools,
and internal configs. Files outside `.evolab` are user inputs or user-visible
outputs.

