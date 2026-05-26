# Trainer Backends

Trainer backends turn recorded trajectories or exported datasets into trainable
state refs. They are used by evolution/export flows rather than normal task
execution.

## Module Map

- `base.py`: trainer contract.
- `blank.py`: no-op trainer for unsupported or intentionally empty paths.
- `sft.py`: supervised fine-tuning dataset/training helper.
- `opsd.py`: OPSD export/training helper.
- `agent0_sage.py`: Agent0/SAGE-style fake trainer support.

## Development Rules

Trainers should write artifacts under the configured artifact root and return
manifest/state refs through contracts. They should not modify active runtime
state unless promotion is explicitly requested.

