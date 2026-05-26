# Reward Backends

`evolab.backends.rewards` contains reward calculators used by training and
evaluation export paths.

## Module Map

- `base.py`: reward request, score, and calculator interface.
- `tool_calls.py`: reward based on tool call patterns.
- `composite.py`: weighted composition of multiple calculators.

## Development Rules

Reward calculators should be deterministic for the same input example. They
should return explicit per-sample scores and avoid reading mutable Lab state
unless the request includes the needed artifacts.

