# Trace2Skill

Trace2Skill turns execution traces into candidate skill updates. It is a
development path for distilling repeated successful behaviors into reusable
skill packages.

## Module Map

- `adapter.py`: converts raw runtime traces into trace pool records.
- `analysts.py`: analyzes traces for reusable behavior and failure patterns.
- `conflicts.py`: detects conflicts between proposed skill changes.
- `consolidation.py`: merges compatible proposals.
- `evolver.py`: orchestrates proposal generation and graph/package updates.
- `outcome.py`: outcome and evidence helpers.
- `schema.py`: trace-to-skill data models.
- `trace_pool.py`: trace storage and retrieval.
- `validator.py`: validates proposed skill packages and graph edits.

## Boundaries

Trace2Skill should emit bounded, reviewable package or graph changes. It should
not bypass validation, write secrets, or mutate task outputs. Promotion into a
stable skill graph is a separate controlled step.

