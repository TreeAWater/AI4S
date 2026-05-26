# Skill Backends

This package implements skill retrieval and skill evolution. Skills are reusable
task capabilities that can declare required tools, package metadata, examples,
and graph relationships.

## Module Map

- `base.py`: skill backend interface.
- `fake.py`: deterministic skill backend for tests and examples.
- `graph.py`: graph-backed retrieval backend.
- `store.py`: graph file loading and package hydration.
- `registry.py`: package scanning and group loading.
- `package_loader.py`: converts skill package folders into candidate skills.
- `graph_schema.py` and `package_schema.py`: graph and package contracts.
- `searcher.py` and `graph_indexer.py`: graph traversal and retrieval ranking.
- `evolution.py`: post-run skill update analysis and persistence.
- `trace2skill/`: trace-to-skill distillation utilities.

## Boundaries

Skill backends return `SkillBundle` contracts and required tool names. They do
not execute tools, call LLMs for task work, or mutate role pools. Skill package
data used by a session should live under `.evolab/skills` or a configured dev
fixture path, not as user-visible output.

