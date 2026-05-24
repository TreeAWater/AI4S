# Skill Storage and Registry

EvoLab stores reusable scientific skills as independent packages and uses the skill graph as an index over those packages. This keeps the graph small enough to manage while still preserving graph-aware retrieval over scientific capabilities, task categories, and skill relationships.

## Why Packages

The graph should not grow into a single large file containing every procedure, validation case, resource, and domain package. A skill package owns the full reusable skill contract:

- `SKILL.md` for human-readable procedure and operating guidance.
- `metadata.yaml` or `metadata.json` for the machine-readable `CandidateSkill` fields.
- `resources/` for reusable supporting files.
- `tests/` for package-local smoke, synthetic, system, or benchmark checks.

`GraphSkillBackend` hydrates package content into `CandidateSkill` objects only when retrieval needs those skills.

## Lightweight Graphs

Graph JSON files keep:

- scientific process capability roots, such as `Literature`, `Analysis`, `Validation`, and `Execution`;
- intermediate scientific task/category nodes at arbitrary depth;
- lightweight skill nodes with `id`, `name`, `summary`, `package_ref`, `group`, `status`, `tags`, and `metadata`;
- category membership and relationship edges.

The graph can still load legacy embedded `CandidateSkill` entries for backward compatibility. New scientific IE data uses `package_ref` skill nodes.

## Skill Groups

Skill group configs live under `configs/skills/groups/`. A group declares:

- the graph it belongs to;
- skill roots to scan;
- domain package paths that can be passed to tasks;
- include and exclude patterns;
- the default active status.

This is similar in spirit to manager/tree systems such as AgentSkillOS: groups make skill libraries pluggable, while the EvoLab graph remains scientific-domain-oriented and retrieves only a small relevant `SkillBundle` for each subagent run.

## Adding A Reusable Scientific Skill

1. Create a package under `skills/<library>/<skill_slug>/`.
2. Add `SKILL.md`.
3. Add `metadata.yaml` or `metadata.json` with the full reusable skill contract.
4. Add package resources or tests if needed.
5. Add a lightweight graph skill node with `package_ref`.
6. Add `belongs_to_category` and any `related_to`, `depends_on`, or other relation edges.

No backend Python code should be edited for a new skill package.

## Adding A Domain Package

Domain packages live under `domain_packages/`. They can contain schemas, ontologies, policies, negative patterns, and task configs. Domain-specific terms such as biological component names may appear there.

Domain packages must not become stable reusable skill IDs. For example, use a reusable skill like `skill.schema_guided_field_mapping.v1` with a biology schema resource, not `skill.promoter_extraction.v1`.

## Retrieval Flow

`GraphSkillBackend.get(...)` uses:

1. `GraphSkillStore` to load graph JSON and hydrate skill packages.
2. `SkillRegistry` to scan skill roots and group configs.
3. `GraphTreeIndexer` to build recursive category indexes.
4. `GraphSkillSearcher` to traverse matching category paths, score skill refs, expand one-hop relationships, deduplicate results, aggregate required tools, and return trace metadata.

`GraphSkillBackend.look_at(...)` runs conservative Post-run Skill Evolution v1. It records observations, generates proposals and policy decisions, applies only bounded metadata usage/failure stats, and stages candidate skills, required-tool changes, and relationship changes for review.
