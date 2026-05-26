# SkillBackend Post-run Evolution V1

EvoLab Post-run Skill Evolution v1 turns `SkillObservationRequest` records into structured, deterministic skill update proposals. It is intentionally conservative: it records observations, applies only bounded metadata statistics to existing skills, and stages anything that could change stable behavior for later review.

## Implemented

`GraphSkillBackend.look_at(...)` now:

1. accepts the existing observation payload;
2. builds a compact observation summary;
3. generates `SkillUpdateProposal` records;
4. evaluates proposals with `SkillEvolutionPolicy`;
5. applies safe metadata-only updates;
6. persists observations, proposals, decisions, staged candidates, staged updates, and applied updates as JSONL;
7. returns a `SkillUpdateResult` with proposal and decision details in `update_summary` and `metadata`.

## Automatically Applied

Only safe, bounded metadata updates are automatically applied to existing graph skill nodes:

- `usage_stats_update`
- `failure_note_update`

The applied metadata lives under each graph skill node's `metadata.evolution_stats`:

- `usage_count`
- `success_count`
- `failure_count`
- `last_observed_at`
- `last_status`
- `recent_failure_reasons`

These updates do not change skill IDs, required tools, graph relationships, package content, or stable retrieval semantics.

## Staged For Review

The following proposals are persisted but not applied automatically:

- `example_trace_memory_update`
- `required_tools_update_proposal`
- `candidate_skill_creation`
- `relationship_update_proposal`
- general `metadata_update`

Candidate skills are written as `CandidateSkillRecord` entries with `status: staged`. They are not inserted into the stable graph in v1.

## Persistence

By default, `GraphSkillBackend` writes evolution records under:

```text
<graph_dir>/backend_state/skill_evolution/
```

Files:

- `observations.jsonl`
- `proposals.jsonl`
- `decisions.jsonl`
- `staged_candidates.jsonl`
- `staged_updates.jsonl`
- `applied_updates.jsonl`

The legacy `<graph>.updates.jsonl` summary log is still written for compatibility.

## Future Work

V1 does not implement:

- automatic required tool edits;
- automatic candidate skill promotion;
- graph rewiring;
- LLM-based skill synthesis;
- package metadata mutation;
- cross-run promotion policies.

Those belong in v1.5/v2 after review, validation, and rollback policies exist.
