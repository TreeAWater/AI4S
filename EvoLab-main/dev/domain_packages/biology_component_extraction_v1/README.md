# Biology Component Extraction Domain Package v1

This directory is a task-specific domain package for biological component extraction. It is intentionally separate from the stable scientific IE seed skill graph.

Reusable stable skills handle document intake, paper structure parsing, section localization, evidence attribution, table understanding, schema interpretation, field mapping, record construction, domain entity grounding, validation, and evaluation.

This package supplies biology-specific resources for those reusable skills:

- `biology_component_schema.json` defines the target biological component record,
  including promoter-sequence fields used by the current component extraction
  experiment.
- `biological_component_ontology.yaml` defines component types and controlled vocabulary.
- `biological_sequence_policy.yaml` defines sequence normalization and validation policy.
- `biological_evidence_policy.yaml` defines evidence requirements.
- `biological_negative_patterns.yaml` defines false-positive patterns such as primer, barcode, adapter, linker, restriction site, PCR oligo, qPCR primer, and sequencing index.

These biology terms are resource content. They must not be promoted into stable skill IDs such as promoter extraction, RBS extraction, terminator extraction, or microbe trait extraction.

The active experiment config is:

```text
configs/biology_component_extraction_v1_generic_subagents.yaml
```

That config keeps biology-specific requirements in natural-language task text
and uses only generic reusable subagents: `SurveyAgent`, `DesignAgent`,
`ExecAgent`, `CriticAgent`, and `WriteAgent`.
