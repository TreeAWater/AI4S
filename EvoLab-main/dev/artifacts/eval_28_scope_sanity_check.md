# 28-Article Processed-Output Scope Sanity Check

This is a processed-output evaluation, not a completed 28-article benchmark. The two unfinished articles are excluded because the original run stopped on OpenRouter quota/balance failure.

## Scope

- Planned articles: 28
- Processed/validated articles evaluated: 26
- Unfinished articles excluded: 2
- Prediction artifact evaluated after source-gated rebuild: `/tmp/evolab-28-biology-dynamic-openrouter-before-20260522T011453Z/evaluation_28_processed_source_gated_after_debug/predictions_from_source_gated_validated_records.jsonl`
- Rebuilt validated artifact root: `/tmp/evolab-28-biology-dynamic-openrouter-before-20260522T011453Z/evaluation_28_processed_source_gated_after_debug/validated_artifacts`
- GT root: `/root/bio/Evaluation`
- Matching: article-aligned using existing GT title alignment, same-article only, deduplicated by normalized sequence, exact/substring/reverse-complement policy.

## Sanity Checks

| Check | Result | Evidence |
|---|---|---|
| Correct prediction artifact | Pass | Uses accepted `validated_records.json` records rebuilt from per-article candidate tables. |
| Raw candidates evaluated | Pass | Candidate rows/records are not passed to evaluator; only accepted records are aggregated. |
| Diagnostic/global mixed artifacts | Pass | Rebuild reads each processed article directory once and writes a fresh prediction JSONL. |
| Duplicate inflation | Mostly pass | Predictions are deduplicated by normalized sequence within article before scoring. |
| Article alignment | Pass | Uses the existing normalized-title article alignment file. |
| Same-article matching only | Pass | Each work item is evaluated against only its matched GT publication subset. |
| Unfinished articles excluded | Pass | Excluded: `characterization_context_dependent_effects_synthetic_promoters`, `tn7_device_calibrated_heterologous_gene_expression_pseudomonas_putida`. |

## Conclusion

The remaining low score is a real source classification and recall problem, not a raw-candidate or cross-article evaluation-scope bug. The source gate reduces broad false positives but still needs better source ranking and cross-table ID-to-sequence/evidence linking for high-GT-count articles.
