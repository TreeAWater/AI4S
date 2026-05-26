# 10-Article Dynamic OpenRouter Evaluation - Partial Before Report

This is **not** a completed 10-article benchmark. The clean-run was interrupted by an external OpenRouter quota error after 6 work items had produced validated artifacts.

- Run root: `/tmp/evolab-10-biology-dynamic-openrouter-stage-before-20260521T184728Z`
- Log: `/tmp/evolab-10-biology-dynamic-openrouter-stage-before-20260521T184728Z.log`
- Subset file: `artifacts/eval_10_article_subset.txt`
- Prediction artifact: `/tmp/evolab-10-biology-dynamic-openrouter-stage-before-20260521T184728Z/partial_evaluation_inputs/partial_validated_records.jsonl`
- Evaluation output: `/tmp/evolab-10-biology-dynamic-openrouter-stage-before-20260521T184728Z/evaluation_partial_fixed10_denominator`
- External blocker: OpenRouter 403 daily key limit exceeded

## Aggregate Metrics Against Fixed 10-Article Denominator

- GT: 10403
- Pred: 9745
- TP / FP / FN: 9744 / 1 / 659
- Precision / Recall / F1: 0.999897 / 0.936653 / 0.967242

## Processed Before Quota

| Article | Accepted records |
|---|---:|
| ai_knowledge_sigma70_design | 5 |
| automated_design_thousands_nonrepetitive_parts_engineering_stable_genetic | 4351 |
| automated_model_predictive_design_synthetic_promoters_control_transcriptional | 5389 |
| characterization_context_dependent_effects_synthetic_promoters | 0 |
| characterization_divergent_promoters_pmaia_phyd_from_gordonia_co | 0 |
| characterization_endogenous_promoters_yarrowia_lipolytica_biomanufacturing_applications | 0 |

## Per-Article Metrics

| Article | GT | Pred | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| automated_model_predictive_design_synthetic_promoters_control_transcriptional | 5391 | 5389 | 5389 | 0 | 2 | 1.0 | 0.999629 | 0.999814 |
| A parts list of promoters and gRNA scaffolds for mammalian genome engineering and molecular recording | 150 | 0 | 0 | 0 | 150 | 0.0 | 0.0 | 0.0 |
| ai_knowledge_sigma70_design | 36 | 5 | 5 | 0 | 31 | 1.0 | 0.138889 | 0.243902 |
| automated_design_thousands_nonrepetitive_parts_engineering_stable_genetic | 4350 | 4351 | 4350 | 1 | 0 | 0.99977 | 1.0 | 0.999885 |
| Characterization of Context-Dependent Effects on Synthetic Promoters | 56 | 0 | 0 | 0 | 56 | 0.0 | 0.0 | 0.0 |
| Characterization of divergent promoters PmaiA and Phyd from Gordonia: Co-expression and regulation by CRP | 2 | 0 | 0 | 0 | 2 | 0.0 | 0.0 | 0.0 |
| Characterization of the endogenous promoters in Yarrowia lipolytica for the biomanufacturing applications | 74 | 0 | 0 | 0 | 74 | 0.0 | 0.0 | 0.0 |
| Characterization of Zymomonas mobilis promoters that are functional in Escherichia coli | 3 | 0 | 0 | 0 | 3 | 0.0 | 0.0 | 0.0 |
| Composability of regulatory sequences controlling transcription and translation in Escherichia coli | 95 | 0 | 0 | 0 | 95 | 0.0 | 0.0 | 0.0 |
| Construction and characterization of a mutant library for the P23 constitutive promoter in lactic acid bacteria | 246 | 0 | 0 | 0 | 246 | 0.0 | 0.0 | 0.0 |

## Interpretation

Precision is already high in the partial outputs. Remaining false negatives in this partial report are dominated by work items that had not reached extraction/finalization before the external quota failure, plus low output for `ai_knowledge_sigma70_design`. This report should be replaced by a full before/after 10-article run when OpenRouter quota is available.
