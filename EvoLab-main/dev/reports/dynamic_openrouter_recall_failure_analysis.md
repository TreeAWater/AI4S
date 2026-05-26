# Dynamic OpenRouter 10-Article Recall Failure Analysis

Scope: fixed 10-article subset in `artifacts/eval_10_article_subset.txt`.

Run inspected: `/tmp/evolab-10-biology-dynamic-openrouter-stage-before-20260521T184728Z`

Status: the clean-run did not complete. OpenRouter returned an external 403 daily key limit error during the sixth work item's writer stage. No work items remained claimed/running after process exit; the task queue contains a failed task record with the API error. The report below is therefore a partial post-hoc analysis of artifacts already produced before interruption, not a completed 10-article before/after cycle.

## Aggregate Partial Metrics Against 10-Article Denominator

- GT: 10403
- Pred: 9745
- TP / FP / FN: 9744 / 1 / 659
- Precision / Recall / F1: 0.999897 / 0.936653 / 0.967242

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

## Main FN Sources

| Article | FN | Likely source | Generic fix direction |
|---|---:|---|---|
| automated_model_predictive_design_synthetic_promoters_control_transcriptional | 2 | partial extraction coverage | inspect candidate rows and validator rejection reasons; improve generic table discovery and source handoff |
| A parts list of promoters and gRNA scaffolds for mammalian genome engineering and molecular recording | 150 | work item not processed before quota or zero accepted records before interruption | rerun when API quota is available; preserve zero-output recovery and validated-artifact handoff checks |
| ai_knowledge_sigma70_design | 31 | near-zero output despite completed candidate/validation artifacts; likely table/source coverage or acceptance gate too narrow | inspect candidate_records versus rejected_records; improve generic high-confidence candidate promotion without weakening primer/plasmid filters |
| Characterization of Context-Dependent Effects on Synthetic Promoters | 56 | work item not processed before quota or zero accepted records before interruption | rerun when API quota is available; preserve zero-output recovery and validated-artifact handoff checks |
| Characterization of divergent promoters PmaiA and Phyd from Gordonia: Co-expression and regulation by CRP | 2 | work item not processed before quota or zero accepted records before interruption | rerun when API quota is available; preserve zero-output recovery and validated-artifact handoff checks |
| Characterization of the endogenous promoters in Yarrowia lipolytica for the biomanufacturing applications | 74 | work item not processed before quota or zero accepted records before interruption | rerun when API quota is available; preserve zero-output recovery and validated-artifact handoff checks |
| Characterization of Zymomonas mobilis promoters that are functional in Escherichia coli | 3 | work item not processed before quota or zero accepted records before interruption | rerun when API quota is available; preserve zero-output recovery and validated-artifact handoff checks |
| Composability of regulatory sequences controlling transcription and translation in Escherichia coli | 95 | work item not processed before quota or zero accepted records before interruption | rerun when API quota is available; preserve zero-output recovery and validated-artifact handoff checks |
| Construction and characterization of a mutant library for the P23 constitutive promoter in lactic acid bacteria | 246 | work item not processed before quota or zero accepted records before interruption | rerun when API quota is available; preserve zero-output recovery and validated-artifact handoff checks |

## False Positive Examples

| Article | Sequence | Component | Likely category | Generic prevention |
|---|---|---|---|---|
| automated_design_thousands_nonrepetitive_parts_engineering_stable_genetic | `TTGACGGCTAGCTCAGTCCTAGGTACAGTGCTAGC` | J23100 | evaluator unmatched extra candidate | compare source evidence and component semantics; keep precision gate active |

## Artifact/Runtime Findings

- Stage 0 runtime stability is improved: the latest 3-article run exited with ledger status `completed`, no claimed/running queue entries, and final aggregate artifacts present.
- The 10-article run did not show a repeated finalization loop. It stopped on an external OpenRouter quota error.
- Validated artifacts existed for 6 work items before quota interruption; 4 of the fixed 10 were not reached.
- The evaluator was fixed generically to align slug article ids with title/path GT identifiers and to match sequences within aligned article groups.

## Remaining Work Once Quota Recovers

1. Rerun the same fixed 10-article clean-run from a fresh lab root.
2. Evaluate the completed aggregate `biology_component_records.jsonl` with `--article-ids-file artifacts/eval_10_article_subset.txt`.
3. If recall remains below target on completed outputs, inspect rejected candidate records and final writer handoff by article.
