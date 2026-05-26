# Dynamic OpenRouter Precision Improvement Summary

## Scope

- 3-article baseline run: `/tmp/evolab-3-biology-dynamic-openrouter-accuracy-20260521T094024Z`
- 3-article precision run: `/tmp/evolab-3-biology-dynamic-openrouter-precision-20260521T104616Z`
- 10-article expansion run: `/tmp/evolab-10-biology-dynamic-openrouter-precision-20260521T120435Z`
- Configs:
  - `configs/tasks/biology_component_extraction_v1_3_article_work_items_dynamic_qwen30b.yaml`
  - `configs/tasks/biology_component_extraction_v1_10_article_work_items_dynamic_qwen30b.yaml`
- Planner and worker used OpenRouter-compatible API through `OPENAI_API_KEY`; no local LMDeploy endpoint was used.
- GT was used only for post-run evaluation and error analysis.

## Generic Fixes Implemented

1. Added a precision-oriented, schema-driven table selection gate for scientific sequence extraction.
   - Keeps primary component tables per source.
   - Stages or rejects broad secondary assay/design/detail tables such as barcode, internal-control, prediction-only, construct, scaffold, plasmid, primer, and non-target sequence tables.
   - This is controlled by config metadata, not hard-coded to article titles or GT.

2. Propagated source/table evidence into candidate rows and records.
   - Records now carry table context, source table metadata, and an `acceptance_reason`.
   - Validation can inspect whether a DNA-like field is supported as the target biological component rather than just any sequence.

3. Strengthened final validation for the promoter sequence extraction profile.
   - Records with insufficient component semantics or unsupported evidence are rejected before final records.
   - Prediction-only/generated-only and non-target sequence artifacts are filtered when the source context does not support target-component status.

4. Added exact and reverse-complement-aware sequence deduplication.
   - Duplicate records caused by repeated sources or equivalent orientation are collapsed before validation/finalization.

5. Fixed a generic dynamic runtime issue found during 10-article expansion.
   - In per-work-item dynamic mode, if a later work item fails dynamic planning after earlier dynamic workflows have already run, EvoLab now records that work item as failed and continues remaining work items.
   - It no longer switches the entire partially executed dynamic task back into the static MetaAgent path.

## 3-Article Metrics

| Article | Before Pred | Before FP | Before Precision | Before Recall | Before F1 | After Pred | After FP | After Precision | After Recall | After F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `parts_list_promoters_grna_scaffolds_mammalian_genome_engineering` | 1629 | 1479 | 0.092081 | 1.000000 | 0.168634 | 209 | 59 | 0.717703 | 1.000000 | 0.835655 |
| `ai_knowledge_sigma70_design` | 5 | 0 | 1.000000 | 0.138889 | 0.243902 | 5 | 0 | 1.000000 | 0.138889 | 0.243902 |
| `automated_design_thousands_nonrepetitive_parts_engineering_stable_genetic` | 6073 | 1723 | 0.716285 | 1.000000 | 0.834693 | 4351 | 1 | 0.999770 | 1.000000 | 0.999885 |

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Micro precision | 0.584534 | 0.986857 | +0.402323 |
| Micro recall | 0.993166 | 0.993166 | 0.000000 |
| Micro F1 | 0.735931 | 0.990001 | +0.254070 |
| Macro precision | 0.602789 | 0.905824 | +0.303035 |
| Macro recall | 0.712963 | 0.712963 | 0.000000 |
| Macro F1 | 0.415743 | 0.693147 | +0.277404 |
| False positives | 3202 | 60 | -3142 |

For the original 3-article benchmark, the precision target was met without recall loss:

- Micro precision >= 0.90: yes, 0.986857.
- Micro recall >= 0.90: yes, 0.993166.

## 10-Article Expansion

The 10-article run exited without process crash and produced aggregate final artifacts, but the EvoLab ledger status is `failed` because some dynamic nodes recorded guard/planning failures. The run remained in dynamic/per-work-item mode after the runtime fix; it did not fall back into static MetaAgent execution.

Output artifacts:

- Report: `/tmp/evolab-10-biology-dynamic-openrouter-precision-20260521T120435Z/evaluation_report.md`
- Summary: `/tmp/evolab-10-biology-dynamic-openrouter-precision-20260521T120435Z/evaluation_summary.json`
- Per article: `/tmp/evolab-10-biology-dynamic-openrouter-precision-20260521T120435Z/evaluation_per_article.jsonl`
- Dynamic summary: `/tmp/evolab-10-biology-dynamic-openrouter-precision-20260521T120435Z/dynamic-workflow-summary.json`

Overall 10-article metrics:

| Metric | Value |
|---|---:|
| GT sequences | 10403 |
| Predicted sequences | 4582 |
| TP | 4505 |
| FP | 77 |
| FN | 5898 |
| Micro precision | 0.983195 |
| Micro recall | 0.433048 |
| Micro F1 | 0.601268 |
| Macro precision | 0.271747 |
| Macro recall | 0.213889 |
| Macro F1 | 0.207944 |

Per-article 10-run metrics:

| Article | GT | Pred | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `parts_list_promoters_grna_scaffolds_mammalian_genome_engineering` | 150 | 209 | 150 | 59 | 0 | 0.717703 | 1.000000 | 0.835655 |
| `ai_knowledge_sigma70_design` | 36 | 5 | 5 | 0 | 31 | 1.000000 | 0.138889 | 0.243902 |
| `automated_design_thousands_nonrepetitive_parts_engineering_stable_genetic` | 4350 | 4351 | 4350 | 1 | 0 | 0.999770 | 1.000000 | 0.999885 |
| `automated_model_predictive_design_synthetic_promoters_control_transcriptional` | 5391 | 17 | 0 | 17 | 5391 | 0.000000 | 0.000000 | 0.000000 |
| `characterization_context_dependent_effects_synthetic_promoters` | 56 | 0 | 0 | 0 | 56 | 0.000000 | 0.000000 | 0.000000 |
| `characterization_zymomonas_mobilis_promoters_that_are_functional_escherichia` | 3 | 0 | 0 | 0 | 3 | 0.000000 | 0.000000 | 0.000000 |
| `characterization_divergent_promoters_pmaia_phyd_from_gordonia_co` | 2 | 0 | 0 | 0 | 2 | 0.000000 | 0.000000 | 0.000000 |
| `characterization_endogenous_promoters_yarrowia_lipolytica_biomanufacturing_applications` | 74 | 0 | 0 | 0 | 74 | 0.000000 | 0.000000 | 0.000000 |
| `composability_regulatory_sequences_controlling_transcription_translation_escherichia_coli` | 95 | 0 | 0 | 0 | 95 | 0.000000 | 0.000000 | 0.000000 |
| `construction_characterization_mutant_library_p23_constitutive_promoter_lactic` | 246 | 0 | 0 | 0 | 246 | 0.000000 | 0.000000 | 0.000000 |

10-article target status:

- Micro precision >= 0.90: yes, 0.983195.
- Micro recall >= 0.90: no, 0.433048.
- Conclusion: the precision gate generalizes for false-positive control, but the current dynamic workflow still misses many articles or emits zero records. Accuracy work should now focus on recall/coverage, not on loosening the precision gate.

## 28-Article Expansion

The 28-article run was not launched in this iteration. The 10-article expansion did not meet the recall target and recorded dynamic node failures, so running 28 articles would spend API/runtime budget before the remaining generic workflow/coverage issues are fixed.

## Remaining Blockers

1. Several dynamic workflows still write empty final records even when the article has GT sequences.
   - Likely causes: insufficient table/file recovery, writer nodes writing empty outputs despite non-empty candidate artifacts, and planner under-specification for small or text-heavy papers.

2. Dynamic workflow observability can overwrite workflow artifacts when planners reuse workflow IDs.
   - This does not affect extracted final records, but it makes dynamic summary incomplete for larger per-work-item batches.

3. Some dynamic nodes still end as `guard_failed` after producing useful artifacts.
   - The runtime preserves artifacts, but ledger status remains failed.

4. Precision on `parts_list` remains below 0.90 at the per-article level.
   - The main remaining FP set is likely broader Table_S1 rows beyond the GT subset. A safer next step is a generic evidence-strength/staging rule, not article-specific filtering.

## Recommended Next Steps

1. Add generic recall recovery for zero-output articles that have candidate tables but no final records.
2. Make dynamic per-work-item persisted workflow paths collision-proof by including work_item_id/spec hash.
3. Improve writer behavior so non-empty validated artifacts are not replaced by empty per-work-item JSONL.
4. Add an evidence-strength scoring stage that can stage weakly supported records without deleting potentially valid primary-component records.
