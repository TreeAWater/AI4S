# 10-Article Accuracy Improvement Summary

## Scope

This summarizes the previous fixed 10-article dynamic OpenRouter workflow improvement round. It does not report a completed 28-article benchmark and does not include unfinished 28-article work.

## Bugs Found

1. **Dynamic finalization was fragile**
   - Dynamic writer nodes sometimes failed to emit `final_records.jsonl` even when upstream `validated_records.json` already existed.
   - Repeated finalization tool calls such as `write_jsonl` / `serialize_final_records` could loop after final artifacts were already present.

2. **Artifact handoff between dynamic agents was incomplete**
   - Downstream dynamic nodes saw artifact names but not always concrete artifact references and readable paths.
   - Context nodes sometimes wrote reports instead of the declared `context_summary.json`, causing avoidable dependency failures.

3. **Evaluation alignment was too weak**
   - Predictions and GT needed article-aligned matching by article id, title/slug/path-derived keys, not global matching.
   - The evaluator needed a fixed subset filter and per-article deduplication so the 10-article denominator was correct.

4. **Precision failed on broad biological tables**
   - DNA-like columns from primers, barcodes, scaffolds, plasmids, constructs, generated/predicted sequences, replicate details, and model-evaluation tables were being accepted too broadly.

5. **Recall failed when valid intermediate records were not finalized**
   - Some articles had candidate/validated records in intermediate artifacts, but those records were not reliably propagated into aggregate final JSONL outputs.

## Main Fixes

1. **Runtime and finalization fixes**
   - Added dynamic final-record recovery from upstream validated/candidate handoff artifacts.
   - Added aggregate final-record writing for dynamic scientific extraction runs.
   - Added graceful termination when repeated finalization tool calls are suppressed after final artifacts already exist.

2. **Artifact handoff fixes**
   - Passed concrete `available_input_artifact_refs` to downstream dynamic subagents.
   - Recovered missing `context_summary.json` from report evidence when the context node produced useful report artifacts.
   - Stored dynamic workflow artifacts under work-item-aware paths to avoid workflow artifact collisions.

3. **Evaluator fixes**
   - Added `--article-id` / `--article-ids-file` filtering.
   - Added normalized article key matching from article id, slug, title, and source paths.
   - Changed deduplication from global sequence-only to per-article sequence deduplication.
   - Restricted sequence matching to GT records from the same article while preserving exact, substring, and reverse-complement matching.

4. **Filtering and validation gates**
   - Added table triage for primary component tables.
   - Added promoter/regulatory sequence-field scoring.
   - Added rejection/staging for prediction-only/generated-only rows, primer/barcode/scaffold/plasmid/construct/vector/adapter/oligo fields, barcode replicate detail rows, external reference dataset rows, and model-evaluation detail rows.
   - Added reverse-complement-aware duplicate collapse.

## Main Files / Modules Modified

- `evolab/runtime/task_runtime.py`
  - Dynamic artifact passing, context-summary recovery, final-record recovery, aggregate final output writing, repeated-finalization termination.

- `evolab/runtime/dynamic_workflow.py`
  - Dynamic workflow artifact-storage path isolation and stronger guidance toward explicit extraction handoff artifacts.

- `evolab/tools/scientific_artifacts.py`
  - Table triage, candidate-row/record construction, validation gates, accepted-record serialization, deduplication, and handoff artifact preservation.

- `scripts/evaluate_promoter_sequences.py`
  - Article-filtered, article-aligned sequence evaluation and per-article deduplication.

- Tests mainly under:
  - `tests/test_dynamic_extraction_workflow.py`
  - `tests/test_dynamic_workflow_runtime.py`
  - `tests/test_promoter_sequence_evaluator.py`
  - `tests/test_scientific_artifact_tools.py`

## Source of the Improvement

The improvement was not mainly prompt tuning.

The largest gains came from:

1. **Runtime fixes** that prevented finalization loops and allowed dynamic workflows to complete cleanly.
2. **Artifact handoff fixes** that preserved and propagated validated records into final outputs.
3. **Evaluator fixes** that made the denominator and matching article-aligned.
4. **Filtering gates** in scientific artifact tools that removed broad non-target DNA-like records from final outputs.

Prompt/config changes existed mainly as workflow/task guidance, but the decisive changes were executable runtime, artifact, evaluator, and validation/filtering changes.

## Filtering Gates Added

| Gate | Where Applied | Purpose |
| --- | --- | --- |
| Primary component table triage | `evolab/tools/scientific_artifacts.py` table selection / row extraction | Prefer tables with target component sequence semantics; stage broad secondary/detail tables. |
| Promoter/regulatory sequence field scoring | `build_candidate_records` / `_promoter_sequence_field_score` | Choose target sequence columns and avoid non-target sequence fields. |
| Non-target artifact rejection | `validate_candidate_records` / `_profile_validation_issues` | Reject primers, barcodes, scaffolds, plasmids, constructs, vectors, adapters, oligos, and similar artifacts. |
| Prediction-only rejection | table selection and record validation | Reject generated/predicted-only tables or rows unless supported by measurement/component evidence. |
| External/model-detail rejection | record validation | Reject external reference dataset rows and model-evaluation residual/detail rows as primary final components. |
| Duplicate collapse | candidate record building and evaluator deduplication | Collapse exact/reverse-complement duplicates within an article. |
| Article-aligned evaluation gate | `scripts/evaluate_promoter_sequences.py` | Match predictions only against GT from the same article/subset. |

## Final 10-Article Result

The fixed 10-article processed-output evaluation reported:

- GT sequence count: **10,403**
- Predicted sequence count: **9,745**
- TP: **9,744**
- FP: **1**
- FN: **659**
- Precision: **0.999897**
- Recall: **0.936653**
- F1: **0.967242**

The run was still partially affected by OpenRouter quota, so this was a fixed-subset processed-output evaluation rather than a completely clean 10-article run.

## Why Precision and Recall Improved

- Precision improved because final records were restricted to accepted, article-specific, evidence-bearing records and broad non-target DNA-like artifacts were filtered before final serialization/evaluation.
- Recall improved because validated intermediate records were no longer lost when writer/finalization nodes failed or looped, and dynamic artifact references were passed forward more reliably.
- Evaluation became more trustworthy because predictions were aligned to the same article’s GT and deduplicated per article.

## Lessons for 28-Article Debugging

1. First verify scope: evaluate only accepted final records for the intended article subset, not candidates or mixed artifacts from multiple runs.
2. Keep article-aligned evaluation mandatory; global matching can hide or distort failures.
3. Preserve strict precision gates, but ensure validated records are not lost during final aggregation.
4. Treat high-count FP explosions as likely artifact-scope, table-selection, or validated-record preservation bugs before changing extraction prompts.
5. Watch for broad supplementary tables that pass target semantics too easily; table triage must remain generic but stricter on source/table evidence.
6. Do not preserve older broad `validated_records.json` over newer stricter validation decisions; validation artifacts are policy decisions, not just coverage artifacts.
7. Continue using reusable tests for finalization, artifact handoff, evaluator alignment, and biological artifact filtering before scaling runs.
