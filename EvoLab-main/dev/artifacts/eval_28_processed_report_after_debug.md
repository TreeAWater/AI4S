# 28-Article Processed-Output Report After Source-Gated Debug

This is a processed-output evaluation, not a completed 28-article benchmark. The two unfinished articles remain excluded because the original run stopped on OpenRouter quota/balance failure.

- Prediction artifact: `/tmp/evolab-28-biology-dynamic-openrouter-before-20260522T011453Z/evaluation_28_processed_source_gated_after_debug/predictions_from_source_gated_validated_records.jsonl`
- Evaluation artifacts: `/tmp/evolab-28-biology-dynamic-openrouter-before-20260522T011453Z/evaluation_28_processed_source_gated_after_debug/article_aligned_evaluator`
- Matching policy: article-aligned by existing GT title alignment; same-article only; deduplicated normalized sequences; exact/substring/reverse-complement matching.

## Metrics

| Metric | Before latest source gate | After latest source gate |
|---|---:|---:|
| gt_sequence_count | 43026 | 43025 |
| predicted_sequence_count | 61421 | 23352 |
| true_positive | 24751 | 19906 |
| false_positive | 36670 | 3446 |
| false_negative | 18275 | 23119 |
| precision | 0.402973 | 0.852432 |
| recall | 0.575257 | 0.462661 |
| f1 | 0.473944 | 0.599786 |
