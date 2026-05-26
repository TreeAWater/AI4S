# 28-Article Dynamic Biology Extraction Benchmark Status

## Result

The 28-article benchmark did not complete. The run was stopped by an external OpenRouter quota error before all work items were processed:

```text
OpenRouter HTTP 402: requested up to 2048 tokens, but the key could only afford 1096.
```

The configured local Qwen fallback endpoint was checked at `http://127.0.0.1:8000/v1/chat/completions` and was unavailable with connection refused. Therefore no after-debug rerun was started in this session.

## Run Configuration

- Config: `configs/tasks/biology_component_extraction_v1_28_article_work_items_dynamic_qwen30b.yaml`
- Dataset: `/root/bio/dataset/Constitutive_promoters_md_30_new`
- GT: `/root/bio/Evaluation`
- Lab root: `/tmp/evolab-28-biology-dynamic-openrouter-before-20260522T011453Z`
- Planner backend: OpenRouter `openai/gpt-4.1-nano`
- Worker backend: OpenRouter `qwen/qwen3-30b-a3b-instruct-2507`
- API key env: `OPENAI_API_KEY`

## Runtime Summary

| Item | Value |
|---|---:|
| Fixed subset size | 28 |
| Dynamic workflow specs created | 27 |
| Work items with validated artifacts | 26 |
| Work items with article-level final JSONL artifacts | 12 |
| Remaining claimed/running queue files | 0 |
| Failed queue files | 1 |
| Exit status | 1 |

Missing validated artifacts:

- `characterization_context_dependent_effects_synthetic_promoters`
- `tn7_device_calibrated_heterologous_gene_expression_pseudomonas_putida`

## Diagnostic Evaluation

This is not a completed 28-article benchmark metric. It is a diagnostic exact/reverse-complement-only evaluation over the fixed 28-article GT denominator using the 26 available validated artifacts. The full substring evaluator was attempted but was too slow on the large over-extracted partial artifact, so these numbers are not the final benchmark score.

| Scope | GT | Pred | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Blocked partial run, exact/RC diagnostic | 12690 | 63940 | 49 | 63891 | 12641 | 0.000766 | 0.003861 | 0.001279 |

Detailed diagnostic files:

- `artifacts/eval_28_article_report_before.json`
- `artifacts/eval_28_article_report_before.md`
- `artifacts/eval_28_article_bad_cases_before.json`
- `artifacts/eval_28_article_bad_cases_before.md`

## Generic Fixes Applied

- Added dynamic final-answer artifact recovery in `TaskRuntime`: when a dynamic subagent returns a declared output artifact as JSON in `final_answer`, the runtime now persists that artifact and can terminate the node cleanly instead of looping on finalization tools.
- Added regression tests for wrapped and single-artifact unwrapped JSON final-answer recovery.
- Added a dynamic 28-article config using the same dynamic workflow mode as the successful smaller runs.
- Reduced the 28-article dynamic config `max_output_tokens` from 2048 to 1024 for both planner and worker backends to avoid the exact OpenRouter 402 condition observed in this run.

## Before/After

No after-debug 28-article rerun was performed because the only available configured remote backend hit a credit limit and the local fallback endpoint was not running.

| Run | Completed? | Precision >= 90%? | Recall >= 90%? | Notes |
|---|---|---|---|---|
| Before OpenRouter run | No | Not evaluated as final | Not evaluated as final | External HTTP 402 quota stopped the run. |
| After rerun | Not run | Not evaluated | Not evaluated | Blocked by backend availability. |

## Commands Used

```bash
pytest tests/test_dynamic_workflow_runtime.py -q
pytest tests/test_dynamic_extraction_workflow.py -q
pytest tests/test_scientific_artifact_tools.py -q
pytest tests/test_promoter_sequence_evaluator.py -q
pytest tests/test_graph_skill_backend.py -q

set -a; source .env; set +a
OUT=/tmp/evolab-28-biology-dynamic-openrouter-before-20260522T011453Z
python3 -m evolab.cli clean-run \
  configs/tasks/biology_component_extraction_v1_28_article_work_items_dynamic_qwen30b.yaml \
  --lab-root "$OUT" \
  > "$OUT.log" 2>&1
```

## Next Step

When backend quota is restored, rerun the same fixed subset with the updated 1024-token config. If local Qwen is preferred, start a compatible OpenAI-style local endpoint first and add a matching dynamic config rather than changing task logic.
