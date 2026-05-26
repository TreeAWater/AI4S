# LABBench2 External Repo Local Changes

Created: 2026-05-26

`benchmark_repo/labbench2` is an external repository. Local patches made there should be recorded in this document so they can be replayed, reviewed, or dropped when syncing upstream.

## 2026-05-26: Configurable LLM Judge

Purpose:

- Change the default judge model to `openai:gpt-5.4-mini@low`.
- Make the judge model configurable from the evaluation CLI.
- Parse `@low`, `@medium`, and `@high` judge suffixes through the existing model config path so reasoning effort reaches the API settings.

Touched files:

- `benchmark_repo/labbench2/evals/evaluators.py`
- `benchmark_repo/labbench2/evals/run_evals.py`
- `benchmark_repo/labbench2/evals/__init__.py`
- `benchmark_repo/labbench2/tests/unit/test_evaluators.py`
- `benchmark_repo/labbench2/tests/unit/test_run_evals.py`

Behavior:

- `DEFAULT_JUDGE_MODEL = "openai:gpt-5.4-mini@low"`.
- `LLMJudgeEvaluator` strips the suffix when passing the model name to `Agent`.
- `judge_model_settings()` uses `get_model_config()` and preserves `temperature` and `timeout`.
- `run_evals.py` accepts:
  - `--judge-model`
  - `--judge-temperature`
  - `--judge-timeout`

Verification:

```bash
cd benchmark_repo/labbench2
conda run -n labbench2 pytest tests/unit/test_evaluators.py tests/unit/test_run_evals.py -q
```

Result observed on 2026-05-26:

```text
19 passed, 9 warnings
```

Warnings were existing `pydantic_ai` deprecation/provider warnings, not test failures.

## 2026-05-26: Main-Repo Evaluation Slice Docs

Purpose:

- Keep evaluation slice metadata outside the external LabBench2 repo.
- Provide a fixed 50-question API smoke slice and a ready `--ids-file`.

Files added in the main repo:

- `doc/labbench2_prop50_slice.md`
- `eval_slices/labbench2_prop50_ids.txt`

No LabBench2 source files were changed for this slice documentation.

## 2026-05-26: Optional Per-Case Progress Lines

Purpose:

- Add an explicit opt-in progress mode for long API runs.
- Print one line when each case completes, so `tee` logs show `x/total` progress even when the rich progress bar does not stream cleanly.

Touched files:

- `benchmark_repo/labbench2/evals/run_evals.py`
- `benchmark_repo/labbench2/tests/unit/test_run_evals.py`

Behavior:

- New CLI flag: `--progress-lines`.
- Default behavior is unchanged when the flag is absent.
- With the flag, each completed case prints:

```text
Progress: 3/50 done <id> <tag> <type>
```

or `failed` when the per-case report is a failure.

Verification:

```bash
cd benchmark_repo/labbench2
conda run -n labbench2 pytest tests/unit/test_run_evals.py -q
```

Result observed on 2026-05-26:

```text
11 passed, 7 warnings
```
