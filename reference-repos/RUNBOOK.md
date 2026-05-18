# Reference Repos Codex CLI Runbook

Run all repositories in the fixed order:

```bash
bash reference-repos/run_all.sh
```

Run one repository by prompt file:

```bash
codex exec \
  --cd /Users/taw/project/AI4S \
  --sandbox danger-full-access \
  -c 'approval_policy="never"' \
  -m gpt-5.3-codex \
  -c 'model_reasoning_effort="high"' \
  --json \
  -o reference-repos/results/<NN>-<slug>.md \
  - < reference-repos/prompts/<NN>-<slug>.md \
  > reference-repos/logs/<NN>-<slug>.jsonl
```

After each run, inspect the result file and update `DEPLOYMENT_SUMMARY.md`.
