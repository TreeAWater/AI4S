# 02-autoresearchclaw

## Status

Configured.

## Repository

- Repo: `aiming-lab/AutoResearchClaw`
- URL: `https://github.com/aiming-lab/AutoResearchClaw.git`
- Commit: `84dad0a`
- Local path: `reference-repos/repos/autoresearchclaw`

## Conda Environment

- Prefix: `reference-repos/envs/autoresearchclaw`
- Python: `3.11.15`
- Installed package: editable `researchclaw-0.3.1`

## Smoke Test

Logged at `reference-repos/smoke/02-autoresearchclaw.txt`.

Passed:
- `python -m compileall -q researchclaw`
- `python -c "import researchclaw"`
- `researchclaw --help`
- `python -m pip check`

Note: `compileall` also printed `Can't list 'sibyl'` and `Can't list 'arc'` because those names are package paths in the build metadata but not top-level directories in this checkout; the command still completed and `pip check` passed.

No API key, long pipeline, dashboard server, or external model call was run.
