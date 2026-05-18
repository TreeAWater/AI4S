# 03-medgeclaw

## Status

Configured.

## Repository

- Repo: `xjtulyc/MedgeClaw`
- URL: `https://github.com/xjtulyc/MedgeClaw.git`
- Commit: `fef51d3`
- Local path: `reference-repos/repos/medgeclaw`

## Conda Environment

- Prefix: `reference-repos/envs/medgeclaw`
- Python: `3.11.15`

## Smoke Test

Logged at `reference-repos/smoke/03-medgeclaw.txt`.

Passed:
- `python -m compileall -q sync.py`
- Key files present: `README.md`, `README_ZH.md`, `MEDGECLAW.md`, `IDENTITY.md`, `setup.sh`, `medgeclaw-init.sh`
- Docs inspection found `7` files under `docs/`
- README headings and quick-start sections were readable

No user-level OpenClaw/Claude configuration was written, and no provider/API test was run.
