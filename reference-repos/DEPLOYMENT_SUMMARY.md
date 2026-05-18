# Reference Repos Deployment Summary

Workspace: `/Users/taw/project/AI4S`

Deployment root: `/Users/taw/project/AI4S/reference-repos`

Policy:
- Each repository was handled through a visible downstream `codex exec` thread where possible.
- Downstream model: `gpt-5.3-codex`.
- Downstream reasoning effort: `high`.
- Each repository has an isolated conda prefix environment under `reference-repos/envs/`.
- Repositories and conda environments stay inside `reference-repos/`.
- No user-level Claude/OpenClaw/agent configuration was modified.
- Smoke tests avoid API keys, GPU-only workloads, long runs, and paid model calls.

## Final Status

| Order | Repository | Commit | Local path | Conda env | Status | Smoke test | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `SakanaAI/AI-Scientist` | `1de1dbc` | `reference-repos/repos/ai-scientist` | `reference-repos/envs/ai-scientist` | Configured | `pip check`, `compileall`, `import ai_scientist`, `launch_scientist.py --help` passed | Upgraded `datasets` to `4.8.5` and kept `fsspec==2026.2.0` to satisfy both `datasets` and `aider-chat`. |
| 2 | `aiming-lab/AutoResearchClaw` | `84dad0a` | `reference-repos/repos/autoresearchclaw` | `reference-repos/envs/autoresearchclaw` | Configured | `compileall`, `import researchclaw`, `researchclaw --help`, `pip check` passed | Installed editable package `researchclaw-0.3.1`. |
| 3 | `xjtulyc/MedgeClaw` | `fef51d3` | `reference-repos/repos/medgeclaw` | `reference-repos/envs/medgeclaw` | Configured | `compileall sync.py`, key-file and docs inspection passed | This repo is primarily scripts/docs/OpenClaw integration; no user-level config was installed. |
| 4 | `wanshuiyin/Auto-claude-code-research-in-sleep` | `ed638f3` | `reference-repos/repos/auto-claude-code-research-in-sleep` | `reference-repos/envs/auto-claude-code-research-in-sleep` | Configured | selected `compileall`, key-file check, `74` `SKILL.md` files, H1 check passed | Skill library/MCP assets only; no global `.claude` install was run. |
| 5 | `Orchestra-Research/AI-research-SKILLs` | `28f2d29` | `reference-repos/repos/ai-research-skills` | `reference-repos/envs/ai-research-skills` | Configured | key-file check, `98` `SKILL.md` files, package metadata inspection passed | Skill library/package assets only; no marketplace/global install was run. |
| 6 | `zjunlp/SkillNet` | `294a607` | `reference-repos/repos/skillnet` | `reference-repos/envs/skillnet` | Configured | `pip check`, `compileall`, `import skillnet_ai`, `skillnet --help` passed | Standard editable install now succeeds with `pip install -e reference-repos/repos/skillnet/skillnet-ai`; previous PyPI DNS failure no longer reproduced. |

## Verification Artifacts

- Global clone/env check: `reference-repos/smoke/global.txt`
- Per-repo smoke logs:
  - `reference-repos/smoke/01-ai-scientist.txt`
  - `reference-repos/smoke/02-autoresearchclaw.txt`
  - `reference-repos/smoke/03-medgeclaw.txt`
  - `reference-repos/smoke/04-auto-claude-code-research-in-sleep.txt`
  - `reference-repos/smoke/05-ai-research-skills.txt`
  - `reference-repos/smoke/06-skillnet.txt`
- Downstream result summaries: `reference-repos/results/01-*.md` through `reference-repos/results/06-*.md`
- Downstream thread ids: `reference-repos/results/*.thread_id`

## Reproducible Checks

```bash
find reference-repos/repos -maxdepth 2 -type d -name .git | sort
for slug in ai-scientist autoresearchclaw medgeclaw auto-claude-code-research-in-sleep ai-research-skills skillnet; do
  conda run --prefix "/Users/taw/project/AI4S/reference-repos/envs/$slug" python -V
done
```

## Known Residual Issues

- Downstream `codex exec` sessions intermittently could not resolve `github.com`, `repo.anaconda.com`, or `pypi.org`; upstream shell networking worked well enough to finish clone/install/verification.
- API-backed, GPU-backed, Docker-backed, or long-running research pipelines were not executed by design.
