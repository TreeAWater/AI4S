# Downstream deployment task: MedgeClaw

You are a downstream Codex CLI worker. Deploy exactly one repository in the current workspace.

Repository:
- Name: `xjtulyc/MedgeClaw`
- URL: `https://github.com/xjtulyc/MedgeClaw`
- Local path: `reference-repos/repos/medgeclaw`
- Conda prefix: `reference-repos/envs/medgeclaw`
- Result file: `reference-repos/results/03-medgeclaw.md`

Constraints:
- Work only inside `/Users/taw/project/AI4S/reference-repos` for cloned code, environments, logs, scratch files, and notes.
- Use conda for the environment. Create the environment with `conda create --prefix reference-repos/envs/medgeclaw ...`.
- Do not modify user-level configuration such as `~/.claude`, `~/.orchestra`, `~/.codex`, OpenClaw global installs, shell startup files, or system Python.
- Do not require real API keys, GPU-only workloads, long biomedical workflows, Docker services, or paid model calls.
- Treat this as a skill library if the repository has no runnable application entrypoint.
- If installation is blocked, capture the exact failing command and concise reason, then still produce a useful result summary.

Required work:
1. Clone the repository to the local path if it is not already present.
2. Inspect README, dependency files, and skill layout.
3. Create or reuse the conda prefix environment for this repository.
4. Install only dependencies needed for local inspection and no-API smoke tests.
5. Run a local smoke test. For a skill library, validate key files exist, count/list representative skills, and run any lightweight syntax/metadata checks available.
6. Write a final Markdown summary in the normal final response. Include status, local path, conda env path, install commands used, smoke test command/result, and unresolved issues.

