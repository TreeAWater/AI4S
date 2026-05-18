# Downstream deployment task: AutoResearchClaw

You are a downstream Codex CLI worker. Deploy exactly one repository in the current workspace.

Repository:
- Name: `aiming-lab/AutoResearchClaw`
- URL: `https://github.com/aiming-lab/AutoResearchClaw.git`
- Local path: `reference-repos/repos/autoresearchclaw`
- Conda prefix: `reference-repos/envs/autoresearchclaw`
- Result file: `reference-repos/results/02-autoresearchclaw.md`

Constraints:
- Work only inside `/Users/taw/project/AI4S/reference-repos` for cloned code, environments, logs, scratch files, and notes.
- Use conda for the environment. Create the environment with `conda create --prefix reference-repos/envs/autoresearchclaw ...`.
- Do not modify user-level configuration such as `~/.claude`, `~/.orchestra`, `~/.codex`, OpenClaw global installs, shell startup files, or system Python.
- Do not require real API keys, GPU-only workloads, long research runs, Docker services, or paid model calls.
- Prefer official README and repository dependency files.
- If installation is blocked, capture the exact failing command and concise reason, then still produce a useful result summary.

Required work:
1. Clone the repository to the local path if it is not already present.
2. Inspect README and dependency files to choose a conservative Python version and install method.
3. Create or reuse the conda prefix environment for this repository.
4. Install dependencies needed for local inspection and no-API smoke tests.
5. Run a local smoke test that does not call external LLM APIs. Good candidates include import checks, CLI help, package metadata checks, or `python -m compileall` on project modules.
6. Write a final Markdown summary in the normal final response. Include status, local path, conda env path, install commands used, smoke test command/result, and unresolved issues.

