- **Status:** Success (existing checkout reused; no blocking issues)
- **Local path:** `reference-repos/repos/ai-research-skills`
- **Conda env path:** `reference-repos/envs/ai-research-skills`
- **Result file:** [05-ai-research-skills.md](/Users/taw/project/AI4S/reference-repos/results/05-ai-research-skills.md)

**Install/setup commands used**
```bash
conda --version
conda run -p reference-repos/envs/ai-research-skills python -V
```
No extra dependencies were installed because local inspection/no-API smoke tests only required existing shell tools plus Python in the reused conda env.

**Smoke test commands/results**
```bash
cd reference-repos/repos/ai-research-skills
test -f README.md && test -f LICENSE && test -f .claude-plugin/marketplace.json
rg --files -g 'SKILL.md' | wc -l
```
- Key files present.
- `SKILL.md` count: **98**.

```bash
cd reference-repos/repos/ai-research-skills
# frontmatter + required keys check for all SKILL.md
# required keys: name, description, version
```
- Checked: **98**
- Bad frontmatter starts: **0**
- Missing required keys: **0**
- Note: `10-optimization/ml-training-recipes/SKILL.md` uses CRLF line endings, but structure is valid.

**Unresolved issues**
- None.