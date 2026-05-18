# FULL_TEST_REPORT

生成时间：2026-05-18  
范围：`reference-repos/results/01-06` 六个离线 smoke 测试结果汇总（未重跑测试）

## 总体结论

| 仓库 | 结论 | 关键说明 |
|---|---|---|
| `SakanaAI/AI-Scientist` | 成功 | `datasets`/`fsspec` 版本兼容告警已修复，离线 smoke 通过 |
| `aiming-lab/AutoResearchClaw` | 成功 | 离线 smoke 通过 |
| `xjtulyc/MedgeClaw` | 成功 | 离线 smoke 通过 |
| `wanshuiyin/Auto-claude-code-research-in-sleep` | 成功 | 离线 smoke 通过 |
| `Orchestra-Research/AI-research-SKILLs` | 成功 | 离线 smoke 通过 |
| `zjunlp/SkillNet` | 成功 | 标准 editable 安装已通过，离线 smoke 通过 |

## 逐仓库汇总

### 1) ai-scientist
- 仓库名与 URL：`SakanaAI/AI-Scientist` / `https://github.com/SakanaAI/AI-Scientist.git`
- 本地 repo 路径：`reference-repos/repos/ai-scientist`
- conda env 路径：`reference-repos/envs/ai-scientist`
- Python 版本：`3.11.15`
- smoke 测试结论：**成功**
- 修复记录：
  - 已将 `datasets` 从 `2.19.1` 升级到 `4.8.5`，并保留 `fsspec 2026.2.0` 以满足 `aider-chat` 的精确依赖。
  - `python -m pip check` 已通过。
- 未解决问题：无

### 2) autoresearchclaw
- 仓库名与 URL：`aiming-lab/AutoResearchClaw` / `https://github.com/aiming-lab/AutoResearchClaw.git`
- 本地 repo 路径：`reference-repos/repos/autoresearchclaw`
- conda env 路径：`reference-repos/envs/autoresearchclaw`
- Python 版本：`3.11.15`
- smoke 测试结论：**成功**
- 未解决问题：无

### 3) medgeclaw
- 仓库名与 URL：`xjtulyc/MedgeClaw` / `https://github.com/xjtulyc/MedgeClaw.git`
- 本地 repo 路径：`reference-repos/repos/medgeclaw`
- conda env 路径：`reference-repos/envs/medgeclaw`
- Python 版本：`3.11.15`
- smoke 测试结论：**成功**
- 未解决问题：无

### 4) auto-claude-code-research-in-sleep
- 仓库名与 URL：`wanshuiyin/Auto-claude-code-research-in-sleep` / `https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep.git`
- 本地 repo 路径：`reference-repos/repos/auto-claude-code-research-in-sleep`
- conda env 路径：`reference-repos/envs/auto-claude-code-research-in-sleep`
- Python 版本：`3.11.15`
- smoke 测试结论：**成功**
- 未解决问题：无

### 5) ai-research-skills
- 仓库名与 URL：`Orchestra-Research/AI-research-SKILLs` / `https://github.com/Orchestra-Research/AI-research-SKILLs.git`
- 本地 repo 路径：`reference-repos/repos/ai-research-skills`
- conda env 路径：`reference-repos/envs/ai-research-skills`
- Python 版本：`3.11.15`
- smoke 测试结论：**成功**
- 未解决问题：无

### 6) skillnet
- 仓库名与 URL：`zjunlp/SkillNet` / `https://github.com/zjunlp/SkillNet.git`
- 本地 repo 路径：`reference-repos/repos/skillnet`
- conda env 路径：`reference-repos/envs/skillnet`
- Python 版本：`3.11.15`
- smoke 测试结论：**成功**
- 修复记录：
  - 标准 `pip install -e reference-repos/repos/skillnet/skillnet-ai` 已通过，不再依赖 `--no-deps` workaround。
  - `python -m pip check`、导入、CLI help、compileall 均已通过。
- 未解决问题：无

## 结果来源

- `reference-repos/results/01-ai-scientist.md`
- `reference-repos/results/02-autoresearchclaw.md`
- `reference-repos/results/03-medgeclaw.md`
- `reference-repos/results/04-auto-claude-code-research-in-sleep.md`
- `reference-repos/results/05-ai-research-skills.md`
- `reference-repos/results/06-skillnet.md`
