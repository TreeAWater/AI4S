# EvoLab × labbench2 接入方案（Max 模式）

> 制定日期：2026-05-24
> 状态：方案已基本确定，待执行
> 关联实验：DeepSeek V4 Pro thinking 在 seqqa2 子集上的跨任务自进化测试

---

## 1. 通用决策（所有阶段适用）

| # | 决策项 | 选择 | 备注 |
|---|---|---|---|
| 1 | Embedding 服务 | **OpenRouter** | 同一个 OPENROUTER_API_KEY 同时走主推理 + memory 提取 + embedding，一站式 |
| 2 | HITL（人工介入）策略 | **自动 yes** | mock human adapter 对所有人工 confirm 请求一律 approve |
| 3 | EvoLab 配置 | **Max 模式**（凡是 EvoLab 里有的都开） | 详见第 3 节 |
| 4 | 主推理模型 | DeepSeek V4 Flash | 官方 |
| 5 | Memory 提取模型 | DeepSeek V4 Flash | 官方 |

---

## 2. 测试阶段配置（仅测试阶段适用）

| # | 配置项 | 选择 | 备注 |
|---|---|---|---|
| 1 | Lab root 路径 | `/Users/taw/project/AI4S/lab/test/` | 测试用，正式跑时改路径 |
| 2 | EvolveWorker 触发频率 | **每题触发一次** | 测试阶段最激进；正式跑时再调 |
| 3 | 单次测试题量 | **5 题** | 跑完停下来 review，验证 pipeline + memory + skill graph 正常 |

---

## 3. Max 模式

凡是 EvoLab 里能开功能的全开

---

## 4. 下次开始时的 checklist

1. 确认 OpenRouter key 已 export 到 shell（`echo ${OPENROUTER_API_KEY:0:6}` 看前缀）
2. `git status` 确认 working tree 干净
3. 进入对应 conda env（labbench2 或新开 evolab）
