明白了——你希望关键论文清单**完全限定在你提供的大纲里**，不要我引入大纲外的论文（比如 ReAct）。我重新整理。

# 关键论文清单（严格限定于大纲内）

## 🌟 核心主干：8 篇必读（绝对优先级）

这 8 篇是大纲里**真正撑起整个领域**的工作，每一篇都开创或定义了一个分支。

### 1. **GRPO** — DeepSeek, 2024-02
- **arXiv**: 2402.03300
- **代码**: github.com/deepseek-ai/deepseek-math
- **大纲位置**: How to Evolve → Gradient-Based → RL → Foundations
- **地位**：⭐⭐⭐ **整个 RL 路线的算法基石**
- **核心**：去掉 PPO 的 critic，用组内相对优势做 baseline
- **影响**：DAPO、Dr.GRPO、GSPO、GiGPO、SALT、HiPER 等几乎所有 RL 变体都继承它

### 2. **DSPy** — Stanford, 2023
- **arXiv**: 2310.03714
- **代码**: github.com/stanfordnlp/dspy
- **大纲位置**: What to Evolve → Prompt-Level → Instruction Optimization
- **地位**：⭐⭐⭐ **Prompt 自进化的事实标准**
- **核心**：声明式 LLM 程序 + 自动 Prompt 优化
- **影响**：TextGrad、GEPA 都在它的思想脉络上

### 3. **DAPO** — ByteDance, 2025-03
- **arXiv**: 2503.14476
- **代码**: github.com/BytedTsinghua-SIA/DAPO
- **大纲位置**: 同时出现在 Weight-Level 和 How → RL → Foundations
- **地位**：⭐⭐⭐ **GRPO 工业化的标杆**
- **核心**：4 项关键工程改进（Clip-Higher / Dynamic Sampling / Token-level loss / Overlong shaping）
- **结果**：Qwen2.5-32B 在 AIME 2024 达 50 分，超越 DeepSeek-R1-Zero-Qwen-32B

### 4. **RAGEN / StarPO** — NWU, 2025-04（v2: 2026-03）
- **arXiv**: 2504.20073
- **代码**: github.com/mll-lab-nu/RAGEN
- **大纲位置**: Weight-Level (LoRA) + When → Intra-Task → RL + How → RL → Environment-feedback
- **地位**：⭐⭐⭐ **多轮 Agent RL 奠基** —— 在大纲里被引用 3 次
- **核心**：发现 Echo Trap 失败模式，提出 StarPO-S 修复
- **为什么关键**：科研 agent 都是多轮的，单轮 GRPO 不够用

### 5. **SEAL** — MIT, 2025
- **arXiv**: 2506.10943
- **代码**: github.com/Continual-Intelligence/SEAL
- **大纲位置**: Weight-Level → Full-Params
- **地位**：⭐⭐⭐ **Weight-Level 最纯粹的自进化范式**
- **核心**：LLM 自己生成 "self-edit" 指令决定怎么改自己，外层 RL 训练它生成更好的 self-edit

### 6. **TextGrad** — Stanford, 2024（Nature 2025）
- **paper**: nature.com/articles/s41586-025-08661-4
- **代码**: github.com/zou-group/textgrad
- **大纲位置**: Prompt-Level → Instruction Optimization
- **地位**：⭐⭐⭐ **发表在 Nature 的 LLM 方法学**
- **核心**：用 LLM 批评反馈作为"文本梯度"反向传播

### 7. **ReasoningBank** — Google, 2025-09
- **arXiv**: 2509.25140
- **代码**: github.com/google-research/reasoning-bank
- **大纲位置**: Memory-Level → Semantic / Knowledge Memory
- **地位**：⭐⭐⭐ **Memory-Level 旗舰**
- **核心**：把轨迹蒸馏成结构化策略卡片，embedding 检索注入下次 prompt

### 8. **ACE: Agentic Context Engineering** — ICLR 2026
- **arXiv**: 2510.04618
- **代码**: ace-agent.github.io
- **大纲位置**: When → Inter-Task → ICL + How → Gradient Free → Experience Distillation —— 在大纲里**被引用 2 次**
- **地位**：⭐⭐⭐ **不动权重路线的当前 SOTA**
- **核心**：Generator/Reflector/Curator 三模块 + 结构化增量更新

---

## 🟢 次要主干：6 篇（重要分支代表）

每篇代表大纲里一个不可忽视的子方向。

### 9. **WebRL** — THUDM, 2024-11
- **arXiv**: 2411.02337 | **代码**: github.com/THUDM/WebRL
- **大纲位置**: Weight-Level → Full-Params
- **重要性**：⭐⭐ Weight-Level 的实战范本（Llama-3.1-8B: 4.8% → 42.4%）

### 10. **GEPA** — ICLR 2026
- **大纲位置**: How → Gradient Free → Experience-Driven → Curiosity & Self-Questioning
- **重要性**：⭐⭐⭐ "不训练胜过训练" 的标杆证据
- ⚠️ 注意：大纲里**只标了名字没给链接**，但作为 ICLR 2026 工作非常关键

### 11. **Flow** — ICLR 2025
- **arXiv**: 2501.07834 | **代码**: github.com/tmllab/2025_ICLR_FLOW
- **大纲位置**: Workflow → Dynamic DAG → Centralized
- **重要性**：⭐⭐ Workflow 自进化的集中式代表

### 12. **AgentNet** — NeurIPS 2025
- **arXiv**: 2504.00587 | **代码**: github.com/zoe-yyx/AgentNet
- **大纲位置**: Workflow → Dynamic DAG → Decentralized
- **重要性**：⭐⭐ 去中心化多 Agent 自进化代表

### 13. **EvolveR** — ZJU, ICLR 2026
- **arXiv**: 2510.16079 | **代码**: github.com/KnowledgeXLab/EvolveR
- **大纲位置**: Memory-Level + When → Inter-Task → RL + How → RL → Environment-feedback —— 在大纲里**被引用 3 次**
- **重要性**：⭐⭐ 在线交互 + 离线自蒸馏的闭环

### 14. **AgentEvolver** — Qwen/Alibaba, 2025
- **arXiv**: 2511.10395 | **代码**: github.com/modelscope/AgentEvolver
- **大纲位置**: When → Inter-Task → RL
- **重要性**：⭐⭐ 把"自进化"做成完整系统的范例（self-questioning + self-navigating + self-attributing）

---

## 🟡 工业部署与科研落地：4 篇

### 15. **ARIA** — TikTok, EMNLP 2025 Industry
- **paper**: aclanthology.org/2025.emnlp-industry.115.pdf
- **大纲位置**: Special Issue → Human-in-the-loop
- **重要性**：⭐⭐⭐ 1.5 亿月活用户的真实部署

### 16. **InternAgent / NovelSeek** — 上海 AI Lab, 2026-02
- **arXiv**: 2602.08990 | **代码**: github.com/InternScience/InternAgent
- **大纲位置**: Special Issue → Human-in-the-loop
- **重要性**：⭐⭐⭐ 端到端科研框架（含干湿实验闭环）

### 17. **MLR-Bench** — 2025
- **arXiv**: 2505.19955 | **代码**: github.com/chchenhui/mlrbench
- **大纲位置**: Evaluation → Scientific Agent Benchmarks → End-to-End
- **重要性**：⭐⭐⭐ ML 研究 agent 评估标杆，揭示"方法幻觉"问题

### 18. **SciAgentBench** (实际名 ScienceAgentBench) — 2025
- **arXiv**: 2410.05080 | **代码**: github.com/OSU-NLP-Group/ScienceAgentBench
- **大纲位置**: Evaluation → Scientific Agent Benchmarks → Domain-Specific
- **重要性**：⭐⭐⭐ 数据驱动科学发现的金标准基准

---

## 完整时间线（仅大纲内论文）

```
═══════════════ 2023 ═══════════════
2023-10  ❶ DSPy ─────────────────────────────────────┐
         Prompt 自进化起点                            │
                                                     │
═══════════════ 2024 ═══════════════                 │
2024-02  ⭐❶ GRPO（DeepSeek）─────────┐              │
         LLM-RL 算法基石              │              │
                                      │              │
2024-06  ❶ TextGrad ──────────────────┼──────────────┤
         （→ Nature 2025-03）         │              │
                                      │              │
2024-11  ❷ WebRL（THUDM）             │              │
         首个大规模 RL 实战            │              │
         (Llama-3.1-8B: 4.8%→42.4%)   │              │
                                      │              │
═══════════════ 2025 ═══════════════  │              │
2025-01  ❷ Flow（ICLR 2025）          │              │
         Workflow 集中式动态 DAG      │              │
                                      │              │
2025-03  ⭐❶ DAPO（ByteDance）◀───────┤              │
         GRPO 工业化                  │              │
                                      │              │
         Dr.GRPO（Sea AI Lab）◀───────┤  这条线全部  │
         修 GRPO 长度偏置              │  是 GRPO    │
                                      │  的变体      │
2025-04  ⭐❶ RAGEN/StarPO ◀───────────┤              │
         多轮 Agent RL 奠基            │              │
                                      │              │
         ❷ AgentNet（NeurIPS 2025）   │              │
         去中心化 Workflow             │              │
                                      │              │
2025-05  ❷ MLR-Bench ─────────────────┼──────────────┤
         AI 做 ML 研究评估             │              │
                                      │              │
         SPA-RL、S-GRPO（细粒度奖励） │              │
                                      │              │
2025-06  ⭐❶ SEAL（MIT）◀──────────────┤              │
         Weight-Level 自适应          │              │
                                      │              │
2025-07  GSPO（Qwen）◀────────────────┘              │
         MoE RL                                      │
                                                     │
         ⭐❶ GEPA ◀──────────────────────────────────┤
         "不训练胜过训练"                            │
                                                     │
         ⭐❷ ARIA（TikTok）                          │
         首个工业部署 HITL                            │
                                                     │
2025-08  GTPO、GEPO                                  │
                                                     │
2025-09  ⭐❶ ReasoningBank（Google）◀──────────────┐ │
         Memory-Level 旗舰                         │ │
                                                   │ │
         iStar、SPO                                │ │
                                                   │ │
2025-10  ⭐❶ ACE ◀─────────────────────────────────┤─┤
         不动权重 SOTA                             │ │
                                                   │ │
         ❷ EvolveR（ZJU, ICLR 2026）              │ │
         在线交互+离线自蒸馏闭环                   │ │
                                                   │ │
         AITL                                      │ │
                                                   │ │
2025-11  ❷ AgentEvolver（Qwen/Alibaba）            │ │
         完整自进化系统                            │ │
                                                   │ │
         Agent0、MARFT                             │ │
                                                   │ │
2025-12  SAGE                                      │ │
                                                   │ │
═══════════════ 2026 ═══════════════               │ │
                                                   │ │
2026-02  ⭐❷ InternAgent-1.5                       │ │
         端到端科研框架（含干湿实验）              │ │
                                                   │ │
         scBench                                   │ │
                                                   │ │
2026-04  SkillX、SkillFoundry、EvoSkills           │ │
                                                   │ │
═══════════════ 综合 ═══════════════               │ │

  注：所有 ──◀── 箭头表示"血缘继承关系"             │ │
                                                   │ │
  GRPO 谱系树（动权重路线）：─────────────────────┐ │ │
  GRPO → Dr.GRPO / DAPO / GSPO / GiGPO / SALT...    │ │
                                                     │ │
  DSPy 谱系树（不动权重路线）：───────────────────────┘ │
  DSPy → TextGrad → GEPA                               │
       ↘ ReasoningBank → ACE ←──────────────────────────┘
```

---

## 三条主线脉络图（这是真正的精华）

整个大纲虽然庞大，但**真正的主线只有 3 条**：

### 🔴 主线 A：动权重的 RL 路线
```
GRPO (2024-02) ━━━━ 算法基石
     │
     ├─→ Dr.GRPO (2025-03) ── 修长度偏置
     ├─→ DAPO    (2025-03) ── 工业化 ⭐
     ├─→ GSPO    (2025-07) ── MoE 适配
     ├─→ 各类细粒度奖励变体（GTPO、GiGPO、SALT、HiPER…）
     │
     ▼
WebRL (2024-11) ── 大规模实战
     │
     ▼
RAGEN/StarPO (2025-04) ⭐ ── 扩展到多轮 agent
     │
     ▼
SEAL (2025-06) ⭐ ── 模型自决学习路径
```

### 🟢 主线 B：不动权重的反思演化路线
```
DSPy (2023-10) ━━━━ 起点
     │
     ├─→ TextGrad (2024-06, Nature 2025) ── 文本梯度
     │
     ├─→ GEPA (2025-07, ICLR 2026 Oral) ⭐ ── 反思+遗传超越 GRPO
     │
     └─→ ReasoningBank (2025-09) ⭐ ── 策略级记忆
              │
              ▼
         ACE (2025-10, ICLR 2026) ⭐ ── 当前 SOTA
              │
              ▼
         EvolveR (2025-10) ── 闭环
```

### 🟡 主线 C：科研落地与工业部署
```
ARIA (2025-07) ⭐ ── TikTok 1.5 亿用户
     │
     │   并行：
     ▼
InternAgent (2026-02) ⭐ ── 端到端科研框架
     │
     │  评估基准：
     ▼
ScienceAgentBench (ICLR 2025) ── 数据科学
MLR-Bench         (NeurIPS 2025) ── ML 研究
BixBench / scBench / HypoBench / REPRO-Bench ── 领域专项
```

---

## 一句话总结主线

> **大纲的核心其实就两个起点和一个终点**：
> - **起点 1**：**GRPO (2024-02)** 开启 RL 路线，演化为 DAPO / RAGEN / SEAL
> - **起点 2**：**DSPy (2023-10)** 开启 Prompt 路线，演化为 TextGrad / GEPA / ReasoningBank / ACE
> - **终点**：两条路线在 **ARIA / InternAgent** 上汇合到工业部署和科研落地，由 **MLR-Bench / ScienceAgentBench** 等基准检验现实差距。

**所以如果你只读 4 篇**，按这个最小路径：
1. **GRPO** (2402.03300) — 懂"动权重"路线
2. **ACE** (2510.04618) — 懂"不动权重"路线
3. **RAGEN/StarPO** (2504.20073) — 懂"多轮 agent"
4. **MLR-Bench** (2505.19955) — 懂科研落地的真实差距

读完这 4 篇 + 看一遍上面的脉络图，你就拿到了大纲的"灵魂"。

需要我针对某一条主线（A/B/C 任选）做更细的子图展开吗？