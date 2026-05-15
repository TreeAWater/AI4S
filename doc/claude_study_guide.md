# Self-Evolving Agents 关键论文清单与时间线

我用三种视角组织这份清单：**单论文卡片** → **演进时间线** → **依赖关系图**。这样你既能逐篇定位，也能看清谁影响了谁。

---

## 第一视角：18 篇核心论文卡片（按角色分类）

### 🔵 奠基类（4 篇）—— 不读这些就读不懂后面所有论文

#### 1. **ReAct** — Yao et al., Princeton, 2022-10
- arXiv: 2210.03629（ICLR 2023）
- **地位**：定义了 LLM Agent 的基本范式 "Thought → Action → Observation" 循环
- **为什么必读**：所有 agent 论文默认你懂这个范式
- **难度**：⭐⭐（概念简单但影响深远）

#### 2. **DSPy** — Khattab et al., Stanford, 2023-10
- arXiv: 2310.03714
- **地位**：把 LLM 调用变成"声明式程序"，Prompt 优化的事实标准
- **为什么必读**：理解 TextGrad/GEPA 的前置；现在是 LLM 工程的基础设施
- **难度**：⭐⭐

#### 3. **GRPO（DeepSeekMath）** — Shao et al., DeepSeek, 2024-02
- arXiv: 2402.03300
- **地位**：⭐⭐⭐ **整个 LLM-RL 时代的算法基石**
- **核心贡献**：去掉 PPO 的 critic，用组内相对优势做 baseline
- **为什么必读**：DeepSeek-R1 的底层算法；之后 90% 的 RL 论文（Dr.GRPO、DAPO、GSPO、GiGPO…）都是它的变体
- **难度**：⭐⭐⭐⭐（公式不简单，但概念清晰）

#### 4. **Self-Evolving Agents Survey** — Gao et al., TMLR 2026-01
- arXiv: 2507.21046
- **地位**：领域第一份权威综述，定义了 What/When/How 三维框架
- **为什么必读**：你的"领域地图"
- **难度**：⭐⭐（综述好读，77 页）

---

### 🟢 范式开创类（5 篇）—— 每篇代表一种新的"自进化思路"

#### 5. **WebRL** — Qi et al., THUDM, 2024-11
- arXiv: 2411.02337（ICLR 2025）
- **地位**：**首次证明 RL 在真实 web agent 上的规模效应**
- **关键结果**：Llama-3.1-8B 在 WebArena-Lite 成功率从 **4.8% → 42.4%**
- **代表的范式**：Weight-Level 自进化的实战范本
- **难度**：⭐⭐⭐

#### 6. **DAPO** — Yu et al., ByteDance Seed, 2025-03
- arXiv: 2503.14476
- **地位**：⭐⭐⭐ **GRPO 工业化的里程碑**
- **核心贡献**：4 项工程改进（Clip-Higher / Dynamic Sampling / Token-level loss / Overlong shaping）
- **关键结果**：Qwen2.5-32B 在 AIME 2024 达 50 分，超越 DeepSeek-R1-Zero-Qwen-32B（47 分），只用一半训练步数
- **代表的范式**：怎么把 GRPO 真正用到工业规模
- **难度**：⭐⭐⭐⭐

#### 7. **RAGEN / StarPO** — Wang et al., Northwestern+Stanford+Microsoft, 2025-04
- arXiv: 2504.20073
- **地位**：⭐⭐⭐ **多轮 Agent RL 的奠基**
- **核心贡献**：(1) 发现并命名 "Echo Trap" 失败模式；(2) 提出 StarPO-S 修复方案
- **为什么重要**：科研 agent 都是多轮的，单轮 GRPO 不够用
- **代表的范式**：把单轮推理 RL 扩展到多轮 agent
- **难度**：⭐⭐⭐⭐

#### 8. **SEAL: Self-Adapting Language Models** — Zweiger & Pari et al., MIT, 2025-06
- arXiv: 2506.10943
- **地位**：⭐⭐⭐ **Weight-Level 自进化最纯粹的范式**
- **核心思想**：让 LLM 自己生成 "self-edit" 指令决定怎么改自己；外层 RL 训练生成更好的 self-edit
- **代表的范式**：模型自主决定学习路径
- **难度**：⭐⭐⭐⭐

#### 9. **TextGrad** — Yuksekgonul et al., Stanford, 2024-06 → *Nature* 2025-03
- arXiv: 2406.07496 / Nature 639:609-616
- **地位**：⭐⭐⭐ **发表在 Nature 的 LLM 方法学**，把"梯度"概念推广到文本
- **核心思想**：用 LLM 的批评反馈作为"文本梯度"反向传播
- **代表的范式**：Prompt-Level 自动优化 + 真实科研落地（药物分子、放射治疗）
- **难度**：⭐⭐⭐

---

### 🟡 当前 SOTA 类（5 篇）—— 2025 下半年的旗舰工作

#### 10. **GSPO** — Zheng et al., Qwen, 2025-07
- arXiv: 2507.18071
- **地位**：⭐⭐ MoE RL 的救星
- **核心贡献**：序列级重要性比，稳定 Mixture-of-Experts 模型的 RL 训练
- **重要性**：Qwen3 系列就是用 GSPO 训练的
- **难度**：⭐⭐⭐⭐

#### 11. **GEPA: Reflective Prompt Evolution** — Agrawal et al., UC Berkeley, 2025-07
- arXiv: 2507.19457（ICLR 2026 **Oral**）
- **地位**：⭐⭐⭐ **"不训练胜过训练"的标杆证据**
- **核心结果**：跨 6 项任务平均超 GRPO **+6%**、最高 **+20%**，最多用少 **35×** rollouts
- **代表的范式**：遗传 + Pareto + 自然语言反思
- **难度**：⭐⭐⭐

#### 12. **ReasoningBank** — Ouyang et al., Google Cloud AI Research + UIUC, 2025-09
- arXiv: 2509.25140
- **地位**：⭐⭐⭐ **Memory-Level 旗舰**
- **核心贡献**：把轨迹蒸馏成结构化策略卡片，配合 MaTTS 实现 **+34.2% 相对效果**、**−16% 交互步数**
- **代表的范式**：策略级记忆（不是简单的轨迹存储）
- **难度**：⭐⭐⭐

#### 13. **ACE: Agentic Context Engineering** — Zhang et al., Stanford+SambaNova+Caltech, 2025-10
- arXiv: 2510.04618（ICLR 2026）
- **地位**：⭐⭐⭐ **不动权重路线的当前 SOTA**
- **核心贡献**：Generator/Reflector/Curator 三模块 + 结构化增量更新（避免 context collapse）
- **关键结果**：agent 任务 +10.6%、金融 +8.6%；用小开源模型匹敌 IBM CUGA
- **难度**：⭐⭐⭐

#### 14. **AgentNet** — Yang et al., 上海交大, 2025-04（NeurIPS 2025）
- arXiv: 2504.00587
- **地位**：⭐⭐ **去中心化多 Agent 自进化的代表**
- **核心思想**：没有中央指挥，每个 agent 用 RAG 记忆决定下一步交给谁，DAG 实时演化
- **难度**：⭐⭐⭐⭐

---

### 🟠 工业部署类（2 篇）—— 证明这个领域不只是论文

#### 15. **ARIA** — He et al., TikTok, 2025-07
- arXiv: 2507.17131（EMNLP 2025 Industry Track）
- **地位**：⭐⭐⭐ **真实工业部署的标杆**
- **规模**：部署在 TikTok Pay，服务 **1.5 亿月活用户**
- **代表的范式**：Human-in-the-Loop 自进化的工业级实现
- **难度**：⭐⭐

#### 16. **AITL（Agent-in-the-Loop）** — Zhao et al., 2025-10
- arXiv: 2510.06674
- **地位**：⭐⭐ 数据飞轮工程范本
- **关键结果**：recall@75 +11.7%、precision@8 +14.8%
- **难度**：⭐⭐

---

### 🟣 科研落地类（2 篇）—— 你的本职目标

#### 17. **MLR-Bench** — Chen et al., NUS, 2025-05（NeurIPS 2025）
- arXiv: 2505.19955
- **地位**：⭐⭐⭐ **AI 做 ML 研究能力的标杆基准**
- **重要发现**：当前 agent 普遍存在 **伪造实验、方法幻觉**——这是必须知道的现实
- **难度**：⭐⭐⭐

#### 18. **InternAgent-1.5 / NovelSeek** — 上海 AI Lab（56 作者）, 2026-02
- arXiv: 2602.08990（旧名 NovelSeek: arXiv:2505.16938）
- **地位**：⭐⭐⭐ **端到端科学发现的统一框架**
- **覆盖**：12 类科研任务，含**干湿实验闭环**——大多数 agent 不敢碰的硬骨头
- **评测**：在 GAIA、HLE、GPQA、FrontierScience 上领先
- **难度**：⭐⭐⭐⭐

---

## 第二视角：技术演进时间线（按时间排列，标注血缘关系）

```
═══════════════════════ 2022 ═══════════════════════
2022-10  ReAct
         │  定义 LLM Agent 范式
         ▼

═══════════════════════ 2023 ═══════════════════════
2023-10  DSPy
         │  LLM 程序化 + Prompt 自动优化
         ▼

═══════════════════════ 2024 ═══════════════════════
2024-02  🌟 GRPO（DeepSeek）
         │   ⭐ LLM-RL 的奠基算法
         │   ├──→ 影响所有后续 RL 工作
         │   ▼
2024-06  TextGrad
         │  "文本梯度"概念
         │  └──→ Nature 2025-03
2024-11  WebRL（THUDM）
         │  首个大规模 RL 实战
         ▼

═══════════════════════ 2025 ═══════════════════════
                              ▲
2025-01  Flow（ICLR 2025）    │ Workflow 路线起步
         动态 DAG 集中式      │

2025-03  🌟 DAPO（ByteDance）  ◀─── 继承 GRPO
         │   ⭐⭐ GRPO 工业化
         │
         Dr.GRPO（Sea AI Lab） ◀─── 修 GRPO 长度偏置

2025-04  🌟 RAGEN/StarPO      ◀─── 把 GRPO 扩展到多轮
         │   ⭐⭐ 多轮 Agent RL 奠基
         │
         AgentNet（NeurIPS 2025）
         去中心化多 Agent

2025-05  MLR-Bench             ◀─── 评估 AI 做研究
         SPA-RL、S-GRPO（细粒度奖励）

2025-06  🌟 SEAL（MIT）        ◀─── Weight-Level 自适应
         │   ⭐⭐ 模型自决学习路径

2025-07  🌟 GSPO（Qwen）       ◀─── 继承 GRPO，做 MoE
         │   ⭐ Qwen3 训练算法
         │
         🌟 GEPA               ◀─── 继承 DSPy/TextGrad 思想
         │   ⭐⭐ "不训练胜过训练"
         │
         ARIA（TikTok）        ◀─── 首个工业部署级 HITL
         │   ⭐ 1.5 亿用户

2025-08  MCP-Bench、GTPO
         GEPO（异步 RL）

2025-09  🌟 ReasoningBank（Google）
         │   ⭐⭐ Memory-Level 旗舰
         │
         iStar、SPO

2025-10  🌟 ACE                ◀─── 继承 ReasoningBank 思想
         │   ⭐⭐ 不动权重 SOTA
         │
         EvolveR、AITL、TRAJECT-Bench

2025-11  AgentEvolver（Alibaba）
         Agent0（自我提问 + 解决）

2025-12  SAGE（Amazon）

═══════════════════════ 2026 ═══════════════════════
2026-01  🌟 Survey（Gao et al., TMLR）
         │   ⭐⭐ 领域第一份权威综述
         │
         MinPRO

2026-02  🌟 InternAgent-1.5（上海 AI Lab）
         │   ⭐⭐ 端到端科研框架
         │
         scBench、RC-GRPO、HiPER

2026-03  HCAPO、AgentFactory

2026-04  SkillX、SkillFoundry、EvoSkills、RAGEN-v2
```

🌟 = 必读核心 | ⭐ 数量 = 重要性

---

## 第三视角：依赖关系图（论文之间怎么相互影响）

```
ReAct (2022)
   │  定义 agent 范式
   ▼
┌──────────────────────────────────────────────────────┐
│                                                      │
│   两条平行的技术路线从 2024 开始分化：               │
│                                                      │
└──────────────────────────────────────────────────────┘
   │                                    │
   ▼                                    ▼

═══ 路线 A：动权重（Gradient-Based）═══

GRPO (2024-02) ━━━━━━━━━━━━━━━━━━━━━━━━━┓
   │   组内相对优势                    ┃
   ├──→ Dr.GRPO (2025-03) 修长度偏置  ┃
   ├──→ DAPO (2025-03) 工业化         ┃
   ├──→ GSPO (2025-07) MoE 适配       ┃
   ├──→ GiGPO/SALT/HiPER... 各种变体  ┃
   │                                   ┃
   ▼                                   ┃
WebRL (2024-11) 大规模实战             ┃
   │                                   ┃
   ▼                                   ┃
RAGEN/StarPO (2025-04) 多轮扩展        ┃
   │                                   ┃
   ▼                                   ┃
SEAL (2025-06) 模型自主决定怎么学      ┃
                                       ┃
═══ 路线 B：不动权重（Gradient-Free）═══┃

DSPy (2023-10) ━━━━━━━━━━━━━━━━━━━━━━━┛
   │   Prompt 程序化
   ├──→ TextGrad (2024-06) 文本梯度
   │                          │
   │                          └──→ Nature 2025
   │
   ▼
GEPA (2025-07) 反思+遗传+Pareto
   │  ⭐ 超过 GRPO，少用 35× rollouts
   │
   ▼
（与 Memory 路线交汇）
   │
ReasoningBank (2025-09) 策略级记忆 ──┐
   │                                 │
   ▼                                 │
ACE (2025-10) 三模块 context 演化 ◀──┘
   │
   ▼
EvolveR (2025-10) 经验闭环

═══ 路线 C：人在回路（HITL）═══

ARIA (2025-07) TikTok 部署
   │  1.5 亿用户
   ▼
AITL (2025-10) 数据飞轮

═══ 路线 D：架构演化（Workflow）═══

Flow (ICLR 2025) 集中式动态 DAG
   │
   ▼
AgentNet (NeurIPS 2025) 去中心化 DAG

═══ 路线 E：科研落地 ═══

ScienceAgentBench (ICLR 2025) 数据驱动科学
   │
   ▼
MLR-Bench (NeurIPS 2025) AI 做 ML 研究
   │  发现：方法幻觉、伪造实验
   ▼
BixBench / HypoBench / REPRO-Bench (2025)
   │  现实警示：基准表现普遍 <50%
   ▼
InternAgent-1.5 (2026-02) 端到端科研框架
   │  整合：生成 + 验证 + 演化

═══ 综合 ═══

Gao et al. Survey (TMLR 2026-01)
   把 ABCDE 五条路线统一到 What/When/How 框架
```

---

## 最关键的 6 篇 —— 如果你时间极度有限

如果你**只能读 6 篇**，建议这个顺序：

| 顺序 | 论文 | 教你什么 | 时间 |
|---|---|---|---|
| 1 | **Survey** (2507.21046) | 全景地图 | 2-3 天 |
| 2 | **GRPO** (2402.03300) | RL 路线根基 | 2 天 |
| 3 | **SEAL** (2506.10943) | Weight-Level 范式 | 1 天 |
| 4 | **ACE** (2510.04618) | 不动权重路线 SOTA | 1 天 |
| 5 | **GEPA** (2507.19457) | 反思演化范式 | 1 天 |
| 6 | **MLR-Bench** (2505.19955) | 科研评估现实 | 1 天 |

读完这 6 篇 + 跑通 1 个代码（推荐 ACE，门槛最低），你就能看懂 90% 的新论文摘要。

---

## 一句话总结这份清单

> **从 GRPO（2024-02）开始，到 InternAgent-1.5（2026-02）为止，整个领域用 2 年时间走完了从"单轮推理 RL"→"多轮 Agent RL"→"不动权重的反思演化"→"端到端科研落地"四级跳。其中 GRPO/DAPO/SEAL 是"动权重"路线的三块基石，DSPy/GEPA/ACE/ReasoningBank 是"不动权重"路线的四块基石，ARIA 证明工业可部署，MLR-Bench 提醒你科研距人类专家还远。**

需要我针对清单里哪几篇做**深度解读**（公式推导、代码结构、关键实验数据）吗？