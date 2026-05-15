# 过程奖励模型与信用分配深度调研 (2024-2025)

> 研究维度：LLM强化学习中的过程奖励模型(PRM)、结果奖励模型(ORM)及各类信用分配方法
> 调研时间：2025年
> 覆盖论文：47+篇核心论文

---

## 目录

1. [概述与核心发现](#1-概述与核心发现)
2. [PRM基础系列](#2-prm基础系列)
3. [ORM vs PRM 对比分析](#3-orm-vs-prm-对比分析)
4. [自动化过程标注方法](#4-自动化过程标注方法)
5. [Token-level 信用分配](#5-token-level-信用分配)
6. [Step-level 信用分配](#6-step-level-信用分配)
7. [Segment-level 信用分配](#7-segment-level-信用分配)
8. [Agent-level 信用分配](#8-agent-level-信用分配)
9. [方法分类对比表](#9-方法分类对比表)
10. [Benchmark汇总](#10-benchmark汇总)
11. [实现细节汇总](#11-实现细节汇总)
12. [关键洞见与研究趋势](#12-关键洞见与研究趋势)

---

## 1. 概述与核心发现

### 1.1 研究背景

信用分配(Credit Assignment, CA)是LLM+RL的核心挑战，特别是在长推理链(可达30K tokens)上。现有方法在token-level、step-level、segment-level和turn-level四个粒度上进行信用分配 [^1^]。

### 1.2 核心发现

1. **信用分配是LLM RL的中心挑战** [^1^]，其重要性随从推理到agentic设置的迁移而增长。从单一生成轨迹(~1K-30K tokens)到多轮agent交互(~100K-1M tokens)的转变，使信用分配从优化便利变为训练必需。

2. **推理RL中信用分配已趋于成熟** [^1^]：Token-level (VinePPO)、segment-level (SPO, SCAR)和step-level (PURE, HICRA, SPRO)方法在确定性转移、单生成轨迹和可验证结果假设下提供了有效解决方案。

3. **Agentic RL中信用分配尚处初期** [^1^]：随机环境、部分可观察性、异构动作、超长horizon和非可验证中间状态等 qualitatively 更难的挑战呼唤新方法。

4. **LLM-as-Critic正成为独特范式** [^1^]：利用LLM对中间状态进行语义评估的能力(CAPO, SWEET-RL)开辟了传统RL中没有的方法论维度。

### 1.3 方法谱系

```
信用分配粒度谱系：
Token-level → Step-level → Segment-level → Turn-level
(VinePPO)    (PURE/SPRO)   (SPO/SCAR)     (ArCHer/SWEET-RL)
  ↓              ↓              ↓                  ↓
最细粒度      主流方法       中间粒度         最粗粒度
最精确        实用平衡       计算友好         适合多轮agent
计算最昂贵    广泛使用       精度折中         适合长horizon
```

---

## 2. PRM基础系列

### 2.1 Let's Verify Step by Step (OpenAI, 2023)

```
Claim: 首次大规模证明过程监督优于结果监督，创建PRM800K数据集
Source: Training Verifiers to Solve Math Word Problems / Let's Verify Step by Step
URL: https://arxiv.org/abs/2110.14168 / Lightman et al. 2023
Date: NeurIPS 2021 / 2023 (PRM800K)
Excerpt: "PRM significantly outperforms ORM by providing step-level supervision..."
Context: PRM的奠基性工作，通过人工标注80万步级标签训练过程奖励模型
Confidence: high
```

**实现细节**：
- **数据集**: PRM800K — 人工标注的800K步级标签，来自MATH数据集 [^2^]
- **模型**: GPT-4-based PRM
- **标注方法**: 人类专家标注每步的正确性
- **实验结果**: PRM在best-of-K选择中显著优于ORM (78.2% vs 72.4% vs 69.6% majority voting)
- **Benchmark**: MATH (Hendrycks et al., 2021)

### 2.2 Math-Shepherd (ACL 2024)

```
Claim: 自动构建PRM训练数据，无需人工标注，通过MC估计每步正确性
Source: Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations
URL: https://arxiv.org/abs/2312.08935
Date: ACL 2024
Excerpt: "Math-Shepherd introduced an automated verification pipeline where mathematical reasoning steps are validated using symbolic tools and consistency-checking heuristics" [^2^]
Context: 首个全自动PRM标注框架，降低PRM训练门槛
Confidence: high
```

**实现细节**：
- **基础模型**: Mistral-7B
- **标注方法**: MC估计 — 从每步采样多个后续轨迹，计算到达正确答案的比例作为该步分数 [^3^]
- **数据集**: 自动生成的大规模过程监督数据
- **Benchmark**: GSM8K, MATH
- **核心创新**: 用MC rollout替代人工标注，使PRM训练可扩展

### 2.3 OmegaPRM (2024)

```
Claim: 使用分治式MCTS高效定位推理链中的第一个错误，自动生成150万+过程标注
Source: Improve Mathematical Reasoning in Language Models by Automated Process Supervision
URL: https://arxiv.org/abs/2406.06592
Date: 2024
Excerpt: "OmegaPRM employs a binary search strategy to swiftly locate the first error in a chain-of-thought..." [^4^]
Context: Math-Shepherd的后续改进，通过MCTS大幅降低标注成本
Confidence: high
```

**实现细节**：
- **算法**: 分治式MCTS + 二分搜索定位首个错误步
- **数据规模**: 150万+自动过程监督标注 [^4^]
- **基础模型**: Gemini Pro (从51%提升到69.4% on MATH500)
- **效率**: 相比人工标注降低约75倍成本 [^5^]
- **Benchmark**: MATH500 (69.4%), GSM8K (93.6%)
- **标注方法**: 构建蒙特卡洛搜索树，用二分搜索定位首个错误

---

## 3. ORM vs PRM 对比分析

### 3.1 核心对比

| 维度 | ORM (结果奖励模型) | PRM (过程奖励模型) |
|------|-------------------|-------------------|
| **监督粒度** | 仅最终结果 | 每步/每token |
| **标注成本** | 低（只需最终结果标签） | 高（需步级标签） |
| **信用分配** | 稀疏，延迟反馈 | 密集，即时反馈 |
| **奖励黑客风险** | 较低 | 较高（可能优化表面正确性） |
| **推理可解释性** | 低 | 高（可定位错误步） |
| **测试时扩展** | 弱（仅选最终答案） | 强（可引导搜索/剪枝） |
| **适用场景** | 短推理、可验证任务 | 长推理、多步任务 |

### 3.2 关键研究发现

**PRM优势** [^6^]：
- PRM在test-time scaling中持续优于ORM [Lightman et al., 2023]
- PRM更有效地指导树搜索（如MCTS），因步级信号可剪枝错误分支 [^6^]
- PRM支持step-wise credit assignment和trajectory-wise reward shaping

**ORM优势** [^6^]：
- ORM更简单、更便宜，只需最终结果标签
- ORM在简单任务上表现足够好
- 最新研究表明纯ORM配合大规模RL也能激发强推理（DeepSeek-R1）

**混合方法趋势**：
- PAPO [2026]：解耦的ORM+PRM优势组合，避免奖励黑客 [^7^]
- Hierarchical Reward Models (HRM)：结合粗粒度结果和细粒度过程分数 [^8^]
- PURE-PRM+VR：仅10%可验证奖励+PRM即可达到最佳性能 [^9^]

### 3.3 何时使用PRM vs ORM

根据文献综述的建议 [^1^]：

| 场景 | 推荐方法 | 理由 |
|------|---------|------|
| 短CoT (GSM8K, MATH简单题) | GRPO (ORM baseline) | 足够简单，ORM够用 |
| 长CoT竞赛题 (AIME, IMO) | PRM / PURE / HICRA | 需要细粒度信用分配 |
| 多轮agent (WebShop, ALFWorld) | GiGPO / AgentPRM | 步级信用在无env交互时必要 |
| 超长horizon (SWE-bench) | ArCHer / CARL | 稀疏信用+层级结构 |
| 计算受限 | GRPO / SPRO / T-REG | 无需辅助模型 |

---

## 4. 自动化过程标注方法

### 4.1 自动化标注方法谱系

| 方法 | 年份 | 核心思想 | 标注成本 |
|------|------|---------|---------|
| Math-Shepherd | 2024 | MC估计：从每步采样rollout | 高（需大量采样） |
| OmegaPRM | 2024 | MCTS+二分搜索定位首个错误 | 中（MCTS优化） |
| AlphaMath | 2024 | 从结果监督推导伪过程监督 | 低 |
| PRIME | 2025 | 隐式PRM：从策略和参考模型推导 | 极低（无需标注） |
| FreePRM | 2025 | 弱监督：用结果正确性生成伪标签 | 低 |
| ThinkPRM | 2025 | 生成式PRM：用CoT验证每步 | 极低（1% PRM800K标签） |
| AURORA | 2025 | 集成提示+反向验证 | 低 |

### 4.2 关键方法详解

#### AlphaMath (Chen et al., 2024)

```
Claim: 从结果监督直接推导伪过程监督，完全消除步级标注需求
Source: AlphaMath
URL: https://arxiv.org (相关论文)
Date: 2024
Excerpt: "In AlphaMath, researchers propose deriving pseudo-process supervision directly from outcome supervision, thereby eliminating the need for stepwise labels altogether" [^2^]
Context: 最激进的自动化方法，但可能引入噪声
Confidence: medium
```

#### PRIME (Cui et al., 2025)

```
Claim: 通过隐式过程奖励实现在线PRM更新，无需任何步级标注
Source: Process Reinforcement through Implicit Rewards
URL: https://arxiv.org/abs/2502.01456
Date: 2025
Excerpt: "PRIME enables online PRM updates using only policy rollouts and outcome labels through implicit process rewards" [^10^]
Context: 用策略和参考模型的log-likelihood ratio推导token-level dense rewards
Confidence: high
```

**实现细节** [^10^]：
- **核心思想**: 隐式PRM将ORM训练为 $r_\phi(y) := \beta \log \frac{\pi_\phi(y)}{\pi_{ref}(y)}$，token-level奖励为 $r_\phi(y_t) := \beta \log \frac{\pi_\phi(y_t|y_{<t})}{\pi_{ref}(y_t|y_{<t})}$
- **优势**: 仅需结果标签即可在线更新PRM，解决reward hacking问题
- **训练**: 策略和隐式PRM同时更新，PRM用cross-entropy loss
- **实验结果**: 从Qwen2.5-Math-7B-Base出发，平均提升15.1%
- **模型**: Eurus-2-7B-PRIME
- **Benchmark**: 多个数学和代码推理benchmark

#### ThinkPRM (Khalifa et al., 2025)

```
Claim: 生成式CoT验证器，仅需1%的PRM800K标签即可超越判别式PRM
Source: Process Reward Models That Think
URL: https://arxiv.org/abs/2504.16828
Date: 2025
Excerpt: "ThinkPRM surpasses both LLM-as-a-Judge and discriminative verifiers using only 1% of the process labels in PRM800K" [^11^]
Context: 利用长CoT模型的推理能力进行步级验证
Confidence: high
```

**实现细节** [^11^]：
- **核心思想**: 训练PRM生成verification chain-of-thought (CoT)后再做判断
- **数据效率**: 仅需1% PRM800K过程标签
- **Benchmark**: ProcessBench, MATH-500, AIME'24, GPQA-Diamond, LiveCodeBench
- **结果**: 超越使用完整PRM800K训练的判别式PRM (OOD: +8% on GPQA, +4.5% on LCB)
- **优势**: 可scale verification compute，在相同token budget下比LLM-as-a-Judge高7.2%

#### FreePRM (Sun et al., 2025)

```
Claim: 无需任何真实过程标签训练PRM，使用弱监督伪标签
Source: FreePRM: Training Process Reward Models Without Ground Truth Process Labels
URL: https://arxiv.org/abs/2506.03570
Date: 2025
Excerpt: "FreePRM achieves an average F1 score of 53.0% on ProcessBench, outperforming fully supervised PRM trained on Math-Shepherd by +24.1%" [^12^]
Context: 弱监督PRM训练新范式
Confidence: high
```

---

## 5. Token-level 信用分配

### 5.1 VinePPO (ICML 2025)

```
Claim: 用无偏蒙特卡洛token-level价值估计替代PPO中的学习价值网络
Source: VinePPO: Unlocking RL Potential for LLM Reasoning through Refined Credit Assignment
URL: https://arxiv.org/abs/2410.01679
Date: ICML 2025
Excerpt: "VinePPO replaces the learned value network in PPO with unbiased Monte Carlo value estimates at the token level" [^1^]
Context: Token-level CA的理论最优美方法，但计算成本高
Confidence: high
```

**实现细节** [^1^] [^13^]：
- **核心算法**: 在每个token位置分叉K个独立continuation ("vine")，评估每个的outcome reward，估计 $V(s_t) \approx \frac{1}{K} \sum_{k=1}^K R(\tau_t^{(k)})$
- **优势函数**: $\hat{A}_t = R(\tau) - V(s_t)$ — 无偏估计，无critic function approximation error
- **计算复杂度**: $O(K \cdot L)$ 额外forward passes per training trajectory，L为序列长度
- **基准**: GSM8K, MATH
- **关键发现**: Credit assignment quality（而非policy optimization）是主要瓶颈
- **辅助模型**: 无需（No auxiliary model）

### 5.2 RED (Reward Redistribution, 2024)

```
Claim: 从off-the-shelf奖励模型的内部表示中提取token-level奖励
Source: RED: Unleashing Token-Level Rewards from Holistic Feedback via Reward Redistribution
URL: https://arxiv.org/abs/2411.08302
Date: 2024
Excerpt: "RED probes the RM's internal representations to estimate token-level reward contributions via linear regression" [^1^]
Context: 最实用的token-level方法之一，零额外RL训练成本
Confidence: high
```

**实现细节** [^14^]：
- **方法**: 在RM的hidden states上训练轻量级probe，预测每token对总奖励的边际贡献
- **成本**: 零额外RL训练 — 纯粹事后重分配
- **实现**: 结合token-wise和sequence-wise奖励的凸组合
- **发现**: 预训练奖励模型已编码丰富的信用分配信息，只是未被充分利用

### 5.3 T-REG (Token-Level Reward Regularization, 2024)

```
Claim: 无需任何外部模型生成token-level奖励信号
Source: T-REG: Preference Optimization with Token-Level Reward Regularization
URL: https://arxiv.org/abs/2412.02685
Date: 2024
Excerpt: "T-REG uses a contrastive self-prompting strategy: the model generates both correct and incorrect solutions, then compares token-level log-probability differences" [^1^]
Context: 最简洁的自监督token-level CA方法
Confidence: high
```

**实现细节** [^15^]：
- **方法**: 对比自提示策略 — 生成正确和错误解，比较token-level log-probability差异
- **成本**: 无需奖励模型、critic或额外rollout
- **Benchmark**: Alpaca Eval 2 (提升3.8%), Arena-Hard (提升4.4%)
- **关键**: 自生成token-level奖励作为正则化，引导更有效地分配sequence-level奖励

### 5.4 TEMPO (2025)

```
Claim: 利用响应前缀树的branch-gated TD修正实现精确token-level信用分配
Source: Exploiting Tree Structure for Credit Assignment in RL Training of LLMs
URL: https://arxiv.org/abs/2509.18314
Date: 2025
Excerpt: "TEMPO augments the group-relative outcome signal of GRPO with branch-gated temporal-difference corrections derived from the tree" [^16^]
Context: 无需学习价值网络或额外judge的critic-free token-level方法
Confidence: high
```

**实现细节** [^16^]：
- **核心创新**: Prefix-to-Tree (P2T) — 将一组响应转换为前缀树，通过聚合后代outcome计算非参数化前缀值 $V(s_t)$
- **算法**: Tree-Estimated Mean Prefix Value for Policy Optimization
- **Branch-gated TD**: 在非分支token处TD项为零（退化为GRPO），在分支token处提供精确token-level credit
- **计算**: 无需学习价值网络、PRM或judge
- **模型**: Qwen3-1.7B/4B
- **Benchmark**: MATH, MedQA (in-dist); GSM-HARD, AMC23, MedMCQA, MMLU-Medical (OOD)
- **结果**: 优于PPO和GRPO，收敛速度提升高达1.6倍

---

## 6. Step-level 信用分配

### 6.1 PURE (ICML 2025)

```
Claim: 通过min-form信用分配解决PRM导致的奖励黑客问题
Source: Stop Summation: Min-Form Credit Assignment Is All Process Reward Model Needs for Reasoning
URL: https://arxiv.org/abs/2504.15275
Date: ICML 2025
Excerpt: "PURE introduces a min-form credit assignment that formulates the value function as the minimum of future rewards" [^9^]
Context: PRM-based RL的关键突破，解决summation-form CA导致的训练崩溃
Confidence: high
```

**实现细节** [^9^]：
- **核心创新**: Min-form credit assignment — $V(s_t) = \min_{\tau \geq t} r_\tau$ 替代标准summation
- **效果**: 限制价值函数范围，更合理分配advantages，显著缓解reward hacking
- **模型**: Qwen2.5-7B, Qwen2.5-Math-7B, Qwen2.5-Math-15B
- **训练**: veRL框架，constant learning rate $10^{-6}$ (PURE-VR), $5 \times 10^{-7}$ (PURE-PRM)
- **超参数**: prompt batch size 64, group size 8, max generation length 8192, temperature 1.0, KL coefficient $10^{-3}$
- **结果**: PRM+min-form仅需30%步数即可达到与VR方法相当的推理性能
- **最佳配置**: PURE-PRM+VR (10%可验证奖励) — AMC23: 82.5%, 5个benchmark平均53.3%
- **发现**: 标准sum-form CA在训练初期就导致训练崩溃

### 6.2 SPRO (2025)

```
Claim: 无需PRM的critic-free过程感知RL，从策略本身推导过程奖励
Source: Self-Guided Process Reward Optimization with Redefined Step-wise Advantage for Process Reinforcement Learning
URL: https://arxiv.org/abs/2507.01551
Date: 2025
Excerpt: "SPRO demonstrates that process rewards can be derived intrinsically from the policy model itself" [^17^]
Context: 工业级可部署的PRM-free过程强化学习
Confidence: high
```

**实现细节** [^17^]：
- **核心创新**: 
  1. 理论上证明过程奖励可从策略模型本身推导
  2. Cumulative Process Reward (CPR) — 隐式聚合前缀序列中所有前步的过程奖励
  3. Masked Step Advantage (MSA) — 严格per-step比较
- **计算**: 无额外计算开销（与GRPO相同）
- **效率**: 3.4倍训练效率提升，17.5%测试准确率提升
- **关键优势**: 保持高policy entropy，减少平均响应长度约1/3，防止reward hacking
- **Benchmark**: 数学和代码benchmarks

### 6.3 CAPO (Credit Assignment Policy Optimization, 2025)

```
Claim: 使用LLM作为生成式PRM (GenPRM)进行自我批评
Source: CAPO
URL: (相关论文)
Date: 2025
Excerpt: "CAPO uses the LLM as a Generative PRM — given a reasoning trajectory, the same LLM generates natural-language critiques of each step" [^1^]
Context: LLM-as-Critic范式的代表方法
Confidence: high
```

**实现细节** [^1^]：
- **核心思想**: 利用LLM的自我批评能力 — 给定推理轨迹，LLM生成每步的自然语言批评
- **优势**: 完全自包含 — 无需单独奖励模型、critic网络或MC rollout
- **风险**: 自我评估偏差 — 模型可能系统性高估自己的步骤
- **缓解**: 校准技术
- **Benchmark**: MATH-500 (31.0% vs GRPO 27.2%, +3.8%), AIME'24 (9.7% vs GRPO 3.6%, +6.1%)
- **模型**: Qwen2.5-7B

### 6.4 ACPO (Attribution-based Credit for RLVR, 2025)

```
Claim: 结合信用分配与课程学习，使用归因方法分解结果奖励
Source: ACPO
URL: (相关论文)
Date: 2025
Excerpt: "ACPO computes factorized hierarchical rewards that decompose the outcome reward into step contributions using attribution methods" [^1^]
Context: 信用分配与数据选择的协同
Confidence: medium
```

**实现细节** [^1^]：
- **方法**: 使用梯度归因等方法将结果奖励分解为步级贡献
- **课程学习**: 信用集中在少数步骤的问题（清晰分叉点）优先早期训练
- **发现**: 信用分配不仅是奖励重分配，而是让整个训练流程更高效

### 6.5 HICRA (Hierarchy-Aware Credit Assignment, 2025)

```
Claim: 识别并关注高影响力规划token而非均匀分配学习信号
Source: HICRA
URL: (相关论文)
Date: 2025
Excerpt: "HICRA identifies a two-phase learning dynamic: models first acquire procedural skills and then develop strategic planning" [^1^]
Context: 层级感知信用分配的先驱
Confidence: high
```

**实现细节** [^1^]：
- **核心发现**: 两阶段学习动态 — 先获得程序性技能（常规计算），后发展策略规划（高层问题分解）
- **方法**: 关注高影响力规划token的信用分配
- **模型**: Qwen3-4B-Instruct
- **结果**: AIME'24 (73.1% vs GRPO 68.5%, +4.6%), AIME'25 (65.1% vs GRPO 60.0%, +5.1%)

---

## 7. Segment-level 信用分配

### 7.1 SPO (Segment Policy Optimization, 2025)

```
Claim: 利用segment-level优势估计实现更精确的信用分配，无需critic模型
Source: Segment Policy Optimization: Effective Segment-Level Credit Assignment in RL for Large Language Models
URL: https://arxiv.org/abs/2505.23564
Date: 2025
Excerpt: "SPO leverages segment-level advantage estimation at an intermediate granularity, achieving a better balance" [^18^]
Context: 介于token-level和trajectory-level的中间粒度方法
Confidence: high
```

**实现细节** [^18^]：
- **核心组件**:
  1. 灵活segment分区
  2. 精确segment优势估计（基于MC，无需critic）
  3. 使用segment优势的策略优化（含概率mask策略）
- **两个实例**:
  - **SPO-chain**: 短CoT — cutpoint-based分区 + chain-based优势估计，GSM8K上比PPO/GRPO提升6-12个百分点
  - **SPO-tree**: 长CoT — tree-based优势估计，大幅降低MC估计成本，MATH500上比GRPO提升7-11个百分点
- **模型**: DeepSeek-R1-Distill-Qwen-1.5B, RhoMath-1.1B
- **结果**: MATH-500 (4K ctx) 82.8% vs GRPO 75.2% (+7.6); GSM8K 56.7% vs GRPO 45.7% (+11.0)

### 7.2 SCAR (Shapley Credit Assignment Rewards, 2025)

```
Claim: 使用Shapley值从合作博弈论分配序列级奖励到各token
Source: SCAR: Shapley Credit Assignment for More Efficient RLHF
URL: https://arxiv.org/abs/2505.20417
Date: 2025
Excerpt: "SCAR distributes the total sequence-level reward among constituent tokens or text spans based on their principled marginal contributions" [^19^]
Context: 有博弈论基础的信用分配方法
Confidence: high
```

**实现细节** [^19^]：
- **理论基础**: 合作博弈论中的Shapley值
- **方法**: 基于各token/span的边际贡献分配总序列奖励
- **优势**: 无需辅助critic模型或细粒度人工标注
- **理论保证**: 保持原始最优策略
- **Benchmark**: 情感控制、文本摘要、指令调优
- **结果**: 相比标准RLHF和基于attention的dense reward baseline收敛更快、最终奖励更高

---

## 8. Agent-level 信用分配

### 8.1 ArCHer (ICML 2024)

```
Claim: 首个面向多轮LLM agent的分层信用分配架构
Source: ArCHer: Actor-Critic with Hierarchical Evaluation for Language Model Agents
URL: https://arxiv.org (相关论文)
Date: ICML 2024
Excerpt: "ArCHer is the pioneering work on hierarchical credit assignment for multi-turn LLM agents" [^1^]
Context: Agentic RL信用分配的奠基性工作
Confidence: high
```

**实现细节** [^1^]：
- **架构**: 显式两级架构
  - **高级off-policy critic**: 学习turn-level Q-function $Q^H(s_t, a_t)$
  - **低级on-policy actor**: 优化turn内token-level policy $\pi_\theta(y|s_t)$
- **更新**: 高级critic用off-policy TD更新（从replay buffer学习），低级actor用高级Q-values作为turn-level rewards
- **创新**: 正式认识到多轮LLM RL需要与单轮推理RL根本不同的信用分配
- **辅助模型**: 需要critic model
- **Benchmark**: Multi-turn dialogue tasks

### 8.2 SWEET-RL (Meta/FAIR, 2025)

```
Claim: 引入privileged (asymmetric) critic概念用于多轮agent训练
Source: SWEET-RL
URL: (相关论文)
Date: 2025
Excerpt: "SWEET-RL trains a critic that conditions on privileged information to provide high-quality turn-level reward signals" [^1^]
Context: 利用训练/推理不对称性提供高质量信用信号
Confidence: high
```

**实现细节** [^1^]：
- **核心思想**: 利用训练时拥有但推理时没有的信息（ground truth答案、完整未来轨迹、环境状态变量）
- **方法**: 训练一个条件于特权信息的critic提供高质量turn-level奖励
- **优化**: 使用DPO-style优化actor（仅看到标准观察）
- **优势**: 优雅绕过中间状态不可验证的挑战
- **辅助模型**: 需要privileged critic

### 8.3 GiGPO (NeurIPS 2025)

```
Claim: 将GRPO的组比较原则从episode级扩展到step级，critic-free
Source: Group-in-Group Policy Optimization for LLM Agent Training
URL: https://arxiv.org/abs/2505.10978
Date: NeurIPS 2025
Excerpt: "GiGPO introduces a two-level advantage estimation: at the outer level, trajectories are grouped and compared; at the inner level, steps within a single trajectory are compared via anchor state grouping" [^20^]
Context: 当前agentic RL中最先进的critic-free step-level方法
Confidence: high
```

**实现细节** [^20^]：
- **两级优势估计**:
  - **外层**: 轨迹分组比较（如标准GRPO）
  - **内层**: 单条轨迹内step比较 — anchor state grouping（共享相似前缀的步骤分组）
- **优势**: 无需学习价值函数，保持GRPO的critic-free特性
- **计算**: 与GRPO相同的GPU内存开销，无额外LLM rollout时间成本
- **模型**: Qwen2.5-1.5B/3B/7B-Instruct
- **Benchmark**: ALFWorld (>12% over GRPO), WebShop (>9% over GRPO), QA tasks (42.1% on 3B, 47.2% on 7B)

### 8.4 AgentPRM (2025)

```
Claim: 将PRM范式从推理适配到agentic设置，使用TD+GAE替代MC标注
Source: AgentPRM: Process Reward Models for LLM Agents via Step-Wise Promise and Progress
URL: https://arxiv.org/abs/2511.08325
Date: 2025
Excerpt: "AgentPRM reports 8x better sample efficiency compared to MC-based PRM training" [^21^]
Context: 证明TD范式在agentic设置中的实际必要性
Confidence: high
```

**实现细节** [^21^]：
- **核心洞察**: MC标注在agentic环境中过于昂贵（需重新执行环境交互）
- **方法**: 使用TD学习训练step-level critic: $V(s_t) \leftarrow V(s_t) + \alpha[r_t + \gamma V(s_{t+1}) - V(s_t)]$，配合GAE
- **双评分机制**: "promise"（成功可能性）+ "progress"（步间进展）
- **效率**: 比MC-based PRM训练高8倍样本效率
- **Benchmark**: Tool-use, code generation, web navigation tasks

---

## 9. 方法分类对比表

### 9.1 统一对比表

| 方法 | 粒度 | 方法论 | 设置 | 类型 | 辅助模型? | 计算成本 | 会议 | 年份 |
|------|------|--------|------|------|----------|---------|------|------|
| **VinePPO** | Token | MC | 推理 | Credit | 否 | 高 | ICML | 2025 |
| **RED** | Token | 重分配 | 推理 | Credit | RM | 低 | — | 2024 |
| **T-REG** | Token | 自生成 | 推理 | Credit | 否 | 低 | — | 2024 |
| **From r to Q*** | Token | 隐式 | 推理 | Credit | 否 | — | — | 2024 |
| **SPO** | Segment | MC | 推理 | Credit | 否 | 中 | — | 2025 |
| **SCAR** | Segment | 博弈论 | 推理 | Credit | 否 | 高 | — | 2025 |
| **TEMPO** | Token/Seg | Tree-TD | 推理 | Credit | 否 | 中 | — | 2025 |
| **PURE** | Step | Min-form PRM | 推理 | Credit | PRM | 中 | ICML | 2025 |
| **SPRO** | Step | Masked Adv. | 推理 | Credit | 否 | 中 | — | 2025 |
| **CAPO** | Step | LLM-as-Critic | 推理 | Credit | LLM | 中 | — | 2025 |
| **ACPO** | Step | 归因 | 推理 | Credit | 否 | 中 | — | 2025 |
| **HICRA** | Step | 层级 | 推理 | Credit | 否 | 中 | — | 2025 |
| **FinePO** | Sub-step | Fine PRM | 推理 | Credit | PRM | 高 | — | 2026 |
| **PRL** | Step | Entropy-RL | 推理 | Credit | 否 | 中 | — | 2026 |
| **InT** | Step | 干预 | 推理 | Credit | 否 | 中 | — | 2026 |
| **ArCHer** | Turn | TD (层级) | Agent | Credit | Critic | 中 | ICML | 2024 |
| **StepAgent** | Step | 隐式+IRL | Agent | Credit | 否 | 中 | — | 2024 |
| **GiGPO** | Step | MC (组) | Agent | Credit | 否 | 低 | NeurIPS | 2025 |
| **SWEET-RL** | Turn | Privileged Critic | Agent | Credit | Critic | 中 | — | 2025 |
| **AgentPRM** | Step | TD+GAE | Agent | Credit | Critic | 中 | — | 2025 |
| **Turn-PPO** | Turn | Turn-level MDP | Agent | Credit | Critic | 中 | EACL | 2025 |

*表格来源: Credit Assignment in Reinforcement Learning for Large Language Models [^1^]*

### 9.2 定量性能对比

| 方法 | 基础模型 | Benchmark | 得分 | Baseline | 提升 |
|------|---------|-----------|------|----------|------|
| SPO | DeepSeek-R1-Distill-Qwen-1.5B | MATH-500 (4K ctx) | 82.8% | GRPO 75.2% | +7.6 |
| SPO | RhoMath-1.1B | GSM8K | 56.7% | GRPO 45.7% | +11.0 |
| PURE | Qwen2.5-Math-7B | MATH-500 | 82.6% | — | — |
| PURE | Qwen2.5-Math-7B | AIME'24 | 20.0% | — | — |
| SPRO | Eurus-2-7B-SFT | MATH-500 | 53.6% | GRPO 51.8% | +1.8 |
| SPRO | Eurus-2-7B-SFT | AMC | 31.9% | GRPO 23.6% | +8.3 |
| CAPO | Qwen2.5-7B | MATH-500 | 31.0% | GRPO 27.2% | +3.8 |
| CAPO | Qwen2.5-7B | AIME'24 | 9.7% | GRPO 3.6% | +6.1 |
| HICRA | Qwen3-4B-Instruct | AIME'24 | 73.1% | GRPO 68.5% | +4.6 |
| HICRA | Qwen3-4B-Instruct | AIME'25 | 65.1% | GRPO 60.0% | +5.1 |
| GiGPO | Qwen2.5 | ALFWorld | — | GRPO | +12% |
| GiGPO | Qwen2.5 | WebShop | — | GRPO | +9% |

*表格来源: 各论文原始结果 [^1^]*

---

## 10. Benchmark汇总

### 10.1 PRM评估Benchmark

| Benchmark | 任务类型 | 标注方式 | 规模 | 特点 |
|-----------|---------|---------|------|------|
| **PRM800K** | 数学推理 | 人工标注 | 800K步 | 首个大规模人工标注PRM数据集 [^2^] |
| **ProcessBench** | 竞赛级数学 | 专家标注 | — | 定位最早错误步 [^6^] |
| **PRMBench** | 数学推理 | 人工标注 | 多步评估 | 细粒度错误类型评估 [^6^] |
| **AgentProcessBench** | Agent任务 | 人工标注 | ~2K | 工具使用agent的步级标注 [^22^] |
| **NVProcessBench** | 非可验证任务 | 人工标注 | ~2K | 针对非可验证多轮轨迹 [^23^] |
| **Socratic-PRMBench** | 教学对话 | — | — | 苏格拉底式教学场景 |

### 10.2 训练/评估用Benchmark

| Benchmark | 领域 | 难度 | 常用方法 |
|-----------|------|------|---------|
| GSM8K | 小学数学 | 易 | SPO, VinePPO, TEMPO |
| MATH / MATH-500 | 高中竞赛 | 中 | 几乎所有方法 |
| AIME'24/'25 | 数学邀请赛 | 难 | PURE, HICRA, CAPO |
| AMC23 | 数学竞赛 | 中-难 | SPRO, PURE |
| OlympiadBench | 奥林匹克 | 极难 | PAPO |
| ALFWorld | 室内导航agent | 中 | GiGPO, ArCHer |
| WebShop | 网上购物agent | 中-难 | GiGPO |
| SWE-bench | 软件工程 | 极难 | SWEET-RL, ArCHer |

---

## 11. 实现细节汇总

### 11.1 PURE详细配置 [^9^]

```python
# PURE 训练配置
models = ["Qwen2.5-7B", "Qwen2.5-Math-7B", "Qwen2.5-Math-15B"]
framework = "veRL"
learning_rate = {"PURE-VR": 1e-6, "PURE-PRM": 5e-7}
max_training_steps = {"Qwen2.5-Math": 500, "Qwen2.5-7B": 1000}
prompt_batch_size = 64
group_size = 8  # 每prompt生成8个响应
train_mini_batch_size = 512
max_generation_length = 8192
sampling_temperature = 1.0
kl_coefficient = 1e-3
transform_temperature = 0.1  # Eq(5)中的温度
checkpoint_save_interval = 50

# 数据集
rl_dataset = "SimpleRL RFT dataset"  # ~8000 problems from MATH (difficulty 3-5)
reward_types = ["PURE-PRM", "PURE-VR", "PURE-PRM+VR"]
```

### 11.2 SPRO核心特点 [^17^]

- **无需辅助PRM** — 从策略本身推导过程信号
- **CPR (Cumulative Process Reward)** — 隐式聚合前缀过程奖励
- **MSA (Masked Step Advantage)** — 严格per-step组内比较
- **训练效率**: 3.4x提升
- **响应长度**: 减少约1/3
- **Entropy**: 稳定维持高水平

### 11.3 TEMPO核心配置 [^16^]

```python
# TEMPO 算法
base_model = ["Qwen3-1.7B", "Qwen3-4B"]
baselines = ["PPO", "GRPO", "HEPO"]
reward_type = "binary"  # 最终答案正确性
episodes_per_question = 6
dataset_passes = 10  # 每问题60 episodes

def P2T(responses):
    """Prefix-to-Tree: 将响应组转换为前缀树"""
    # 每个节点: token prefix
    # 分支: 不同响应在某token处分歧
    # 价值: 所有后代完成正确性的平均
    pass

def TEMPO_advantage(token, tree):
    """Branch-gated TD correction"""
    if is_branch_token(token):
        return GRPO_advantage + TD_correction(token, tree)
    else:
        return GRPO_advantage  # TD term = 0
```

### 11.4 GiGPO核心配置 [^20^]

```python
# GiGPO 两级优势估计
def GiGPO_advantage(action_t, trajectory_i):
    A_E = episode_level_relative_advantage(trajectory_i)  # 全局
    A_S = step_level_relative_advantage(action_t)  # 局部
    return A_E + omega * A_S

# Anchor State Grouping:
# - 识别跨轨迹的重复环境状态
# - 来自相同状态的actions分组
# - 组内计算micro相对优势
```

### 11.5 VinePPO核心配置

```python
# VinePPO: MC Token-level Value Estimation
def VinePPO_value(state_t, K=16):
    """从state_t分叉K个continuation"""
    continuations = [sample_from_policy(state_t) for _ in range(K)]
    rewards = [outcome_reward(cont) for cont in continuations]
    return mean(rewards)

def VinePPO_advantage(trajectory, t):
    V_t = VinePPO_value(trajectory[:t])
    return outcome_reward(trajectory) - V_t

# 复杂度: O(K * L) 额外forward passes
# K: 每token分叉数, L: 序列长度
```

---

## 12. 关键洞见与研究趋势

### 12.1 五大关键洞见 [^1^]

1. **信用分配是LLM RL的中心挑战** [强实证] — 从推理到agentic设置的重要性递增

2. **推理RL中信用分配已成熟** [强实证] — Token-level (VinePPO)、segment-level (SPO, SCAR)、step-level (PURE, HICRA, SPRO)在确定性假设下有效

3. **Agentic RL中信用分配尚处初期** [有限但提示性] — 随机环境、部分可观察性、超长horizon等新挑战

4. **LLM-as-Critic是独特范式** [有限但提示性] — 使用LLM进行中间状态语义评估(CAPO, SWEET-RL)是LLM时代特有的方法论

5. **Implicit Credit Assignment (ICA)正兴起** — PRIME, DPO隐含Q值等从偏好/结果模型推导过程信号的方法减少标注依赖

### 12.2 当前最佳实践

| 场景 | 推荐方法 | 关键考量 |
|------|---------|---------|
| 数学推理 (GSM8K, MATH) | GRPO (baseline), PURE, SPO, SPRO | PRM监督容易获得 |
| 竞赛数学 (AIME, IMO) | VinePPO, HICRA, CAPO | 计算预算随CoT长度增长 |
| 工具使用agent | GiGPO, AgentPRM, Turn-PPO | Critic-free更实用 |
| 网页导航 | SWEET-RL, HCAPO | Privileged critic利用训练信息 |
| 软件工程 | ArCHer, CARL | 稀疏信用+ hindsight分析 |
| 计算受限 | GRPO, T-REG, SPRO | Critic-free，低开销 |

### 12.3 开放问题

1. **多智能体信用分配**: 跨agent的信用分解（M-GRPO, SHARP）
2. **超长horizon**: 50-100+轮的信用分配（SWE-bench类任务）
3. **探索-信用交互**: 如何在信用分配中有效激励探索
4. **非可验证中间状态**: 当没有ground truth时如何评估中间步骤
5. **PRM的泛化性**: PRM从数学到其他领域（常识推理、代码）的迁移
6. **计算-精度权衡**: 更细粒度信用分配是否值得额外计算成本

### 12.4 演进趋势

```
2023: PRM基础 (Let's Verify, PRM800K)
  ↓
2024 H1: 自动标注 (Math-Shepherd, OmegaPRM, AlphaMath)
  ↓
2024 H2: Token-level CA (VinePPO, RED, T-REG)
  ↓
2025 H1: Step-level CA (PURE, SPRO, CAPO, ACPO, HICRA)
  ↓
2025 H2: Segment-level + Agent-level (SPO, TEMPO, GiGPO, AgentPRM)
  ↓
2026: Implicit PRM + Multi-agent (PRIME evolution, SHARP, HCAPO)
```

---

## 参考文献索引

[^1^]: Credit Assignment in Reinforcement Learning for Large Language Models (2026). arXiv:2604.09459. — *核心综述论文，涵盖47种方法*

[^2^]: A Survey of Process Reward Models: From Outcome Signals to Process Supervisions for Large Language Models (2026). arXiv:2510.08049.

[^3^]: Wang et al. (2024). Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations. ACL 2024.

[^4^]: Luo et al. (2024). Improve Mathematical Reasoning in Language Models by Automated Process Supervision (OmegaPRM). arXiv:2406.06592.

[^5^]: OmegaPRM achieves 75x lower annotation cost compared to human annotation (source: [^1^] Section 3.3).

[^6^]: Song et al. (2025). PRMBench: a fine-grained and challenging benchmark for process-level reward models. arXiv:2501.03124.

[^7^]: PAPO: Process-Aware Policy Optimization (2026). arXiv:2603.26535. — *解耦ORM+PRM优势*

[^8^]: Hierarchical Reward Models (HRM), referenced in [^2^] Section 7.

[^9^]: Cheng et al. (2025). Stop Summation: Min-Form Credit Assignment Is All Process Reward Model Needs for Reasoning (PURE). arXiv:2504.15275.

[^10^]: Cui et al. (2025). Process Reinforcement through Implicit Rewards (PRIME). arXiv:2502.01456.

[^11^]: Khalifa et al. (2025). Process Reward Models That Think (ThinkPRM). arXiv:2504.16828.

[^12^]: Sun et al. (2025). FreePRM: Training Process Reward Models Without Ground Truth Process Labels. arXiv:2506.03570.

[^13^]: Kazemnejad et al. (2025). VinePPO: Unlocking RL Potential for LLM Reasoning through Refined Credit Assignment. ICML 2025. arXiv:2410.01679.

[^14^]: Li et al. (2024). RED: Unleashing Token-Level Rewards from Holistic Feedback via Reward Redistribution. arXiv:2411.08302.

[^15^]: Zhou et al. (2024). T-REG: Preference Optimization with Token-Level Reward Regularization. arXiv:2412.02685.

[^16^]: Tran et al. (2025). Exploiting Tree Structure for Credit Assignment in RL Training of LLMs (TEMPO). arXiv:2509.18314.

[^17^]: Kong et al. (2025). Self-Guided Process Reward Optimization (SPRO). arXiv:2507.01551.

[^18^]: Guo et al. (2025). Segment Policy Optimization (SPO). arXiv:2505.23564.

[^19^]: Cao et al. (2025). SCAR: Shapley Credit Assignment for More Efficient RLHF. arXiv:2505.20417.

[^20^]: Feng et al. (2025). Group-in-Group Policy Optimization (GiGPO). NeurIPS 2025. arXiv:2505.10978.

[^21^]: Xi et al. (2025). AgentPRM: Process Reward Models for LLM Agents via Step-Wise Promise and Progress. arXiv:2511.08325.

[^22^]: AgentProcessBench (2026). arXiv:2603.14465.

[^23^]: NVProcessBench, referenced in Hybrid Reward Normalization for Process-supervised Non-verifiable Agentic Tasks (2025). arXiv:2509.25598.

[^24^]: Lightman et al. (2023). Let's Verify Step by Step (PRM800K).

[^25^]: Chen et al. (2024). AlphaMath — pseudo-process supervision from outcome.

[^26^]: Xie et al. (2025). CAPO: Credit Assignment Policy Optimization.

[^27^]: Yin et al. (2025). ACPO: Attribution-based Credit for RLVR.

[^28^]: Wang et al. (2025c). HICRA: Hierarchy-Aware Credit Assignment.

[^29^]: Zhou et al. (2024). ArCHer: Actor-Critic with Hierarchical Evaluation. ICML 2024.

[^30^]: Zhou et al. (2025). SWEET-RL: Privileged Critic for Multi-Turn Agents (Meta/FAIR).

---

*报告生成时间: 2025年*
*覆盖论文: 47+篇*
*主要会议: ICML, NeurIPS, ICLR, ACL, EMNLP, EACL*
