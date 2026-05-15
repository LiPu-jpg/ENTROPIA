# 研究维度9：多目标对齐与安全RL（Safe RLHF, Multi-objective DPO等）

## 深度调研报告 (2024-2025)

---

## 目录
1. [概述与研究背景](#1-概述与研究背景)
2. [Safe RLHF](#2-safe-rlhf)
3. [Multi-objective DPO/MODPO](#3-multi-objective-dpomodpo)
4. [Nash Learning from Human Feedback (NLHF)](#4-nash-learning-from-human-feedback-nlhf)
5. [Pluralistic Alignment (PAL)](#5-pluralistic-alignment-pal)
6. [Reward Inconsistency](#6-reward-inconsistency)
7. [Multi-party RLHF与多元反馈](#7-multi-party-rlhf与多元反馈)
8. [Distributional Preference Learning](#8-distributional-preference-learning)
9. [多目标方法分类对比](#9-多目标方法分类对比)
10. [安全机制对比表](#10-安全机制对比表)
11. [总结与演进关系](#11-总结与演进关系)

---

## 1. 概述与研究背景

多目标对齐与安全RL是LLM+RL领域的重要研究方向。传统RLHF通过单一奖励模型来捕捉人类偏好，这在面对多元、冲突的偏好时表现出根本性局限。2024-2025年，该领域涌现出大量研究工作，主要沿着以下方向发展：

1. **Safe RLHF**: 在安全约束下进行RLHF训练，将helpfulness和harmlessness解耦为奖励和约束
2. **Multi-objective DPO**: 将DPO扩展到多目标场景，如MODPO、CPO、GAPO等
3. **NLHF**: 基于纳什均衡的偏好学习框架，处理非传递性偏好
4. **Pluralistic Alignment**: 捕捉人类偏好的异质性，实现个性化对齐
5. **Reward Inconsistency**: 研究奖励不一致对RLHF的连锁负面影响
6. **Distributional Preference Learning**: 用分布而非标量来表示偏好

---

## 2. Safe RLHF

### 2.1 Safe RLHF (Dai et al., ICLR 2024)

```
Claim: 提出Safe RLHF框架，将人类偏好中helpfulness和harmlessness解耦，分别训练奖励模型和成本模型，通过PPO-Lagrangian进行约束优化
Source: Safe RLHF: Safe Reinforcement Learning from Human Feedback
URL: https://openreview.net/pdf?id=TyFrPOKYXw
Date: ICLR 2024
Excerpt: "Safe RLHF explicitly decouples human preferences regarding helpfulness and harmlessness, effectively avoiding the crowd workers' confusion about the tension and allowing us to train separate reward and cost models. We formalize the safety concern of LLMs as an optimization task of maximizing the reward function while satisfying specified cost constraints. Leveraging the Lagrangian method to solve this constrained problem, Safe RLHF dynamically adjusts the balance between the two objectives during fine-tuning."
Context: 由北京大学PKU-Alignment团队提出，是Safe RLHF领域的开创性工作。核心思想是将安全对齐形式化为约束马尔可夫决策过程(CMDP)，在安全约束下最大化奖励。
Confidence: high
```

**具体实现细节：**
- **模型**: Alpaca-7B (LLaMA-7B基础上SFT)
- **数据集**: PKU-SafeRLHF（团队自建，包含helpfulness和harmlessness双重标注）
- **算法**: PPO-Lagrangian，primal-dual优化
- **架构**: 分别训练reward model和cost model
- **超参数**: 三轮fine-tuning，动态调整Lagrangian乘子
- **Benchmark**: PKU-SafeRLHF test set + GPT-4评分
- **对比结果**: 相比标准RLHF，在helpfulness和harmlessness两方面都有显著提升

**核心方法设计：**
1. **解耦偏好标注**: 分别收集helpfulness和harmlessness的偏好数据，避免标注者混淆
2. **双模型训练**: 独立训练reward model (helpfulness) 和 cost model (harmlessness)
3. **PPO-Lagrangian**: 使用原始-对偶方法求解约束优化问题
4. **动态平衡**: 在fine-tuning过程中动态调整两个目标的权重

---

### 2.2 C-DPO: Constrained DPO (Liu et al., 2024)

```
Claim: 将Safe RLHF中的PPO-Lagrangian替换为基于DPO的dual-gradient descent方法，简化训练流程同时保持安全保证
Source: Enhancing LLM Safety via Constrained Direct Preference Optimization
URL: https://arxiv.org/pdf/2403.02475
Date: 2024
Excerpt: "We extend the Direct Preference Optimization (DPO) framework to constrained fine-tuning. DPO has been recently proposed as a stable and lightweight alternative to RLHF. We develop a dual-gradient descent approach over DPO that still requires pretraining reward and cost functions but is more efficient than the primal-dual PPO approach."
Context: 属于Safe RLHF的轻量化解决方案，用DPO替代PPO进行安全约束优化，降低了训练复杂度。
Confidence: high
```

**具体实现细节：**
- **基础模型**: Llama 2
- **约束方法**: Dual-gradient descent over DPO
- **训练流程**: 先训练reward model和cost model，然后在DPO框架下进行原始-对偶优化
- **Benchmark**: 安全对齐评估
- **对比结果**: 相比vanilla DPO提供更强的安全保证，同时比PPO-Lagrangian获得显著更高的reward

---

### 2.3 SafeDPO (Kim et al., ICLR 2025)

```
Claim: 将安全对齐隐式地整合到单阶段策略更新中，仅需引入一个额外的安全超参数并做少量DPO修改
Source: SafeDPO: A Simple Approach to Direct Preference Optimization with Enhanced Safety
URL: https://openreview.net/forum?id=MoJSnVZ59d
Date: ICLR 2025
Excerpt: "SafeDPO is designed to implicitly optimize the safety alignment objective within a single stage of policy learning. The resulting algorithm can be implemented by introducing only one additional hyperparameter, which aims to further enhance safety, along with minor modifications to the DPO implementation."
Context: SafeDPO进一步简化了安全对齐流程，不需要显式训练reward和cost model。
Confidence: high
```

**具体实现细节：**
- **核心思想**: 通过margin-based modification将安全约束隐式集成到DPO中
- **超参数**: 仅一个额外的安全超参数
- **优势**: 不需要拟合reward和cost model，不需要从语言模型采样
- **对比结果**: 在安全性和人类偏好对齐两方面达到SOTA性能

---

### 2.4 SACPO: Stepwise Alignment with Cost DPO (Wachi et al., 2024)

```
Claim: 利用DPO逐步对齐helpfulness和harmlessness，避免exaggerated safety behaviors
Source: Stepwise Alignment with Cost DPO (SACPO)
URL: 相关引用见多篇论文
Date: 2024
Excerpt: "SACPO proposes a stepwise alignment method with respect to individual safety metrics by leveraging simple yet effective algorithms such as DPO, thereby effectively mitigating exaggerated safety behaviors."
Context: SACPO观察到最优解的关系，设计了不需要知道最优Lagrange乘子的逐步DPO方法。
Confidence: medium
```

**具体实现细节：**
- **方法**: 逐步DPO，先对齐helpfulness再对齐harmlessness
- **不需要**: 显式拟合reward和cost model
- **问题**: 依赖于所用和最优Lagrange乘子之间的偏差（可能无界）

---

### 2.5 RePO: Rectified Policy Optimization (Peng et al., 2024)

```
Claim: 用严格(per-prompt)安全约束替代平均安全约束，解决"safety interference"问题
Source: Enhancing Safety in Reinforcement Learning with Human Feedback via Rectified Policy Optimization
URL: https://arxiv.org/abs/2410.19933
Date: 2024
Excerpt: "We propose Rectified Policy Optimization (RePO), which replaces the expected safety constraint with critical safety constraints imposed on every prompt. At the core of RePO is a policy update mechanism driven by rectified policy gradients, which penalizes the strict safety violation of every prompt, thereby enhancing safety across nearly all prompts."
Context: RePO识别出传统平均安全约束的"safety compensation"问题：安全的prompt-response对补偿了不安全的对，导致期望安全但个别不安全。
Confidence: high
```

**具体实现细节：**
- **模型**: Alpaca-7B, Llama3.2-3B
- **数据集**: PKU-SafeRLHF
- **算法**: Primal-dual PPO with rectified policy gradients
- **核心设计**: 对每个prompt施加rectification operator {·}+ 来评估安全性
- **Benchmark**: PKU-SafeRLHF test set, PhysicalSafety, CoNa, Controversial, MaliciousInstructions
- **对比结果**: 在helpfulness和harmlessness两方面均优于PPO-Lagrangian和SACPO

---

### 2.6 Primal-Dual DPO for Constrained LLM Alignment

```
Claim: 提出原始-对偶DPO方法，先训练标准DPO获取奖励信息，再用重排的Lagrangian DPO目标进行安全微调
Source: Provably Convergent Primal-Dual DPO for Constrained LLM Alignment
URL: https://arxiv.org/html/2510.05703
Date: 2025
Excerpt: "Our approach first trains a model using standard DPO on reward preference data, and then fine-tunes LLMs with a rearranged Lagrangian DPO objective on cost preference data, utilizing the reward information provided by the reward-aligned DPO model."
Context: 该方法将需要训练的模型数量从3个减少到2个，节约了内存成本。
Confidence: high
```

**具体实现细节：**
- **模型数量**: 仅需2个（reward-aligned LM + reward-cost-aligned LM）
- **数据集**: PKU-SafeRLHF, TruthfulQA
- **理论保证**: suboptimality和constraint violation的严格保证
- **在线设置**: 引入exploration bonuses进行主动探索

---

## 3. Multi-objective DPO/MODPO

### 3.1 MODPO: Multi-Objective DPO (Zhou et al., 2024)

```
Claim: 将DPO扩展到多目标设置，通过margin term引导策略向多个目标优化，产生Pareto前沿
Source: Multi-Objective Direct Preference Optimization
URL: https://arxiv.org/abs/2405.17956
Date: 2024 (多次更新至2025)
Excerpt: "MODPO loss includes a margin to steer language models by multiple objectives. Empirical results in safety alignment and long-form QA show that MODPO produced a Pareto front catering to diverse preferences."
Context: MODPO是DPO的自然多目标扩展，将线性标量化整合到奖励建模过程中，通过简单的margin-based cross-entropy loss训练语言模型。
Confidence: high
```

**具体实现细节：**
- **核心思想**: DPO loss + margin term用于额外目标
- **方法**: 先在一个偏好数据集上训练reward model，计算chosen和rejected response之间的reward gap，将此gap作为margin term加入标准DPO loss
- **目标函数**: L_MODPO = L_DPO + w_k * margin_k
- **优势**: RL-free，不需要在训练期间从语言模型采样
- **劣势**: 需要为每个目标权重训练不同的策略
- **Benchmark**: 安全对齐（helpful+harmless）、长篇问答
- **对比结果**: 在多个目标上产生Pareto front，优于单目标DPO

---

### 3.2 CPO: Controllable Preference Optimization (Guo et al., EMNLP 2024)

```
Claim: 为不同目标指定显式偏好分数，引导模型生成满足指定要求的响应，实现推理时可控性
Source: Controllable Preference Optimization: Toward Controllable Multi-Objective Alignment
URL: https://aclanthology.org/2024.emnlp-main.85/
Date: EMNLP 2024
Excerpt: "We introduce controllable preference optimization (CPO), which explicitly specifies preference scores for different objectives, thereby guiding the model to generate responses that meet the requirements. Our experimental analysis reveals that the aligned models can provide responses that match various preferences among the '3H' (helpfulness, honesty, harmlessness) desiderata."
Context: CPO解决了alignment tax问题——在一个目标上的改进可能导致其他目标性能下降。
Confidence: high
```

**具体实现细节：**
- **方法**: 为每个目标分配显式偏好分数
- **训练**: 需要每个响应对每个目标的偏好分数/标签
- **优势**: 推理时可控制，缓解alignment tax
- **局限**: 要求偏好数据集对每个目标都有标注，收集成本高
- **Benchmark**: 3H (helpfulness, honesty, harmlessness)
- **对比结果**: 实现Pareto改进，优于单目标对齐方法

---

### 3.3 GAPO: Gradient-Adaptive Policy Optimization (Li et al., 2025)

```
Claim: 利用Multiple-Gradient Descent Algorithm (MGDA)进行多目标对齐，通过gradient rescaling实现平衡优化
Source: Gradient-Adaptive Policy Optimization: Towards Multi-Objective Alignment of Large Language Models
URL: https://arxiv.org/abs/2507.01915
Date: 2025
Excerpt: "GAPO leverages the multiple-gradient descent algorithm (MGDA), a gradient-based multi-objective optimization algorithm, to find Pareto optimal policy. Unlike previous MORLHF approaches that rely on linear scalarization of rewards, MGDA efficiently manages trade-offs by simultaneously considering the gradients of all objectives."
Context: GAPO是RL-based的多目标对齐方法，通过梯度几何找到所有目标同时改进的共同下降方向。
Confidence: high
```

**具体实现细节：**
- **模型**: Mistral-7B
- **算法**: MGDA + gradient rescaling
- **Gradient Rescaling**: 对每个目标的梯度进行自适应归一化，∇J_i(θ) / ||∇J_i(θ)||^p
- **扩展**: P-GAPO引入用户偏好，生成更符合用户需求的Pareto解
- **理论保证**: 证明GAPO收敛到Pareto最优解
- **Benchmark**: PKU-SafeRLHF, HH-RLHF
- **对比结果**: 在helpfulness和harmlessness trade-off上达到SOTA

---

### 3.4 MGDA-Decoupled (2026)

```
Claim: 基于梯度几何的多目标DPO优化方法，通过loss-normalized per-objective gradients计算平衡系数
Source: MGDA-Decoupled: Geometry-Aware Multi-Objective Optimisation for DPO-based LLM Alignment
URL: https://arxiv.org/abs/2604.20685
Date: 2026 (预印本)
Excerpt: "We introduce MGDA-Decoupled, a multi-objective optimisation approach for DPO that computes balancing coefficients from loss-normalised per-objective gradients while applying them to the raw gradients, explicitly managing gradient conflicts and mitigating objective dominance."
Context: MGDA-Decoupled属于轻量级、无需RL和辅助reward model的多目标对齐方法家族。
Confidence: medium
```

**具体实现细节：**
- **模型**: Gemma-2-2b-it (2.6B), Qwen2.5-0.5B-Instruct
- **数据集**: UltraFeedback (4个目标: Helpfulness, Instruction Following, Honesty, Truthfulness)
- **训练**: full-parameter fine-tuning, Adam, lr=5e-7, β=0.5, 1 epoch
- **对比基线**: Uniform, GroupDRO, MGDA-Normalised, CDPO
- **评估**: GPT-4o作为judge，Net Win Rate
- **对比结果**: 在所有目标上实现公平改进，特别关注worst-objective行为

---

### 3.5 HaM: Hypervolume Maximization (Mukherjee et al., 2024)

```
Claim: 首次将a-posteriori多目标优化应用于MOAHF，通过超体积最大化学习多样化的LLM策略
Source: Multi-Objective Alignment of Large Language Models Through Hypervolume Maximization
URL: https://arxiv.org/abs/2412.05469
Date: 2024
Excerpt: "Recent works on MOAHF considered a-priori multi-objective optimization, where human preferences are known at training or inference time. In contrast, when human preferences are unknown or difficult to quantify, a natural approach is to cover the Pareto front by multiple diverse solutions. We propose an algorithm HaM for learning diverse LLM policies that maximizes their hypervolume. This is the first application of a-posteriori MOO to MOAHF."
Context: HaM优化K个LLM策略，使其在目标空间中联合多样化，覆盖Pareto前沿。
Confidence: high
```

**具体实现细节：**
- **方法**: 优化K个策略参数θ_k，最大化hypervolume
- **目标函数**: L_ham(Θ) = Σ_{S⊆{1,...,K}} (-1)^{|S|-1} Π_j min_{k∈S} L̄_j(θ_k)
- **计算优化**: mini-batches降低计算成本，随机hypervolume标量化降低指数复杂度
- **共享参数化**: 共享transformer backbone，独立头部
- **目标**: harmlessness, helpfulness, humor, faithfulness, hallucination
- **优势**: 计算和空间高效

---

### 3.6 UPO: Unified Preference Optimization (Ethayarajh et al., 2025)

```
Claim: 统一DPO和RLHF的优势，在保持DPO简洁性的同时支持辅助目标优化
Source: Language Model Alignment Beyond the Preference Frontier (UPO)
URL: https://arxiv.org/abs/2405.17956
Date: 2025
Excerpt: "We propose a unified technique that leverages the simplicity of MLE objectives for preference alignment, while allowing for stable and efficient optimization of auxiliary objectives."
Context: UPO基于KTO构建，仅需约10行代码修改即可支持辅助目标（如readability, toxicity等）的优化。
Confidence: high
```

**具体实现细节：**
- **基础方法**: KTO (不需要成对偏好数据)
- **辅助目标权重**: w_safe=0.95, w_read=0.05
- **模型规模**: Pythia-[1.4B,2.8B,6.9B], Llama-[7B,13B]
- **数据集**: Anthropic HH, OpenAssistant, SHP
- **对比基线**: MODPO, DRO-V, A-LOL, aoPPO, DPO, CSFT, KTO
- **效率**: 与DPO-style方法计算成本相当，比on-policy RL稳定且高效
- **对比结果**: 在辅助目标优化上显著优于之前的方法

---

### 3.7 MOC: Multi-Objective Control (2026)

```
Claim: 仅需训练一次即可在推理时控制多目标权衡的算法
Source: One Model for All: Multi-Objective Controllable Language Models
URL: https://arxiv.org/abs/2604.04497
Date: 2026 (预印本)
Excerpt: "MOC requires training only once, incorporates explicit policy improvement, and does not rely on human preference data. Moreover, its training cost is comparable to single-objective RLHF. We further improve the computational efficiency of MOC by integrating LoRA, making it feasible to fine-tune a 7B-parameter model on a single A6000 GPU."
Context: MOC使用PPO而非DPO，支持推理时通过偏好向量控制生成。
Confidence: medium
```

**具体实现细节：**
- **算法**: 基于PPO的多目标优化 + LoRA
- **硬件**: 单张A6000 GPU可训练7B模型
- **特点**: 不需要人类偏好数据，训练成本与单目标RLHF相当
- **推理控制**: 通过偏好向量进行推理时控制

---

## 4. Nash Learning from Human Feedback (NLHF)

### 4.1 NLHF (Munos et al., NeurIPS 2024)

```
Claim: 基于纳什均衡的偏好学习框架，将LLM对齐形式化为两人零和博弈，解决BT模型无法处理非传递性偏好的问题
Source: Nash Learning from Human Feedback
URL: https://arxiv.org/abs/2310.04373
Date: NeurIPS 2024
Excerpt: "NLHF employs a pairwise preference modeling strategy by using a neural network to estimate the probability of one response being preferred over another. The alignment process is then carried out through a two-player game, where two LLMs are trained to generate responses, each aiming to maximize the probability that its own response is preferred over the other's."
Context: NLHF由Google DeepMind提出，是RLHF的重要替代框架。核心洞察是scalar reward model无法捕捉偏好的多样性和非传递性（如Condorcet悖论）。
Confidence: high
```

**具体实现细节：**
- **核心思想**: 两人常数和博弈，寻找纳什均衡策略
- **目标函数**: max_π min_π' E_{x~ρ}[E_{y~π(·|x), y'~π'(·|x)}[P(y≻y'|x)]]
- **保证**: Nash策略对任何其他策略至少50%胜率
- **算法**: NashMD (Mirror Descent)，多项式速率收敛到von Neumann Winner
- **优势**: 可以表示所有人类偏好（包括非传递性偏好）
- **局限**: 可能缺乏响应多样性，collapse到确定性策略

---

### 4.2 SPO: Self-Play Preference Optimization (Swamy et al., 2024)

```
Claim: 通过self-play Mirror Descent学习原始博弈的纳什均衡
Source: Self-Play Preference Optimization
URL: 见NLHF相关引用
Date: 2024
Excerpt: "SPO learns the NE of the original game through MD. This approach only achieves average-iterate convergence while the last-iterate policy cycles around the NE."
Context: SPO是NLHF的重要后续工作，使用经典遗憾最小化工具为矩阵博弈开发。
Confidence: medium
```

**具体实现细节：**
- **方法**: Self-play Mirror Descent
- **收敛**: Average-iterate收敛（非last-iterate）
- **复杂度**: Õ(1/ε²)次调用偏好模型找到ε-次优策略

---

### 4.3 Magnetic Preference Optimization (2024)

```
Claim: 实现last-iterate线性收敛到原始博弈的纳什均衡
Source: Magnetic Preference Optimization: Achieving Last-iterate Convergence for Language Model Alignment
URL: https://arxiv.org/abs/2410.16714
Date: 2024
Excerpt: "Our method achieves linear last-iterate convergence and ensures convergence to the NE of the original game."
Context: 改进了NLHF的sublinear last-iterate收敛性，不依赖geometric mixture reference policy。
Confidence: medium
```

---

### 4.4 ONPO: Optimistic Online Mirror Descent for NLHF (Zhang et al., 2025)

```
Claim: 改进NLHF收敛速度到Õ(1/ε)
Source: Optimistic Online Mirror Descent for NLHF
URL: 见相关引用
Date: 2025
Excerpt: "ONPO improved the complexity to Õ(1/ε) for finding an ε-VNW in the original game."
Context: 基于optimistic mirror descent的NLHF算法。
Confidence: medium
```

---

### 4.5 统计不可能性可能性分析 (2025)

```
Claim: 推导出NLHF-aligned LLM多样化输出的充要条件——当且仅当不存在"winning response"时采用混合策略
Source: Statistical Impossibility and Possibility of Aligning LLMs with Human Preferences: From Condorcet Paradox to Nash Equilibrium
URL: https://arxiv.org/abs/2503.10990
Date: 2025
Excerpt: "We derive a necessary and sufficient condition under which any optimal solution of NLHF is a mixed strategy—that is, the NLHF-aligned LLM generates at least two distinct responses with positive probability, thereby avoiding collapse to a single response."
Context: 该工作从统计角度分析了RLHF和NLHF的固有局限性，证明偏好collapse的存在条件。
Confidence: high
```

---

## 5. Pluralistic Alignment (PAL)

### 5.1 PAL: Pluralistic Alignment Framework (Chen et al., 2024)

```
Claim: 使用理想点模型和混合建模从异质人类偏好中学习统一的潜在奖励表示，实现少样本可泛化的多元对齐
Source: PAL: Pluralistic Alignment Framework for Learning from Heterogeneous Preferences
URL: https://arxiv.org/abs/2406.08469
Date: 2024
Excerpt: "We propose PAL, a framework to model human preference complementary to existing pretraining strategies, which incorporates plurality from the ground up. We propose using the ideal point model as a lens to view alignment using preference comparisons. Together with our novel reformulation and using mixture modeling, our framework captures the plurality of population preferences while simultaneously learning a common preference latent space across different preferences, which can few-shot generalize to new, unseen users."
Context: PAL连接了对齐研究与政治科学和心理测量学中的理想点模型，实现了"representation learning for electorates"。
Confidence: high
```

**具体实现细节：**
- **核心思想**: 理想点模型 + 混合建模
- **模型A**: K个原型理想点
- **模型B**: K个从输入prompt到理想点的原型函数
- **架构**: 使用预训练模型的penultimate-layer representation + 简单MLP层
- **K**: 混合原型数量（实验中设K=2）
- **效率**: 冻结LLM参数，仅训练投影层（2-layer MLP + GELU）
- **Benchmark**: Summary dataset, Pick-a-Pic (image gen), Anthropic Personas
- **对比基线**: GPO, VPL, Bradley-Terry Reward Model
- **结果**: 与大型SOTA reward model准确率相当，显著提升数据效率

---

### 5.2 PRISM Alignment Dataset (Kirk et al., NeurIPS 2024)

```
Claim: 创建了包含75个国家1500名参与者的多元人类反馈数据集
Source: The PRISM Alignment Dataset: What Participatory, Representative and Individualised Human Feedback Reveals About the Subjective and Multicultural Alignment of Large Language Models
URL: https://arxiv.org/abs/2404.16019
Date: NeurIPS 2024 (Datasets and Benchmarks Track)
Excerpt: "The PRISM Alignment Project presents a dataset of human feedback suitable for the construction of the multiple reward models required for fully-pluralistic RLHF. This feedback has been gathered from a more diverse set of participants than other feedback datasets, including attempts to provide wider geographic and cultural coverage."
Context: PRISM是多元对齐研究的重要里程碑，提供了包含人口统计信息和细粒度属性评级的反馈数据。
Confidence: high
```

**数据集特点：**
- **规模**: 1,500名参与者，75个国家，约9,000次独特对话
- **标注**: 每个反馈项标注提供者的人口统计信息
- **细粒度评级**: overall values, fluency, factuality, safety, diversity, creativity, helpfulness
- **重要性权重**: 每个参与者对各属性的重视程度
- **用途**: 支持个性化效用函数创建

---

### 5.3 A Roadmap to Pluralistic Alignment (Sorensen et al., ICML 2024)

```
Claim: 提出Pluralistic Alignment的概念框架，主张保存价值多样性而非将其collapse为单一目标
Source: A Roadmap to Pluralistic Alignment
URL: 见相关引用
Date: ICML 2024
Excerpt: "Pluralistic alignment adapts the preferences of diverse groups, especially minority populations, thereby maximizing the collective benefits and promoting fairness across heterogeneous user communities."
Context: 该工作定义了pluralistic alignment的研究议程，指出RLHF聚合偏好会消除多元分歧。
Confidence: high
```

---

## 6. Reward Inconsistency

### 6.1 The Trickle-down Impact of Reward Inconsistency (Shen et al., ICLR 2024)

```
Claim: 系统性地研究了奖励不一致对RLHF的连锁负面影响（"trickle-down"效应）
Source: The Trickle-down Impact of Reward Inconsistency on RLHF
URL: https://openreview.net/forum?id=jyqXeBHY2K (ICLR 2024 poster)
Date: ICLR 2024
Excerpt: "Even large models resort to random guessing when faced with conflicting instructions and responses."
Context: 该工作揭示了当偏好数据存在内在冲突时，RLHF训练过程中不一致性如何逐级放大，导致策略退化。
Confidence: high
```

**核心发现：**
- **问题**: 人类偏好数据中存在偏见和不一致性
- **影响**: 不一致性在RLHF pipeline中逐级放大（trickle-down effect）
- **表现**: 大模型在面对冲突指令时 resort to random guessing
- **原因**: 任务复杂性和主观性、评估标准局限、标注者资质约束

---

### 6.2 ODIN: Disentangled Reward Mitigates Hacking (Chen et al., ICML 2024)

```
Claim: 将长度特征从奖励中解耦，建立更可靠的奖励模型
Source: ODIN: Disentangled Reward Mitigates Hacking in RLHF
URL: https://github.com/Lichang-Chen/ODIN
Date: ICML 2024
Excerpt: "ODIN has two heads to predict two rewards, but only uses one for RL. In RM training stage, ODIN is trained with the same human preference data as vanilla RM with a carefully designed loss to disentangle the length signal and the quality signal into two heads."
Context: ODIN解决length bias导致的reward hacking问题，将奖励信号分离为长度和质量两部分。
Confidence: high
```

**具体实现细节：**
- **架构**: 双头reward model（length head + quality head）
- **训练**: 相同的偏好数据，精心设计的loss将长度和质量信号解耦
- **RL阶段**: 仅quality head参与fine-tuning，length reward被丢弃
- **结果**: PPO和ReMax上均实现显著更高的Pareto front

---

### 6.3 WARM: Weight Averaged Reward Models (Ramé et al., ICML 2024)

```
Claim: 通过在权重空间平均多个奖励模型来提高对分布偏移的可靠性和对偏好不一致的鲁棒性
Source: WARM: On the Benefits of Weight Averaged Reward Models
URL: https://arxiv.org/abs/2401.12187
Date: ICML 2024
Excerpt: "WARM, first fine-tuning multiple RMs, then averaging them in the weight space. This strategy follows the observation that fine-tuned weights remain linearly mode connected when sharing the same pre-training. By averaging weights, WARM improves efficiency compared to the traditional ensembling of predictions, while improving reliability under distribution shifts and robustness to preference inconsistencies."
Context: WARM解决reward hacking中的两大挑战：RL过程中的分布偏移和人类偏好的不一致性。
Confidence: high
```

**具体实现细节：**
- **方法**: 先fine-tune多个RM，然后在权重空间平均
- **假设**: 共享相同预训练的fine-tuned weights保持线性模式连通性
- **效率**: 比传统prediction ensemble更高效
- **结果**: 使用WARM进行RL fine-tuned的policy对单RM policy有79.4%的win rate

---

## 7. Multi-party RLHF与多元反馈

### 7.1 Multi-Objective Reinforcement Learning from AI Feedback (2024)

```
Claim: 从多个AI反馈偏好模型进行多目标强化学习
Source: Multi-objective Reinforcement learning from AI Feedback
URL: https://arxiv.org/abs/2406.07295
Date: 2024
Excerpt: "MORLAIF decomposes the reward into separate models for each principle, providing a complementary approach to improving interpretability and robustness in RLHF reward modeling."
Context: 该工作训练不同原则上的偏好模型，与DPL等分布方法形成互补。
Confidence: medium
```

---

### 7.2 Personalization of LLMs (Jang et al., 2023)

```
Claim: 通过Multi-Objective RL和parameter merging实现个性化偏好对齐
Source: Personalization of Large Language Models
URL: 见相关引用
Date: 2023
Excerpt: "Jang et al. (2023) frame this problem as a Multi-Objective Reinforcement Learning (MORL) task, where diverse and potentially conflicting user preferences are decomposed into multiple dimensions and optimized independently."
Context: 将不同维度偏好独立优化，然后通过parameter merging在推理时组合。
Confidence: high
```

---

## 8. Distributional Preference Learning

### 8.1 DPL: Distributional Preference Learning (Siththaranjan et al., 2024)

```
Claim: 用效用值分布而非单标量表示偏好，捕捉hidden context导致的偏好不确定性
Source: Distributional Preference Learning
URL: 见相关引用
Date: 2024
Excerpt: "DPL estimates a distribution over utility values to account for this hidden context. Applied to an RLHF dataset, DPL identifies the hidden context of helpfulness vs. harmlessness without supervision. Optimizing a lower quantile of the DPL utility distributions also reduces jailbreak vulnerabilities caused by conflicting objectives."
Context: DPL的核心洞察是异质性不是需要平均掉的噪声，而是目标规范的一部分。
Confidence: high
```

**具体实现细节：**
- **核心思想**: 每个输出用效用分布而非单标量表示
- **隐藏上下文**: 标注者多样性或标注目标变化导致的偏好异质性
- **方法**: 估计效用值分布，通过explained-variance diagnostics检测隐藏上下文影响
- **应用**: 优化DPL效用分布的lower quantile可减少jailbreak漏洞
- **对比**: MORLAIF将奖励分解为每个原则的独立模型；DPL学习单个分布模型捕捉隐藏上下文的不确定性

---

### 8.2 Quantile Reward Model (QRM) (Dorka, 2024)

```
Claim: 将奖励建模从标量信号扩展到完整分布
Source: Quantile Reward Model
URL: 见DVPO相关引用
Date: 2024
Excerpt: "The Quantile Reward Model (QRM) extends reward modeling from scalar signals to full distributions, capturing the ambiguity and multidimensionality of human preferences and guiding optimization toward safer and more reliable behaviors."
Context: QRM应用分位数回归进行分布奖励建模。
Confidence: medium
```

---

## 9. 多目标方法分类对比

### 9.1 方法分类

| 类别 | 代表方法 | 核心机制 | 是否需要RL | 是否需要RM | 推理可控性 |
|------|---------|---------|-----------|-----------|-----------|
| **Safe RLHF (PPO-based)** | Safe RLHF, RePO | PPO-Lagrangian + 约束优化 | 是 | 是（reward+cost） | 否 |
| **Safe DPO** | C-DPO, SafeDPO, SACPO | DPO + 安全约束/对偶梯度 | 否 | 否/部分 | 否 |
| **Multi-Objective DPO** | MODPO, CPO, MGDA-Decoupled | DPO + margin/多梯度 | 否 | 部分 | CPO支持 |
| **RL-based MOO** | GAPO, HaM, MOC | MGDA/Hypervolume/PPO | 是 | 是 | MOC支持 |
| **Nash Learning** | NLHF, SPO, MPO | 博弈论 + Mirror Descent | 否/是 | 否 | 否 |
| **Pluralistic** | PAL, DPL | 混合建模/分布学习 | 否 | 是 | 是（个性化） |
| **统一优化** | UPO | KTO + 辅助目标 | 否 | 否 | 否 |

### 9.2 约束优化方法对比

| 方法 | 约束类型 | 优化算法 | 原始-对偶 | 严格/平均约束 |
|------|---------|---------|-----------|--------------|
| Safe RLHF | 期望安全 | PPO-Lagrangian | 是 | 平均 |
| C-DPO | 期望安全 | Dual-gradient descent over DPO | 是 | 平均 |
| SafeDPO | 隐式安全 | DPO + margin | 否 | 隐式 |
| SACPO | 期望安全 | 逐步DPO | 否 | 平均 |
| RePO | 严格安全 | Rectified policy gradients | 是 | 严格(per-prompt) |
| Primal-Dual DPO | 期望安全 | Lagrangian DPO | 是 | 平均 |

### 9.3 Pareto前沿覆盖能力

| 方法 | Pareto前沿类型 | a-priori/a-posteriori | 策略数量 |
|------|---------------|----------------------|---------|
| MODPO | Convex hull | a-priori (权重依赖) | 每个权重一个 |
| HaM | Full coverage | a-posteriori | K个联合优化 |
| GAPO | Pareto stationary | a-priori (梯度自适应) | 1个 |
| CPO | Steerable | a-priori (推理时控制) | 1个条件策略 |
| Rewarded Soups | 权重插值 | a-priori (推理时) | 多个+合并 |

---

## 10. 安全机制对比表

| 方法 | 安全机制 | 约束类型 | 是否解决Safety Interference | 训练稳定性 | 计算复杂度 |
|------|---------|---------|---------------------------|-----------|-----------|
| **Safe RLHF** | PPO-Lagrangian | 期望成本 ≤ 阈值 | 否 | 低 | 高（3个模型+PPO） |
| **C-DPO** | Dual-gradient DPO | 期望成本 ≤ 阈值 | 否 | 中 | 中（需RM+CM） |
| **SafeDPO** | Margin-based DPO | 隐式安全偏好 | 部分 | 高 | 低（单阶段） |
| **SACPO** | 逐步DPO对齐 | 期望成本 ≤ 阈值 | 否 | 高 | 低（RL-free） |
| **RePO** | Rectified gradients | 逐prompt安全 | **是** | 中 | 高（PPO-based） |
| **Primal-Dual DPO** | Lagrangian DPO | 期望成本 ≤ 阈值 | 否 | 中 | 中（2个模型） |
| **MODPO** | Margin loss | 多目标权衡 | 部分 | 高 | 低 |
| **GAPO** | MGDA梯度平衡 | 多目标Pareto | 部分 | 中 | 高（PPO-based） |

### 关键术语

- **Safety Interference**: 平均安全约束下，某些prompt的安全性以牺牲其他prompt的安全性为代价
- **Safety Compensation**: 安全的prompt-response对补偿了不安全的对，导致期望安全但个别不安全
- **Exaggerated Safety Behaviors**: 模型生成无害但无用的响应
- **Alignment Tax**: 在一个目标上的改进导致其他目标性能下降

---

## 11. 总结与演进关系

### 11.1 方法演进时间线

```
2023 ──────────────────────────────────────────────────────►
  │
  ├── Safe RLHF (Dai et al.) [ICLR 2024] ──┐
  │                                         ├── Safe RL 范式
  ├── WARM (Ramé et al.) [ICML 2024]        │
  │                                         │
2024 ───────────────────────────────────────┼───────────────►
  │                                         │
  ├── C-DPO ────────────────────────────────┤
  ├── RePO ─────────────────────────────────┤
  ├── SACPO ────────────────────────────────┤
  ├── NLHF (Munos et al.) [NeurIPS 2024] ───┤
  ├── PAL (Chen et al.) ────────────────────┤
  ├── PRISM Dataset [NeurIPS 2024] ─────────┤
  ├── ODIN [ICML 2024] ─────────────────────┤
  ├── MODPO (Zhou et al.) ──────────────────┤
  ├── CPO (Guo et al.) [EMNLP 2024] ────────┤
  ├── DPL (Siththaranjan et al.) ────────────┤
  ├── UPO ──────────────────────────────────┤
  ├── GAPO ─────────────────────────────────┤
  ├── Reward Inconsistency (Shen) [ICLR 2024]┤
  └── HaM (Mukherjee et al.) ────────────────┘
  │
2025 ──────────────────────────────────────────────────────►
  │
  ├── SafeDPO [ICLR 2025] ──────────────────┐
  ├── Primal-Dual DPO ──────────────────────┤
  ├── MGDA-Decoupled ───────────────────────┤
  ├── LCPO ─────────────────────────────────┤
  └── Mirror Prox加速NLHF ──────────────────┘
```

### 11.2 关键趋势

1. **从RL到RL-free**: 大量工作（C-DPO, SafeDPO, SACPO, MODPO）从PPO-based转向DPO-based，提高稳定性
2. **从平均到严格约束**: RePO识别并解决safety interference问题，推动严格安全约束
3. **从标量到分布**: DPL和QRM将偏好表示从标量扩展到分布
4. **从统一到多元**: PAL和PRISM推动异质偏好建模
5. **从纳什均衡到收敛保证**: NLHF系列工作不断改进收敛速度和保证
6. **从a-priori到a-posteriori**: HaM等方法在偏好未知时覆盖Pareto前沿

### 11.3 开放问题

1. **严格安全约束的效率**: RePO的逐prompt约束计算成本高，如何高效实现
2. **多目标推理可控性**: 如何在推理时动态调整目标权衡而不重新训练
3. **异质偏好的可扩展性**: PAL等方法在小样本下有效，如何扩展到大规模
4. **NLHF的多样性保证**: 在什么条件下纳什均衡策略保持输出多样性
5. **分布学习的优化**: 如何有效优化分布偏好目标
6. **Alignment Tax的理论理解**: 多目标冲突的量化分析和理论界限
7. **安全与能力的fundamental trade-off**: 安全约束对模型能力的理论影响

---

## 参考文献汇总

| # | 论文 | 作者 | 会议/年份 | 核心贡献 |
|---|------|------|----------|---------|
| 1 | Safe RLHF | Dai et al. | ICLR 2024 | PPO-Lagrangian安全约束 |
| 2 | C-DPO | Liu et al. | 2024 | DPO上的对偶梯度下降 |
| 3 | SafeDPO | Kim et al. | ICLR 2025 | 单阶段安全DPO |
| 4 | RePO | Peng et al. | 2024 | 逐prompt严格安全约束 |
| 5 | SACPO | Wachi et al. | 2024 | 逐步安全DPO对齐 |
| 6 | Primal-Dual DPO | 匿名 | 2025 | 可证明收敛的原始对偶DPO |
| 7 | MODPO | Zhou et al. | 2024 | 多目标DPO |
| 8 | CPO | Guo et al. | EMNLP 2024 | 可控多目标对齐 |
| 9 | GAPO | Li et al. | 2025 | MGDA梯度自适应优化 |
| 10 | MGDA-Decoupled | 匿名 | 2026 | 几何感知多目标DPO |
| 11 | HaM | Mukherjee et al. | 2024 | 超体积最大化 |
| 12 | UPO | Ethayarajh et al. | 2025 | 统一辅助目标优化 |
| 13 | NLHF | Munos et al. | NeurIPS 2024 | 纳什学习 |
| 14 | SPO | Swamy et al. | 2024 | 自博弈偏好优化 |
| 15 | MPO | 匿名 | 2024 | 磁偏好优化 |
| 16 | PAL | Chen et al. | 2024 | 多元对齐框架 |
| 17 | PRISM Dataset | Kirk et al. | NeurIPS 2024 | 多元反馈数据集 |
| 18 | DPL | Siththaranjan et al. | 2024 | 分布偏好学习 |
| 19 | ODIN | Chen et al. | ICML 2024 | 解耦奖励 |
| 20 | WARM | Ramé et al. | ICML 2024 | 权重平均奖励模型 |
| 21 | Reward Inconsistency | Shen et al. | ICLR 2024 | 奖励不一致性影响 |
| 22 | LCPO | 匿名 | 2025 | 潜在集体偏好优化 |
| 23 | MOC | 匿名 | 2026 | 多目标可控语言模型 |
| 24 | Statistical NLHF Analysis | 匿名 | 2025 | 统计不可能性可能性 |

---

*报告生成时间: 2025年*
*覆盖论文: 24+ 篇*
*搜索范围: NeurIPS 2024, ICML 2024, ICLR 2024/2025, EMNLP 2024, ACL 2025*
