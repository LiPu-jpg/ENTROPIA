# 维度10: 课程学习与数据筛选策略（FastCuRL, 难度过滤, 数据合成等）

> 深度调研2024-2025年LLM+RL中的课程学习与数据筛选策略
> 覆盖论文：20+篇顶会/预印本论文
> 最后更新：2025年7月

---

## 目录

1. [总体概述](#1-总体概述)
2. [课程强化学习方法](#2-课程强化学习方法)
   - 2.1 [FastCuRL: 阶段式上下文缩放](#21-fastcurl)
   - 2.2 [SPEED-RL: 在线课程学习](#22-speed-rl)
   - 2.3 [CurES: 梯度分析驱动的课程学习](#23-cures)
   - 2.4 [CLPO: 课程学习 meets 策略优化](#24-clpo)
   - 2.5 [AdaRFT: 自适应课程强化微调](#25-adarft)
   - 2.6 [HAMMER: Hamiltonian好奇心课程](#26-hammer)
3. [数据剪枝与选择方法](#3-数据剪枝与选择方法)
   - 3.1 [LIMR: Less is More for RL Scaling](#31-limr)
   - 3.2 [LearnAlign: 梯度对齐数据选择](#32-learnalign)
   - 3.3 [PODS: Rollout下采样](#33-pods)
   - 3.4 [GAIN-RL: 角度信息驱动选择](#34-gain-rl)
   - 3.5 [HIVE: 在线验证Prompt选择](#35-hive)
   - 3.6 [BOTS: 贝叶斯在线任务选择](#36-bots)
4. [在线难度过滤方法](#4-在线难度过滤方法)
   - 4.1 [Online Difficulty Filtering](#41-online-difficulty-filtering)
   - 4.2 [DAPO的动态采样](#42-dapo动态采样)
   - 4.3 [ThinkPrune: 迭代长度剪枝](#43-thinkprune)
5. [长度感知与自适应采样](#5-长度感知与自适应采样)
   - 5.1 [LSPO: 长度感知动态采样](#51-lspo)
   - 5.2 [DAST: 难度自适应慢思考](#52-dast)
6. [数据合成方法](#6-数据合成方法)
   - 6.1 [DeepScaleR: 数学推理数据构建](#61-deepscaler)
   - 6.2 [Self-Instruct: 自举指令生成](#62-self-instruct)
   - 6.3 [ULTRAFEEDBACK: AI反馈数据合成](#63-ultrafeedback)
   - 6.4 [AlpaGasus: 质量过滤数据选择](#64-alpagasus)
7. [方法对比与分类](#7-方法对比与分类)
   - 7.1 [数据筛选方法对比表](#71-数据筛选方法对比表)
   - 7.2 [课程学习策略分类](#72-课程学习策略分类)
   - 7.3 [效率提升数据汇总](#73-效率提升数据汇总)
8. [关键洞察与趋势](#8-关键洞察与趋势)

---

## 1. 总体概述

课程学习（Curriculum Learning）和数据筛选是LLM+RL训练的关键优化手段。2024-2025年，随着DeepSeek-R1、OpenAI o1等推理模型的成功，研究社区大量探索如何通过智能数据选择和课程安排来加速RL训练、降低计算成本并提高最终性能。

**核心发现：**
- **数据规模并非决定性因素**：LIMR等工作表明，1,389个精心选择的样本可以超过8,523个完整数据集的效果
- **中等难度样本最关键**：SPEED-RL、CurES、BOTS等方法均发现，pass rate接近0.5的样本提供最高学习信号
- **上下文长度需要课程式扩展**：FastCuRL证明阶段式上下文缩放可减少50%+的训练步骤
- **Rollout并非全部有用**：PODS等方法通过下采样最有信息的rollout实现1.7x+加速
- **在线自适应优于静态课程**：AdaRFT、CLPO、GAIN-RL等方法通过模型自身信号动态调整课程

---

## 2. 课程强化学习方法

### 2.1 FastCuRL

```
Claim: FastCuRL是一个课程强化学习框架，通过阶段式上下文缩放（stage-wise context scaling）
交替进行CoT压缩和扩展，显著提高R1-like推理模型的训练效率，仅需50%训练步骤即可超越DeepScaleR。
Source: FastCuRL: Curriculum Reinforcement Learning with Stage-wise Context Scaling for Efficient Training R1-like Reasoning Models
URL: https://arxiv.org/abs/2503.17287
Date: 2025-03-21 (arXiv), v6: 2025-09-20
Excerpt: "FastCuRL-1.5B-Preview achieves better performance and reduces computational resource 
consumption by more than 50%, with all training phases efficiently executed using a single node with 8 GPUs."
Context: 基于DeepScaleR的观察：8K上下文时42%输出被截断，24K时熵崩溃导致过早收敛
Confidence: high
```

**方法核心：**
- **交替压缩-扩展循环**：先压缩CoT推理输出，再扩展，重复此过程
- **阶段式上下文缩放**：根据训练阶段动态调整上下文长度，避免过早使用过长上下文
- **数据复杂度控制**：同时控制训练样本的复杂度，确保训练效率

**具体实现细节：**
- 基础模型：DeepSeek-R1-Distill-Qwen-1.5B
- 训练硬件：单节点8 GPUs
- 训练步骤：仅为DeepScaleR的50%
- Benchmark：AIME 2024 (49.6%), AMC 2023, MATH 500, Minerva Math, OlympiadBench

**关键发现：**
1. 8K上下文时约42%模型输出被截断，降低训练效率
2. 24K上下文时模型熵崩溃，导致探索能力下降和过早收敛
3. 同时控制上下文长度和数据复杂度能显著提升训练效率
4. 长度感知分割和渐进窗口扩展都是必要的，移除任一都会使增益减半

---

### 2.2 SPEED-RL

```
Claim: SPEED（Selective Prompting with Efficient Estimation of Difficulty）是一种自适应在线RL课程方法，
通过两阶段推理（筛选+继续）选择中等难度样本，实现2-6倍训练加速。
Source: Faster Training of Reasoning Models via Online Curriculum Learning
URL: https://arxiv.org/abs/2506.09016
Date: 2025-06-10 (ICML Workshop)
Excerpt: "SPEED achieves target validation accuracies 2-6 times faster compared with baseline RL 
algorithms across nearly all benchmarks and experimental runs."
Context: 理论证明中等难度prompt提高梯度估计器的信噪比(SNR)，加速收敛
Confidence: high
```

**方法核心：**
- **两阶段推理方案**：
  1. **筛选阶段**：每个prompt生成少量rollout（N_init ≈ 4-8），识别pass rate远离极端值（0%或100%）的"合格"prompt
  2. **继续阶段**：对合格prompt生成剩余rollout，确保计算资源分配给高SNR样本
- **与算法无关**：可无缝集成到GRPO、PPO、RLOO、REINFORCE等标准RL算法

**具体实现细节：**
- 模型：Qwen2.5-Math-1.5B, Qwen2.5-Math-7B
- RL算法：RLOO, DAPO
- 训练数据：NuminaMath (220k), DAPO-17k, DeepScaleR (400k)
- 筛选大小：N_init = 4-8, N_cont = 补齐到总N (16-64)
- 理论分析：中等难度prompt（pass rate ≈ 0.5）提供最高SNR的梯度估计

**实验结果：**
- Qwen2.5-Math-7B在DAPO-1k上达到validation accuracy 0.45：SPEED-RLOO需7.6小时，vanilla RLOO需约3.4倍时间
- SPEED维持训练accuracy更接近0.5（相比vanilla RLOO）
- 梯度范数显著大于基线方法

---

### 2.3 CurES

```
Claim: CurES从梯度优化角度系统分析课程学习，理论揭示prompt采样分布决定梯度下降收敛速度，
rollout数量分配影响梯度一致性，提出贝叶斯后验估计实现高效样本选择和rollout分配。
Source: CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs
URL: https://arxiv.org/abs/2510.01037
Date: 2025-10-01
Excerpt: "CurES outperforms GRPO by +3.30 points and +4.82 points with 1.5B and 7B models, 
respectively, and exceeds the best prior sample efficient methods by +2.12 points on average."
Context: 从梯度角度分析prompt采样分布和rollout数量分配对训练效率的影响
Confidence: high
```

**方法核心：**
- **理论分析**：
  - Prompt采样分布直接控制梯度下降的收敛速度
  - Rollout数量分配影响整体梯度更新的一致性和稳定性
- **贝叶斯后验估计**：
  - 先通过少量rollout估计prompt难度（模型问答准确率）
  - 根据估计的准确率重新分配prompt采样概率和rollout数量
  - 通过后验估计逐步提升准确率估计的置信度

**具体实现细节：**
- 模型：Qwen2.5-Math-1.5B, 7B
- 基线：GRPO, RPP, Speed-RL, GVM
- 8个数学推理benchmark
- 动态分配：对中等难度prompt分配更多rollout预算

**实验结果：**
- 1.5B模型上超过GRPO +3.30 points
- 7B模型上超过GRPO +4.82 points
- 超过最佳 prior sample-efficient 方法平均 +2.12 points
- 更快的收敛速度

---

### 2.4 CLPO

```
Claim: CLPO是首个深度整合动态在线课程学习与RL微调的算法，利用模型rollout表现进行实时难度评估，
构建在线课程并驱动自适应问题重构（模型自己作为教师）。
Source: CLPO: Curriculum Learning meets Policy Optimization for LLM Reasoning
URL: https://arxiv.org/abs/2509.25004
Date: 2025-09-29
Excerpt: "CLPO achieves state-of-the-art performance across eight challenging mathematical and 
general reasoning benchmarks, with an average pass@1 improvement of 6.96% over other methods."
Context: 将课程学习带入新维度，第一个深度融合动态在线课程学习与RL微调的算法
Confidence: high
```

**方法核心：**
- **实时难度评估**：利用模型自身rollout表现进行在线难度评估
- **在线课程构建**：根据实时性能动态构建课程
- **自适应问题重构**：
  - 对中等难度问题进行多样化以促进泛化
  - 对高难度问题进行简化以使其可学习
- **教学反馈循环**：模型充当自己的教师

**实验结果：**
- 在8个数学和通用推理benchmark上达到SOTA
- 平均pass@1提升6.96%相比其他方法

---

### 2.5 AdaRFT

```
Claim: AdaRFT是一种轻量级、与模型无关的自适应课程学习策略，基于奖励反馈动态调整任务难度，
使训练问题保持在适合当前模型能力的难度范围内，无需修改RL算法或手动数据管理。
Source: Efficient Reinforcement Finetuning via Adaptive Curriculum Learning
URL: https://arxiv.org/abs/2504.05520
Date: 2025-04 (arXiv), 2026-02修订
Excerpt: "AdaRFT introduces a lightweight, model-agnostic curriculum learning strategy that 
dynamically adjusts task difficulty based on reward feedback."
Context: 课程学习在LLM RFT领域的应用，克服了固定课程缺乏适应性和在线过滤计算开销大的问题
Confidence: high
```

**方法核心：**
- **自适应难度调整**：根据模型奖励信号动态调整目标难度T
- **离线难度估计**：使用预计算的难度分数（基于内在复杂度而非模型依赖）
- **目标难度更新**：
  - 步长η、灵敏度α、目标奖励β
  - 难度范围d_min, d_max
  - 选择最接近目标难度T的样本

**具体实现细节：**
- 兼容RL算法：PPO, GRPO, REINFORCE++
- 难度分数：可使用人工标注或模型估计
- 奖励信号驱动：连续调整目标难度以匹配模型能力

**关键优势：**
- 无需手动数据管理或模型特定预处理
- 课程自动适应模型能力
- 适用于固定数据设置

---

### 2.6 HAMMER

```
Claim: HAMMER将多样性度量（常用于数据集评估）转化为动态RL过程，通过最小语义Hamiltonian路径
排序训练样本，在早期训练中保持更多探索，稳定收敛。
Source: HAMMER: Hamiltonian Curiosity Augmented Large Language Model Reinforcement
URL: https://arxiv.org/abs/2509.25240
Date: 2025-09-25
Excerpt: "HAMMER stimulates model curiosity and consistently achieves a 3% to 4% average 
accuracy gain across diverse inference benchmarks."
Context: 解决基于难度的课程学习中的局部优化问题（早期持续训练简单样本导致策略失去探索能力）
Confidence: high
```

**方法核心：**
- **Hamiltonian好奇心顺序**：通过最小语义Hamiltonian路径排序训练序列
- **多样性驱动排序**：将数据集评估中的多样性度量转化为动态RL过程
- **高效启发式算法**：计算Hamiltonian Curiosity Order的低成本算法
- **理论保证**：保留最优策略，通过收紧泛化界促进收敛

**实验结果：**
- 集成到DAPO和GRPO中，一致提升样本效率
- 准确率提升3-4%
- 与昂贵的基于难度的课程RL相比，显著降低计算开销

---

## 3. 数据剪枝与选择方法

### 3.1 LIMR

```
Claim: LIMR提出学习影响度量（Learning Impact Measurement），通过衡量每个样本奖励轨迹
与平均学习曲线的对齐程度来评估训练数据的有效性，仅用1,389个精选样本就超过8,523个样本的完整数据集。
Source: LIMR: Less is More for RL Scaling
URL: https://arxiv.org/abs/2502.11886
Date: 2025-02-17
Excerpt: "A strategically selected subset of just 1,389 samples can outperform the full 
8,523-sample dataset... Our RL-based LIMR achieves 16.7% higher accuracy on AIME24 and 
outperforms LIMO and s1 by 13.0% and 22.2% on MATH500."
Context: 挑战RL训练数据规模越大越好的假设，证明精确的样本选择而非数据规模才是关键
Confidence: high
```

**方法核心：**
- **学习影响度量（LIM）**：
  - 评估和优先化训练样本与模型学习轨迹的对齐程度
  - 衡量每个样本的奖励轨迹与平均学习曲线的匹配度
- **自动化数据评估**：可扩展实现，无需人工干预
- **从基础模型直接训练**：不从distilled模型开始

**具体实现细节：**
- 精选样本：1,389个（vs. 完整数据集8,523个）
- 模型规模：7B（对比LIMO和s1主要使用32B模型）
- Benchmark：AIME24 (+16.7%), MATH500 (+13.0% vs LIMO, +22.2% vs s1)

**关键发现：**
- 7B-scale的SFT显著低于32B-scale（LIMO和s1的SFT方法在7B上效果差）
- RL-based的LIMR在7B-scale上大幅超越SFT方法
- 精确的样本选择比数据规模更重要

---

### 3.2 LearnAlign

```
Claim: LearnAlign是一种基于梯度对齐的实用数据选择方法，引入基于成功率的数据可学习性度量
来克服梯度范数中的响应长度偏差，高效识别可学习且有代表性的推理数据。
Source: LearnAlign: Data Selection for LLM Reinforcement Learning with Improved Gradient Alignment
URL: https://arxiv.org/abs/2506.11480
Date: 2025-06-13
Excerpt: "It reduces data requirements by up to 1,000 data points with better performance 
(77.5%) than that on the full dataset on the GSM8K benchmark (77.0%)."
Context: 针对RLVR范式设计的数据选择方法，克服SFT数据选择在RL中效果有限的问题
Confidence: high
```

**方法核心：**
- **梯度对齐评分**：
  - 使用一阶Taylor展开近似训练损失变化，估计每个数据点的影响
  - 将影响力转化为数据点与训练数据集梯度的对齐分数
- **可学习性度量**：
  - 引入基于成功率的学习价值来替代梯度范数
  - 克服响应长度偏差
  - 表示每个数据点的学习潜力（不受长度影响）

**具体实现细节：**
- 模型：Qwen2.5-1.5B-Instruct, 3B, 7B
- 训练数据：GSM8K, DAPO-MATH-17K
- 超参数：8 rollouts/sample, temperature 1.0, lr 1e-6, KL coeff 0.04, clip epsilon 0.2
- Batch size: 48 (GSM8K), 64 (DAPO-MATH-17K)
- Warmup: 300 (GSM8K), 1000 (DAPO-MATH-17K) samples

**实验结果：**
- 1000样本(13.4%)达到77.5%，匹配全量数据(77.0%)
- 2000样本(26.8%)达到78.3%，超越全量数据
- 在GSM8K、MATH500、AMC2023、AIME2024、CRUX上验证有效

---

### 3.3 PODS (Policy Optimization with Down-Sampling)

```
Claim: PODS通过在可扩展的推理阶段生成大批量rollout，但在策略更新阶段仅选择最有信息的子集进行训练，
解决推理与策略更新之间的计算不对称性，实现至少1.7倍加速。
Source: Not All Rollouts are Useful: Down-Sampling Rollouts in LLM Reinforcement Learning
URL: https://arxiv.org/abs/2504.13818
Date: 2025-04 (v3: 2025-10-01)
Excerpt: "PODS delivers at least a 1.7x speedup and reaching higher final accuracy across 
diverse model architectures, scales, and deployment scenarios."
Context: 核心观察：并非所有rollout对模型改进贡献相等，超出一定规模后额外rollout提供递减回报
Confidence: high
```

**方法核心：**
- **最大方差下采样（Max-Variance Down-Sampling）**：
  - 保留奖励光谱极端的rollout（正确和错误的）
  - 保持强对比信号
  - O(n log n)时间复杂度求解最优子集
- **计算不对称解决**：
  - 推理阶段：生成n个rollout（ embarrassingly parallel）
  - 更新阶段：仅训练m < n个信息丰富的样本

**具体实现细节：**
- 推荐下采样比：2-4（即n=32或64, m=16）
- 资源受限设置：可使用高达16的下采样比
- 最优rollout池大小：n=64
- 训练batch size m：在{16,8,4,2}中表现稳健，m≤4时性能下降

**实验设置：**
- 模型：多种架构和规模
- 训练数据：GSM8K等数学推理数据集
- 硬件：单卡L40S到多卡部署

**实用指南：**
- 下采样比2-4提供性能和效率的有效平衡
- 资源受限时可使用高达16的下采样比
- 内存充足时可从更大的rollout池中获益

---

### 3.4 GAIN-RL

```
Claim: GAIN-RL利用模型固有的角度集中信号（token隐藏状态向量角分布与梯度的相关性）
动态选择训练数据，实现超过2.5倍训练加速，用一半数据即可超越全量数据训练的vanilla GRPO。
Source: Angles Don't Lie: Unlocking Training-Efficient RL Through the Model's Own Signals
URL: https://arxiv.org/abs/2506.02281
Date: 2025-06-02 (NeurIPS 2025)
Excerpt: "GAIN-RL (GRPO) achieves over a 2.5x acceleration in training efficiency across 
diverse mathematical and coding tasks and varying model scales."
Context: 发现角度集中信号能有效反映LLM从特定数据学习的能力
Confidence: high
```

**方法核心：**
- **角度集中信号**：
  - 理论证明token隐藏状态向量角分布与梯度相关
  - 数据点的高角度集中预示更强的学习偏好
- **梯度驱动导航**：
  - 利用模型内在信号动态选择每轮训练数据
  - 确保持续的 impactful 梯度更新
- **数据选择洞察**：
  - 优先高角度集中数据，丢弃低角度集中数据
  - 仅用50%数据即可超越vanilla GRPO全量数据训练

**具体实现细节：**
- 目标准确率β=0.5（维持强梯度）
- 灵敏度参数：α=2（准确率）, γ=0.5（角度集中）
- Batch size: 1024, sampling n: 1024
- 框架：VerL, 8 NVIDIA A100 GPUs

**实验结果：**
- GSM8K: 3.33x加速, 4.72%最终准确率提升
- MATH: 2.5x加速
- AMC23: 2x加速
- PPO上也有效：平均2.2x训练加速

---

### 3.5 HIVE

```
Claim: HIVE是一个分层框架，通过历史信息选择（基于奖励轨迹和响应熵）和在线验证选择
（使用prompt熵作为高效代理）精确追踪模型的"学习边缘"，实现最多3.8倍rollout加速。
Source: HIVE: Online-Verified Prompt Selection for Efficient RL Training of Large Reasoning Model
URL: https://arxiv.org/abs/2603.25184
Date: 2026-03-26 (arXiv)
Excerpt: "HIVE achieves up to 3.8x speedup in rollout and 2.2x faster total training time... 
reducing up to 9.2 million rollouts while consistently maintaining or even exceeding reasoning accuracy."
Context: 解决历史指标在训练中快速过时（metadata staleness）的问题
Confidence: high
```

**方法核心：**
- **两阶段分层选择**：
  1. **历史信息选择**：利用奖励历史和响应熵定义"学习边缘"的选择概率
  2. **在线验证选择**：基于当前模型参数计算prompt熵，验证候选样本的当前效用
- **Prompt熵作为代理**：
  - V(x) = 平均token-level熵
  - 理论保证rank一致性
  - 仅需单次前向传播（O(1) vs O(G·L_r)）
- **动态中位数阈值**：保留top 50%高熵样本

**具体实现细节：**
- 模型：Qwen2.5-Math-1.5B/7B, DeepSeek-R1-Distill-Qwen-1.5B, Llama-3.2-3B, Qwen2.5-14B/32B
- 训练数据：DAPO+MATH, Open-R1 30k
- 8x A100 GPUs, 训练1000 steps
- Batch size: 256/128, rollout batch: 384/192, mini-batch: 512
- 每prompt采样8个响应, lr 1e-6

**实验结果：**
- Qwen2.5-Math-7B: rollout减少70% (13.1M→3.9M), 3.4x加速
- Total training: 198.4h→85.8h
- Rollout: 153.7h→40.2h (3.8x加速)
- 准确率59.7%，超越GRESO (58.6%) 和 DS (57.8%)

---

### 3.6 BOTS

```
Claim: BOTS是一个统一的贝叶斯框架，通过维护任务难度的后验估计进行在线任务选择，
联合利用直接评估的显式证据和从相关任务推断的隐式证据，在多个领域和模型规模上一致提升数据效率。
Source: BOTS: A Unified Framework for Bayesian Online Task Selection in LLM Reinforcement Finetuning
URL: https://arxiv.org/abs/2510.26374
Date: 2025-10-30
Excerpt: "For the 1.5B model, BOTS obtains 36% acceleration (TTB(100%) = 0.64) in math; 
For the 7B model, 50% acceleration (TTB(100%) = 0.50) in logic."
Context: 首个统一的贝叶斯在线任务选择框架，解决非平稳性和部分可观测性
Confidence: high
```

**方法核心：**
- **贝叶斯基础**：基于贝叶斯推理，自适应模型演化中的能力变化
- **两证据源整合**：
  - 显式证据：直接评估选定任务的成功/失败计数
  - 隐式证据：通过超轻量级插值插件估计未评估任务的难度（无需额外rollout）
- **Thompson采样**：确保探索与利用的原则性平衡

**具体实现细节：**
- 超参数：λ=0.1 (显式证据权重), ρ=0.1 (更新率)
- 目标难度：p*=0.5
- 插值插件：计算开销可忽略

**实验结果：**
- 1.5B模型：数学领域36%加速 (TTB=0.64)
- 7B模型：逻辑领域50%加速 (TTB=0.50)
- 在18个指标中，1.5B模型获得8个第一、9个第二；7B模型获得6个第一、8个第二

---

## 4. 在线难度过滤方法

### 4.1 Online Difficulty Filtering

```
Claim: 在线难度过滤动态评估每个训练步骤中prompt的难度（通过采样rollout的成功率），
过滤掉太容易或太难的样本，确保训练批次由中等难度样本组成，最大化GRPO中的可学习性。
Source: Online Difficulty Filtering for Reasoning Oriented Reinforcement Learning
URL: https://arxiv.org/abs/2504.03380
Date: 2025-04 (arXiv), 2026-01修订
Excerpt: "Online difficulty filtering with fixed batch size...dynamically assesses difficulty 
on the fly in each training step and applies difficulty filtering logic."
Context: 基于理论分析：中等难度样本在GRPO中产生平衡模型更新，促进有效学习
Confidence: high
```

**方法核心：**
- **在线难度评估**：
  - 使用采样rollout的成功率p(x)衡量每个prompt的难度
  - 预定义难度阈值T_Low和T_High
- **异步采样**：
  - 确保固定batch size (|B|=N)
  - 并行异步采样和填充
- **可学习性最大化**：
  - 平衡难度鼓励GRPO中对坏轨迹的惩罚和好轨迹的强化

**具体实现细节：**
- 模型：Qwen2.5-3B
- RL算法：GRPO
- 每个训练步骤：16 rollouts for 16 prompts
- 训练数据：NuminaMath子集
- 硬件：8x NVIDIA A100 (80GB)
- SFT: 1,107个问题 (从NuminaMath采样+DeepSeek-R1生成)
- RL: 256 steps, 每步16 rollouts x 16 prompts
- 奖励：格式(1.0) + 语言(1.0) + 准确率(1.0) = 总分3.0

**实现要点：**
- 异步实现兼容SGLang框架
- 每个prompt的访问计数递增，确保同一次迭代不重复处理
- 达到batch容量后停止活跃rollout进程

---

### 4.2 DAPO的动态采样

DAPO（Decoupled Clip and Dynamic Sampling Policy Optimization）是在线难度过滤的基线方法：

- **动态采样**：丢弃准确率为0%或100%的prompt（无梯度信号）
- **解耦裁剪**：不对称裁剪范围 ε_low ≠ ε_high
- **去除KL散度项**：鼓励探索
- ** overlong penalty**：惩罚过长响应

DAPO的过滤是SPEED-RL、LSPO、CurES等方法的基础组件。

---

### 4.3 ThinkPrune

```
Claim: ThinkPrune通过RL训练中的长度裁剪和迭代长度剪枝，将长思考LLM的思考长度减少一半，
同时仅损失2%的性能。
Source: ThinkPrune: Pruning Long Chain-of-Thought of LLMs via Reinforcement Learning
URL: https://arxiv.org/abs/2504.01296
Date: 2025-04-02 (TMLR)
Excerpt: "On AIME24, the reasoning length of DeepSeek-R1-Distill-Qwen-1.5B can be reduced 
by half with only 2% drop in performance."
Context: 现有长度减少方法主要关注强制提前退出，而非让LLM适应性地优化思考过程
Confidence: high
```

**方法核心：**
- **RL with Length Clipping**：
  - 在标准GRPO基础上添加长度限制L
  - 超过L的输出被裁剪，无法产生有效答案→零奖励
  - 无需超参数调优或奖励工程
- **迭代长度剪枝**：
  - 多轮RL训练，每轮后逐渐降低最大长度限制
  - 从较高长度预算开始，逐步收紧

**具体实现细节：**
- 模型：DeepSeek-R1-Distill-Qwen-1.5B, DeepScaleR-1.5B-Preview, QwQ-32B
- 训练数据：AIME和AMC历史问题 (2,470个)
- One-shot长度限制：4,000/3,000/2,000 tokens
- 迭代：从较高预算开始，逐轮降低

**关键发现：**
- 剪枝后LLM能绕过不必要步骤，同时保持核心推理完整
- 对未饱和模型（仅SFT）和饱和模型（经RL训练）均有效
- 相比budget-forcing方法有更好的长度-性能权衡

---

## 5. 长度感知与自适应采样

### 5.1 LSPO

```
Claim: LSPO是一种元RL-VR算法，基于每个问题的平均响应长度动态选择训练数据，
保留最短和最长响应的问题，聚焦于最可能产生有意义改进的数据。
Source: LSPO: Length-aware Dynamic Sampling for Policy Optimization in LLM Reasoning
URL: https://arxiv.org/abs/2510.01459
Date: 2025-10-01
Excerpt: "LSPO consistently improves learning effectiveness... filtering prompts whose average 
response lengths fall in the middle, retaining only those in the shortest and longest ranges."
Context: 基于"过度思考"行为分析：响应长度与输出质量之间存在强相关性
Confidence: high
```

**方法核心：**
- **长度感知动态采样**：
  - 每轮训练迭代中计算每个剩余prompt的平均响应长度
  - 过滤掉中间长度的prompt
  - 保留最短和最长范围的prompt
- **两阶段过滤**：
  1. 准确率过滤：丢弃uniformly 0或1准确率的prompt
  2. 长度过滤：基于响应长度分布保留极端值

**具体实现细节：**
- 模型：Qwen-2.5-Math-7B, Qwen3-4B-Base, Llama-3.2-4B-Instruct
- 基算法：GRPO, DAPO, GSPO
- 训练数据：DAPO-17K, MATH训练集
- 默认超参数：L_low=0.3, L_high=0.65, L_max=0.95
- Batch size: ≥256
- 每问题采样8个响应，temperature 1.0
- Mini batch size: 32, training batch: 512

**实验结果：**
- 在多个基础模型和数据集上一致提升学习效果
- 在AIME-25, Olympiad, Minerva-Math上avg@32准确率提升
- 与GRPO, DAPO, GSPO等基算法均有互补增益

---

### 5.2 DAST

```
Claim: DAST通过Token Length Budget (TLB) 度量实现难度自适应慢思考，
使模型能根据问题难度自主调整CoT长度——简单问题简洁回答，复杂问题深度推理。
Source: DAST: Difficulty-Adaptive Slow-Thinking for Large Reasoning Models
URL: https://aclanthology.org/2025.emnlp-industry.160/
Date: EMNLP 2025 Industry Track
Excerpt: "DAST effectively mitigates overthinking, substantially lowering costs and latency—while 
crucially preserving high accuracy on complex problems."
Context: 解决慢思考模型在简单问题上"过度思考"导致的资源浪费
Confidence: high
```

**方法核心：**
- **Token Length Budget (TLB)**：
  - L_budget = p·L_r̄ + (1-p)·L_max
  - p = 采样准确率（正确响应数/总采样数）
  - 高准确率→接近平均正确长度；低准确率→接近最大长度
- **Budget-aware偏好优化**：
  - 比较实际token长度与TLB校准奖励分数
  - 基于校准奖励构建成对偏好训练数据
  - 使用SimPO微调原始推理模型

**具体实现细节：**
- 模型：DeepSeek-R1-Distill-Qwen-7B, 32B
- Benchmark：MATH-500, AIME 2024, GPQA
- 对比baseline：CCoT, Chain of Draft, SFTShortest, SimPOShortest

**关键发现：**
- TLB与问题难度呈强正相关
- 有效缓解过度思考，同时保留复杂问题上的高准确率

---

## 6. 数据合成方法

### 6.1 DeepScaleR

```
Claim: DeepScaleR提出迭代式增长策略（iterative lengthening），将上下文长度从8K逐步增加到24K，
训练DeepSeek-R1-Distill-Qwen-1.5B模型超越OpenAI o1-preview，仅需3,800 A100 GPU小时
（相比原版70,000小时）。
Source: DeepScaleR: Surpassing o1-preview with a 1.5B Model by Scaling RL
URL: https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL
Date: 2025
Excerpt: "Iteratively increases the context length from 8K to 24K to train the 
DeepSeek-R1-Distill-Qwen-1.5B model toward more concise reasoning, outperforming OpenAI's o1-preview."
Context: 首个证明小模型(1.5B)通过RL scaling可以超越大模型推理能力的工作
Confidence: high
```

**方法核心：**
- **迭代式增长策略**：
  - Stage 1: 8K上下文 → 约1,000 steps → 响应长度开始增加
  - Stage 2: 16K上下文 → 约500 steps → 性能改善平台期
  - Stage 3: 24K上下文 → 继续训练
- **训练数据**：约40,000个高质量数学问题（AIME, AMC, Omni-Math, STILL）
- **目标**：引导模型更有效利用上下文，提高长CoT推理质量

**具体实现细节：**
- 基础模型：DeepSeek-R1-Distill-Qwen-1.5B
- 训练成本：3,800 A100 GPU hours（vs. 原版70,000小时）
- 训练数据规模：~40k
- 上下文长度：8K → 16K → 24K

**关键洞察（FastCuRL的观察）：**
- 8K时约42-45%输出被截断
- 24K时熵崩溃，探索能力下降
- 这些观察直接启发了FastCuRL的设计

---

### 6.2 Self-Instruct

```
Claim: Self-Instruct是一种半自动化框架，利用模型自身生成能力进行指令调优，
仅需175个种子任务即可生成52K指令，使GPT3在Super-NaturalInstructions上提升33%。
Source: Self-Instruct: Aligning Language Models with Self-Generated Instructions
URL: https://arxiv.org/abs/2212.10560
Date: ACL 2023
Excerpt: "Self-Instruct can boost GPT3 performance by a large margin (+33.1%) and nearly 
matches the performance of InstructGPT_001."
Context: 几乎无需标注的方法，使用模型自身生成来对齐预训练语言模型
Confidence: high
```

**方法核心：**
- **迭代自举算法**：
  1. 从种子任务池中随机采样任务
  2. 提示模型生成新指令和对应实例
  3. 使用启发式规则过滤低质量或重复指令
  4. 将有效任务添加回任务池
  5. 重复直到达到目标数量
- **三阶段过滤**：
  - 指令级：过滤与种子集ROUGE-L相似度高的
  - 输入/输出级：确保格式有效性
  - 多样性保证：使用多种模板编码

**实验数据：**
- 种子任务：175个手写任务
- 生成指令：~52K
- 实例对：~82K (输入+输出)
- GPT3在SuperNI上：6.8% → 39.9% (+33.1%)
- 接近InstructGPT_001 (40.8%)

---

### 6.3 ULTRAFEEDBACK

```
Claim: ULTRAFEEDBACK是一个大规模、高质量、多样化的AI反馈数据集（100万+ GPT-4反馈），
通过精心设计的注释流程减轻标注偏差，建立了可扩展的偏好数据构建流水线。
Source: UltraFeedback: Boosting Language Models with Scaled AI Feedback
URL: https://arxiv.org/abs/2310.01377
Date: 2023-10-02 (v2: 2024-07-16)
Excerpt: "A large-scale, high-quality, and diversified AI feedback dataset, which contains 
over 1 million GPT-4 feedback for 250k user-assistant conversations."
Context: 探索超越人类反馈的AI反馈自动收集，规模和多样性是关键
Confidence: high
```

**方法核心：**
- **规模与多样性**：
  - 25万用户-助手对话
  - 100万+ GPT-4反馈
  - 覆盖多个领域和任务类型
- **多模型响应生成**：
  - 使用多个源模型生成对比响应
  - 包括不同能力水平的模型
- **细致注释指令**：
  - 数值评分 + 文本反馈
  - 多种质量维度（正确性、有用性、安全性等）
- **偏差缓解技术**：
  - 位置偏差处理
  - 长度偏差处理
  - 模型身份隐藏

**衍生模型：**
- UltraRM：奖励模型
- UltraLM-13B-PPO：对话语言模型
- UltraCM：评论模型

---

### 6.4 AlpaGasus

```
Claim: AlpaGasus使用强大的LLM（如ChatGPT）自动评估和过滤Alpaca数据集中的低质量样本，
从52K中选出仅1K或9K高质量样本进行训练，效果优于使用全部数据。
Source: AlpaGasus: Training a Better Alpaca with Fewer Data
URL: (Referenced in multiple papers, including https://arxiv.org/html/2402.04833)
Date: 2023-2024
Excerpt: "AlpaGasus-1k/9k contains 1k/9k high-quality examples filtered from the original 
Alpaca-52k dataset using strong LLMs to automatically detect and filter out low-quality data."
Context: 数据质量优于数据数量的理念，使用强模型作为质量评估器
Confidence: high
```

**方法核心：**
- **LLM-based质量过滤**：
  - 使用ChatGPT等强模型评估每个样本质量
  - 自动检测低质量样本
  - 保留高质量子集
- **小数据高效训练**：
  - 1K和9K两个版本
  - 优于原始52K数据集

**对比实验（vs. Length-based选择）：**
- Alpaca-1k-longest（基于长度选择）在多个评估集上优于AlpaGasus-1k和AlpaGasus-9k
- 表明响应长度是更强的指令微调数据选择标准

---

## 7. 方法对比与分类

### 7.1 数据筛选方法对比表

| 方法 | 类别 | 筛选信号 | 在线/离线 | 时间开销 | 训练加速 | 数据减少 | 来源 |
|------|------|----------|-----------|----------|----------|----------|------|
| **LIMR** | 轨迹对齐 | 奖励轨迹与学习曲线对齐 | 离线 | 高（需预训练） | N/A | 1,389/8,523 (16%) | arXiv 2025 |
| **LearnAlign** | 梯度对齐 | 梯度方向+成功率可学习性 | 离线 | 中 | N/A | 1,000/7,500 (13%) | arXiv 2025 |
| **PODS** | Rollout下采样 | 奖励方差（极端保留） | 在线 | 低 | 1.7x+ | n→m (n/m=2-16) | arXiv 2025 |
| **GAIN-RL** | 模型信号 | Token隐藏状态角度集中 | 在线 | 低 | 2.5x+ | 50% | NeurIPS 2025 |
| **HIVE** | 分层选择 | Prompt熵+历史轨迹 | 在线 | 极低 | 3.8x rollout | 70% rollouts | arXiv 2026 |
| **BOTS** | 贝叶斯推断 | 任务难度后验估计 | 在线 | 极低 | 1.4-2x | 36-50% | arXiv 2025 |
| **SPEED-RL** | 难度过滤 | 经验pass rate (≈0.5) | 在线 | 中 | 2-6x | 动态 | ICMLW 2025 |
| **CurES** | 梯度优化 | 贝叶斯后验估计 | 在线 | 低 | 更快收敛 | 动态 | arXiv 2025 |
| **Online Difficulty Filtering** | 难度过滤 | 样本成功率 | 在线 | 中 | 中等 | 动态 | arXiv 2025 |

### 7.2 课程学习策略分类

| 策略类型 | 代表方法 | 难度度量 | 课程调整频率 | 核心思想 |
|----------|----------|----------|--------------|----------|
| **上下文长度课程** | FastCuRL, DeepScaleR | 输入长度 | 阶段性（8K→16K→24K） | 渐进扩展上下文窗口 |
| **Rollout数量课程** | CurES, GVM | 估计准确率 | 每步动态 | 根据难度分配rollout预算 |
| **Prompt采样课程** | SPEED-RL, BOTS, HIVE | 经验pass rate | 每步动态 | 选择中等难度样本 |
| **任务排序课程** | HAMMER | 语义多样性 | 一次性 | Hamiltonian路径最大化多样性 |
| **自适应难度课程** | AdaRFT, CLPO | 预计算/实时难度 | 每步动态 | 奖励反馈调整目标难度 |
| **长度感知课程** | LSPO, DAST | 响应长度 | 每步动态 | 基于响应长度过滤 |
| **迭代剪枝课程** | ThinkPrune | 长度限制 | 多轮迭代 | 逐轮收紧长度限制 |
| **梯度分析课程** | CurES, GAIN-RL | 梯度SNR/方差 | 每步动态 | 优化梯度信号质量 |

### 7.3 效率提升数据汇总

| 方法 | 模型规模 | 训练加速 | 数据减少 | 性能影响 | Benchmark |
|------|----------|----------|----------|----------|-----------|
| **FastCuRL** | 1.5B | 2x (50% steps) | N/A | 超越DeepScaleR | AIME 49.6% |
| **SPEED-RL** | 1.5B/7B | 2-6x | 动态 | 匹配/超越 | DAPO-17k, NuminaMath |
| **CurES** | 1.5B/7B | 更快收敛 | 动态 | +3.30/+4.82 vs GRPO | 8 math benchmarks |
| **LIMR** | 7B | N/A | 84%→16% | +16.7% AIME24 | AIME, MATH500 |
| **LearnAlign** | 1.5B-7B | N/A | ~87%→13% | 77.5% vs 77.0% | GSM8K |
| **PODS** | 多种 | 1.7x+ wall-clock | 2-16x rollout↓ | 更高最终准确率 | GSM8K |
| **GAIN-RL** | 0.5B-7B | 2-3.3x | 50% | +4.72% GSM8K | GSM8K, MATH, AMC |
| **HIVE** | 1.5B-32B | 3.8x rollout, 2.2x total | 70% rollouts | 59.7% vs 57.8% | 6 math benchmarks |
| **BOTS** | 1.5B/7B | 1.4-2x | 36-50% | 8/18 第一 | Math, Code, Logic |
| **AdaRFT** | 多种 | 更快收敛 | 自适应 | 匹配手动课程 | Math reasoning |
| **HAMMER** | 多种 | 更高样本效率 | N/A | +3-4% accuracy | Diverse benchmarks |
| **ThinkPrune** | 1.5B-32B | N/A | 50%长度↓ | -2% AIME24 | AIME24 |
| **LSPO** | 4B-7B | 稳定训练 | 30-40% middle | 一致提升 | AIME, Olympiad |

---

## 8. 关键洞察与趋势

### 8.1 核心发现

1. **中等难度样本最关键**
   - 几乎所有方法（SPEED-RL, CurES, BOTS, Online Difficulty Filtering）都发现pass rate ≈ 0.5的样本提供最高学习信号
   - 太容易→无梯度信号；太难→无法学习
   - "学习边缘"（learning edge）概念被HIVE、GAIN-RL等方法反复验证

2. **在线自适应优于静态课程**
   - FastCuRL和DeepScaleR的阶段性上下文扩展是有效的但较粗糙
   - AdaRFT、CLPO、CurES等的动态自适应课程表现更好
   - 关键挑战：实时难度估计的计算开销

3. **数据规模远非决定性**
   - LIMR: 16%数据 > 100%数据
   - LearnAlign: 13%数据 > 100%数据
   - HIVE: 30%数据达到100%数据效果
   - "Less is More"成为共识

4. **模型自身信号极具价值**
   - GAIN-RL: token隐藏状态角度集中
   - HIVE: prompt熵作为效用代理
   - CurES: 梯度分析指导样本选择
   - CLPO: rollout表现作为难度指标

5. **Rollout效率有巨大提升空间**
   - PODS: 并非所有rollout有用
   - HIVE: 70% rollouts可省
   - SPEED-RL: 两阶段筛选节省大量计算

### 8.2 演进关系

```
DeepScaleR (2025.01)
    ├── 上下文长度迭代扩展
    └── 问题：8K截断+24K熵崩溃
            ↓
    FastCuRL (2025.03) ──课程化上下文缩放
            ↓
    SPEED-RL (2025.06) ──在线难度过滤
    PODS (2025.04) ──────Rollout下采样
    LIMR (2025.02) ──────数据轨迹对齐
            ↓
    CurES (2025.10) ─────梯度分析+贝叶斯
    CLPO (2025.09) ──────自适应问题重构
    GAIN-RL (2025.06) ───角度信号驱动
    HIVE (2026.03) ──────在线验证分层
    BOTS (2025.10) ──────贝叶斯任务选择
```

### 8.3 开放问题

1. **跨领域泛化**：当前方法主要在数学推理上验证，代码、通用推理等领域的效果有待验证
2. **大规模模型验证**：大部分工作在7B以下模型验证，更大模型(32B+)上的效果未知
3. **多目标优化**：难度、多样性、长度、领域覆盖等多维度联合优化
4. **理论理解**：课程学习的理论保证仍然有限，特别是非凸优化设置下
5. **组合策略**：不同策略（课程+剪枝+合成）的最优组合方式尚未探索

---

*报告生成时间：2025年7月*
*覆盖论文：20+篇，主要来自NeurIPS 2025, ICML 2025, ACL 2025, EMNLP 2025, ICLR 2025, 以及arXiv 2024-2025预印本*
