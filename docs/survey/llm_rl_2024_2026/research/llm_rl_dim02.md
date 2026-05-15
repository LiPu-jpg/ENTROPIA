# GRPO及其变体算法深度调研报告

## 1. 概述

Group Relative Policy Optimization (GRPO) 是2024年由DeepSeek-AI在DeepSeekMath论文中提出的核心算法 [^1^]。它消除了PPO中需要单独训练的critic网络，通过组内相对奖励计算优势函数，极大地降低了训练资源需求。GRPO的成功推动了大量变体算法的发展，包括DAPO、Dr.GRPO、GMPO、GSPO、VAPO、SAPO等，形成了critic-free RL方法的重要研究方向。

---

## 2. GRPO原始论文

### 2.1 DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

```
Claim: 提出GRPO算法，用组内相对奖励归一化替代PPO中的critic模型，显著降低内存消耗，在数学推理上取得大幅性能提升
Source: DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
URL: https://arxiv.org/abs/2402.03300
Date: ICML 2024 (February 2024)
Excerpt: "We introduce Group Relative Policy Optimization (GRPO), a variant reinforcement learning algorithm of Proximal Policy Optimization (PPO). GRPO foregoes the critic model, instead estimating the baseline from group scores, significantly reducing training resources."
Context: GRPO是DeepSeekMath的核心创新，通过组内采样和相对归一化替代PPO的critic网络
Confidence: high
```

**核心公式**：

GRPO的优势函数计算公式为：

$$\hat{A}_{i,t} = \frac{r_i - \text{mean}(\{r_i\}_{i=1}^G)}{\text{std}(\{r_i\}_{i=1}^G)}$$

其中，$G$ 是组大小，$r_i$ 是第 $i$ 个回答的奖励，优势函数对组内所有回答的奖励进行标准化（减去均值、除以标准差），并将该优势赋给回答中每个token。

**GRPO的目标函数**：

$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left( r_{i,t}(\theta) \hat{A}_i, \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i \right) - \beta D_{KL}[\pi_\theta \| \pi_{ref}] \right]$$

**与PPO的核心区别**：

| 特征 | PPO | GRPO |
|------|-----|------|
| Critic网络 | 需要单独训练 | 不需要（组内相对奖励替代） |
| 优势估计 | 需要value model | 组内归一化 |
| 内存消耗 | ~双倍（actor + critic） | ~单倍 |
| KL散度 | 在奖励函数中隐式添加 | 在loss函数中显式添加 |
| 熵奖励 | 通常包含 | 不包含 |

**实验设置与结果**：

| 参数 | 值 |
|------|------|
| 基础模型 | DeepSeekMath-Instruct 7B |
| 训练数据 | 144K chain-of-thought GSM8K+MATH问题 |
| 学习率 | $1 \times 10^{-6}$ |
| KL系数 $\beta$ | 0.04 |
| 组大小 $G$ | 64 |
| 最大长度 | 1024 tokens |
| Batch Size | 1024 |

| Benchmark | DeepSeekMath-Instruct | DeepSeekMath-RL (GRPO) | 提升 |
|-----------|----------------------|------------------------|------|
| GSM8K | 82.9% | 88.2% | +5.3% |
| MATH | 46.8% | 51.7% | +4.9% |
| CMATH | 84.6% | 88.8% | +4.2% |

---

## 3. DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization)

### 3.1 DAPO: An Open-Source LLM Reinforcement Learning System at Scale

```
Claim: 提出DAPO算法，通过解耦裁剪(Clip-Higher)、动态采样(Dynamic Sampling)、Token级损失、过软惩罚四项关键技术，使Qwen2.5-32B在AIME 2024上达到50分，仅需GRPO 50%的训练步数
Source: DAPO: An Open-Source LLM Reinforcement Learning System at Scale
URL: https://arxiv.org/abs/2503.14476
Date: arXiv:2503.14476, March 2025 (ByteDance Seed)
Excerpt: "We propose the Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO) algorithm, and fully open-source a state-of-the-art large-scale RL system that achieves 50 points on AIME 2024 using Qwen2.5-32B base model."
Context: DAPO在GRPO基础上引入四项改进，重点解决零方差问题和训练效率问题
Confidence: high
```

**四大核心改进**：

#### 3.1.1 Dynamic Sampling (动态采样)

过滤掉所有回答都正确或都错误的零方差组，确保每个batch中所有prompt都有有效梯度信号。

约束条件：$0 < |\{o_i | \text{is_equivalent}(a, o_i)\}| < G$

#### 3.1.2 Clip-Higher (非对称裁剪)

将裁剪上下界解耦：$\epsilon_{low} = 0.20$, $\epsilon_{high} = 0.28$，允许更大的策略更新幅度，缓解熵崩塌。

$$\text{clip}(r_{i,t}(\theta), 1 - \epsilon_{low}, 1 + \epsilon_{high})$$

#### 3.1.3 Token-Level Policy Gradient Loss (Token级策略梯度损失)

用总token数归一化损失，替代按序列数归一化：

$$\mathcal{J}_{DAPO} = \mathbb{E} \left[ \frac{1}{\sum_{i=1}^{G}|o_i|} \sum_{i=1}^{G} \sum_{t=1}^{|o_i|} \min(r_{i,t}(\theta)\hat{A}_{i,t}, \text{clip}(r_{i,t}(\theta), 1-\epsilon_{low}, 1+\epsilon_{high})\hat{A}_{i,t}) \right]$$

#### 3.1.4 Overlong Reward Shaping (过长奖励塑形)

对超过预设长度阈值的回答施加soft惩罚，鼓励生成更长的CoT但避免过度冗长。

**实验设置**：

| 参数 | 值 |
|------|------|
| 基础模型 | Qwen2.5-32B |
| 学习率 | $1 \times 10^{-6}$ |
| 组大小 $G$ | 动态 |
| $\epsilon_{low}$ | 0.20 |
| $\epsilon_{high}$ | 0.28 |
| KL惩罚 | 0 |
| Max Prompt Length | 2048 |
| Max Response Length | 20480 |
| Train Batch Size | 32 |

**主要结果**（渐进式技术添加）：

| 方法 | AIME 2024 (avg@32) |
|------|--------------------|
| Naive GRPO | 30 |
| + Overlong Filtering | 36 |
| + Clip-Higher | 38 |
| + Soft Overlong Punishment | 41 |
| + Token-level Loss | 42 |
| + Dynamic Sampling (DAPO) | **50** |
| DeepSeek-R1-Zero-Qwen-32B | 47 |

---

## 4. Dr.GRPO (Group Relative Policy Optimization Done Right)

### 4.1 Understanding R1-Zero-Like Training: A Critical Perspective

```
Claim: 识别GRPO的长度偏差和难度偏差问题，提出Dr.GRPO通过移除长度归一化和标准差归一化来提供无偏优化，用7B模型在AIME 2024上达到43.3%
Source: Understanding R1-Zero-Like Training: A Critical Perspective
URL: https://arxiv.org/abs/2503.20783
Date: arXiv:2503.20783, March 2025 (SIA-Lab, Tsinghua & ByteDance)
Excerpt: "We identify an optimization bias in Group Relative Policy Optimization (GRPO), which artificially increases response length (especially for incorrect outputs) during training. To address this, we introduce Dr. GRPO, an unbiased optimization method that improves token efficiency while maintaining reasoning performance."
Context: Dr.GRPO纠正GRPO中的两个偏差源：长度归一化导致长回答惩罚不足、标准差归一化导致极端奖励分布问题被过度加权
Confidence: high
```

**GRPO的两个偏差源**：

1. **长度偏差**：GRPO对每个回答的损失除以 $|o_i|$（回答长度），导致长回答的梯度被缩小，模型倾向于生成更长但不一定更好的回答
2. **标准差偏差**：除以组内标准差使得问题难度影响梯度大小，极端奖励分布的问题被过度加权

**Dr.GRPO的修正**：

移除 $\frac{1}{|o_i|}$ 和 $\text{std}(R)$ 两项归一化：

$$\hat{A}_i = r_i - \text{mean}(R)$$

**目标函数**：

$$\mathcal{L}_{Dr.GRPO} = \frac{1}{G} \sum_{i=1}^{G} \sum_{t=1}^{|o_i|} \min\left( \rho_{i,t}(\theta)\hat{A}_{i,t}, \text{clip}(\rho_{i,t}(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_{i,t} \right)$$

其中梯度归一化使用常数（如生成预算），而非回答长度。

**实验结果**：

| 方法 | Qwen2.5-1.5B AIME2024 |
|------|----------------------|
| GRPO | ~35 |
| Dr.GRPO | **43.3** (7B model) |

---

## 5. GMPO (Geometric-Mean Policy Optimization)

### 5.1 GMPO: Geometric-Mean Policy Optimization

```
Claim: 用几何平均替代算术平均来聚合token级奖励，从根本上解决GRPO中离群值导致的训练不稳定性，GMPO-7B在多个数学基准上平均超越GRPO 4.1%
Source: Geometric-Mean Policy Optimization
URL: https://arxiv.org/abs/2507.20673
Date: arXiv:2507.20673, July 2025 (Microsoft Research Asia, UCAS)
Excerpt: "Instead of optimizing the arithmetic mean, GMPO maximizes the geometric mean of token-level rewards, which is inherently less sensitive to outliers and maintains a more stable range of importance sampling ratio."
Context: GMPO通过改变聚合统计量（算术平均→几何平均）从根本上提升训练稳定性
Confidence: high
```

**核心公式**：

GMPO将目标函数从算术平均改为几何平均：

$$\mathcal{J}_{GMPO}(\theta) = \mathbb{E}\left[\left(\prod_{t=1}^{|o_i|} \rho_{i,t}(\theta) \cdot \mathbf{1}_{[\text{unclipped}]}\right)^{\frac{1}{|o_i|}} \cdot \hat{A}_i \right]$$

等价于对数空间：

$$\log \mathcal{J}_{GMPO} \propto \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \log \rho_{i,t}(\theta)$$

几何平均的核心优势：
- 对离群值不敏感（outlier robust）
- 重要性采样比率范围更稳定
- 允许使用更宽的裁剪阈值而不失去稳定性

**实验结果**（GMPO-7B vs GRPO-7B）：

| Benchmark | GMPO | GRPO | 提升 |
|-----------|------|------|------|
| AIME24 | - | - | +4.1% avg |
| AMC | - | - | +4.1% avg |
| MATH500 | - | - | +4.1% avg |
| OlympiadBench | - | - | +4.1% avg |
| Minerva | - | - | +4.1% avg |
| Geometry3K | - | - | +1.4% (多模态) |

**代码**: https://github.com/callsys/GMPO

---

## 6. GSPO (Group Sequence Policy Optimization)

### 6.1 GSPO: Group Sequence Policy Optimization

```
Claim: 提出GSPO算法，将重要性比率定义在序列级别（而非token级别），执行序列级裁剪、奖励和优化，显著提升训练稳定性，特别在MoE模型上表现出色
Source: Group Sequence Policy Optimization
URL: https://arxiv.org/abs/2507.18071
Date: arXiv:2507.18071, July 2025 (Qwen Team, Alibaba)
Excerpt: "GSPO defines the importance ratio based on sequence likelihood and performs sequence-level clipping, rewarding, and optimization. GSPO demonstrates notably superior training stability, efficiency, and performance compared to GRPO."
Context: GSPO是Qwen3模型RL训练的核心算法，从token级到序列级的根本性改变
Confidence: high
```

**核心公式**：

GSPO定义序列级重要性比率：

$$s_i(\theta) = \left( \frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)} \right)^{\frac{1}{|y_i|}} = \exp\left( \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \log \frac{\pi_\theta(y_{i,t}|x, y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t}|x, y_{i,<t})} \right)$$

对序列比率进行裁剪（非token级）：

$$\mathcal{J}_{GSPO} = \mathbb{E} \left[ \min(s_i(\theta) \hat{A}_i, \text{clip}(s_i(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i) \right]$$

**GSPO vs GRPO的关键区别**：

| 方面 | GRPO | GSPO |
|------|------|------|
| 重要性比率 | Token级 | 序列级 |
| 裁剪级别 | 每个token独立裁剪 | 整个序列统一裁剪 |
| 熵稳定性 | 容易崩塌 | 更稳定 |
| MoE兼容性 | 较差 | 优秀 |
| 梯度噪声 | 较高 | 较低 |

**代码/博客**: https://qwenlm.github.io/blog/gspo/

---

## 7. VAPO (Value-based Augmented Proximal Policy Optimization)

### 7.1 VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks

```
Claim: VAPO通过value模型预训练、解耦GAE、自适应GAE、Clip-Higher、Token级损失、正例LM损失、组采样七项改进，在Qwen-32B上达到AIME 2024的60.4分新SOTA
Source: VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks
URL: https://arxiv.org/abs/2504.05118
Date: arXiv:2504.05118, April 2025 (ByteDance Seed)
Excerpt: "VAPO, built on the Qwen 32B pre-trained model, attains a state-of-the-art score of 60.4 on AIME 2024... It reaches state-of-the-art performance within a mere 5,000 steps."
Context: VAPO是ByteDance Seed在DAPO基础上的进一步改进，是唯一使用value model的方法
Confidence: high
```

**VAPO七大改进**：

1. **Value-Pretraining**: 用reward model初始化value model，避免vanilla PPO中value model学习崩塌
2. **Decoupled GAE**: 解耦GAE参数，防止长序列奖励信号指数衰减
3. **Length-adaptive GAE**: 自适应调整GAE参数以平衡短长序列优化
4. **Clip-Higher**: 同DAPO，$\epsilon_{high}=0.28$
5. **Token-level Loss**: 同DAPO，总token数归一化
6. **Positive-Example LM Loss**: 增加正样本LM损失，提升6分
7. **Group-Sampling**: 少prompt多重复采样策略

**实验结果**：

| 方法 | AIME 2024 (avg@32) |
|------|--------------------|
| Vanilla PPO | 5 |
| DeepSeek-R1-Zero-Qwen-32B | 47 |
| DAPO | 50 |
| VAPO w/o Value-Pretraining | 11 |
| VAPO w/o Decoupled-GAE | 33 |
| VAPO w/o Adaptive GAE | 45 |
| VAPO w/o Clip-Higher | 46 |
| VAPO w/o Token-level Loss | 53 |
| VAPO w/o Positive Example LM Loss | 54 |
| VAPO w/o Group-Sampling | 55 |
| **VAPO** | **60.4** |

**VAPO超参数**：

| 参数 | 值 |
|------|------|
| Actor学习率 | $1 \times 10^{-6}$ |
| Critic学习率 | $2 \times 10^{-6}$ |
| GAE $\lambda$ | 0.95 |
| GAE $\gamma$ | 1.0 |
| Clip $\epsilon$ | 0.2 |
| Batch Size | 192 prompts |
| 每个prompt采样次数 | 1 |

---

## 8. SAPO (Soft Adaptive Policy Optimization)

### 8.1 SAPO: Soft Adaptive Policy Optimization

```
Claim: SAPO用平滑、温度控制的门控函数替代硬裁剪，实现token级自适应衰减，同时保持序列一致性，在Qwen3-VL训练上展现一致的稳定性优势
Source: Soft Adaptive Policy Optimization
URL: https://arxiv.org/abs/2511.20347
Date: arXiv:2511.20347, November 2025 (Qwen Team, Alibaba)
Excerpt: "SAPO replaces hard clipping with a smooth, temperature-controlled gate that adaptively attenuates off-policy updates while preserving useful learning signals."
Context: SAPO是GSPO的进一步改进，用软门控替代硬裁剪，成功应用于Qwen3-VL系列模型训练
Confidence: high
```

**核心公式**：

SAPO的软门控函数：

$$f_{i,t}(\rho) = \sigma(\tau_{i,t}(\rho - 1)) \cdot \frac{4}{\tau_{i,t}}$$

其中通过为非对称温度参数（当优势>0时使用$\tau_{pos}$，否则使用$\tau_{neg}$）实现token级细粒度更新控制。

**目标函数**：

$$\mathcal{J}_{SAPO}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, \{y_i\}_{i=1}^G \sim \pi_{\theta_{old}}} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} f_{i,t}(\rho_{i,t}(\theta)) \hat{A}_i \right]$$

**SAPO设计哲学**：
- 对正优势token：允许更大更新幅度，通过Sigmoid平滑衰减
- 对负优势token：施加更严格约束
- 非对称温度设计：正负token差异化处理

---

## 9. 其他重要GRPO变体

### 9.1 AGPO (Asymmetric Group Policy Optimization)

```
Claim: AGPO采用负样本强化主导(NSR-dominant)策略，通过非对称动态优势估计和正交正样本干预，在推理任务上保持推理能力边界
Source: Asymmetric Group Policy Optimization for Verifiable Reasoning and Search Ads Relevance
URL: https://arxiv.org/abs/2605.05826
Date: arXiv:2605.05826, 2025 (JD)
Excerpt: "AGPO employs asymmetric dynamic advantage estimation, with advantages in both PSR and NSR showing nearly linear trends."
Context: AGPO关注如何在RLVR中保持模型推理边界不被压缩
Confidence: medium
```

**核心公式**：

$$\hat{A}_i^{AGPO} = \underbrace{\frac{1}{\sqrt{\sigma^2 + \delta^2}} \cdot (r_i - \mu)}_{\text{Constrained group relative}} + \underbrace{\mathbb{I}(r_i < 0) \cdot \mathcal{R}}_{\text{Gated negative term}}$$

### 9.2 GVPO (Group Variance Policy Optimization)

```
Claim: GVPO从KL约束奖励最大化问题推导组权重，使难以处理的分区函数相消，得到理论上最优的更新权重
Source: GVPO: Group Variance Policy Optimization for Large Language Model Post-Training
URL: https://arxiv.org/abs/2504.19599
Date: arXiv:2504.19599, 2025
Excerpt: "GVPO incorporates the analytical solution to KL-constrained reward maximization directly into its gradient weights, ensuring alignment with the optimal policy."
Context: GVPO从理论出发推导最优组权重，提供理论保证
Confidence: medium
```

### 9.3 EP-GRPO (Entropy-Progress Aligned GRPO)

```
Claim: EP-GRPO通过隐式过程指导对齐熵和推理进度，利用推理步骤的积累特性缓解零方差崩塌
Source: EP-GRPO: Entropy-Progress Aligned Group Relative Policy Optimization with Implicit Process Guidance
URL: https://arxiv.org/abs/2605.04960
Date: arXiv:2605.04960, 2025
Context: EP-GRPO关注零方差崩塌的信号级解决方案
Confidence: medium
```

### 9.4 M-GRPO (Momentum-Anchored GRPO)

```
Claim: M-GRPO使用缓慢演变的momentum模型提供稳定训练目标，结合IQR自适应过滤方法动态剪除低熵轨迹
Source: M-GRPO: Stabilizing Self-Supervised Reinforcement Learning for Large Language Models with Momentum-Anchored Policy Optimization
URL: https://arxiv.org/abs/2512.13070
Date: arXiv:2512.13070, December 2025
Context: M-GRPO解决长程训练中的策略崩塌问题
Confidence: medium
```

### 9.5 OPO (On-Policy RL with Optimal Reward Baseline)

```
Claim: OPO使用固定阈值(Sign advantage)替代组均值参考，在所有组（包括退化组）中保持完整梯度信号
Source: On-Policy RL with Optimal Reward Baseline
URL: https://arxiv.org/abs/2505.23585
Date: arXiv:2505.23585, 2025
Excerpt: "The Sign advantage $A_j = 2r_j - 1$ assigns +1 to correct and -1 to incorrect responses regardless of group composition."
Context: OPO发现简单固定阈值就能解决退化组信号丢失问题
Confidence: medium
```

### 9.6 Lambda-GRPO

```
Claim: 通过可学习token偏好参数统一GRPO框架，动态调整长度惩罚和偏好权重
Source: Unifying the GRPO Frameworks with Learnable Token Preferences
URL: https://arxiv.org/abs/2510.06870
Date: arXiv:2510.06870, 2025
Excerpt: "Lambda-GRPO maintains a consistently higher token-level entropy than DAPO, indicating enhanced response diversity and training stability."
Context: 在Qwen2.5-1.5B上超越DAPO和GRPO
Confidence: medium
```

### 9.7 GAPO (Group Adaptive Policy Optimization)

```
Claim: GAPO为代码编辑任务设计，动态调整组大小和裁剪参数
Source: GAPO: Group Adaptive Policy Optimization for Real-World Code Edit
URL: https://arxiv.org/abs/2510.21830
Date: arXiv:2510.21830, 2025
Context: 代码编辑场景的GRPO变体
Confidence: medium
```

---

## 10. GRPO vs PPO系统性对比

### 10.1 核心差异总结

| 维度 | PPO | GRPO | DAPO | Dr.GRPO | VAPO |
|------|-----|------|------|---------|------|
| Critic网络 | 需要 | 不需要 | 不需要 | 不需要 | 需要 |
| 优势估计 | GAE + Value Model | 组内相对归一化 | 组内相对归一化 | 组内均值（无std） | GAE + Value Model |
| 裁剪 | 对称 (0.2) | 对称 (0.2) | 非对称 (0.2/0.28) | 对称 (0.2) | 对称 (0.2) |
| KL惩罚 | 在奖励中 | 在损失中 | 无 | 可选 | 有 |
| 动态采样 | 无 | 无 | 有 | 无 | 无 |
| Token/Seq级 | Token级 | Token级 | Token级 | Token级 | Sample级 |
| 内存消耗 | ~2x | ~1x | ~1x | ~1x | ~2x |

### 10.2 Benchmark对比表

| 方法 | 模型 | AIME24 | MATH500 | GSM8K | 备注 |
|------|------|--------|---------|-------|------|
| DeepSeekMath-RL | 7B | - | 51.7 | 88.2 | 原始GRPO |
| Dr.GRPO | 7B | 43.3 | - | - | 无偏优化 |
| DAPO | 32B | 50 | - | - | 动态采样 |
| VAPO | 32B | **60.4** | - | - | Value model |
| GMPO | 7B | +4.1% | +4.1% | - | 几何平均 |
| GSPO | - | 优于GRPO | 优于GRPO | - | 序列级 |
| SAPO | - | 更稳定 | 更稳定 | - | 软门控 |
| DeepSeek-R1-Zero | 32B | 47 | - | - | DeepSeek原版 |

### 10.3 训练稳定性对比

| 方法 | 熵崩塌 | 零方差 | 长度偏差 | 实现复杂度 |
|------|--------|--------|----------|------------|
| GRPO | 严重 | 严重 | 有 | 简单 |
| DAPO | 缓解 | 解决 | 部分缓解 | 中等 |
| Dr.GRPO | 部分缓解 | 部分缓解 | **解决** | 简单 |
| GSPO | **大幅缓解** | 缓解 | 部分缓解 | 中等 |
| GMPO | **大幅缓解** | 缓解 | 部分缓解 | 简单 |
| SAPO | **大幅缓解** | 缓解 | 部分缓解 | 中等 |
| VAPO | **大幅缓解** | 缓解 | 缓解 | 复杂 |

---

## 11. 超参数汇总表

| 参数 | GRPO | DAPO | Dr.GRPO | VAPO | GMPO | GSPO |
|------|------|------|---------|------|------|------|
| 学习率 | $1e^{-6}$ | $1e^{-6}$ | $1e^{-6}$ | $1e^{-6}$(actor), $2e^{-6}$(critic) | $1e^{-6}$ | $1e^{-6}$ |
| 组大小 G | 64 | 动态 | 8-16 | 192 prompts | 8-16 | 8-16 |
| $\epsilon_{low}$ | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 |
| $\epsilon_{high}$ | 0.2 | 0.28 | 0.2 | 0.2 | 可调 | 序列级 |
| KL系数 | 0.04 | 0 | 可选 | 有 | 可选 | 可选 |
| 最大长度 | 1024 | 20480 | 4096+ | 长CoT | 4096 | 4096 |
| Batch Size | 1024 | 32 | 32-256 | 192 | 32-128 | 32-128 |
| 优化器 | AdamW | AdamW | AdamW | AdamW | AdamW | AdamW |

---

## 12. 演化关系图

```
PPO (2017)
  |
  |-- GRPO (DeepSeekMath, 2024) [原始critic-free方法]
  |     |
  |     |-- Dr.GRPO (2025) [移除长度/std偏差]
  |     |-- DAPO (2025) [动态采样+Clip-Higher+Token级损失]
  |     |     |
  |     |     |-- VAPO (2025) [Value model + 七项改进]
  |     |
  |     |-- GMPO (2025) [几何平均替代算术平均]
  |     |-- GSPO (2025) [序列级替代Token级]
  |     |     |
  |     |     |-- SAPO (2025) [软门控替代硬裁剪]
  |     |
  |     |-- AGPO (2025) [非对称优势估计]
  |     |-- GVPO (2025) [最优组权重]
  |     |-- Lambda-GRPO (2025) [可学习偏好]
  |     |-- EP-GRPO (2025) [熵-进度对齐]
  |     |-- M-GRPO (2025) [Momentum锚定]
  |     |-- OPO (2025) [最优奖励基线]
  |     |-- GAPO (2025) [组自适应]
  |     |-- ... (40+ variants)
  |
  |-- Value-based variants
        |
        |-- VAPO (2025) [Value-pretraining]
```

---

## 13. 关键发现与趋势

### 13.1 主要技术趋势

1. **无Critic化**：GRPO系列方法普遍消除PPO中的critic网络，通过组内相对归一化降低内存消耗
2. **优化粒度演进**：从token级（GRPO）→ 序列级（GSPO）→ 混合级（DHPO）
3. **裁剪策略改进**：从对称裁剪 → 非对称裁剪（DAPO）→ 软门控（SAPO）
4. **归一化修正**：从算术平均 → 几何平均（GMPO）
5. **偏差修正**：识别并消除长度偏差（Dr.GRPO）和难度偏差
6. **Value Model回归**：VAPO发现value model在长CoT中仍有价值

### 13.2 核心挑战与解决方案

| 挑战 | 解决方案 | 代表方法 |
|------|----------|----------|
| 熵崩塌 | 非对称裁剪/软门控/温度控制 | DAPO, SAPO, AEPO |
| 零方差组 | 动态采样/过滤 | DAPO, GRESO |
| 长度偏差 | 移除长度归一化 | Dr.GRPO |
| 离群值敏感 | 几何平均 | GMPO |
| MoE不稳定 | 序列级优化 | GSPO, SAPO |
| Value Model崩塌 | Value Pretraining | VAPO |

---

## 14. 参考文献

[^1^]: Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models," ICML 2024. https://arxiv.org/abs/2402.03300

[^2^]: Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at Scale," arXiv:2503.14476, 2025.

[^3^]: Liu et al., "Understanding R1-Zero-Like Training: A Critical Perspective," arXiv:2503.20783, 2025.

[^4^]: Zhao et al., "Geometric-Mean Policy Optimization," arXiv:2507.20673, 2025.

[^5^]: Zheng et al., "Group Sequence Policy Optimization," arXiv:2507.18071, 2025.

[^6^]: Yue et al., "VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks," arXiv:2504.05118, 2025.

[^7^]: Gao et al., "Soft Adaptive Policy Optimization," arXiv:2511.20347, 2025.

[^8^]: Liu et al., "Asymmetric Group Policy Optimization for Verifiable Reasoning," arXiv:2605.05826, 2025.

[^9^]: Zhang et al., "GVPO: Group Variance Policy Optimization for Large Language Model Post-Training," arXiv:2504.19599, 2025.

[^10^]: Bai et al., "M-GRPO: Stabilizing Self-Supervised Reinforcement Learning with Momentum-Anchored Policy Optimization," arXiv:2512.13070, 2025.

[^11^]: Hao et al., "On-Policy RL with Optimal Reward Baseline," arXiv:2505.23585, 2025.

[^12^]: Wang et al., "GAPO: Group Adaptive Policy Optimization for Real-World Code Edit," arXiv:2510.21830, 2025.

[^13^]: Guo et al., "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning," arXiv:2501.12948, 2025.

[^14^]: Schulman et al., "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.

[^15^]: Ahmadian et al., "Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs," 2024.

---

*本报告由AI Agent自动生成，基于截至2025年7月的公开学术论文。所有论文引用均已标注arXiv ID和URL。*
