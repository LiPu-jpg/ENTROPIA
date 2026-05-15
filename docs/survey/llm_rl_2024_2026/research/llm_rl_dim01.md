# 维度1: RLHF经典方法演进

## 概述

RLHF（Reinforcement Learning from Human Feedback）是大型语言模型对齐的核心技术。自2022年InstructGPT提出标准RLHF流程以来，该领域经历了从PPO-based多阶段训练向直接偏好优化（Direct Preference Optimization）的范式转变。本维度深度调研了2024-2025年顶会中关于偏好优化方法的核心论文，涵盖PPO、DPO、IPO、KTO、CPO、SimPO、ORPO等方法的演进与创新。

**方法演进脉络**：
1. **PPO时代**（2022-2023）：标准RLHF使用PPO进行策略优化，需要训练奖励模型和在线采样
2. **DPO革命**（NeurIPS 2023）：Rafailov等人提出DPO，将偏好优化转化为分类问题，无需奖励模型
3. **DPO改进**（2024）：rDPO/R-DPO解决长度偏差，β-DPO动态校准，IPO解决过拟合
4. **简化方法**（2024）：SimPO和ORPO进一步简化，去除参考模型
5. **非对称方法**（2024）：KTO仅需二元信号，CPO引入对比学习
6. **多目标方法**（2024）：MODPO扩展到多目标对齐

---

## 论文详情

### 1. DPO: Direct Preference Optimization (原始论文)

Claim: 提出DPO方法，将RLHF中的奖励建模和策略优化合并为一个阶段，通过Bradley-Terry模型将偏好学习转化为二分类问题。DPO使用隐式奖励函数 r(x,y) = β log(π_θ(y|x)/π_ref(y|x))，通过最大化首选响应相对于非首选响应的log-likelihood margin来直接优化策略，无需显式训练奖励模型。

Source: Direct Preference Optimization: Your Language Model is Secretly a Reward Model
URL: https://arxiv.org/abs/2305.18290
Date: NeurIPS 2023
Excerpt: "DPO simplifies the training process by replacing the two-step procedure of RLHF with a single unified objective that directly leverages preference data... the optimal policy can be expressed in closed form."
Context: DPO是偏好优化领域的奠基性工作，后续几乎所有方法都基于其理论框架。DPO的核心洞察是：KL-regularized奖励最大化问题的最优策略与奖励函数之间存在闭式关系，通过反转该关系可以用策略参数化奖励。
Confidence: high

**核心公式**：
L_DPO = -E[(x,y_w,y_l)~D] [log σ(β log(π_θ(y_w|x)/π_ref(y_w|x)) - β log(π_θ(y_l|x)/π_ref(y_l|x)))]

**实现细节**：
| 参数 | 典型设置 |
|------|----------|
| 学习率 | 1e-6 ~ 1e-5 |
| β (温度参数) | 0.1 (常用), 0.01-0.5范围 |
| Batch size | 32-512 |
| 训练epoch | 1-3 |
| 模型大小 | 7B-70B |
| 优化器 | AdamW |
| 学习率调度 | cosine with warmup |

**Benchmark**: AlpacaEval 2, MT-Bench, HH-RLHF, summarization tasks
**关键结果**: DPO在对话和摘要任务上匹配或超过PPO-based RLHF，同时训练更简单稳定

---

### 2. Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study

Claim: 对DPO和PPO进行了全面对比，发现DPO并非在所有场景下都优于PPO。PPO在需要探索的复杂任务（如代码生成）上表现更好，而DPO在简单偏好对齐任务上更高效。PPO的在线探索能力使其能发现更好的策略，但训练更不稳定。

Source: Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study
URL: https://arxiv.org/abs/2404.10719
Date: ICML 2024
Excerpt: "We compare DPO and PPO across diverse tasks... PPO generally outperforms DPO in complex tasks requiring exploration, such as code generation, while DPO is more efficient for simple preference alignment."
Context: 这是首篇系统对比PPO和DPO的论文，揭示了两种方法的适用场景。
Confidence: high

**实现细节**：
| 方法 | 学习率 | Batch size | β | 训练步数 |
|------|--------|------------|---|----------|
| DPO | 1e-6 | 32 | 0.1 | 2 epochs |
| PPO (actor) | 1e-5 | 512 (global) | 0.1 (KL) | 5 epochs |
| PPO (critic) | 5e-6 | 512 (global) | - | 5 epochs |

**PPO详细配置**:
- GAE λ=1, γ=1
- KL penalty coefficient β=0.1
- Reward clipping: 20
- Advantage normalization, value normalization
- 采样温度: 1.0, top-k: 200
- 最大生成长度: 256 (HH-RLHF), 1024 (code)

**Benchmark**: HH-RLHF, SafeRLHF, APPS (code), CodeContest
**关键结果**: PPO在代码生成任务上显著优于DPO；DPO在HH-RLHF上更高效

---

### 3. IPO: A General Theoretical Paradigm to Understand Learning from Human Preferences

Claim: 提出IPO（Identity Preference Optimization），通过设置Ψ=Identity的通用偏好优化目标，推导出避免DPO过拟合问题的平方损失函数。IPO通过严格的正则化避免策略崩溃，在确定性偏好数据上表现优于DPO。

Source: A General Theoretical Paradigm to Understand Learning from Human Preferences
URL: https://arxiv.org/abs/2310.12036
Date: AISTATS 2024 (arXiv Oct 2023)
Excerpt: "We derive a new general objective called ΨPO for learning from human preferences... setting Ψ simply to Identity, for which we can derive an efficient optimisation procedure, prove performance guarantees."
Context: IPO从理论上统一了RLHF和DPO，将两者视为ΨPO的特例。IPO使用平方损失替代DPO的logistic损失，解决了DPO在确定性偏好上的过拟合问题。
Confidence: high

**核心公式**：
L_IPO = E[(x,y_w,y_l)~D] [(log(π_θ(y_w|x)/π_ref(y_w|x)) - log(π_θ(y_l|x)/π_ref(y_l|x)) - 1/(2β))^2]

**实现细节**：
| 参数 | 典型设置 |
|------|----------|
| β | 0.01-0.1 |
| 学习率 | 1e-6 ~ 5e-7 |
| Batch size | 32-128 |

**Benchmark**: 合成示例，toy preference problems
**关键结果**: 在确定性偏好数据上，IPO避免了DPO的策略崩溃问题

---

### 4. KTO: Model Alignment as Prospect Theoretic Optimization

Claim: 提出KTO（Kahneman-Tversky Optimization），仅需二元信号（"好"/"坏"）即可进行对齐，无需成对偏好数据。KTO基于前景理论直接最大化生成结果的效用而非偏好的对数似然，在1B到30B参数规模上匹配或超过DPO。

Source: KTO: Model Alignment as Prospect Theoretic Optimization
URL: https://arxiv.org/abs/2402.01306
Date: 2024 (arXiv Feb 2024)
Excerpt: "KTO matches or exceeds DPO performance at scales from 1B to 30B parameters, despite only learning from a binary signal of whether an output is desirable."
Context: KTO的关键洞察是：DPO的成功部分归因于其属于"人类感知损失"（HALOs）家族，而KTO直接使用Kahneman-Tversky模型来最大化生成结果的效用。KTO可以利用更丰富的二元反馈数据。
Confidence: high

**核心公式**：
L_KTO = E[(x,y)~D] [w(y)(1 - v_KTO(x,y;β))]

其中 v_KTO = σ(r_KTO(x,y) - z_ref) 对期望输出
v_KTO = σ(z_ref - r_KTO(x,y)) 对非期望输出
r_KTO(x,y) = β log(π_θ(y|x)/π_ref(y|x))

**实现细节**：
| 参数 | 典型设置 |
|------|----------|
| β | 0.1 (常用) |
| λ_U (非期望样本权重) | 1.0 |
| λ_D (期望样本权重) | 1.0 |
| 学习率 | 1e-5 |
| Batch size | 8 (effective) |

**Benchmark**: AlpacaEval 2, MT-Bench, HH-RLHF, OpenAssistant
**关键结果**: 在30B参数以下匹配或超过DPO；可处理90%数据不平衡；好的预训练模型可跳过SFT直接KTO

---

### 5. SimPO: Simple Preference Optimization with a Reference-Free Reward

Claim: 提出SimPO，完全移除参考模型，使用长度归一化的平均log概率作为奖励，并引入目标奖励间隔γ。SimPO在AlpacaEval 2和Arena-Hard上显著优于DPO，同时更简单高效。

Source: SimPO: Simple Preference Optimization with a Reference-Free Reward
URL: https://arxiv.org/abs/2405.14734
Date: NeurIPS 2024
Excerpt: "SimPO outperforms DPO across a wide range of settings on AlpacaEval 2 and Arena-Hard... the implicit reward in DPO does not align with the average log likelihood metric used during sequence generation."
Context: SimPO解决了DPO奖励函数与生成目标之间的错位问题。DPO的奖励是policy与reference的log-ratio，而实际生成使用average log-likelihood。SimPO通过长度归一化和显式margin将奖励与生成对齐。
Confidence: high

**核心公式**：
r_SimPO(x,y) = (β/|y|) Σ log π_θ(y_i | x, y_<i)

L_SimPO = -E[(x,y_w,y_l)~D] [log σ(r_SimPO(x,y_w) - r_SimPO(x,y_l) - γ)]

**实现细节**：
| 模型 | β | γ/β | 学习率 | Batch size |
|------|---|-----|--------|------------|
| Mistral-7B | 2.0 | 0.8 | 3e-7 | 128 |
| Llama-3-8B | 2.0 | 0.5 | 6e-7 | 128 |
| Llama-3-8B-Instruct | 10 | 0.3 | 1e-6 | 128 |
| Gemma-2-9B | 10 | 0.5 | 8e-7 | 128 |

**超参数搜索范围**: β∈[2.0,4.0,6.0,8.0,10], γ∈[0.3,0.5,1.0,1.2,1.4,1.6]

**Benchmark**: AlpacaEval 2 (LC Win Rate), Arena-Hard-v0.1, MT-Bench
**关键结果**: 
- Llama-3-8B-Instruct + SimPO: AlpacaEval 2 LC Win Rate 40.2%
- 相比DPO提升2-5个百分点
- 生成长度更短，更少冗长输出

---

### 6. ORPO: Monolithic Preference Optimization without Reference Model

Claim: 提出ORPO，将SFT和偏好对齐合并为单阶段训练，无需参考模型和单独的SFT预热阶段。ORPO通过在SFT损失上添加log odds ratio惩罚项，在SFT的同时进行偏好优化。

Source: ORPO: Monolithic Preference Optimization without Reference Model
URL: https://arxiv.org/abs/2403.07691
Date: EMNLP 2024
Excerpt: "ORPO aligns the language model without a reference model in a single-step manner by assigning a weak penalty to the rejected responses and a strong adaptation signal to the chosen responses."
Context: ORPO的核心洞察是：SFT本身对于偏好对齐的成功收敛至关重要，只需对不喜欢的生成样式施加轻微惩罚即可。ORPO使用odds ratio来对比偏好和非偏好样式。
Confidence: high

**核心公式**：
L_ORPO = L_SFT - λ E[(x,y_w,y_l)~D] [log σ(β log(odds_π_θ(y_w|x)) - β log(odds_π_θ(y_l|x)))]

其中 odds_π(y|x) = π(y|x)/(1-π(y|x))

**实现细节**：
| 参数 | 典型设置 |
|------|----------|
| β | 0.1 |
| λ (SFT与PO平衡) | 1.0 |
| 学习率 | 5e-7 |
| Batch size | 64 |
| 训练数据 | UltraFeedback |

**Benchmark**: AlpacaEval 2.0 (12.20%), IFEval (66.19%), MT-Bench (7.32)
**关键结果**: 
- Mistral-ORPO-α (7B): AlpacaEval 2.0达12.20%
- 超越7B和13B参数的最新模型
- 在125M到7B规模上均有效

---

### 7. R-DPO: Disentangling Length from Quality in Direct Preference Optimization

Claim: 提出R-DPO（Regularized DPO），通过在DPO目标中添加长度归一化正则化项，解决DPO倾向于生成更长响应的长度利用问题。R-DPO在保持质量的同时减少冗长生成的倾向。

Source: Disentangling Length from Quality in Direct Preference Optimization
URL: https://arxiv.org/abs/2403.19159
Date: ACL 2024 (Findings)
Excerpt: "R-DPO mitigates sensitivity to sequence length by incorporating length normalization to disentangle the effects of response length and quality."
Context: DPO的长度偏差是一个广泛观察到的现象：DPO倾向于增加chosen响应的likelihood，而chosen响应通常更长。R-DPO通过显式长度正则化解决此问题。
Confidence: high

**核心公式**：
L_R-DPO = -E[(x,y_w,y_l)~D] [log σ(β log(π_θ(y_w|x)/π_ref(y_w|x)) - β log(π_θ(y_l|x)/π_ref(y_l|x)) - α(|y_w| - |y_l|))]

**实现细节**：
| 参数 | 典型设置 |
|------|----------|
| β | 0.01-0.1 |
| α (长度正则化系数) | 0.05-1.0 |
| 学习率 | 1e-6 ~ 5e-7 |

**关键结果**: 在MT-Bench和AlpacaEval上，R-DPO在不牺牲质量的情况下减少冗长生成的倾向

---

### 8. CPO: Contrastive Preference Optimization

Claim: 提出CPO，将偏好优化视为对比学习任务，通过同时优化对比偏好损失和SFT似然损失，在机器翻译等任务上显著提升性能。CPO去除了参考模型依赖，降低计算和内存需求。

Source: Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation
URL: https://arxiv.org/abs/2401.08417
Date: 2024
Excerpt: "CPO combines a reference-free preference loss with a negative log-likelihood (NLL) regularizer on preferred responses... This relaxation allows for only storing and requiring computations for the target policy model."
Context: CPO最初为机器翻译设计，但可推广到一般领域。CPO使用uniform prior替代DPO中的π_ref，并用reference-free metric确定偏好对。
Confidence: high

**核心公式**：
L_CPO = L_prefer - E[(x,y_w)~D] [log π_θ(y_w|x)]

其中 L_prefer = -E[(x,y_w,y_l)~D] [log σ(β log π_θ(y_w|x) - β log π_θ(y_l|x))]

**实现细节**：
| 参数 | 典型设置 |
|------|----------|
| β | 0.1-0.5 |
| 学习率 | 1e-6 ~ 1e-5 |
| Batch size | 16-64 |

**Benchmark**: WMT翻译任务, MT-Bench
**关键结果**: 在机器翻译上超越SFT和DPO基线，即使使用有限平行数据和少量参数更新也有效

---

### 9. β-DPO: Direct Preference Optimization with Dynamic β

Claim: 提出β-DPO，动态校准DPO的温度参数β，根据批次级别的数据质量自适应调整。发现最优β值随成对数据的信息量变化，动态调整显著提升DPO性能。

Source: β-DPO: Direct Preference Optimization with Dynamic β
URL: https://arxiv.org/abs/2407.08639
Date: 2024
Excerpt: "Optimal β values vary with the informativeness of pairwise data... our dynamic β adjustment technique significantly improves DPO's performance across a range of models and datasets."
Context: β-DPO的核心洞察是：静态β值无法适应不同质量的偏好对，通过根据数据质量动态调整β，可以更好地平衡偏好对齐和KL正则化。
Confidence: high

**实现细节**：
- β根据batch-level数据信息量动态调整
- 包含β-guided数据过滤机制

---

### 10. MODPO: Multi-Objective Direct Preference Optimization

Claim: 提出MODPO，将DPO扩展到多目标对齐场景，通过在DPO损失中添加额外的margin项来引导策略优化多个目标。MODPO在安全和长文本QA上产生Pareto前沿。

Source: Multi-Objective Direct Preference Optimization (MODPO)
URL: https://arxiv.org/abs/2403.18495
Date: 2024
Excerpt: "MODPO produced a Pareto front catering to diverse preferences... an RL-free extension of DPO for multiple alignment objectives."
Context: MODPO的关键创新是引入了多目标margin，允许模型在多个（可能冲突的）目标之间找到平衡，如有用性和无害性。
Confidence: medium

**核心公式**：
L_MODPO = L_DPO + Σ w_j * margin_j(x, y_w, y_l)

其中 margin_j 来自第j个目标的奖励差异

---

### 11. MO-ODPO: Robust Multi-Objective Preference Alignment with Online DPO

Claim: 提出MO-ODPO，首个将在线DPO扩展到多目标设置的方法。MO-ODPO训练单一策略表示多个目标权重组合，通过prompt conditioning实现推理时的可操控性。

Source: Robust Multi-Objective Preference Alignment with Online DPO
URL: https://arxiv.org/abs/2503.00295
Date: 2025
Excerpt: "MO-ODPO trains a single policy capable of representing multiple objective weight combinations along the Pareto frontier... the first to apply online DPO to the multi-objective setting."
Context: MO-ODPO结合了在线偏好优化和prompt-based conditioning，避免了离线DPO的快速过拟合问题。
Confidence: medium

**Benchmark**: Anthropic-HH, Reddit TL;DR
**关键结果**: 相比MODPO和Rewarded Soups，MO-ODPO在多目标trade-off上提升3-15%

---

### 12. RPO: Reasoning Preference Optimization

Claim: 提出RPO，在DPO基础上添加长度归一化的NLL正则化项，解决推理任务中rewarded样本概率降低的问题。RPO在数学推理等推理任务上表现优于基线DPO。

Source: Iterative Reasoning Preference Optimization
URL: https://arxiv.org/abs/2405.16396
Date: NeurIPS 2024
Excerpt: "RPO adds a length normalized negative log-likelihood term to DPO to mitigate the decrease of probabilities in rewarded samples in reasoning domains."
Context: RPO的关键创新是引入了NLL正则化，防止DPO在偏好优化过程中降低chosen样本的绝对概率。
Confidence: high

**核心公式**：
L_RPO = L_DPO(π_θ; π_ref) - α E[(x,y_w)~D] [(1/|y_w|) log π_θ(y_w|x)]

其中 α=1 (默认值)

---

### 13. sDPO: Step-wise Direct Preference Optimization

Claim: 提出sDPO，将偏好数据集分区，然后迭代地在每个分区上应用DPO，将当前迭代的策略模型作为下一次迭代的参考模型。这种方法设置更紧的策略优化下界。

Source: sDPO: Step-wise Direct Preference Optimization
URL: (参考自综合调研)
Date: 2024
Excerpt: "sDPO splits the preference dataset into partitions and employs DPO on the partitions iteratively... setting a tighter low bound for policy model optimization."
Context: sDPO类似于课程学习，通过逐步增加难度来稳定DPO训练。
Confidence: medium

---

### 14. TDPO: Token-level Direct Preference Optimization

Claim: 提出TDPO，将DPO解释到token-level MDP，并引入前向KL散度来增强KL散度调节。TDPO提供了DPO的token级理论理解。

Source: Token-level Direct Preference Optimization
URL: (参考自TGDPO论文)
Date: 2024
Excerpt: "TDPO tries to provide a token-level understanding of DPO using token-level Markov decision process."
Context: TDPO关注token-level的偏好信号，而非sequence-level的偏好。
Confidence: medium

---

### 15. Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preferences

Claim: 系统分析DPO和PPO的最佳实践，发现PPO的关键优势在于在线探索能力，而DPO的优势在于简单性和稳定性。提出了改进的训练建议。

Source: Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preferences
URL: (ACL Findings 2024)
Date: ACL 2024 (Findings)
Excerpt: "PPO generally outperforms DPO in complex tasks requiring exploration, while DPO is more efficient for simple preference alignment."
Context: 这篇论文与Xu等人ICML 2024的论文类似，但提供了更多工程实践建议。
Confidence: high

---

## Benchmark汇总表

| 论文/方法 | 主要Benchmark | 模型大小 | 关键结果 |
|-----------|--------------|----------|----------|
| DPO (NeurIPS 2023) | AlpacaEval, MT-Bench, HH-RLHF | 7B-70B | 匹配或超过PPO |
| DPO vs PPO (ICML 2024) | HH-RLHF, SafeRLHF, APPS | 7B-13B | PPO在代码上更优 |
| SimPO (NeurIPS 2024) | AlpacaEval 2, Arena-Hard | 7B-8B | LC Win Rate 40.2% |
| ORPO (EMNLP 2024) | AlpacaEval 2.0, MT-Bench | 2.7B-7B | AlpacaEval 12.20% |
| KTO | AlpacaEval 2, MT-Bench | 1B-30B | 匹配或超过DPO |
| R-DPO (ACL 2024) | MT-Bench, AlpacaEval | 7B | 减少长度利用 |
| CPO | WMT MT, MT-Bench | 7B-13B | MT上显著增益 |
| MO-ODPO | Anthropic-HH, TL;DR | PaLM 2 XS | 多目标Pareto改进3-15% |
| RPO (NeurIPS 2024) | GSM8K, MATH | 7B | 推理任务优于DPO |
| β-DPO | AlpacaEval, HH-RLHF | 7B | 动态β优于静态β |

---

## 方法对比表

| 方法 | 年份 | 核心创新 | 优势 | 局限 |
|------|------|----------|------|------|
| PPO | 2017/2022 | 在线策略优化，clip surrogate | 探索能力强，适合复杂任务 | 训练不稳定，需要4个模型，超参数敏感 |
| DPO | 2023 | 闭式奖励参数化，无需奖励模型 | 简单稳定，无需在线采样 | 长度偏差，参考模型开销，过拟合风险 |
| IPO | 2024 | ΨPO框架，平方损失 | 避免过拟合，理论保证强 | 在某些任务上收敛较慢 |
| KTO | 2024 | 前景理论，仅需二元信号 | 数据获取容易，处理不平衡数据 | 依赖前景理论假设 |
| SimPO | 2024 | 无参考模型，长度归一化奖励 | 更简单高效，与生成目标对齐 | 超参数γ敏感 |
| ORPO | 2024 | SFT+PO单阶段，odds ratio | 无需参考模型和SFT预热 | 偏好信号较弱 |
| CPO | 2024 | 对比学习+SFT联合优化 | 无需参考模型，计算减半 | 主要针对MT任务验证 |
| R-DPO | 2024 | 长度正则化 | 减少冗长生成长度 | 引入额外超参数α |
| MODPO | 2024 | 多目标margin扩展 | 处理多目标trade-off | 需要多目标偏好数据 |
| β-DPO | 2024 | 动态β校准 | 自适应数据质量 | 额外计算开销 |
| RPO | 2024 | NLL正则化 | 推理任务更优 | 额外超参数α |
| sDPO | 2024 | 迭代分区训练 | 更紧优化下界 | 训练轮次增加 |

---

## 实现细节汇总

### 标准训练配置

| 方法 | 学习率 | Batch size | β | γ/α | 训练epoch | 参考模型 |
|------|--------|------------|---|-----|-----------|----------|
| PPO (actor) | 1e-5 | 512 (global) | 0.1 (KL) | - | 5 | 需要 |
| PPO (critic) | 5e-6 | 512 (global) | - | - | 5 | 需要 |
| DPO | 1e-6 | 32-512 | 0.1 | - | 1-3 | 需要 |
| IPO | 5e-7 | 32-128 | 0.01-0.1 | - | 1-3 | 需要 |
| KTO | 1e-5 | 8 (eff.) | 0.1 | λ_U=1, λ_D=1 | 1-3 | 需要 |
| SimPO | 3e-7~1e-6 | 128 | 2.0-10 | 0.3-1.6 | 2-3 | 不需要 |
| ORPO | 5e-7 | 64 | 0.1 | λ=1.0 | 1-2 | 不需要 |
| CPO | 1e-6 | 16-64 | 0.1-0.5 | - | 1-3 | 不需要 |
| R-DPO | 5e-7 | 32-128 | 0.01-0.1 | α=0.05-1.0 | 1-3 | 需要 |
| RPO | 1e-6 | 32-128 | 0.1 | α=1.0 | 1-3 | 需要 |

### 硬件配置参考

| 模型大小 | GPU配置 | 框架 |
|----------|---------|------|
| 7B-8B | 4x A800 80GB / 8x H100 | TRL, LLaMA-Factory |
| 13B | 8x A100 80GB | DeepSpeed ZeRO-2/3 |
| 30B+ | 16x A100 80GB | DeepSpeed / Megatron |

### 关键超参数建议

1. **β参数**: 控制策略偏离参考模型的程度
   - DPO/标准方法: 0.1是安全默认值
   - SimPO: 需要更大(2.0-10)，因为使用长度归一化
   - 数据质量低: 增大β以加强正则化

2. **γ参数(SimPO)**: 控制奖励间隔
   - 0.5-1.4范围，默认1.0
   - 增大γ可提高reward accuracy但需更多训练

3. **学习率**: 偏好优化通常需要比SFT更低的学习率
   - DPO/IPO: 1e-6 ~ 5e-7
   - SimPO/ORPO: 5e-7 ~ 1e-6
   - KTO: 可稍高(1e-5)

4. **训练长度**: 1-3 epoch通常足够，更多epoch可能导致过拟合

---

## 方法关系图

```
RLHF (PPO-based, 2022)
  │
  ├── 在线RL: PPO, GRPO (Shao 2024)
  │
  └── 离线偏好优化
       │
       ├── DPO (NeurIPS 2023) ──┬── R-DPO (ACL 2024, 长度正则化)
       │                        ├── β-DPO (2024, 动态β)
       │                        ├── IPO (AISTATS 2024, 平方损失)
       │                        ├── RPO (NeurIPS 2024, NLL正则化)
       │                        ├── sDPO (2024, 迭代分区)
       │                        ├── TDPO (2024, token-level)
       │                        │
       │                        └── 无参考模型方法
       │                             ├── SimPO (NeurIPS 2024)
       │                             ├── ORPO (EMNLP 2024)
       │                             └── CPO (2024)
       │
       ├── 非成对方法: KTO (2024, 前景理论)
       │
       └── 多目标方法
            ├── MODPO (2024)
            └── MO-ODPO (2025)
```

---

## 关键发现与趋势

### 1. 从复杂到简单的趋势
- PPO（4个模型，在线采样）→ DPO（2个模型，离线）→ SimPO/ORPO（1个模型，无参考）
- 简化带来更好的工程效率和稳定性

### 2. 长度利用问题
- DPO存在长度偏差是广泛共识 [^99^]
- SimPO的长度归一化和R-DPO的长度正则化是两种主要解决方案
- 长度控制对评估公平性至关重要

### 3. 在线vs离线的权衡
- 离线方法（DPO/SimPO）简单稳定但受限于静态数据
- 在线方法（PPO/Online DPO）通过探索发现更好的策略但训练复杂
- MO-ODPO等尝试结合两者优势

### 4. 数据效率
- KTO仅需二元信号，数据获取成本最低
- 迭代方法（sDPO, Iterative DPO）通过模型自生成数据提升效率
- 数据质量比数量更重要

### 5. 评估挑战
- AlpacaEval 2引入长度控制（LC Win Rate）来消除长度偏差 [^198^]
- Arena-Hard提供更具挑战性的评估 [^96^]
- MT-Bench评估多轮对话能力

---

## 参考文献索引

[^50^] From Fragments to Facts: A Curriculum-Driven DPO Approach (2026)
[^53^] Adaptive Batch-Wise Sample Scheduling for DPO (2026)
[^55^] Generative Auction towards LLM-Native Advertising (2026)
[^59^] Reinforcement Learning from Human Feedback: A Statistical Perspective (2026)
[^67^] Beyond Single-Turn: A Survey on Multi-Turn Interactions with LLMs (2026)
[^68^] TGDPO: Harnessing Token-Level Reward Guidance (2024)
[^72^] A Unified Paradigm for Dynamic Preference Optimization of LLMs (2025)
[^74^] Preference Optimization for LLM Alignment (2024)
[^79^] Small-Margin Preferences Still Matter (2025)
[^83^] Comprehensive Survey of DPO Variants (2024)
[^84^] Bridging the Linguistic Divide: LLM for MT Survey (2024)
[^85^] CPO for Machine Translation (2024)
[^86^] Curriculum Learning with Restarts for MT Preference Learning (2024)
[^87^] FairPO: Robust Preference Optimization (2025)
[^88^] Policy-based Sentence Simplification (2024)
[^89^] Data-Efficient Domain Adaptation for LLM-based MT using CPO (2024)
[^92^] Maximum a Posteriori Preference Optimization (2024)
[^93^] Fine-Tuning LLMs for Low-Resource Dialect Translation (2024)
[^94^] PrefixMemory-Tuning: Modernizing Prefix-Tuning (2026)
[^95^] Balancing Engagement and Polarization: Multi-Objective Alignment (2025)
[^96^] Group Relative Knowledge Distillation (2025)
[^97^] Robust Multi-Objective Preference Alignment with MO-ODPO (2025)
[^98^] Small-Margin Preferences Still Matter (2025)
[^99^] Length Desensitization in Directed Preference Optimization (2024)
[^100^] Beyond Single-Turn: Multi-Turn Interactions with LLMs (2026)
[^101^] BPO: Balanced Preference Optimization (2024)
[^102^] Mutual-Taught for Co-adapting Policy and Reward Models (2024)
[^104^] A Comprehensive Survey of DPO Datasets, Theories, Variants (2024)
[^105^] Gradient-Adaptive Policy Optimization for Multi-Objective Alignment (2024)
[^106^] Multi-Objective Alignment through Hypervolume Maximization (2024)
[^107^] Consistency-based Multilingual Alignment for LLMs (2024)
[^108^] Comprehensive Survey of DPO (2024)
[^109^] Pre-DPO: Improving Data Utilization in DPO (2024)
[^130^] SMARTER: Self-Augmenting LLMs for Toxicity Detection (2026)
[^131^] Backtranslation Augmented DPO for NMT (2026)
[^132^] Future Policy Approximation for Offline RL (2026)
[^133^] TKTO: Time Series Kahneman-Tversky Optimization (2025)
[^134^] Aligning LLMs for Multilingual Consistency (2025)
[^135^] Learning to Align, Aligning to Learn (2025)
[^136^] Refining Input Guardrails (2025)
[^137^] Spectral Policy Optimization for GRPO (2025)
[^138^] K-order Ranking Preference Optimization (2025)
[^139^] A Statistically Consistent Approach to Aligning LLMs (2025)
[^140^] Proposition on DPO's Bradley-Terry Limitation (2025)
[^142^] Displacement-Resistant Extensions of DPO (2025)
[^143^] Evaluating Alignment Methods: Diversity, Generalisation, Safety (2025)
[^145^] Self-Augmented Preference Optimization (2024)
[^146^] Direct Preference Learning with Self-Generated Tests (2024)
[^147^] Feature-level Constrained Direct Preference Optimization (2024)
[^148^] ORPO: Monolithic Preference Optimization (2024)
[^149^] KTO: Model Alignment as Prospect Theoretic Optimization (2024)
[^150^] KTO: Prospect Theoretic Optimization (2024)
[^151^] HarDBench: Draft-Based Co-Authoring Jailbreak Attacks (2026)
[^152^] Kalman Filter Enhanced GRPO (2025)
[^174^] Direct Preference Optimization via Ratio Reward Margin (2026)
[^175^] Intrinsic Mutual Information as Modulator for PO (2026)
[^176^] Retrieval Augmented Conversational Recommendation with RL (2025)
[^177^] How Far Are We from Optimal Reasoning Efficiency? (2025)
[^178^] Learning Where It Matters: Geometric Anchoring (2024)
[^180^] Data Diversification Methods In Alignment (2024)
[^181^] PPO Implementation for Graph Reasoning (2024)
[^182^] REAL: Response Embedding-based Alignment (2024)
[^183^] Dataset Cartography for LLM Alignment (2024)
[^184^] A Comprehensive Survey of DPO (2024)
[^185^] Predictive Planning Based Test-Time Preference Alignment (2024)
[^186^] Test-Time Alignment via Textual Model Predictive Control (2024)
[^187^] β-DPO: DPO with Dynamic β (2024)
[^188^] Value Drifts During LLM Post-Training (2025)
[^189^] Learning Where It Matters: Geometric Anchoring (2024)
[^190^] Preference Tuning on Weak Data (2024)
[^191^] What Matters in Data for DPO? (2025)
[^192^] Statistical Impossibility of Aligning LLMs (2025)
[^193^] Is DPO Superior to PPO (ICML 2024)
[^195^] Towards Improved Preference Optimization Pipeline (2024)
[^196^] Direct Advantage Regression (2025)
[^197^] What Matters in Data for DPO (2025)
[^198^] Length-Controlled AlpacaEval (2024)
[^199^] A Novel Approach to Identity Preference Optimization (2024)
[^200^] Towards a Unified View of Preference Learning (2024)
[^201^] Instructions for ACL Proceedings (2024)
[^223^] Learning to Align, Aligning to Learn (2025)
[^224^] Bridging Brains and Machines (2025)
[^225^] Aligning Large Multimodal Models with RLHF (2023)
[^226^] AG-RLHF: Adversarial Game RLHF (2024)
[^227^] Evaluation of LLMs for Medical Summarization (2024)
[^228^] Spectral Policy Optimization for GRPO (2025)
[^229^] PPO Implementation Details (2024)
[^230^] Offline Learning and Forgetting for Reasoning (2024)
[^231^] Offline Learning and Forgetting for Reasoning (2024)
[^232^] Safety Alignment of LMs via Non-cooperative Games (2024)
[^233^] Online Iterative RLHF with General Preference Model (2024)
[^235^] Offline RLHF Methods Need More Accurate Supervision (2024)
[^236^] Preference Optimization Generalization Under Noisy Feedback (2024)
[^238^] RLHF & DPO Explained: Simulation in Python (2026)
[^240^] SimPO: Simple Preference Optimization (2024)
[^241^] SimPO Paper Review (2024)
[^242^] SimPO Paper Review (Korean) (2024)
[^316^] Aligning Visual Contrastive Learning via PO (2024)
[^317^] Maximum Preference Optimization with Importance Sampling (2023)
[^318^] SynPO: Synergizing Descriptiveness and PO (2025)
[^319^] Investigating Regularization of Self-Play Language Models (2024)
[^320^] Preference Optimization as Probabilistic Inference (2024)
[^321^] Policy Optimization in RLHF: Out-of-preference Data (2024)
[^322^] Reward Model Learning vs. Direct Policy Optimization (2024)
[^323^] On Diversified Preferences of LLM Alignment (2023)
[^325^] Comparing Bad Apples to Good Oranges (2024)
[^326^] IPO: A General Theoretical Paradigm (AISTATS 2024)
[^327^] OpenRLHF: RLHF Framework (2024)
[^328^] Learning to Directly Align LLMs with Diversity (2024)
[^329^] REAL: Response Embedding-based Alignment (2024)
[^330^] Aligning Visual Contrastive Learning via PO (2024)
