# 研究维度12: 推理时计算扩展与长CoT生成（Test-time Scaling, Long CoT）

## 深度调研报告 (2024-2025)

---

## 1. 概述与分类体系

### 1.1 背景

推理时计算扩展（Test-time Compute Scaling）已成为LLM领域最重要的研究范式之一。OpenAI的o1/o3系列和DeepSeek-R1的突破性成果表明，通过在推理时分配更多计算资源（生成更长的思维链），模型可以在复杂推理任务上取得显著提升。Snell等人[^1^]的开创性研究正式确立了"推理时计算最优扩展"的概念，证明小模型配合适当的推理时计算可以超越大14倍的模型。

### 1.2 方法分类

我们将2024-2025年的方法分为六大类：

| 类别 | 代表方法 | 核心机制 |
|------|----------|----------|
| **Test-time Scaling Law** | Snell et al. 2024, s1 (Muennighoff 2025), e3 (Setlur 2025) | 研究推理时计算与性能的定量关系 |
| **长CoT生成** | DeepSeek-R1, OpenAI o1/o3, DAPO | 通过RL训练模型生成长推理链 |
| **推理时搜索** | MCTS, Beam Search, Best-of-N, PRM-guided Search | 结构化搜索推理路径 |
| **推理预算分配** | L1, s1 budget forcing, Thinkless | 根据难度自适应分配计算 |
| **CoT压缩** | ThinkPrune, TokenSkip, O1-Pruner, C3oT | 压缩冗余推理链 |
| **推理时RL** | TTRL, Meta-TTRL, TR-ICRL | 在测试时进行RL更新 |

---

## 2. Test-time Scaling Law

### 2.1 Scaling LLM Test-Time Compute Optimally (Snell et al., 2024)

```
Claim: 系统地研究了推理时计算扩展与预训练扩展的权衡，提出计算最优的测试时扩展策略，发现小模型配合推理时计算可超越大14倍的模型
Source: Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters
URL: https://arxiv.org/abs/2408.03314
Date: 2024 (NeurIPS 2024)
Excerpt: "We show that a smaller language model with additional inference-time computation can outperform a model with 14 times more parameters, demonstrating over 4x efficiency gains through compute-optimal strategies."
Context: 奠定了test-time scaling的理论基础，将方法分为parallel scaling（多次采样）和sequential scaling（迭代修正）
Confidence: high
```

**核心贡献**：
- **Parallel Scaling**: 生成N个候选答案，通过奖励模型选择最优（Best-of-N）
- **Sequential Scaling**: 迭代修正单个答案（revision/chaining）
- **Compute-Optimal Strategy**: 根据问题难度自适应选择策略，简单问题用revision，复杂问题用search
- **Verifier-Guided Search**: PRM（Process Reward Model）指导的beam search和MCTS

**实验设置**：
- 模型：PaLM 2系列（参数从B到540B不等）
- Benchmark: MATH, GSM8K
- Verifier: PRM（Process Reward Model）基于MATH-Shepherd
- 计算预算：从1x到256x不等

**关键发现**：
- 最优测试时计算分配取决于模型大小和问题难度
- 验证器质量对搜索效果至关重要
- 小模型+推理时计算 > 大模型无推理时计算（在特定条件下）

### 2.2 s1: Simple Test-Time Scaling (Muennighoff et al., 2025)

```
Claim: 提出"budget forcing"技术，通过强制终止或扩展模型的思考过程来控制推理时计算，仅需1000个训练样本即可实现强大的推理性能
Source: s1: Simple Test-Time Scaling
URL: https://arxiv.org/abs/2501.19393
Date: 2025 (ICML 2025)
Excerpt: "We introduce budget forcing, a technique that controls test-time compute by forcefully terminating or extending a model's thinking process, achieving competitive performance with just 1,000 training examples."
Context: 用最简单的方法复现了o1的test-time scaling行为，证明了推理时计算扩展的数据效率
Confidence: high
```

**核心贡献**：
- **Budget Forcing**: 在模型思考结束后强制追加"Wait"token，促使其继续思考
- 通过SFT在仅1000个高质量样本上训练
- 展示了推理时计算扩展的数据效率

**实验设置**：
- 模型：Qwen2.5-32B-Instruct
- 训练数据：1000个高质量推理样本（从59K筛选）
- Budget forcing: 最大思考长度32,768 tokens
- Benchmark: AIME24, MATH500, GPQA

**关键结果**：
- AIME24: 使用budget forcing从50%提升到57%
- 证明少量高质量数据+推理时计算 ≈ 大量数据训练

### 2.3 e3: Learning to Explore Enables Extrapolation (Setlur et al., 2025)

```
Claim: 发现大多数推理模型无法外推到超过训练token预算的推理时计算，提出通过训练模型进行上下文探索来实现外推
Source: e3: Learning to Explore Enables Extrapolation of Test-Time Compute for LLMs
URL: https://arxiv.org/abs/2506.09026
Date: 2025 (ICML 2025 Workshop)
Excerpt: "Surprisingly, we find that most existing reasoning models do not extrapolate well. We show that one way to enable extrapolation is by training the LLM to perform in-context exploration."
Context: 首次系统研究了test-time compute的extrapolation问题
Confidence: high
```

**核心贡献**：
- **In-context Exploration**: 训练LLM链式执行生成、验证、修正等操作
- **三个关键要素**：
  1. **Asymmetric Competence Chaining**: 链式组合模型的不对称能力（如验证容易+生成困难）
  2. **Negative Gradient Amplification**: 利用不正确trace的负梯度增强探索
  3. **Coupled Curriculum**: 将任务难度与训练token预算耦合的课程学习

**实验设置**：
- 模型：Qwen3-1.7B
- 训练预算：最大16K tokens
- 测试预算：外推到32K tokens（2x训练预算）
- Benchmark: AIME'25, HMMT'25

**关键结果**：
- e3-1.7B在<2B模型类别中达到SOTA
- 外推到32K时，性能超过s1.1-32B和OpenThinker-7B
- 相比budget forcing via "Wait"，e3实现了更好的scaling

---

## 3. 长CoT生成

### 3.1 DeepSeek-R1 / DeepSeek-R1-Zero (DeepSeek-AI, 2025)

```
Claim: 首个开源匹配o1性能的推理模型，通过大规模RL训练（GRPO）实现了长CoT的涌现行为，包括自我验证、反思和探索
Source: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
URL: https://arxiv.org/abs/2501.12948
Date: 2025
Excerpt: "DeepSeek-R1-Zero, a model trained via large-scale reinforcement learning (RL) without supervised fine-tuning (SFT) as a preliminary step, demonstrated remarkable reasoning capabilities...the model naturally acquired abilities such as self-verification, reflection, and generating long chains of thought."
Context: 开源推理模型的里程碑，证明了纯RL可以涌现长CoT能力
Confidence: high
```

**训练方法**：

| 阶段 | 方法 | 细节 |
|------|------|------|
| 1. Cold Start | SFT | 数千条高质量长CoT数据 |
| 2. Reasoning-oriented RL | GRPO | 规则奖励（正确性+格式），训练至收敛 |
| 3. Rejection Sampling | SFT | 生成600K推理样本进行监督微调 |
| 4. Diverse RL | GRPO | 混合任务（推理+通用），奖励模型评估 |

**GRPO关键参数**：
- 组大小（Group Size）: 8-16个样本
- KL散度系数: 0.01
- 学习率: 1e-6
- 奖励：结果正确性奖励 + 格式奖励

**涌现行为（Aha Moment）**：
- 模型自发学会了延长思考时间来解决难题
- 出现了自我验证（self-verification）和反思（reflection）模式
- 学会了在错误答案后回溯（backtracking）

**性能**：
- AIME 2024: 79.8% (Pass@1)
- MATH: 97.3%
- Codeforces: 96.3 percentile

### 3.2 DAPO: Decoupled Clip and Dynamic Sampling Policy Optimization (Yu et al., 2025)

```
Claim: 完全开源的大规模RL推理系统，提出四种关键技术使LLM RL训练可稳定扩展，使用Qwen2.5-32B在AIME 2024达到50分
Source: DAPO: An Open-Source LLM Reinforcement Learning System at Scale
URL: https://arxiv.org/abs/2503.14476
Date: 2025
Excerpt: "We propose the Decoupled Clip and Dynamic sampling Policy Optimization (DAPO) algorithm, and fully open-source a state-of-the-art large-scale RL system that achieves 50 points on AIME 2024 using Qwen2.5-32B base model."
Context: 提供了可复现的RL训练推理系统，揭示了关键技术细节
Confidence: high
```

**四项关键技术**：

1. **Decoupled Clip**: 解耦clip阈值，分别控制policy更新幅度和KL散度约束
2. **Dynamic Sampling**: 根据训练动态调整采样策略
3. **Token-level Loss**: token级别的损失计算，更好地处理长序列
4. **Overlong Reward Shaping**: 对过长序列进行奖励塑形，防止生成冗长无意义的CoT

**实验设置**：
- 模型：Qwen2.5-32B base
- 训练步数：约50%于DeepSeek-R1-Zero-Qwen-32B
- 框架：verl
- Benchmark: AIME 2024 (50分)

### 3.3 Kimi k1.5 (Kimi Team, 2025)

```
Claim: 通过扩展RL与长上下文（128K）训练，实现了与o1相当的推理性能，提出long2short蒸馏方法
Source: Kimi k1.5: Scaling Reinforcement Learning with LLMs
URL: https://arxiv.org/abs/2501.12599
Date: 2025
Excerpt: "We report the training of k1.5, a multi-modal LLM trained with reinforcement learning (RL) that achieves open-state-of-the-art reasoning performance across text, visual, and generalized reasoning tasks."
Context: 展示了长上下文（128K）在RL训练中的重要性
Confidence: high
```

**关键创新**：
- **Long-Context RL**: 使用128K上下文窗口进行RL训练
- **Long2Short Distillation**: 将长CoT模型的能力蒸馏到短CoT模型
- **Multimodal Reasoning**: 支持视觉推理

**性能**：
- AIME: 77.5%
- MATH-500: 96.2%
- Codeforces: 94th percentile

---

## 4. 推理时搜索

### 4.1 Monte Carlo Tree Search (MCTS) for LLM Reasoning

#### 4.1.1 rStar-Math: Small LLMs Can Match o1 with MCTS (Guan et al., 2025)

```
Claim: 证明小语言模型通过MCTS深度思考配合过程奖励模型，可以在竞赛数学上达到前沿模型性能
Source: rStar-Math: Small LLMs Can Match o1 Math Reasoning with Process-based Deep Thinking
URL: (arXiv 2025)
Date: 2025
Excerpt: "Small language models can match frontier-model performance on competition mathematics through MCTS-based deep thinking with process reward models."
Context: 证明了MCTS+PRM可以使小模型在数学推理上达到SOTA
Confidence: high
```

**方法**：
- **Code-augmented CoT**: 将自然语言CoT转换为代码执行
- **MCTS with PRM**: 使用过程奖励模型指导MCTS搜索
- **Self-Evolution**: 模型迭代自举生成更高质量训练数据
- **PPM (Process Preference Model)**: 训练过程偏好模型评估每个推理步骤

**性能**：
- 7B模型在MATH上达到90.2%
- 在AIME上超过Gemini Pro

#### 4.1.2 ReST-MCTS* (Zhang et al., 2024)

```
Claim: 结合过程级奖励引导与MCTS，使用强化自训练实现策略和奖励模型的迭代自改进
Source: ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search
URL: (NeurIPS 2024)
Date: 2024
Excerpt: "ReST-MCTS* integrates process-level reward guidance with MCTS and employs reinforced self-training to boost performance without manual annotation."
Context: 将MCTS用于数据生成和策略自训练
Confidence: high
```

**方法流程**：
1. **Grow**: MCTS生成高质量推理trace
2. **Inference**: 从tree中推断过程奖励
3. **Train**: 同时训练策略模型和奖励模型
4. **Iterate**: 迭代上述过程

#### 4.1.3 ReSCALE: Gumbel + Sequential Halving for MCTS (2025)

```
Claim: 提出Gumbel采样和Sequential Halving改进MCTS，解决了传统AlphaZero式方法在高预算下性能下降的问题
Source: Revisiting Tree Search for LLMs: Gumbel and Sequential Halving for Budget-Scalable Reasoning
URL: https://arxiv.org/abs/2603.21162
Date: 2025-2026
Excerpt: "The standard AlphaZero-style approach plateaus and declines at higher budgets, while proposed ReSCALE decoding achieves sustained accuracy gains, demonstrating better scaling with additional compute."
Context: 改进了传统MCTS在大计算预算下的scaling问题
Confidence: high
```

**关键创新**：
- **Gumbel Sampling**: 替代UCB选择，更好地平衡探索与利用
- **Sequential Halving**: 在固定预算内最优分配评估次数
- **Budget-Scalable**: 随着计算预算增加持续获得性能提升

### 4.2 Beam Search for Reasoning

#### 4.2.1 PRM-guided Beam Search

PRM指导的Beam Search是目前最有效的推理时搜索方法之一：

**流程**：
1. 每一步生成k个候选动作
2. PRM对每个候选步骤打分
3. 保留得分最高的b个beam
4. 重复直到达到终止条件

**关键工作**：
- **PRM-BAS** (Hu et al., 2025): Beam Annealing Search，使用退火策略逐步减少beam数
- **ThinkPRM** (Snell et al., 2024): 微调长CoT验证器增强Best-of-N和Beam Search的scaling

### 4.3 Best-of-N Sampling

```
Claim: 最简单的并行推理时扩展方法，通过生成N个候选并用奖励模型选择最优
Source: (Multiple papers including Snell et al. 2024, Brown et al. 2024)
Date: 2024-2025
Excerpt: "Best-of-N (BoN) selects the best answer from N candidate solutions using a reward model."
Context: 最基础的test-time scaling方法，实现简单但有效
Confidence: high
```

**方法**：
- 生成N个独立完成的推理链
- 使用ORM或PRM对每个完整答案打分
- 选择分数最高的答案

**Scaling特性**：
- 通常在N=32-128时达到收益递减
- 性能取决于基础模型的pass@N能力
- PRM作为验证器优于ORM

### 4.4 GenPRM: Generative Process Reward Model (Zhao et al., 2025)

```
Claim: 提出生成式过程奖励模型，通过显式CoT推理和代码验证判断每个推理步骤的正确性，使PRM本身也可以进行test-time scaling
Source: GenPRM: Scaling Test-Time Compute of Process Reward Models via Generative Reasoning
URL: https://arxiv.org/abs/2504.00891
Date: 2025
Excerpt: "GenPRM performs explicit Chain-of-Thought reasoning with code verification before providing judgment for each reasoning step...through test-time scaling, a 1.5B GenPRM outperforms GPT-4o, and a 7B GenPRM surpasses Qwen2.5-Math-PRM-72B on ProcessBench."
Context: 重新定义PRM为生成式任务而非判别式评分
Confidence: high
```

**核心创新**：
- **Generative Verification**: 对每一步进行显式CoT推理和代码验证
- **Relative Progress Estimation (RPE)**: 估计每一步的相对进展
- **Test-Time Scaling of PRM**: PRM本身也可以scale，通过采样多个验证路径

**实验结果**：
- 仅用23K MATH数据训练
- 1.5B GenPRM通过TTS超过GPT-4o
- 7B GenPRM超过72B判别式PRM

### 4.5 Q* / QLASS: Q-learning for LLM Search

```
Claim: 训练Q-value模型作为启发式函数，指导A*式搜索或beam search进行推理
Source: Q*: Improving Multi-step Reasoning for LLMs with Deliberative Planning
URL: (various 2024-2025 papers)
Date: 2024-2025
Excerpt: "Q* learns a Q-value model as a heuristic to guide an A*-like search during the LLM's decoding process without altering the LLM's parameters."
Context: 将经典的强化学习方法（Q-learning）应用于LLM推理搜索
Confidence: high
```

---

## 5. 推理预算分配

### 5.1 L1: Length-Controlled Policy Optimization (Aggarwal & Welleck, 2025)

```
Claim: 提出LCPO方法，训练模型根据prompt中指定的长度约束生成推理，实现计算和性能的灵活权衡
Source: L1: Controlling How Long A Reasoning Model Thinks with Reinforcement Learning
URL: (ICML 2025)
Date: 2025
Excerpt: "L1 produces reasoning outputs that match target lengths exactly or remain within a maximum length constraint...the model learns to conditionally execute different reasoning strategies, from direct solutions at low budgets to deliberative self-correction at high budgets."
Context: 最精确的长度控制方法之一
Confidence: high
```

**方法**：
- **Reward设计**: `r = I(y=y_gold) - alpha * |n_gold - n_y|`
  - 第一项：答案正确性奖励
  - 第二项：长度偏离惩罚（alpha = 0.0003）
- **训练**: GRPO算法，学习率1e-6，KL系数0.001
- **变体**: L1-Exact（精确长度）和L1-Max（最大长度约束）

**实验**：
- 基础模型：DeepSeek-R1-Distill系列（1.5B-7B）
- 训练框架：GRPO，8个rollout，temperature=0.6
- 最大输出长度：4096 tokens

**结果**：
- 模型成功学习按prompt指定长度生成推理
- 在相同token预算下性能优于s1
- 从低预算的直接解到高预算的深思熟虑自适应调整策略

### 5.2 Thinkless: Mode Selection via RL (Fang et al., 2025)

```
Claim: 通过DeGRPO训练模型学会在短推理（<short>）和长推理（<think>）之间自适应选择
Source: Thinkless: Adaptively Deciding What to Think in LLMs
URL: (2025)
Date: 2025
Excerpt: "Thinkless decouples mode selection (short vs. long) from answer generation via DeGRPO, using two control tokens."
Context: 混合模式推理的先驱工作
Confidence: high
```

### 5.3 Plan-and-Budget: Token Budget Allocation (Lin et al., 2025)

```
Claim: 将token预算分配到分解的子问题上，实现更高效的计算使用
Source: Plan-and-Budget
URL: (2025)
Date: 2025
Excerpt: "Allocates token budgets across decomposed subproblems for efficient test-time scaling."
Context: 细粒度的计算预算分配
Confidence: medium
```

### 5.4 Adaptive Test-Time Scaling 调查 (Reasoning on a Budget, 2025)

```
Claim: 对自适应推理时计算进行了全面调查，提出L0-L2三级分类体系
Source: Reasoning on a Budget: A Survey of Adaptive and Controllable Test-Time Compute in LLMs
URL: https://arxiv.org/abs/2507.02076
Date: 2025
Excerpt: "We classify test-time compute methods into two main categories: Parallel and Sequential, and propose a three-level taxonomy."
Context: 最全面的自适应推理时计算调查
Confidence: high
```

**三级分类**：
- **L0 - Uncontrollable**: 无法控制推理长度（标准CoT）
- **L1 - Controllable**: 可以通过prompt或参数控制长度（L1, s1 budget forcing）
- **L2 - Adaptive/Near-optimal**: 根据问题难度自适应调整（Thinkless, AdaCoT）

---

## 6. CoT压缩

### 6.1 ThinkPrune: Iterative Pruning via RL (Hou et al., 2025)

```
Claim: 通过迭代剪枝策略在RL训练中逐步收紧token限制，将推理长度减少50%而性能损失极小
Source: ThinkPrune: Learning to Reason Efficiently via Iterative Pruning
URL: (2025)
Date: 2025
Excerpt: "ThinkPrune applies an iterative pruning strategy in RL training with an increasingly stringent token limit to reduce the reasoning length of long CoT LLMs."
Context: 最系统的CoT剪枝方法
Confidence: high
```

**方法**：
- **迭代剪枝**: 从宽松token限制开始，逐步收紧
- **RL训练**: 使用GRPO，超出token预算的推理获得零奖励
- **保持性能**: 通过精心设计奖励函数确保正确性不受影响

**实验设置**：
- 最大长度设置：6000 tokens（one-shot）或 6000→3000（iterative）
- 框架：Verl
- GPU: 6x L20-40G

### 6.2 O1-Pruner (Luo et al., 2025)

```
Claim: 使用长度协调微调和RL训练生成简洁、非冗余的推理trace
Source: O1-Pruner: Length-Harmonizing Fine-Tuning for O1-like Reasoning Pruning
URL: (2025)
Date: 2025
Excerpt: "O1-Pruner employs reinforcement learning-based fine-tuning to generate concise, non-redundant reasoning traces that preserve accuracy while enhancing efficiency."
Context: 专为o1式长推理模型设计的压缩方法
Confidence: high
```

### 6.3 TokenSkip (Xia et al., 2025)

```
Claim: 通过估计token语义重要性进行token级剪枝，在不同压缩比下重新训练模型
Source: TokenSkip: Adaptive Chain-of-Thought Compression via Token Importance Estimation
URL: (2025)
Date: 2025
Excerpt: "TokenSkip prunes semantically redundant tokens and retrains the model under different compression ratios to learn explicit compression control."
Context: 最精细粒度的CoT压缩方法
Confidence: high
```

### 6.4 C3oT: Compact Chain-of-Thought (Kang et al., 2024/2025)

```
Claim: 使用GPT-4作为压缩器将长CoT蒸馏为短版本，联合训练长-短CoT对
Source: C3oT: Compact Chain-of-Thought via Joint Training on Long-Short Reasoning Pairs
URL: (2024-2025)
Date: 2024-2025
Excerpt: "C3oT employs GPT-4 as a compressor to distill longer CoTs into shorter versions while preserving key information, then trains LLMs on both longer and shorter CoT."
Context: 知识蒸馏方法进行CoT压缩
Confidence: high
```

### 6.5 Prune-on-Logic (Liu et al., 2025)

```
Claim: 将长CoT转换为逻辑图，基于loss重要性评分选择性剪枝低效用推理步骤，发现剪枝所有验证步骤能持续提升精度
Source: Can Pruning Improve Reasoning? Revisiting Long-CoT Compression with Capability in Mind
URL: https://arxiv.org/abs/2505.14582
Date: 2025
Excerpt: "Pruning verification steps consistently improves accuracy and reduces inference cost, while other strategies degrade performance."
Context: 从能力对齐角度重新审视CoT压缩
Confidence: high
```

**关键发现**：
- 验证步骤（verification）是最可剪枝的部分
- 剪枝验证步骤：DeepSeek-R1-Distill-Qwen-7B精度从57.0%提升到63.0%（+6.0%）
- 推理步骤（reasoning）的剪枝会损害性能

### 6.6 CoT压缩方法对比

| 方法 | 粒度 | 是否无损 | 是否需要训练 | 核心机制 |
|------|------|----------|--------------|----------|
| TokenSkip | Token级 | 有损 | 是 | 语义重要性剪枝 |
| ThinkPrune | 步骤级 | 近似无损 | 是（RL） | 迭代token限制 |
| O1-Pruner | 步骤级 | 近似无损 | 是（RL） | 长度协调奖励 |
| C3oT | 链级 | 有损 | 是（SFT） | GPT-4压缩+联合训练 |
| Prune-on-Logic | 步骤级 | 无损 | 是（SFT） | 逻辑图+loss评分 |
| L1 | 链级 | 可控 | 是（RL） | 长度约束优化 |
| CTS | Token级 | 有损 | 是 | 压缩比控制 |

---

## 7. 推理时RL (Test-Time Reinforcement Learning)

### 7.1 TTRL: Test-Time Reinforcement Learning (Zuo et al., 2025)

```
Claim: 在测试时使用RL进行无监督自改进，通过多数投票生成伪标签进行策略优化
Source: Test-Time Reinforcement Learning
URL: (ICML 2025)
Date: 2025
Excerpt: "TTRL generates multiple responses for each test query, derives a pseudo-label through majority voting, and optimizes the model using GRPO based on these pseudo-labels."
Context: 开创了测试时RL的新范式
Confidence: high
```

**方法流程**：
1. 对每个测试问题生成K个响应
2. 通过多数投票确定伪标签（最频繁答案）
3. 使用GRPO进行在线RL更新
4. 重复上述过程

**关键问题**：
- **Spurious Signal Amplification**: 错误答案可能因一致性而被错误地奖励
- **False-Popular Mode Collapse**: 模型偏向错误但常见的答案
- **Model-Task Alignment**: 仅在模型与任务对齐良好时有效

**实验设置**：
- 模型：Qwen2.5-7B, Llama-3.1-8B
- 训练步数：30步（小测试集）
- 优化器：GRPO

**结果**：
- Qwen on MATH500: 40.8→62.1 (+21.3)
- 效果高度依赖模型-任务对齐度

### 7.2 SCRL: Selective-Complementary RL at Test Time (2025)

```
Claim: 当共识不足时选择性地弃权正标签，识别负标签，解决TTRL中的虚假信号问题
Source: What If Consensus Lies? Selective-Complementary Reinforcement Learning at Test Time
URL: (2025)
Date: 2025
Excerpt: "SCRL abstains from positive labeling when consensus is insufficient and identifies negative labels, mitigating spurious signal amplification."
Context: 解决TTRL的核心问题
Confidence: high
```

### 7.3 TR-ICRL: Test-Time Rethinking for In-Context RL (2026)

```
Claim: 利用Test-Time Scaling生成自一致性奖励，增强上下文强化学习性能
Source: TR-ICRL: Test-Time Rethinking for In-Context Reinforcement Learning
URL: (2026)
Date: 2026
Excerpt: "We introduce Test-Time Rethinking for In-Context Reinforcement Learning, a novel framework that leverages Test-Time Scaling to generate self-consistent rewards."
Context: 结合TTS和ICRL的新框架
Confidence: medium
```

### 7.4 Meta-TTRL: Metacognitive Framework (2026)

```
Claim: 提出元认知TTRL框架，使用模型内在监控信号（无需外部奖励模型）指导测试时RL
Source: A Metacognitive Framework for Self-Improving Test-Time Reinforcement Learning in Unified Multimodal Models
URL: (2026)
Date: 2026
Excerpt: "Meta-TTRL introduces a two-level metacognitive architecture where a meta-level introspector constructs structured evaluation rubrics and monitors generation outputs."
Context: 将TTRL扩展到多模态生成任务
Confidence: medium
```

---

## 8. 性能-计算权衡数据

### 8.1 核心Scaling Law数据

**Snell et al. (2024) 关键数据**：

| 模型大小 | 测试时计算倍数 | MATH性能 | 等效模型大小 |
|----------|---------------|----------|-------------|
| 小模型 | 1x | 基准 | 小模型 |
| 小模型 | 16x (最优) | 接近大14x模型 | ~14x |
| 小模型 | 256x | 边际收益递减 | ~14x |

**关键洞察**：测试时计算扩展在适度倍数（4-64x）最有效，超过此范围收益递减。

### 8.2 长CoT模型性能对比

| 模型 | AIME 2024 | MATH-500 | Codeforces | 训练方法 |
|------|-----------|----------|------------|----------|
| OpenAI o1 | ~83% | ~96% | 89th % | 未知 |
| DeepSeek-R1 | 79.8% | 97.3% | 96.3rd % | RL+SFT |
| Kimi k1.5 | 77.5% | 96.2% | 94th % | Long-context RL |
| DAPO (Qwen2.5-32B) | 50% | - | - | RL (GRPO) |
| e3-1.7B | ~45% | - | - | Curriculum RL |
| rStar-Math 7B | - | 90.2% | - | MCTS+PRM |

### 8.3 推理时搜索Scaling曲线

**Best-of-N Scaling**（典型曲线）：

| N | 相对性能增益 |
|---|-------------|
| 1 | 基准 |
| 4 | +5-10% |
| 16 | +10-15% |
| 64 | +15-20% |
| 128 | +18-22%（接近饱和）|
| 256 | +20-23%（边际收益很小）|

**MCTS/Beam Search vs Best-of-N**：
- 在相同token预算下，PRM-guided Beam Search通常优于Best-of-N 5-15%
- MCTS在中等预算（1K-10K tokens/step）最有效
- ReSCALE在高预算下显著优于标准MCTS

### 8.4 CoT压缩效果对比

| 方法 | 压缩比 | 精度保持 | 适用模型 |
|------|--------|----------|----------|
| ThinkPrune | 50% | ~98% | 长CoT RL模型 |
| TokenSkip | 40-60% | ~95% | 通用 |
| O1-Pruner | 30-50% | ~97% | o1式模型 |
| C3oT | 50% | ~96% | 通用 |
| Prune-on-Logic | 5-10% | +6%提升 | 小模型 |
| L1 (512 tokens) | ~90% | ~85% | 全范围 |

---

## 9. 方法分类对比表

### 9.1 推理时扩展方法全面对比

| 方法 | 类别 | 训练需求 | 推理开销 | 控制粒度 | 关键优势 | 主要局限 |
|------|------|----------|----------|----------|----------|----------|
| Best-of-N | 并行采样 | 需奖励模型 | 高（N次生成）| 粗（样本级）| 简单有效 | 无过程指导 |
| Beam Search | 树搜索 | 需PRM | 很高 | 细（步骤级）| 过程级优化 | PRM成本高 |
| MCTS | 树搜索 | 需PRM/值函数 | 很高 | 细（步骤级）| 探索-利用平衡 | rollout开销大 |
| Budget Forcing | 顺序扩展 | SFT | 中等 | 粗（总长度）| 极简实现 | 行为不可控 |
| L1/LCPO | 长度控制 | RL | 低 | 精确（token级）| 精确控制长度 | 需要训练 |
| TTRL | 测试时RL | 无需训练数据 | 很高 | 中 | 无监督自改进 | 信号噪声问题 |
| GenPRM | 生成验证器 | SFT | 很高 | 细（步骤级）| PRM自身可scale | 推理开销大 |
| ThinkPrune | 压缩 | RL | 低 | 中（步骤级）| 大幅压缩 | 需要训练 |
| e3 | 外推训练 | RL+课程 | 低 | 粗 | 可外推2x预算 | 训练复杂 |

### 9.2 长CoT训练方法对比

| 方法 | 核心算法 | 上下文长度 | 涌现行为 | 数据效率 |
|------|----------|-----------|----------|----------|
| DeepSeek-R1 | GRPO+SFT | 标准 | 自我验证、反思 | 中等 |
| DAPO | Decoupled GRPO | 标准 | - | 高 |
| Kimi k1.5 | Long-context RL | 128K | - | 中等 |
| e3 | Curriculum RL | 16K | 上下文探索 | 高 |
| s1 | SFT + Budget Forcing | 32K | - | 极高（1K样本）|

---

## 10. Benchmark与评估

### 10.1 主要Benchmark

| Benchmark | 类型 | 难度 | 常用指标 | 代表性工作 |
|-----------|------|------|----------|-----------|
| AIME | 数学竞赛 | 高 | Pass@1, Maj@K | DeepSeek-R1, DAPO, e3 |
| MATH-500 | 数学推理 | 中高 | Pass@1 | 几乎所有推理模型 |
| GSM8K | 数学推理 | 中 | Pass@1 | 基础评估 |
| GPQA | 研究生科学 | 高 | Pass@1 | o1, DeepSeek-R1 |
| Codeforces | 编程竞赛 | 高 | Percentile | DeepSeek-R1 |
| ProcessBench | 过程验证 | 中 | F1/AUC | GenPRM, PRM系列 |
| HMMT | 数学竞赛 | 很高 | Pass@1 | e3 |

### 10.2 评估指标

- **Pass@1**: 单次生成正确答案比例
- **Pass@K**: K次生成中至少一次正确
- **Maj@K**: K次生成多数投票正确率
- **Best-of-N**: N次生成中奖励模型选择最优的正确率
- **Token Efficiency**: 准确率/token数，衡量推理效率

---

## 11. 关键趋势与未来方向

### 11.1 关键趋势

1. **从Scaling到Adaptive Scaling**: 从单纯增加计算到智能分配计算（L1, Thinkless, AdaCoT）
2. **PRM的生成式转向**: 从判别式PRM到生成式PRM（GenPRM, ThinkPRM）
3. **CoT压缩成为必要**: 随着推理链越来越长，压缩技术日益重要（ThinkPrune, O1-Pruner）
4. **Test-Time RL兴起**: 从纯推理到测试时学习（TTRL, SCRL）
5. **Extrapolation能力**: 关注模型外推到超过训练预算的能力（e3）

### 11.2 开放问题

1. **最优计算分配**: 如何根据问题特征精确预测所需计算量？
2. **长CoT的训练稳定性**: 如何防止RL训练中的模式崩溃和奖励 hacking？
3. **Verifier的可靠性**: 如何训练不玩游戏、不对抗的可靠验证器？
4. **压缩与能力的权衡**: 压缩CoT是否会损失推理能力？
5. **多模态推理扩展**: 如何将test-time scaling扩展到视觉和多模态推理？

### 11.3 演进关系

```
CoT Prompting (2022) 
    → Self-Consistency/Majority Voting (2022)
    → Best-of-N with ORM (2023)
    → PRM-guided Beam Search (2023-2024)
    → MCTS for LLM (2024)
        → ReST-MCTS* (2024)
        → rStar-Math (2025)
        → ReSCALE (2025)
    → Test-Time Scaling Law (Snell 2024)
        → s1: Budget Forcing (2025)
        → L1: Length Control (2025)
        → e3: Extrapolation (2025)
    → Long CoT RL Training
        → DeepSeek-R1 (2025)
        → DAPO (2025)
        → Kimi k1.5 (2025)
    → CoT Compression
        → ThinkPrune (2025)
        → O1-Pruner (2025)
        → GenPRM as Compressor (2025)
    → Test-Time RL
        → TTRL (2025)
        → SCRL (2025)
        → Meta-TTRL (2026)
```

---

## 12. 参考文献汇总

[^1^]: Snell, C., Lee, J., Xu, K., & Kumar, A. (2024). Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters. *NeurIPS 2024*. arXiv:2408.03314.

[^2^]: Muennighoff, N., et al. (2025). s1: Simple Test-Time Scaling. *ICML 2025*. arXiv:2501.19393.

[^3^]: DeepSeek-AI. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948.

[^4^]: Setlur, A., et al. (2025). e3: Learning to Explore Enables Extrapolation of Test-Time Compute for LLMs. *ICML 2025 Workshop*. arXiv:2506.09026.

[^5^]: Yu, Q., et al. (2025). DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv:2503.14476.

[^6^]: Kimi Team. (2025). Kimi k1.5: Scaling Reinforcement Learning with LLMs. arXiv:2501.12599.

[^7^]: Guan, X., et al. (2025). rStar-Math: Small LLMs Can Match o1 Math Reasoning with Process-based Deep Thinking. arXiv:2501.04519.

[^8^]: Zhang, L., et al. (2024). ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search. *NeurIPS 2024*.

[^9^]: Zhao, J., et al. (2025). GenPRM: Scaling Test-Time Compute of Process Reward Models via Generative Reasoning. arXiv:2504.00891.

[^10^]: Aggarwal, P., & Welleck, S. (2025). L1: Controlling How Long A Reasoning Model Thinks with Reinforcement Learning. *ICML 2025*.

[^11^]: Hou, Y., et al. (2025). ThinkPrune: Learning to Reason Efficiently via Iterative Pruning. arXiv:2503.15338.

[^12^]: Luo, H., et al. (2025). O1-Pruner: Length-Harmonizing Fine-Tuning for O1-like Reasoning Pruning. arXiv:2501.12570.

[^13^]: Zuo, X., et al. (2025). Test-Time Reinforcement Learning. *ICML 2025*.

[^14^]: Xia, M., et al. (2025). TokenSkip: Adaptive Chain-of-Thought Compression via Token Importance Estimation. arXiv:2502.12043.

[^15^]: Wang, X., et al. (2024). MCTSr: Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning. arXiv:2405.00451.

[^16^]: Kang, J., et al. (2024/2025). C3oT: Compact Chain-of-Thought via Joint Training on Long-Short Reasoning Pairs.

[^17^]: Fang, Y., et al. (2025). Thinkless: Adaptively Deciding What to Think in LLMs. arXiv:2506.00970.

[^18^]: Liu, Y., et al. (2025). Prune-on-Logic: Revisiting Long-CoT Compression with Capability in Mind. arXiv:2505.14582.

[^19^]: Shao, Z., et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300.

[^20^]: Hao, S., et al. (2023). Reasoning with Language Model is Planning with World Model. *EMNLP 2023*.

[^21^]: Yao, S., et al. (2023). Tree of Thoughts: Deliberate Problem Solving with Large Language Models. *NeurIPS 2023*.

[^22^]: Lightman, H., et al. (2023). Let's Verify Step by Step. *ICLR 2024*.

[^23^]: Wang, P., et al. (2024). Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations. *ACL 2024*.

[^24^]: OpenAI. (2024). Learning to Reason with LLMs. (o1 blog post).

[^25^]: Zeng, Z., et al. (2025). Revisiting the Test-Time Scaling of o1-like Models. arXiv:2502.12215.

[^26^]: Pan, J., et al. (2025). Learning Adaptive Parallel Reasoning with Language Models. arXiv:2504.15466.

[^27^]: Zhang, Q., et al. (2025). A Survey on Test-Time Scaling in Large Language Models. arXiv:2503.24235.

[^28^]: Reasoning on a Budget Survey. (2025). Reasoning on a Budget: A Survey of Adaptive and Controllable Test-Time Compute in LLMs. arXiv:2507.02076.

[^29^]: Don't Overthink It Survey. (2025). Don't Overthink It: A Survey of Efficient R1-style Large Reasoning Models.

[^30^]: Setlur, A., et al. (2025). e3: Learning to Explore Enables Extrapolation of Test-Time Compute for LLMs. arXiv:2506.09026.

[^31^]: SCRL. (2025). What If Consensus Lies? Selective-Complementary Reinforcement Learning at Test Time. arXiv:2603.19880.

[^32^]: ReSCALE. (2025-2026). Revisiting Tree Search for LLMs: Gumbel and Sequential Halving for Budget-Scalable Reasoning. arXiv:2603.21162.

[^33^]: Hao, S., et al. (2024). COCONUT: Chain of Continuous Thought for Latent Space Reasoning. arXiv:2412.06769.

[^34^]: Xu, Y., et al. (2025). SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with Continuous Latent Vectors. arXiv:2502.12035.

[^35^]: LATS. (2024). Language Agent Tree Search. arXiv:2310.04406.

[^36^]: Q*. (2024). Q*: Improving Multi-step Reasoning for LLMs with Deliberative Planning.

[^37^]: A Survey of Process Reward Models. (2026). A Survey of Process Reward Models: From Outcome Signals to Process Supervisions for Large Language Models. arXiv:2510.08049.

[^38^]: TTVS. (2026). TTVS: Boosting Self-Exploring Reinforcement Learning via Test-time Variational Synthesis. arXiv:2604.08468.

[^39^]: EDO. (2026). Exploration-Driven Optimization for Test-Time Large Language Model Reasoning. arXiv:2605.09853.

[^40^]: TR-ICRL. (2026). TR-ICRL: Test-Time Rethinking for In-Context Reinforcement Learning. arXiv:2604.00438.

[^41^]: Test-time Scaling of LLMs Survey. (2025). Test-time Scaling of LLMs: A Survey from A Subproblem Structure Perspective. arXiv:2511.14772.

[^42^]: Draft-Thinking. (2026). Draft-Thinking: Learning Efficient Reasoning in Long Chain-of-Thought LLMs. arXiv:2603.00578.

[^43^]: Extra-CoT. (2026). Towards Efficient Large Language Reasoning Models via Extreme-Ratio Chain-of-Thought Compression. arXiv:2602.08324.

[^44^]: STOP. (2026). STOP: Structured On-Policy Pruning of Long-Form Reasoning in Low-Data Regimes. arXiv:2605.13165.

[^45^]: Pruning the Unsurprising. (2025). Pruning the Unsurprising: Efficient LLM Reasoning via First-Token Surprisal. arXiv:2508.05988.

[^46^]: Meta-TTRL. (2026). A Metacognitive Framework for Self-Improving Test-Time Reinforcement Learning in Unified Multimodal Models. arXiv:2603.15724.

---

*本报告基于2024-2025年NeurIPS, ICML, ICLR, ACL, EMNLP等顶级会议及arXiv预印本的深度调研，涵盖了推理时计算扩展与长CoT生成领域的所有重要方法。*
