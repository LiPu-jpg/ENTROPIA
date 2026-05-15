# 研究维度4: RLVR与可验证奖励训练（DeepSeek-R1类方法）

## 1. 概述

Reinforcement Learning with Verifiable Rewards (RLVR) 是2024-2025年大语言模型后训练领域最重要的范式转变之一。该范式的核心思想是利用可验证的结果（如数学答案正确性、代码测试通过等）作为奖励信号，取代传统RLHF中需要人工标注或学习奖励模型的方式。DeepSeek-R1/R1-Zero通过纯RL（无SFT冷启动）在数学推理上取得了突破性进展，催生了大量后续工作。[^1^]

### RLVR范式演进

| 阶段 | 时期 | 方法 | 核心特征 |
|------|------|------|----------|
| Phase 1 | 2022 | RLHF (PPO) | 人类偏好标注，训练RM |
| Phase 2 | 2023-2024 | DPO系列 | 无需RM，偏好优化 |
| Phase 3 | 2024-present | GRPO/RLVR | 无需critic，可验证奖励 |

---

## 2. DeepSeek-R1/R1-Zero

### 2.1 核心贡献

DeepSeek-R1-Zero是首个通过纯RL（无SFT）直接训练base模型获得强大推理能力的公开研究，验证了LLM的推理能力可以完全通过RL激励，无需SFT。[^2^]

**DeepSeek-R1-Zero核心发现：**
- 从DeepSeek-V3-Base（671B参数，37B激活参数）直接进行大规模RL训练
- 在AIME 2024上从15.6%提升至71.0%（Pass@1），接近OpenAI-o1-0912水平
- 通过majority voting可达86.7%
- 出现了自发的长链推理、自我验证、反思等涌现行为（"Aha Moment"）

### 2.2 训练Pipeline

DeepSeek-R1采用四阶段训练流程：[^3^]

```
Stage 1: Cold Start SFT
  - 使用数千条高质量长CoT样本对DeepSeek-V3-Base进行SFT
  - 数据来源：few-shot prompting、人工标注、R1-Zero生成等多种方式
  - 目的：加速RL前期收敛，提高可读性

Stage 2: Reasoning-oriented RL
  - 在冷启动模型上进行大规模RL训练（同R1-Zero方法）
  - 使用GRPO算法
  - 额外引入语言一致性奖励，解决语言混合问题

Stage 3: Rejection Sampling + SFT
  - 从RL checkpoint通过rejection sampling收集600K推理样本
  - 收集200K非推理样本（写作、QA等通用任务）
  - 在DeepSeek-V3-Base上重新SFT（注意不是在RL模型上继续训练）

Stage 4: RL Alignment
  - 在人类偏好数据上进行RL
  - 优化helpfulness和harmlessness
  - 保持推理能力
```

### 2.3 GRPO算法

Group Relative Policy Optimization (GRPO) 是RLVR的核心算法：[^4^]

**核心思想：**
- 无需critic network，通过组内相对奖励估计advantage
- 对每个prompt采样G个response，计算组内标准化reward作为advantage

**Advantage计算：**
```
A_i = (r_i - mean({r_1, r_2, ..., r_G})) / std({r_1, r_2, ..., r_G})
```

**目标函数：**
```
L_GRPO = -1/G * sum(min(pi_theta/pi_old * A_i, clip(pi_theta/pi_old, 1-eps, 1+eps) * A_i))
```

**关键优势：**
- 无需训练value network，大幅降低显存需求
- 仅需policy model一个模型
- 与vLLM等高效推理框架配合良好

### 2.4 奖励设计

DeepSeek-R1-Zero使用两种规则奖励：[^5^]

| 奖励类型 | 说明 | 取值 |
|----------|------|------|
| Accuracy Reward | 答案正确性（数学用规则验证，代码用测试用例） | 0或1 |
| Format Reward | 要求推理过程放在`<think>`和`</think>`标签内 | 0或1 |

**注意：** 不使用outcome/process neural reward model，因为在大规模RL中容易reward hacking。

### 2.5 涌现行为（Emergent Behaviors）

训练过程中观察到的自发涌现行为：[^6^]

| 行为 | 描述 | 出现时间 |
|------|------|----------|
| Response长度增长 | 随RL训练response长度稳定增长 | 训练早期 |
| 自我验证 | 模型检查自己之前的推理步骤 | 中期 |
| 反思 | 模型重新思考之前的解法 | 中期 |
| Backtracking | 发现错误后回溯尝试新方案 | 中后期 |
| "Aha Moment" | 以拟人化口吻重新思考（如"Wait, let me reconsider..."） | 中期 |

### 2.6 Benchmark结果

| Benchmark | DeepSeek-R1 | o1-1217 | DeepSeek-V3 |
|-----------|-------------|---------|-------------|
| AIME 2024 (Pass@1) | 79.8 | 79.2 | 39.2 |
| MATH-500 (Pass@1) | 97.3 | 96.4 | 90.2 |
| GPQA Diamond | 71.5 | 75.7 | 59.1 |
| LiveCodeBench | 65.9 | 63.4 | 36.2 |
| Codeforces Rating | 2029 | 2061 | 1134 |
| Codeforces Percentile | 96.3 | 96.6 | 58.7 |
| MMLU | 90.8 | 91.8 | 88.5 |

### 2.7 蒸馏结果

使用DeepSeek-R1生成的800K样本对小型模型进行SFT蒸馏：[^7^]

| 模型 | AIME 2024 | MATH-500 | GPQA | LiveCodeBench |
|------|-----------|----------|------|---------------|
| Distill-Qwen-1.5B | 28.9 | 83.9 | 33.8 | 16.9 |
| Distill-Qwen-7B | 55.5 | 92.8 | 49.1 | 37.6 |
| Distill-Qwen-14B | 69.7 | 93.9 | 59.1 | 53.1 |
| Distill-Qwen-32B | 72.6 | 94.3 | 62.1 | 57.2 |
| Distill-Llama-8B | 50.4 | 89.1 | 49.0 | 39.6 |
| Distill-Llama-70B | 70.0 | 94.5 | 65.2 | 57.5 |

---

## 3. DeepSeek-R1复现工作

### 3.1 Open-R1 (HuggingFace)

**基本信息：**
- **Source:** https://github.com/huggingface/open-r1
- **目标:** 完全开源复现DeepSeek-R1
- **框架:** 基于TRL库实现GRPO

**实现计划：**
1. Step 1: 复现R1-Distill模型（从DeepSeek-R1蒸馏高质量语料）
2. Step 2: 复现纯RL pipeline（创建大规模数学、推理、代码数据集）
3. Step 3: 从base模型到RL-tuned的多阶段训练

**技术细节：**
- 使用TRL的vLLM backend支持多节点训练
- 支持单节点训练（8 GPU）使用`vllm_mode="colocate"`
- 支持多节点训练（N+1节点，1节点运行vLLM server）
- 提供Slurm脚本支持

### 3.2 SimpleRL-Reason / SimpleRL-Zoo (HKUST)

**基本信息：**
- **Source:** https://github.com/hkust-nlp/simpleRL-reason
- **论文:** SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild
- **机构:** HKUST

**核心发现：**
- 在10个不同base模型上研究ZeroRL（Llama3-8B, Mistral-7B/24B, DeepSeek-Math-7B, Qwen2.5 0.5B-32B）
- 首次在非Qwen系列的小模型中观察到"Aha Moment"
- 提出ZeroRL训练的关键设计策略

**ZeroRL定义：**
- 直接从pretrained base model开始RL优化，没有任何SFT中间步骤
- 仅使用简单的规则奖励（正确性奖励）
- 可以自发涌现出长链推理和自我反思能力

**关键发现：**[^8^]
1. **SFT冷启动可能限制探索：** 传统SFT作为冷启动会限制模型在RL阶段的探索能力，甚至降低最终可达的准确率上限
2. **Base model直接RL可获得更高上限：** 从Mistral-Small-24B base model直接RL可达49.6% pass@1，而经过100/500步SFT后再RL分别只有47.3%和40.3%
3. **数据难度匹配很重要：** 需要将数据难度与模型能力对齐
4. **Format reward设计影响大：** 需要精心设计format reward

### 3.3 TinyZero (UCB)

**基本信息：**
- **作者:** Jiayi Pan (UCB)
- **GitHub:** https://github.com/Jiayi-Pan/TinyZero
- **成本:** 不到$30 cloud compute
- **硬件:** 单GPU（A100 80GB或RTX 4090）

**实现细节：**
- 使用veRL框架
- 对Qwen-2.5-1.5B/7B应用PPO
- 训练任务：countdown game（用四个数字通过算术运算达到目标值）
- 训练200-400 steps

**涌现行为：**
- 自我验证
- 错误检测后的回溯和修正
- 中间结果的反思
- 扩展的chain-of-thought推理
- "aha moment"行为

**关键结论：**
- 在1.5B参数的小模型上也能通过纯RL涌现出推理能力
- RL方法的选择不重要（PPO/GRPO均可）
- 使用IFT模型初始化收敛更快

---

## 4. QwQ-32B

### 4.1 基本信息

- **模型:** QwQ-32B（Qwen系列推理模型）
- **基础模型:** Qwen2.5-32B
- **参数:** 32.5B（非嵌入参数31.0B）
- **架构:** 64层Transformer，40个Q头，8个KV头（GQA）
- **上下文长度:** 131,072 tokens
- **许可证:** Apache 2.0

### 4.2 两阶段RL训练

QwQ-32B采用独特的两阶段RL训练方法：[^9^]

```
Stage 1: Math & Coding RL
  - 使用outcome-based rewards驱动
  - Math: 使用accuracy verifier验证答案正确性
  - Code: 使用code execution server执行测试用例
  - 观察到性能随训练episode持续提升

Stage 2: General Capability RL
  - 在第一阶段基础上继续训练
  - 使用general reward model + rule-based verifiers
  - 提升instruction following、human preference alignment、agent performance
  - 少量steps即可提升，数学/编码性能无明显下降
```

### 4.3 Benchmark结果

| Benchmark | QwQ-32B | DeepSeek-R1 |
|-----------|---------|-------------|
| AIME 2024 | ~80 | 79.8 |
| LiveCodeBench | ~63 | 65.9 |
| GPQA | ~66 | 71.5 |
| IFEval | 强 | 83.3 |
| BFCL | 强 | - |

---

## 5. Kimi 1.5

### 5.1 基本信息

- **论文:** Kimi k1.5: Scaling Reinforcement Learning with LLMs
- **机构:** Moonshot AI
- **日期:** 2025-01-22
- **arXiv:** 2501.12599

### 5.2 核心创新

Kimi 1.5提出了几个关键创新：[^10^]

**1. Long Context Scaling**
- 将RL上下文窗口扩展到128K
- 观察到性能随上下文长度持续增长
- 使用partial rollouts技术提高训练效率

**2. Improved Policy Optimization**
- 使用long-CoT RL的formulation
- 采用online mirror descent variant进行鲁棒优化
- 有效的sampling strategy、length penalty、data recipe优化

**3. Simplistic Framework**
- 不依赖MCTS、value functions、process reward models
- 纯靠长上下文scaling + 策略优化达到强性能

**4. Multi-modal**
- 在文本和视觉数据上联合训练

### 5.3 四阶段训练流程

```
Stage 1: Pre-training
Stage 2: Vanilla SFT (标准指令微调)
Stage 3: Long-CoT SFT (长链推理监督微调)
  - 构建小规模高质量long-CoT warmup数据集
  - 包含planning、evaluation、reflection、exploration等认知过程
Stage 4: Reinforcement Learning
  - 使用高质量RL prompt set
  - Partial rollouts处理长CoT
```

### 5.4 Partial Rollouts技术

解决长CoT RL训练的关键技术：
- 设定固定输出token预算
- 超过限制的trajectory保存到replay buffer，在下一轮继续
- 只有当前iteration需要on-policy计算
- 之前的segments可从buffer复用
- 异步rollout workers最大化计算效率

### 5.5 Long2Short方法

将长CoT推理能力转移到短CoT模型：

| 方法 | 描述 |
|------|------|
| Model Merging | 长CoT模型与短模型权重平均 |
| Shortest Rejection Sampling | 从长CoT模型生成多个response，选最短正确的 |
| DPO | 短正确解为正样本，长解为负样本 |
| Long2Short RL | 两阶段RL，第二阶段加length penalty |

### 5.6 Benchmark结果

**Long-CoT:**

| Benchmark | Kimi 1.5 | o1 |
|-----------|----------|-----|
| AIME | 77.5 | - |
| MATH 500 | 96.2 | - |
| Codeforces | 94th percentile | - |
| MathVista | 74.9 | - |

**Short-CoT:**

| Benchmark | Kimi 1.5 | GPT-4o | Claude Sonnet 3.5 |
|-----------|----------|--------|-------------------|
| AIME | 60.8 | - | - |
| MATH500 | 94.6 | - | - |
| LiveCodeBench | 47.3 | - | - |

---

## 6. OpenAI o1/o3训练推测

### 6.1 公开信息分析

OpenAI的o1/o3模型采用闭源开发，但基于公开信息和技术报告，可以推测其训练方法：[^11^]

**推测的训练流程：**
1. 为问题域生成Chain of Thought (CoT)
2. 结合人类专家（SFT）和自动化机器（如RL）标注中间CoT步骤
3. 使用标注数据训练base model
4. 在测试时迭代推理，使用process model

**关键技术要素（推测）：**
- Process Reward Model (PRM): 对推理中间步骤进行奖励
- Monte Carlo Tree Search (MCTS): 在推理时进行搜索
- 大规模RL训练
- Human preference alignment

### 6.2 与DeepSeek-R1的对比

| 维度 | OpenAI o1/o3 | DeepSeek-R1 |
|------|-------------|-------------|
| 训练方法 | 闭源 | 开源 |
| CoT生成 | 推测使用MCTS+PRM | 纯GRPO+规则奖励 |
| Inference | 推测使用process model | 直接生成 |
| 是否使用SFT | 是 | R1-Zero无SFT; R1有SFT |
| Test-time compute | 高（o3可达57M tokens） | 中等 |

---

## 7. Code-Specific RLVR

### 7.1 R1-Code-Interpreter

**基本信息：**
- **论文:** R1-Code-Interpreter: LLMs Reason with Code via Supervised and Multi-stage Reinforcement Learning
- **机构:** MIT/Harvard
- **arXiv:** 2505.21668
- **GitHub:** https://github.com/yongchao98/R1-Code-Interpreter

**核心思想：**
训练LLM在推理过程中自主决定何时调用Code Interpreter执行代码，结合文本推理和代码执行解决复杂任务。[^12^]

**训练Pipeline：**
```
Stage 1: SFT
  - 144个推理和规划任务（107训练，37测试）
  - 使用GPT-4o生成6.5K多轮text/code trajectories
  - 只保留正确执行的trajectory

Stage 2: Multi-stage Curriculum RL (GRPO)
  - 根据improvement potential将样本分为4个阶段
  - 高potential样本先训练，逐渐加入低potential样本
  - 平均RL增益从+3.4%提升到+9.3%
```

**奖励设计：**
- 纯outcome-based correctness reward
- 事实推理任务用exact matching
- 规划任务检查约束和目标是否满足
- 不使用format reward（模型已能遵循格式）
- 不使用neural reward model

**实验结果：**
- R1-CI-14B在37个测试任务上从44.1%提升到72.4%
- 超过text-only GPT-4o (58.6%)
- 超过GPT-4o with Code Interpreter (70.9%)
- 涌现出通过代码生成进行自我检查的行为

### 7.2 CodeRL+

**基本信息：**
- **论文:** CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment
- **arXiv:** 2510.18471

**核心创新：**
- 将execution semantics alignment集成到RLVR训练pipeline
- 使模型能推断variable-level execution trajectory
- 提供dense learning signal of execution semantics

**实验结果：**
- 相比RLVR和Distillation baseline平均4.6%相对提升（pass@1）
- 在code-reasoning benchmark上高15.5%
- 在test-output-generation benchmark上高4.4%

### 7.3 SpecRL

**基本信息：**
- **论文:** SpecRL: Reinforcement Learning for Program Specification Synthesis
- 针对形式化验证中的specification synthesis
- 使用GRPO + domain-specific reward

**奖励设计：**
- Progressive reward measuring completeness from negative-test rejection rates
- Offline spectest construction pipeline

---

## 8. Zero-RL方法

### 8.1 定义与核心思想

Zero-RL（Zero Reinforcement Learning）指直接从pretrained base model开始RL优化，不经过任何SFT中间步骤的训练范式。这一范式的关键发现来自DeepSeek-R1-Zero。[^13^]

**核心发现：**
- 大模型的长思考能力（反思、自我验证）可以单纯通过RL激发，不需要SFT
- 只需要简单的correctness-based rule rewards
- 数据难度需要与模型能力对齐
- 稳定的RL算法（如GRPO）至关重要

### 8.2 关键研究对比

| 研究 | 模型大小 | 是否成功涌现推理 | 关键发现 |
|------|----------|------------------|----------|
| DeepSeek-R1-Zero | 671B (37B activated) | 是 | 首个证明纯RL可激发推理 |
| SimpleRL-Zoo | 0.5B-32B | 多数可以 | 不同模型有不同训练动态 |
| TinyZero | 1.5B-7B | 是 | $30成本即可复现 |
| VisualThinker-R1-Zero | 2B (VLM) | 是 | 首次在多模态复现 |

### 8.3 不同Base模型的训练动态

SimpleRL-Zoo的研究发现：[^14^]
- **Qwen2.5系列:** 已有强instruction following和self-reflection，可能掩盖true zero RL effects
- **Mistral系列:** 首次观察到非Qwen模型的"aha moment"
- **Llama系列:** 需要更多调整format reward和数据难度
- **通用规律:**
  - 增加response length不总是与verification等认知行为涌现相关
  - 需要仔细监控training dynamics
  - 不同模型有distinct patterns

---

## 9. 其他重要相关方法

### 9.1 s1 / s1.1

**基本信息：**
- **论文:** s1: Simple test-time scaling
- **作者:** Muennighoff et al. (Stanford)
- **arXiv:** 2501.19393

**核心方法：**
- 仅使用1,000个高质量推理样本（s1K数据集）进行SFT
- 开发**budget forcing**技术控制test-time compute
  - Truncation: 超过思考token限制强制结束思考
  - Extrapolation: 模型试图结束时追加"Wait"促使继续思考
- 在Qwen2.5-32B-Instruct上训练26分钟（16 H100 GPUs）

**s1.1改进：**
- 使用DeepSeek-R1重新生成s1K的reasoning traces（s1K-1.1）
- 性能显著优于s1

**Budget Forcing效果：**
- s1-32B在AIME24上从50%提升到57%（使用budget forcing）
- 超过o1-preview在MATH和AIME24上（最高27%）

### 9.2 DeepScaleR

**基本信息：**
- 使用iterative scaling of GRPO扩展thinking length（8K -> 16K -> 24K）
- 1.5B模型DeepScaleR-1.5B-Preview超越o1-Preview，在AIME上达43.1%

**训练细节：**
- 基于DeepSeek-R1-Distill-Qwen-1.5B
- 训练数据40K math QA pairs（来自AIME 1984-2023、AMC、Omni-MATH、STILL）
- 使用8K->16K->24K的渐进式context length scaling

### 9.3 DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization)

**改进点：**
- 提出higher clipping threshold
- Dynamic sampling strategy: 过滤zero-advantage samples
- 解决GRPO中所有response都正确/错误时advantage collapse的问题
- 采用clip-higher和no KL-loss策略

### 9.4 训练数据对比

| 模型/方法 | 训练数据 | 数据量 | 数据来源 |
|-----------|----------|--------|----------|
| DeepSeek-R1 | 数学、代码、推理 | 数十万 | AIME、MATH、LeetCode等 |
| DeepScaleR | 数学 | 40K | AIME 1984-2023、AMC、Omni-MATH、STILL |
| s1/s1.1 | 数学推理 | 1K | 精选高质量问题 |
| R1-Code-Interpreter | 推理+代码 | 6.5K | 144个多样化任务 |
| SimpleRL-Zoo | 数学 | 15K | GSM8K、MATH子集 |
| Open-R1 | 多样化 | 持续扩展 | 社区贡献 |

---

## 10. 奖励设计对比

### 10.1 各方法奖励设计

| 方法 | Accuracy Reward | Format Reward | Length Penalty | Language Consistency | Other |
|------|----------------|---------------|----------------|---------------------|-------|
| DeepSeek-R1-Zero | 0/1 | 0/1 | No | No | - |
| DeepSeek-R1 | 0/1 | 0/1 | No | Yes | - |
| QwQ-32B Stage 1 | 0/1 | No | No | No | Code execution |
| QwQ-32B Stage 2 | General RM | Rule-based | No | No | - |
| Kimi 1.5 | 0/1 | No | Yes | No | - |
| R1-Code-Interpreter | 0/1 | No | No | No | - |
| CodeRL+ | Execution-based | No | No | No | Variable-level alignment |
| DAPO | 0/1 | No | No | No | Dynamic sampling |

### 10.2 奖励设计演进

1. **简单规则奖励（DeepSeek-R1-Zero）**: 仅accuracy + format
2. **加入语言一致性（DeepSeek-R1）**: 解决language mixing
3. **加入长度惩罚（Kimi 1.5）**: 控制推理长度
4. **加入curriculum（R1-Code-Interpreter）**: 按improvement potential分阶段训练
5. **加入dense signal（CodeRL+）**: variable-level execution semantics
6. **动态采样（DAPO）**: 解决zero-variance问题

---

## 11. 训练Pipeline对比

### 11.1 各方法Pipeline

| 方法 | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|------|---------|---------|---------|---------|
| DeepSeek-R1-Zero | Base Model | RL (GRPO) | - | - |
| DeepSeek-R1 | Cold-start SFT | RL (GRPO) | Rejection Sampling + SFT | RL Alignment |
| QwQ-32B | Cold-start | Math/Code RL | General Capability RL | - |
| Kimi 1.5 | Pretrain | Vanilla SFT | Long-CoT SFT | RL |
| s1.1 | SFT (1K samples) | Budget Forcing (inference) | - | - |
| R1-Code-Interpreter | SFT (6.5K) | Multi-stage Curriculum RL | - | - |
| TinyZero | Base Model | PPO RL | - | - |

### 11.2 冷启动策略对比

| 方法 | 冷启动类型 | 数据量 | 目的 |
|------|-----------|--------|------|
| DeepSeek-R1-Zero | 无冷启动 | 0 | 纯RL探索 |
| DeepSeek-R1 | Long-CoT SFT | ~K级 | 加速收敛，提高可读性 |
| QwQ-32B | Cold-start checkpoint | - | 初始化模型 |
| Kimi 1.5 | Long-CoT SFT | 小规模高质量 | Prime推理策略 |
| R1-Code-Interpreter | SFT | 6.5K | 学习code/text交互 |

---

## 12. Benchmark结果综合对比

### 12.1 主模型对比

| 模型 | AIME 2024 | AIME 2025 | MATH-500 | GPQA | LiveCodeBench | CodeForces |
|------|-----------|-----------|----------|------|---------------|------------|
| DeepSeek-R1 | 79.8 | - | 97.3 | 71.5 | 65.9 | 2029 |
| QwQ-32B | ~80 | - | ~97 | ~66 | ~63 | - |
| Kimi 1.5 (Long-CoT) | 77.5 | - | 96.2 | - | 47.3* | 94th %ile |
| o1-1217 | 79.2 | - | 96.4 | 75.7 | 63.4 | 2061 |
| o3 (high) | - | - | - | - | - | - |

### 12.2 蒸馏/小模型对比

| 模型 | AIME 2024 | MATH-500 | GPQA | LiveCodeBench |
|------|-----------|----------|------|---------------|
| Distill-Qwen-7B | 55.5 | 92.8 | 49.1 | 37.6 |
| Distill-Qwen-32B | 72.6 | 94.3 | 62.1 | 57.2 |
| Distill-Llama-70B | 70.0 | 94.5 | 65.2 | 57.5 |
| s1.1-32B | ~67 | ~94 | - | - |
| DeepScaleR-1.5B | 43.1 | - | - | - |

### 12.3 复现工作对比

| 工作 | 基础模型 | 训练方法 | AIME 2024 | 成本 |
|------|----------|----------|-----------|------|
| Open-R1 | Qwen2.5系列 | GRPO | 有前景 | - |
| SimpleRL-Zoo | 多种0.5B-32B | GRPO ZeroRL | 显著提升 | - |
| TinyZero | Qwen-2.5-1.5B | PPO | countdown任务 | <$30 |
| VisualThinker-R1-Zero | Qwen2-VL-2B | RL | CVBench 59.47% | - |

---

## 13. 关键论文列表

| 论文 | arXiv | 会议 | 核心贡献 |
|------|-------|------|----------|
| DeepSeek-R1 | 2501.12948 | 技术报告 | GRPO纯RL激发推理 |
| Kimi k1.5 | 2501.12599 | 技术报告 | Long context scaling RL |
| QwQ-32B | - | 技术报告 | 两阶段RL |
| SimpleRL-Zoo | 2503.18892 | 预印 | ZeroRL系统研究 |
| s1: Simple test-time scaling | 2501.19393 | ICML 2025 | Budget forcing |
| R1-Code-Interpreter | 2505.21668 | - | Code reasoning RL |
| CodeRL+ | 2510.18471 | - | Execution semantics alignment |
| DAPO | - | - | Dynamic sampling GRPO |
| DeepScaleR | - | - | Iterative context scaling |
| TinyZero | - | - | 低成本R1-Zero复现 |
| VisualThinker-R1-Zero | 2503.05132 | - | 多模态R1-Zero复现 |
| Revisiting GRPO | 2505.22257 | - | On/Off-policy GRPO分析 |
| Open-R1 | - | GitHub | 开源复现pipeline |

---

## 14. 关键实现框架

| 框架 | 支持算法 | 特点 |
|------|----------|------|
| TRL (HuggingFace) | GRPO, PPO, DPO | 最广泛使用的开源库 |
| veRL | GRPO, DAPO, PPO | 高效、支持多模态 |
| OpenRLHF | GRPO, PPO, REINFORCE++ | 分布式训练 |
| Open-R1 | GRPO | DeepSeek-R1复现 |
| EasyR1 | GRPO, REINFORCE++, RLOO | Open-R1 fork |

---

## 15. 总结与趋势

### 15.1 关键结论

1. **RLVR范式有效:** 简单的规则奖励（正确性+格式）足以激发强大的推理能力
2. **纯RL可行:** DeepSeek-R1-Zero证明不需要SFT即可涌现出推理能力
3. **Scaling很重要:** 上下文长度scaling是继续提升的关键维度
4. **蒸馏有效:** 大模型的推理模式可以有效转移到小模型
5. **数据质量>数量:** s1用1K样本即达到强性能
6. **涌现行为可预期:** 自我验证、反思等是RL训练的普遍产物

### 15.2 未来方向

1. **更高效的RL算法:** DAPO、Dr.GRPO等改进持续涌现
2. **多模态RLVR:** 视觉推理、GUI agent等扩展
3. **Long2Short:** 将长推理能力压缩到短推理
4. **Process Supervision:** 从outcome reward向process reward演进
5. **Test-time Compute Scaling:** Budget forcing等技术
6. **Safety:** RLVR训练中的hallucination和reward hacking问题

---

## 参考文献

[^1^]: Lambert et al., "Reinforcement Learning with Verifiable Rewards," 2024. https://arxiv.org/abs/2405.14746

[^2^]: Guo et al., "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning," arXiv:2501.12948, 2025.

[^3^]: DeepSeek-AI, "DeepSeek-R1 Technical Report," 2025.

[^4^]: Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models," 2024.

[^5^]: DeepSeek-R1技术报告, Section 2.2.2 Reward Modeling.

[^6^]: DeepSeek-R1技术报告, Section 2.2.1 Emergent Behaviors.

[^7^]: DeepSeek-R1技术报告, Section 3.2 Distilled Model Evaluation.

[^8^]: Zeng et al., "SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild," arXiv:2503.18892, 2025.

[^9^]: Qwen Team, "QwQ-32B: Embracing the Power of Reinforcement Learning," 2025. https://qwenlm.github.io/blog/qwq-32b/

[^10^]: Kimi Team, "Kimi k1.5: Scaling Reinforcement Learning with LLMs," arXiv:2501.12599, 2025.

[^11^]: ARC Prize Analysis, "DeepSeek-R1 vs OpenAI o1 & o3," 2025.

[^12^]: Chen et al., "R1-Code-Interpreter: LLMs Reason with Code via Supervised and Multi-stage Reinforcement Learning," arXiv:2505.21668, 2025.

[^13^]: DeepSeek-R1-Zero训练分析, 2025.

[^14^]: SimpleRL-Zoo实验分析, 2025.
