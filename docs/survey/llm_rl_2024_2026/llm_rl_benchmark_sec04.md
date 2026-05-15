## 4. RLVR可验证奖励训练与推理模型

Reinforcement Learning with Verifiable Rewards（RLVR，可验证奖励强化学习）是2024—2025年大语言模型后训练领域最具变革性的范式之一。与依赖人类偏好标注的传统RLHF不同，RLVR利用数学答案正确性、代码测试通过等可自动验证的结果作为奖励信号，直接驱动策略优化。这一范式 shift 的核心在于：模型通过与可验证答案的交互自主探索解题路径，而非模仿人类标注的思维链。DeepSeek-R1/R1-Zero的出现标志着该范式的成熟——它不仅证明了纯RL可以激发大规模语言模型的复杂推理能力，更催生了以GRPO（Group Relative Policy Optimization）为核心的训练框架生态，使得从671B参数的超大型模型到1.5B的小模型均可通过RLVR获得显著的能力提升。

### 4.1 DeepSeek-R1/R1-Zero训练解析

#### 4.1.1 R1-Zero：纯RL训练的里程碑

DeepSeek-R1-Zero是首个完全通过强化学习、不经过任何监督微调（Supervised Fine-Tuning, SFT）冷启动而直接从Base模型训练获得强推理能力的公开研究。[^2^] 该实验从DeepSeek-V3-Base（671B参数MoE架构，37B激活参数）出发，仅使用GRPO算法配合简单的规则奖励进行大规模RL训练。在AIME 2024（American Invitational Mathematics Examination）上，模型Pass@1准确率从初始的15.6%提升至71.0%，通过多数投票（majority voting）可进一步达到86.7%——这一水平已接近OpenAI-o1-0912的表现。

R1-Zero的核心意义在于验证了LLM的推理能力可以通过纯粹的RL激励完全激活，无需SFT提供任何先验推理模式。训练过程中，研究者观察到了一系列自发涌现的复杂认知行为：早期response长度稳定增长，中期出现自我验证（self-verification）和反思（reflection），中后期则发展出错题回溯（backtracking）和多解法尝试。[^6^] 其中最具代表性的是所谓的"Aha Moment"——模型会以拟人化口吻重新思考（如"Wait, let me reconsider..."），标志着模型开始自主发现更优的解题策略。这些行为并非预先设计，而是RL优化过程中自然演化的结果，表明预训练模型中潜藏的推理能力可以通过恰当的奖励机制被有效"挖掘"。

#### 4.1.2 R1完整版四阶段Pipeline

在R1-Zero验证纯RL可行性之后，DeepSeek-R1完整版采用四阶段训练Pipeline，在保留推理能力的同时显著改善输出的可读性和通用性：[^3^]

![R1四阶段Pipeline](r1_pipeline_fig.png)

**表1：DeepSeek-R1四阶段训练Pipeline详细流程**

| 阶段 | 输入 | 处理方法 | 输出 | 关键参数/说明 |
|------|------|----------|------|--------------|
| Stage 1: Cold Start SFT | DeepSeek-V3-Base + 数千条高质量长CoT样本 [^3^] | Few-shot prompting、人工标注与R1-Zero生成样本混合，对Base模型SFT | Cold Start模型 | 数据量约K级，目的为加速RL前期收敛并提高可读性 |
| Stage 2: 推理导向RL | Cold Start模型 | 使用GRPO算法大规模RL训练，覆盖数学/代码/推理任务，引入语言一致性奖励 | 推理RL Checkpoint | 同R1-Zero的RL方法，额外解决语言混合问题 |
| Stage 3: Rejection Sampling + SFT | Stage 2 RL checkpoint | Rejection Sampling收集600K推理样本 + 200K非推理样本（写作、QA等），在**V3-Base**上重新SFT | SFT推理模型 | 关键设计：在Base而非RL模型上重训，避免RL偏差累积 |
| Stage 4: 对齐RL | Stage 3 SFT模型 | 在人类偏好数据上进行RL，优化helpfulness与harmlessness | DeepSeek-R1最终模型 | 使用RLHF风格的对齐训练，保持Stage 3获得的推理能力 |

Stage 3的设计尤为关键。DeepSeek团队选择将Stage 2 RL checkpoint生成的样本用于在**V3-Base**上重新SFT，而非在RL模型上继续训练。这一决策基于一个重要的经验观察：大规模RL训练虽然能产生强推理能力，但同时会引入可读性下降、语言混合等副产品。通过在干净的Base模型上重新学习，可以在保留核心推理能力的同时获得更规范的输出格式。[^3^] Stage 4的对齐RL则进一步确保模型在通用对话和安全性方面的表现，形成推理能力与通用能力的平衡。

#### 4.1.3 奖励设计：正确性、格式与语言一致性

DeepSeek-R1-Zero的奖励系统仅包含两种规则奖励，刻意避免使用神经网络奖励模型（包括Outcome Reward Model和Process Reward Model），因为后者在大规模RL训练中容易出现reward hacking。[^5^]

**正确性奖励（Accuracy Reward）**是核心驱动信号。对于数学问题，系统通过规则验证器（如SymPy符号计算）判断最终答案是否正确，正确得1分，错误得0分；对于代码问题，则通过测试用例执行结果判定。这种二元奖励虽然简单，但在GRPO的组内标准化机制下能够有效传播学习信号。

**格式奖励（Format Reward）**要求推理过程必须放在`<think>`和`</think>`标签内。这一设计的目的是促使模型形成结构化的思考模式，而非直接输出答案。格式奖励同样为二元值——符合格式得1分，否则得0分。

在R1完整版的Stage 2中，团队额外引入了**语言一致性奖励（Language Consistency Reward）**，用于解决R1-Zero中出现的语言混合（language mixing）问题——即模型在推理过程中在不同语言间频繁切换，影响输出可读性。该奖励通过检测目标语言的一致性来计算，与前两项奖励共同构成三元奖励系统。实验表明，语言一致性奖励的加入显著改善了输出的可读性，同时对推理性能无显著负面影响。

**表2：DeepSeek-R1系列奖励设计演进**

| 组件 | R1-Zero | R1完整版 | 说明 |
|------|---------|----------|------|
| 正确性奖励 | ✓ (0/1) | ✓ (0/1) | 数学规则验证/代码测试用例 [^5^] |
| 格式奖励 | ✓ (0/1) | ✓ (0/1) | `<think>`标签约束 |
| 语言一致性奖励 | ✗ | ✓ | 解决语言混合问题 |
| 神经网络RM | ✗ | ✗ | 避免reward hacking |

### 4.2 开源复现生态

DeepSeek-R1的技术报告发布后，开源社区迅速展开了大规模复现工作。这些复现不仅验证了R1训练方法的可重复性，更在降低训练成本、适配小模型等方面取得了重要突破。

**表3：DeepSeek-R1开源复现项目对比**

| 项目 | 机构 | 基础模型 | 大小 | RL算法 | 训练成本 | AIME结果 | 硬件配置 |
|------|------|----------|------|--------|----------|----------|----------|
| Open-R1 [^3^] | HuggingFace | Qwen2.5系列 | 7B-32B | GRPO | — | 复现中 | 8 GPU (单节点) / N+1节点 |
| TinyZero [^2^] | UCB | Qwen-2.5 | 1.5B/7B | PPO | **<$30** | Countdown任务 | **单GPU** (A100 80GB/RTX 4090) |
| Open-RS | Knovel等 | R1-Distill-Qwen | 1.5B | GRPO | **$42** | **46.7%** [^8^] | 4×A40 48GB (24h) |
| SimpleRL-Zoo [^8^] | HKUST | 10种模型 | 0.5B-32B | GRPO | 可变 | 显著提升 | 可变 |
| DeepScaleR | Agentica | R1-Distill-Qwen | 1.5B | GRPO | ~$3,629 | **43.1%** | 8→32×A100 80GB |

#### 4.2.1 Open-R1（HuggingFace）

Open-R1是HuggingFace发起的完全开源复现项目，目标是重建DeepSeek-R1的完整训练pipeline。[^3^] 项目基于TRL（Transformer Reinforcement Learning）库实现GRPO算法，采用vLLM作为推理后端支持多节点训练。Open-R1的实施分为三个步骤：首先复现R1-Distill模型（从DeepSeek-R1蒸馏高质量语料），其次复现纯RL pipeline（创建大规模数学、推理、代码数据集），最后实现从Base模型到RL-tuned的多阶段训练。项目提供单节点训练支持（8 GPU使用`vllm_mode="colocate"`）以及多节点训练方案（N+1节点架构，1节点专用于vLLM server），并配套Slurm调度脚本。Open-R1的意义在于将工业级RL训练流程标准化为可复现的开源方案，降低了研究者和开发者进入RLVR领域的门槛。

#### 4.2.2 TinyZero（UCB）

TinyZero由加州大学伯克利分校的Jiayi Pan开发，是DeepSeek R1-Zero的最小开源复现。[^2^] 该项目的核心贡献在于证明了R1-Zero的关键发现——纯RL可激发推理能力——可以在极低预算和单GPU条件下复现。TinyZero使用veRL框架对Qwen-2.5-1.5B/7B模型应用PPO算法，在Countdown算术任务（给定四个数字通过加减乘除运算达到目标值）上训练200-400步。总训练成本不到$30云算力，硬件仅需单张A100 80GB或RTX 4090。

尽管任务复杂度远低于AIME数学竞赛，TinyZero成功复现了R1-Zero的全部涌现行为——自我验证、错误检测后的回溯修正、中间结果的反思以及延长的思维链推理。项目的核心结论具有方法论意义：RL算法的选择（PPO或GRPO）并非关键因素，重要的是可验证奖励机制本身；同时，使用经过指令微调（IFT）的模型初始化可以加速收敛，但从Base模型直接训练同样可行。TinyZero为资源有限的研究者提供了一个低成本的RLVR实验平台。

#### 4.2.3 Open-RS：小模型RL的性价比标杆

Open-RS项目在小模型RL训练的效率优化方面取得了突出成果。该项目以DeepSeek-R1-Distill-Qwen-1.5B为基础模型，使用GRPO算法配合精心设计的**余弦奖励函数**（Cosine Reward），仅消耗7,000个训练样本和$42云算力（4×A40 GPU训练24小时），就在AIME 2024上达到46.7%的Pass@1分数——超越o1-preview的44.6%。[^8^]

Open-RS的奖励设计包含三个组件：准确性奖励（二元判断答案正确性）、余弦奖励（基于response长度的余弦调度缩放准确性奖励，有效控制推理冗长性）以及格式奖励（要求推理过程封装在`<thinking>`标签中）。实验发现，小模型在前50-100步快速获得推理能力，但随后可能因过优化（over-optimization）而退化。为此，Open-RS提出混合难度级别和课程式策略缓解此问题，AMC23准确率从训练前的63%提升至80%，AIME24从约28%提升至46.7%。与DeepScaleR（$3,629成本、43.1% AIME分数）相比，Open-RS以不到1/80的成本实现了更高的性能，展示了蒸馏+RL两阶段策略在小模型上的高效性。

### 4.3 其他重要推理模型

#### 4.3.1 QwQ-32B：两阶段RL与参数效率

QwQ-32B是阿里云Qwen团队推出的推理模型，以32B参数规模达到了与671B参数的DeepSeek-R1相匹敌的推理性能，展示了参数效率与训练方法的协同优化。[^9^] QwQ-32B基于Qwen2.5-32B基础模型，采用独特的两阶段RL训练策略。

第一阶段聚焦数学与代码能力，使用outcome-based rewards驱动：数学任务通过accuracy verifier验证答案正确性，代码任务通过code execution server执行测试用例。第二阶段在第一阶段基础上继续训练，引入general reward model和rule-based verifiers，提升instruction following、human preference alignment和agent performance。实验显示，第二阶段的通用能力训练仅需少量步骤即可实现，且数学/编码性能无明显下降。

QwQ-32B的成功表明，RLVR的核心优势——通过可验证奖励精准优化特定能力——可以有效放大基础模型的性能。在AIME 2024上达到约80%、LiveCodeBench约63%的表现，QwQ-32B以约1/21的参数规模（32B vs 671B）接近甚至部分超越了DeepSeek-R1的水平，证明了RL训练方法在弥补模型规模差距方面的巨大潜力。[^9^]

#### 4.3.2 Kimi 1.5：长上下文Scaling与Partial Rollouts

Kimi 1.5由Moonshot AI开发，其核心创新在于将RL的上下文窗口扩展到128K tokens，并系统研究了长上下文scaling对推理能力的提升作用。[^10^] 与DeepSeek-R1不同，Kimi 1.5不依赖MCTS（Monte Carlo Tree Search）、value functions或process reward models，纯靠长上下文scaling配合策略优化达到强性能。

Kimi 1.5的训练流程包含四个阶段：Pre-training → Vanilla SFT（标准指令微调）→ Long-CoT SFT（长链推理监督微调，构建包含planning、evaluation、reflection等认知过程的warmup数据）→ Reinforcement Learning。在RL阶段，Kimi 1.5引入了**Partial Rollouts**技术解决长CoT RL训练的效率瓶颈：设定固定输出token预算，超过限制的trajectory保存到replay buffer在下一轮继续，只有当前iteration需要on-policy计算，之前的segments可从buffer复用。这一机制配合异步rollout workers最大化计算效率，使得128K上下文的RL训练在工程上成为可能。

此外，Kimi 1.5提出了**Long2Short**方法将长CoT推理能力迁移到短CoT模型，包括模型权重合并（model merging）、最短拒绝采样（shortest rejection sampling）、DPO（以短正确解为正样本、长解为负样本）以及带length penalty的两阶段RL四种技术。Long-CoT模式在AIME上达到77.5%，MATH-500达到96.2%；Short-CoT模式下AIME仍达60.8%，展现了长上下文训练成果向短推理场景的有效迁移。[^10^]

**表4：主要推理模型Benchmark对比**

| 模型 | AIME 2024 | AIME 2025 | MATH-500 | GPQA Diamond | HumanEval | 参数量 |
|------|-----------|-----------|----------|--------------|-----------|--------|
| DeepSeek-R1 [^3^] | **79.8%** | — | **97.3%** | **71.5%** | — | 671B |
| QwQ-32B [^9^] | ~80% | — | ~97% | ~66% | — | 32B |
| Kimi 1.5 Long-CoT [^10^] | 77.5% | — | 96.2% | — | — | — |
| Kimi 1.5 Short-CoT [^10^] | 60.8% | — | 94.6% | — | — | — |
| o1-1217 | 79.2% | — | 96.4% | 75.7% | — | — |
| o3 (high) | — | — | — | — | — | — |
| Open-RS 1.5B [^8^] | 46.7% | — | — | — | — | 1.5B |

表4的数据揭示了推理模型领域的关键格局。DeepSeek-R1与QwQ-32B在AIME 2024和MATH-500上均处于第一梯队，但QwQ-32B以仅32B的参数实现了与671B的R1相近的水平，参数效率显著更优。Kimi 1.5的Long-CoT模式在AIME上达到77.5%，虽略低于R1和QwQ-32B，但其Short-CoT模式的60.8%展示了从长到短能力迁移的实际价值。值得注意的是，Open-RS 1.5B以$42的训练成本达到46.7%的AIME分数，超越o1-preview的44.6%，标志着小模型RL已具备实用的性价比优势。

### 4.4 代码推理RLVR

#### 4.4.1 R1-Code-Interpreter：代码工具集成增强推理

R1-Code-Interpreter（R1-CI）由MIT与Harvard联合开发，将RLVR范式扩展到代码解释器工具的使用场景，训练LLM在推理过程中自主决定何时调用Code Interpreter执行代码。[^12^] 该研究的核心洞见是：文本推理与代码执行的互补结合可以解决单独使用任一方式都难以处理的复杂任务。

R1-CI的训练采用两阶段策略。第一阶段为SFT冷启动：在144个推理和规划任务（107个训练任务、37个测试任务）上，使用GPT-4o生成6.5K多轮text/code trajectories，仅保留正确执行的trajectory。第二阶段为多阶段课程RL（GRPO）：根据improvement potential将样本分为4个阶段，高potential样本先训练，逐渐加入低potential样本。这种课程设计使平均RL增益从+3.4%提升到+9.3%。[^12^]

奖励设计方面，R1-CI采用纯outcome-based correctness reward：事实推理任务使用exact matching，规划任务检查约束和目标是否满足。值得注意的是，训练过程中**不使用format reward**（SFT冷启动已使模型掌握格式要求）和**不使用neural reward model**，这与DeepSeek-R1的设计哲学一致。实验结果显示，R1-CI-14B在37个测试任务上从Base模型的44.1%提升至72.4%，超过text-only GPT-4o（58.6%）和GPT-4o with Code Interpreter（70.9%）。更值得关注的是，模型涌现出通过代码生成进行自我检查的行为——主动编写测试代码验证自身推理的正确性，这种元认知能力是纯文本RLVR难以激发的。

#### 4.4.2 代码RLVR与数学RLVR的训练差异与共享机制

代码推理RLVR与数学推理RLVR在训练机制上存在结构性差异。数学RLVR的验证器相对简单——通过符号计算（如SymPy）或数值比较即可判定答案正确性，验证过程确定性高且计算开销低。代码RLVR的验证器则需要完整的代码执行环境：编译/解释代码、运行测试用例、捕获输出和异常，执行环境的状态管理和安全性约束显著增加了系统复杂度。

在奖励密度方面，数学RLVR通常为稀疏奖励（仅最终结果正确性），代码RLVR同样以测试通过与否作为核心奖励信号。但CodeRL+等研究表明，将execution semantics alignment集成到RLVR训练pipeline——使模型能推断variable-level execution trajectory——可提供dense learning signal，相比RLVR基线平均实现4.6%的相对提升（pass@1）。[^12^]

两类RLVR共享的核心机制是GRPO算法框架和outcome-based reward设计。Search-R1进一步将RLVR范式扩展到搜索工具的使用，通过检索token掩码（retrieved token masking）技术实现稳定RL训练——在RL优化中屏蔽检索回来的token，只更新LLM生成token的梯度。这一技术为工具集成RLVR提供了通用的工程解决方案。从数学到代码再到搜索工具的扩展轨迹表明，RLVR范式的核心思想——可验证奖励驱动策略优化——具有跨领域的通用性，其成功关键在于为每个领域设计可靠的验证器和适配的训练基础设施。
