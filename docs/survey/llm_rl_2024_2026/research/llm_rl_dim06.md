# 研究维度6: Agent RL与工具使用（ToolRL, Search-R1, AgentPRM等）

## 深度调研报告

---

## 1. 概述

Agent RL是LLM+RL领域的重要新兴方向，涉及工具使用、多轮交互、搜索增强等场景。与单轮推理RL（如数学推理）不同，Agent RL需要处理：(1) 多轮交互和延迟奖励；(2) 部分可观测环境；(3) 工具调用和环境反馈；(4) 信用分配跨越多个轮次。本报告系统调研2024-2025年该领域的核心论文和方法。

---

## 2. 核心论文详解

### 2.1 ToolRL: Reward is All Tool Learning Needs

```
Claim: 首次对工具使用RL中的奖励设计进行系统研究，提出原则化的奖励设计框架，使用GRPO训练在工具调用基准上比基础模型提升17%，比SFT模型提升15%
Source: ToolRL: Reward is All Tool Learning Needs
URL: https://arxiv.org/abs/2504.13958
Date: 2025-04-16 (arXiv)
Excerpt: "We present the first comprehensive study on reward design for tool selection and application tasks within the RL paradigm. We systematically explore a wide range of reward strategies, analyzing their types, scales, granularity, and temporal dynamics."
Context: 来自UIUC，关注通用工具选择和应用任务的RL训练，突破现有研究局限于特定工具（搜索、代码工具）的限制
Confidence: high
```

**Agent环境定义：**
- **动作空间**: JSON格式的工具调用，包含工具名称、参数名称和参数内容
- **观察空间**: 工具描述、用户查询、历史工具调用轨迹
- **奖励设计**: 
  - 格式奖励 R_format ∈ {0,1}：评估输出是否遵循指定格式
  - 正确性奖励 R_correctness ∈ [-3,3]：细粒度比较预测工具调用与真实工具调用的工具名、参数名、参数内容
  - 总奖励 R = R_format + R_correctness

**工具类型**: 通用API/函数调用，涵盖多样化工具集（从ToolACE、xLAM、Hammer数据集混合）

**多轮交互机制**: 支持多步工具调用，每轮可调用多个工具，具有不同的任务类型

**使用的Benchmark:**
- BFCL V3（Berkeley Function Call Leaderboard）：4K+实例，包含Non-live、Live、Multi-turn子集
- API-Bank：三级评估框架，73个多样化API工具
- Bamboogle：QA基准，测量最终答案准确率

**具体实现细节：**
- 模型: Qwen-2.5-Instruct系列（7B, 14B）, Llama-3.2-Instruct
- 训练数据: 4K混合样本（2K ToolACE + 1K Hammer Masked + 1K xLAM）
- RL算法: GRPO（主要）和PPO
- 框架: veRL
- GPU: 2×A100(80G) per run
- Batch size: 512，每查询生成4个响应，训练15个epoch
- Temperature: 1.0（rollout），移除KL正则化以鼓励探索
- 训练设置: 支持冷启动（无SFT）和SFT初始化两种模式

**关键发现：**
- 奖励类型：细粒度奖励（分解工具名、参数名、参数内容）优于粗粒度奖励
- 奖励规模：动态平滑奖励规模促进从简单到复杂行为的平滑过渡
- 冷启动GRPO训练有效，无需大量SFT数据
- 展现涌现行为：主动性（proactiveness）和元认知推理

---

### 2.2 Search-R1: Training LLMs to Reason and Leverage Search Engines with RL

```
Claim: 提出Search-R1框架，使LLM能够与搜索引擎交错进行多轮推理和检索，通过检索token掩码实现稳定RL训练，使用简单结果奖励，在7个QA数据集上比RAG基线提升41%（Qwen2.5-7B）
Source: Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning
URL: https://arxiv.org/abs/2503.09516
Date: 2025-03-12 (arXiv)
Excerpt: "Search-R1 introduces the following key innovations: (1) We model the search engine as part of the environment, enabling rollout sequences that interleave LLM token generation with search engine retrievals. (2) Search-R1 supports multi-turn retrieval and reasoning. (3) We adopt a straightforward outcome-based reward function."
Context: 来自 Indiana University / UW，扩展DeepSeek-R1范式到搜索增强推理
Confidence: high
```

**Agent环境定义：**
- **动作空间**: LLM生成<search>查询</search>、<think>推理</think>、<answer>答案</answer>格式的token序列
- **观察空间**: 当前问题 + 历史检索结果（包裹在<information>标签中）
- **奖励设计**: 简单的结果奖励——最终答案正确性（exact match或包含判断）

**工具类型**: 搜索引擎（FAISS本地检索器 + API在线搜索服务）

**多轮交互机制**: 
- 支持多轮交错推理和搜索
- 搜索由<search>和</search>token显式触发
- 检索内容包裹在<information>标签中
- LLM推理步骤包裹在<think>标签中

**使用的Benchmark:**
- 7个QA数据集：NQ, TriviaQA, PopQA（单跳）; HotpotQA, 2WikiMultiHopQA, MuSiQue, Bamboogle（多跳）

**具体实现细节：**
- 模型: Qwen2.5-7B-Instruct, Qwen2.5-3B-Instruct
- RL算法: PPO和GRPO（兼容多种RL算法）
- 关键创新: **Retrieved Token Masking**——在RL优化中屏蔽检索回来的token，只更新LLM生成的token，确保稳定训练
- 使用E5作为检索器
- Rollout group size: 5（Search QA设置）
- 最大轮次: 4轮

**关键发现：**
- 检索token掩码对稳定训练至关重要
- 简单结果奖励在搜索推理场景中足够有效
- 模型学会自验证：在已有足够信息后执行额外检索步骤验证结论
- RL训练使模型能够动态调整检索策略

---

### 2.3 R1-Searcher / R1-Searcher++

```
Claim: 提出两阶段RL方法（SFT冷启动 + RL动态知识获取），使LLM学会在内部知识和外部搜索之间自适应切换，减少42.9%的检索次数，同时超越基线4.3%
Source: R1-Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs via Reinforcement Learning
URL: https://arxiv.org/abs/2505.17005
Date: 2025-05-22 (arXiv)
Excerpt: "R1-Searcher++ employs a two-stage training strategy: an initial SFT Cold-start phase for preliminary format learning, followed by RL for Dynamic Knowledge Acquisition. The RL stage uses outcome-supervision to encourage exploration, incorporates a reward mechanism for internal knowledge utilization."
Context: 来自中国人民大学高瓴人工智能学院
Confidence: high
```

**Agent环境定义：**
- **动作空间**: 搜索查询生成、推理、最终答案
- **观察空间**: 问题文本 + 检索结果
- **奖励设计**: 
  - 结果奖励（答案正确性）
  - 内部知识利用奖励（鼓励在自信时依赖内部知识）
  - 记忆机制（将检索到的知识内化）

**工具类型**: 搜索引擎

**多轮交互机制**: 两阶段训练——第一阶段学习基本格式，第二阶段通过RL学习动态知识获取策略

**使用的Benchmark:**
- 多个检索中心基准（具体数据集在原始R1-Searcher论文中）

**具体实现细节：**
- 模型: Qwen-2.5-7B-Instruct
- 训练策略: 拒绝采样收集格式正确的数据进行SFT冷启动，然后进行RL训练
- 引入记忆机制使模型保留训练过程中遇到的知识

**关键发现：**
- 模型学会在内部知识和外部检索之间自适应切换
- 检索次数减少42.9%，同时性能提升
- 记忆机制持续丰富内部知识

---

### 2.4 Agent-R1: Training Powerful LLM Agents with End-to-End Reinforcement Learning

```
Claim: 提出Agent-R1模块化训练框架，通过扩展MDP框架系统定义LLM Agent的关键组件，支持多轮rollout、精确信用分配和灵活的工具/环境集成，在Multi-hop QA上验证有效性
Source: Agent-R1: Training Powerful LLM Agents with End-to-End Reinforcement Learning
URL: https://arxiv.org/abs/2511.14460
Date: 2025-11-18 (arXiv)
Excerpt: "This paper first revisits and clarifies Reinforcement Learning methodologies for LLM Agents by systematically extending the Markov Decision Process (MDP) framework to comprehensively define the key components of an LLM Agent. Secondly, we introduce Agent-R1, a modular, flexible, and user-friendly training framework."
Context: 来自中国科学技术大学认知智能国家重点实验室
Confidence: high
```

**Agent环境定义（扩展MDP）：**
- **状态空间**: 包含用户指令、历史交互、环境反馈
- **动作空间**: 推理文本 + 工具调用/环境动作
- **状态转移**: 区分agent生成的动作和环境反馈
- **奖励函数**: 支持过程奖励（中间奖励）和结果奖励

**工具类型**: 模块化设计，支持多种工具集成

**多轮交互机制**: 
- 强调区分agent生成动作和环境反馈的重要性
- 支持中间（过程）奖励以有效指导学习
- 引入loss mask和advantage mask

**使用的Benchmark:**
- Multi-hop QA（多跳问答）任务
- AgentGym相关基准

**具体实现细节：**
- 支持PPO和GRPO算法
- Loss Mask: 确保梯度只聚焦于agent生成的token
- Advantage Mask: 实现精确信用分配
- 消融实验显示：禁用loss mask使PPO平均EM从0.3136降至0.3022，GRPO从0.3877降至0.3722

**关键发现：**
- Loss mask和advantage mask是关键设计选择
- 模块化架构支持快速集成不同环境接口和任务场景
- 扩展MDP框架为Agent RL提供概念基础

---

### 2.5 AgentPRM: Process Reward Models for LLM Agents via Step-Wise Promise and Progress

```
Claim: 提出AgentPRM，为Agent任务重新定义过程奖励模型，同时捕捉步骤的即时进展和长期承诺，使用TD+GAE方法比MC方法计算效率高8倍以上
Source: AgentPRM: Process Reward Models for LLM Agents via Step-Wise Promise and Progress
URL: https://arxiv.org/abs/2511.08325
Date: 2025-11-11 (arXiv)
Excerpt: "We propose a re-defined PRM for agent tasks, named AgentPRM, to capture both the interdependence between sequential decisions and their contribution to the final goal. This enables better progress tracking and exploration-exploitation balance."
Context: 来自复旦大学，提出TD-based估计+GAE用于高效PRM训练
Confidence: high
```

**Agent环境定义：**
- **动作空间**: ReAct格式的推理过程 + 动作输出
- **观察空间**: 环境观察（网页状态、文本等）
- **奖励设计**: 
  - 稀疏结果奖励（任务完成时）
  - AgentPRM提供密集的步骤级奖励：
    - Promise: 步骤实现目标的概率（Q值）
    - Progress: 相邻步骤间的依赖关系（优势值）
  - 损失函数: L_AgentPRM = L_Q + β × L_A，β=1.0

**工具类型**: WebShop（网页导航）、BabyAI（文本环境）、TextCraft（合成任务）

**多轮交互机制**: 
- 使用TD学习而非MC采样进行步骤级价值估计
- GAE用于优势估计，λ=0.95
- 捕捉相邻动作间的依赖关系

**使用的Benchmark:**
- WebShop: 100个评估查询
- BabyAI: 90个评估查询  
- TextCraft: 97个评估查询
- AgentGym训练集（300条轨迹初始化）

**具体实现细节：**
- 模型: Qwen-2.5-0.5B/3B/7B-Instruct, Llama-3.1-8B-Instruct
- GPU: A100-80GB
- SFT学习率: 1×10^-5
- RM训练: 最多55个epoch，学习率1×10^-6
- β=1.0, λ=0.95（GAE）
- 温度: 轨迹收集1.0，评估0.7
- MC基线: N_Traj=1, N_mc=16; TD: N_TD=16

**关键发现：**
- AgentPRM比MC基线计算效率高8倍以上
- 在测试时计算扩展时表现更稳定
- 同时捕捉promise（未来成功概率）和progress（步骤间进展）
- 适用于数学任务的泛化

---

### 2.6 ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL

```
Claim: 提出ArCHer分层RL框架，高级离策略TD学习聚合轮次间奖励，低级策略梯度优化每轮内token生成，比PPO等on-policy方法样本效率高约100倍
Source: ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL
URL: https://arxiv.org/abs/2402.19446
Date: 2024-02-29 (ICML 2024)
Excerpt: "Our framework prescribes an off-policy temporal difference learning method for training an utterance-level value function at the high level, and any on-policy policy gradient algorithm for optimizing the token generation at each turn of the interaction at the low level."
Context: 来自Stanford/UC Berkeley，ICML 2024，最早的分层Agent RL框架之一
Confidence: high
```

**Agent环境定义：**
- **高级MDP（轮次级）**: 
  - 状态: 对话历史 + 环境状态
  - 动作: 完整一轮的utterance
  - 奖励: 轮次结束时的延迟奖励
  - 价值函数: 离策略TD学习训练
- **低级MDP（token级）**:
  - 动作: 单个token
  - 策略: 使用高级价值函数作为终端奖励进行策略梯度更新

**工具类型**: 通用框架，适用于各种Agent任务

**多轮交互机制**: 
- 分层架构：高级处理轮次间信用分配，低级处理轮次内token生成
- 高级离策略学习允许样本复用
- 避免在单个token上做Bellman备份或在巨大动作空间上最大化

**使用的Benchmark:**
- 多轮对话任务
- Web交互任务
- 工具使用任务

**具体实现细节：**
- 模型规模: 测试至7B参数
- 高级: 离策略TD学习训练utterance-level价值函数
- 低级: on-policy策略梯度（PPO/REINFORCE）训练token policy
- 样本效率: 比PPO高约100倍
- 支持不同Transformer架构

**关键发现：**
- 分层方法结合离策略和on-policy的优势
- 高级离策略学习实现样本复用和更快收敛
- 低级保留现有单轮RL方法的灵活性
- 随模型规模增大性能提升（测试至7B）

---

### 2.7 SWEET-RL: Privileged Critic for Multi-Turn Agents

```
Claim: 提出SWEET-RL（Meta/FAIR），引入特权（非对称）critic概念，利用训练/推理不对称性，在训练时使用agent在推理时无法访问的信息（正确答案、完整未来轨迹）提供高质量轮次级奖励信号
Source: SWEET-RL: Privileged Critic for Multi-Turn Agents
URL: 引用自 Credit Assignment in RL for LLMs (2026) 综述
Date: 2025
Excerpt: "SWEET-RL trains a critic that conditions on this privileged information to provide high-quality turn-level reward signals, which are then used for DPO-style optimization of the actor."
Context: 来自Meta/FAIR，解决Agent任务中间状态不可验证的挑战
Confidence: high
```

**Agent环境定义：**
- **动作空间**: 每轮的agent动作
- **观察空间**: 标准观察（推理时可访问）
- **特权信息（仅训练时）**: 正确答案、完整未来轨迹、环境状态变量
- **奖励设计**: Critic基于特权信息提供轮次级密集奖励

**工具类型**: 通用多轮Agent任务

**多轮交互机制**: 
- 非对称设计：actor针对实际部分可观测设置优化
- Critic信号受益于训练时的完整信息
- 使用DPO风格优化actor

**关键特点：**
- 巧妙绕过不可验证性挑战
- 即使中间状态从agent视角无法验证，特权critic也能评估
- Actor策略针对实际部分可观测设置优化
- Critic信号从训练时的完整信息中受益

---

### 2.8 GiGPO: Group-in-Group Policy Optimization for LLM Agent Training

```
Claim: 提出GiGPO（NeurIPS 2025），通过锚状态分组机制实现细粒度信用分配，在保持GRPO无critic、低内存、稳定收敛特性的同时，ALFWorld提升>12%，WebShop提升>9%
Source: GiGPO: Group-in-Group Policy Optimization for LLM Agent Training
URL: https://neurips.cc/virtual/2025/poster/118123
Date: 2025 (NeurIPS 2025)
Excerpt: "GiGPO introduces a two-level structure for estimating relative advantage: (i) At the episode-level, GiGPO computes macro relative advantages based on groups of complete trajectories; (ii) At the step-level, GiGPO introduces an anchor state grouping mechanism."
Context: 来自NTU Singapore，NeurIPS 2025
Confidence: high
```

**Agent环境定义：**
- **动作空间**: 每步的agent动作（文本命令）
- **观察空间**: 环境文本观察（网页HTML、房间描述等）
- **奖励设计**: 稀疏的结果奖励（任务成功/失败）

**工具类型**: 不适用（直接环境交互）

**多轮交互机制**: 
- **Episode-level**: 基于完整轨迹组的宏观相对优势
- **Step-level**: 锚状态分组机制——识别轨迹中重复出现的状态（锚状态），构建步骤级组进行微观相对优势估计
- 完全无critic，无辅助模型

**使用的Benchmark:**
- ALFWorld: 文本家庭任务环境（6类任务，50步上限）
- WebShop: 网页购物环境（15步上限，1.1M产品）
- 搜索增强QA任务

**具体实现细节：**
- 模型: Qwen2.5-1.5B/3B/7B-Instruct
- 保持与GRPO相同的GPU内存开销
- 算法复杂度: O(NT)，额外开销可忽略（<0.2%迭代时间）
- 锚状态哈希: ~0.01s/iteration

**关键发现：**
- ALFWorld性能提升>12%，WebShop>9%
- QA任务: 3B模型42.1%，7B模型47.2%
- 保持critic-free、低内存、稳定收敛特性
- 完全无额外模型或rollout开销

---

### 2.9 Turn-PPO: Turn-Level Advantage Estimation for Multi-Turn RL

```
Claim: 提出Turn-PPO，将MDP从token级重新定义到turn级，使用可学习的critic在turn级进行优势估计，在WebShop和Sokoban上优于GRPO和token-PPO
Source: Turn-PPO: Turn-Level Advantage Estimation with PPO for Improved Multi-Turn RL in Agentic LLMs
URL: https://arxiv.org/abs/2512.17008
Date: 2025-12-18 (arXiv)
Excerpt: "We introduce turn-PPO, a variant that operates on a turn-level MDP formulation, as opposed to the commonly used token-level MDP. Our results on the WebShop and Sokoban datasets demonstrate the effectiveness of turn-PPO."
Context: 来自UT Austin / Amazon
Confidence: high
```

**Agent环境定义：**
- **Turn-MDP**:
  - 状态 s_n: 历史所有轮次的(Q_n', R_n')拼接 + 当前查询Q_n
  - 动作 a_n: 当前轮的完整响应R_n
  - 价值函数: 在每轮最后一个token处评估
- **奖励设计**: 最终结果奖励 + 格式惩罚

**多轮交互机制**: 
- 将整个turn视为单个state-action对
- Critic在turn级学习更准确的价值估计
- 使用GAE在turn级计算优势，γ和λ可灵活调节

**使用的Benchmark:**
- WebShop
- Sokoban

**具体实现细节：**
- 模型: Qwen2.5, Qwen3（thinking启用/禁用）
- GPU: 8×H100
- PPO超参数: γ=0.99, λ=0.9
- Rollout temperature: 1.0
- ALFWorld: 50步上限; WebShop: 15步上限
- Critic需要比actor更大的学习率
- 对于PPO，batch多样性比每问题多个rollout更重要

**关键发现：**
- PPO在多轮设置中比GRPO更稳定
- Turn-PPO允许灵活调节γ和λ，token-PPO必须固定为1.0
- Turn级裁剪率更高，防止策略大幅变化时的不稳定更新
- Qwen3配合thinking禁用表现优于Qwen2.5

---

### 2.10 ToRL: Scaling Tool-Integrated RL

```
Claim: 提出ToRL框架，直接从基础模型进行工具集成RL训练（无需SFT约束），使模型通过大规模探索发现最优工具利用策略，在数学任务上超越无工具调用的RL基线
Source: ToRL: Scaling Tool-Integrated RL
URL: https://arxiv.org/abs/2503.23383
Date: 2025-03
Excerpt: "ToRL enables RL training from scratch, allowing models to discover optimal tool utilization strategies through extensive exploration. This scaling approach yields qualitatively different behaviors than methods that build upon predetermined patterns."
Context: 关注代码解释器工具在数学推理中的集成
Confidence: high
```

**Agent环境定义：**
- **动作空间**: 自然语言推理 + Python代码生成
- **观察空间**: 代码执行器输出结果
- **奖励设计**: 结果奖励（数学答案正确性）

**工具类型**: Python代码解释器

**多轮交互机制**: 
- 迭代生成代码、执行、基于输出继续推理
- 将代码执行集成到GRPO训练循环中

**使用的Benchmark:**
- 多个数学数据集（具体名称在论文中）

**关键发现：**
- 从零开始的RL训练使模型发现最优工具使用策略
- 超越无工具集成的RL基线
- 产生与基于预设模式方法本质不同的行为

---

### 2.11 VerlTool: Towards Holistic Agentic Reinforcement Learning with Tool Use

```
Claim: 提出VerlTool统一框架，支持多种工具类型的Agent RL训练，通过异步rollout实现近2倍加速，在6个ARLT领域实现与专门系统相当的性能
Source: VerlTool: Towards Holistic Agentic Reinforcement Learning with Tool Use
URL: https://arxiv.org/abs/2509.01055
Date: 2025-09-01 (arXiv)
Excerpt: "We introduce VerlTool, a unified and modular framework that addresses these limitations through systematic design principles: upstream alignment with VeRL, unified tool management via standardized APIs, asynchronous rollout execution achieving near 2x speedup."
Context: 来自UIUC/Sea AI Lab等
Confidence: high
```

**支持的6个ARLT领域:**
1. 数学推理（代码执行工具）
2. 知识QA（搜索检索工具）
3. SQL生成（SQL执行工具）
4. 视觉推理（视觉处理工具）
5. 网页搜索（网页浏览工具）
6. 软件工程（SWE-Bench）

**框架关键设计:**
- 上游对齐：继承VeRL确保兼容性
- 统一工具管理：标准化API支持多样化工具
- 异步Rollout：按轨迹而非同步批次与工具服务器交互，消除~2倍等待时间
- 多模态支持：文本/图像/视频观察token

---

### 2.12 ReTool / ExCoT / Think2SQL / FTRL / Tool-Zero

**ReTool (Feng et al., 2025):**
- 将工具调用集成到PPO训练中
- 针对Python代码工具的数学问题解决
- 使用SFT冷启动 + RL迭代优化
- 与ToRL的区别：ReTool依赖SFT + 额外标注数据

**ExCoT (Zhai et al., 2025):**
- 使用GRPO + SQL执行器响应增强NL2SQL性能
- 通过全面的奖励设计实现初步成功
- 使用DPO算法

**Think2SQL (Papicchio et al., 2025):**
- 同样针对NL2SQL任务
- GRPO训练，奖励包括：precision, recall, cardinality, 2×format
- 通过挑战性训练问题过滤提升性能

**FTRL (Ye et al., 2025):**
- 提出稳定且可验证的工具使用训练数据合成方法
- 使用成功工具调用与总尝试次数比率计算奖励
- 数据集：2,000+自动构建的工具使用环境

**Tool-Zero (Zeng et al., 2025):**
- 纯RL从零训练工具增强LLM（无需SFT）
- 使用GG-GRPO算法
- 模型: Qwen2.5-7B Base, Qwen2.5-32B Base
- 评估: BFCL, API-Bank, Nexus Raven, Tool-Alpaca, Seal-Tools

---

## 3. Agent环境对比表

| 方法 | 动作空间 | 观察空间 | 奖励类型 | 多轮机制 | 工具类型 |
|------|---------|---------|---------|---------|---------|
| ToolRL | JSON工具调用（工具名+参数） | 工具描述+查询+历史 | 格式+细粒度正确性（[-3,3]） | 多步工具调用 | 通用API/函数 |
| Search-R1 | <search>+<think>+<answer> | 问题+检索结果 | 简单结果奖励 | 多轮交错推理搜索 | 搜索引擎 |
| R1-Searcher++ | 搜索查询+推理+答案 | 问题+检索结果 | 结果+内部知识利用 | 两阶段训练 | 搜索引擎 |
| Agent-R1 | 推理文本+工具调用 | 指令+历史+环境反馈 | 过程+结果奖励 | 扩展MDP，loss/advantage mask | 模块化工具 |
| AgentPRM | ReAct（推理+动作） | 环境观察（网页/文本） | PRM密集步骤奖励 | TD+GAE步骤级评估 | WebShop/BabyAI |
| ArCHer | Token级动作+轮次级utterance | 对话历史+环境状态 | 轮次级延迟奖励 | 分层：高级TD+低级PG | 通用 |
| SWEET-RL | 轮次动作 | 标准观察+特权信息 | 轮次级密集奖励 | 非对称actor-critic | 通用多轮 |
| GiGPO | 文本命令 | 环境文本观察 | 稀疏结果奖励 | 锚状态分组 | 环境交互 |
| Turn-PPO | Turn级响应 | 历史拼接+当前查询 | 结果+格式惩罚 | Turn-MDP | 通用 |
| ToRL | 推理+Python代码 | 代码执行结果 | 结果奖励（答案正确） | 代码生成-执行循环 | Python解释器 |
| VerlTool | 推理+工具调用 | 多模态观察 | 领域特定结果奖励 | 异步多工具交互 | 代码/搜索/SQL/视觉 |

---

## 4. 奖励设计对比

| 方法 | 奖励粒度 | 奖励类型 | 是否需要GT | 可验证性 |
|------|---------|---------|-----------|---------|
| ToolRL | 细粒度（工具名+参数名+参数内容） | 格式+正确性混合 | 是（训练时） | 规则验证 |
| Search-R1 | 轨迹级 | 结果奖励 | 是 | 答案匹配 |
| AgentPRM | 步骤级 | 学习PRM（Promise+Progress） | 否（学习得到） | TD估计 |
| ArCHer | 轮次级 | 延迟奖励+价值函数 | 是 | 环境反馈 |
| SWEET-RL | 轮次级 | 特权Critic评估 | 是（训练时） | 特权信息 |
| GiGPO | 步骤级（锚状态组内） | 组内相对优势 | 是 | 结果奖励 |
| Turn-PPO | Turn级 | 结果+格式 | 是 | 规则验证 |
| ToRL | 轨迹级 | 结果奖励（数学正确） | 是 | 执行验证 |

---

## 5. Benchmark汇总

### 5.1 主要Agent RL基准

| 基准 | 类型 | 描述 | 常用方法 |
|------|------|------|---------|
| **ALFWorld** | 文本 embodied | 家庭任务环境，6类任务（Pick/Look/Clean/Heat/Cool/Pick2） | GiGPO, ArCHer, Turn-PPO, AgentPRM |
| **WebShop** | 网页导航 | 电商购物环境，1.1M产品，搜索/浏览/购买 | GiGPO, Turn-PPO, AgentPRM, SWEET-RL |
| **BFCL** | 工具调用 | Berkeley Function Call Leaderboard，4K+实例 | ToolRL, Tool-Zero, AWPO, MatchTIR |
| **API-Bank** | 工具调用 | 三级评估框架，73个API，多轮对话 | ToolRL, Tool-Zero, AWPO |
| **WebArena** | 网页Agent | 真实网页环境测试自主Agent | 综合评估 |
| **AgentBench** | 综合 | 8个环境，涵盖OS/DB/KG/游戏/网页 | 综合评估 |
| **AgentGym** | 综合 | 多样化设置中进化LLM Agent | AgentPRM, Agent-R1 |

### 5.2 搜索增强QA基准

| 基准 | 类型 | 描述 | 常用方法 |
|------|------|------|---------|
| **NQ** | 单跳QA | Natural Questions | Search-R1, R1-Searcher |
| **TriviaQA** | 单跳QA |  trivia问答 | Search-R1 |
| **PopQA** | 单跳QA | 实体 popularity问答 | Search-R1 |
| **HotpotQA** | 多跳QA | 需要跨文档推理 | Search-R1, R1-Searcher |
| **2WikiMultiHopQA** | 多跳QA | Wikipedia多跳 | Search-R1 |
| **MuSiQue** | 多跳QA | 多场景问答 | Search-R1 |
| **Bamboogle** | 多跳QA | 多样化QA任务 | Search-R1, ToolRL |

### 5.3 代码/SQL工具基准

| 基准 | 类型 | 描述 | 常用方法 |
|------|------|------|---------|
| **Spider** | Text2SQL | 大规模NL2SQL | Think2SQL, ExCoT, SQL-R1 |
| **BIRD** | Text2SQL | 大规模数据库基准 | Think2SQL, SQL-R1, Arctic-Text2SQL-R1 |
| **LiveCodeBench** | 代码生成 |  competitive编程 | ResRL |
| **SWE-Bench** | 软件工程 | 真实GitHub issue解决 | VerlTool |

---

## 6. 方法演进关系图

```
单轮RL基础
├── GRPO (Shao et al., 2024) [DeepSeekMath]
│   ├── ToolRL (2025) ──→ 通用工具调用RL
│   ├── Search-R1 (2025) ──→ 搜索增强推理
│   ├── R1-Searcher++ (2025) ──→ 两阶段搜索RL
│   ├── ToRL (2025) ──→ 代码解释器工具RL
│   ├── GiGPO (NeurIPS 2025) ──→ 组内分组信用分配
│   ├── ExCoT/Think2SQL (2025) ──→ SQL工具RL
│   └── Tool-Zero (2025) ──→ 纯RL工具训练
│
├── PPO基础
│   ├── ArCHer (ICML 2024) ──→ 分层TD+PG
│   ├── Turn-PPO (2025) ──→ Turn级MDP
│   ├── SWEET-RL (2025) ──→ 特权Critic
│   └── ReTool (2025) ──→ PPO+代码工具
│
└── PRM演进
    └── AgentPRM (2025) ──→ TD+GAE步骤级PRM

框架/系统
└── VerlTool (2025) ──→ 统一ARLT框架
    ├── Agent-R1 (2025) ──→ 模块化Agent RL框架
    └── VerlTool ──→ 6领域ARLT支持
```

---

## 7. 关键洞察与趋势

### 7.1 奖励设计原则

1. **细粒度优于粗粒度**: ToolRL的系统研究表明，分解奖励（工具名、参数名、参数内容）优于简单的答案匹配
2. **结果奖励足够**: Search-R1和多篇论文证明，简单结果奖励在配合适当训练机制时足以驱动复杂工具使用行为
3. **密集奖励加速学习**: AgentPRM、SWEET-RL等方法通过密集步骤级奖励显著提升样本效率
4. **动态奖励规模**: ToolRL发现动态调整奖励规模有助于从简单到复杂行为的平滑过渡

### 7.2 信用分配方法对比

| 方法类型 | 代表 | 优点 | 缺点 |
|---------|------|------|------|
| 轨迹级 | GRPO（标准） | 简单、稳定 | 多轮场景下信用分配粗糙 |
| Turn级 | Turn-PPO, SWEET-RL | 适合多轮结构 | 需要额外critic |
| Step级 | AgentPRM, GiGPO | 细粒度 | 实现复杂 |
| 分层 | ArCHer | 样本效率高 | 架构复杂 |
| 特权Critic | SWEET-RL | 绕过不可验证性 | 训练/推理不对称 |

### 7.3 多轮RL关键挑战

1. **状态表示不对齐**: token-MDP在多轮设置中面临状态转移模式不一致问题（Turn-PPO论文指出）
2. **环境交互不稳定**: 多轮设置中环境交互不可控导致采样方差增大
3. **信用分配困难**: 不同轮次对最终奖励贡献不均，统一优势分配引入不准确
4. **延迟奖励**: 稀疏的最终结果奖励使中间步骤学习困难

### 7.4 实现最佳实践

1. **使用veRL框架**: 大多数工具RL工作使用veRL作为基础框架
2. **异步执行**: VerlTool证明异步rollout可实现~2倍加速
3. **检索token掩码**: Search-R1的关键设计——只更新LLM生成token的梯度
4. **Loss Mask**: Agent-R1证明屏蔽环境token的损失是有效训练的关键
5. **冷启动有效**: ToolRL、Tool-Zero等证明无需大量SFT数据即可进行有效RL训练

---

## 8. 参考文献索引

| 编号 | 论文 | 作者 | 年份 | 来源 |
|------|------|------|------|------|
| 1 | ToolRL: Reward is All Tool Learning Needs | Qian et al. | 2025 | arXiv |
| 2 | Search-R1: Training LLMs to Reason and Leverage Search Engines with RL | Jin et al. | 2025 | arXiv |
| 3 | R1-Searcher++: Incentivizing Dynamic Knowledge Acquisition | Song et al. | 2025 | arXiv |
| 4 | Agent-R1: Training Powerful LLM Agents with End-to-End RL | Cheng et al. | 2025 | arXiv |
| 5 | AgentPRM: Process Reward Models for LLM Agents | Xi et al. | 2025 | arXiv |
| 6 | ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL | Zhou et al. | 2024 | ICML 2024 |
| 7 | SWEET-RL: Privileged Critic for Multi-Turn Agents | Zhou et al. | 2025 | Meta/FAIR |
| 8 | GiGPO: Group-in-Group Policy Optimization | Feng et al. | 2025 | NeurIPS 2025 |
| 9 | Turn-PPO: Turn-Level Advantage Estimation with PPO | Li et al. | 2025 | arXiv |
| 10 | ToRL: Scaling Tool-Integrated RL | Li et al. | 2025 | arXiv |
| 11 | VerlTool: Towards Holistic Agentic RL with Tool Use | Jiang et al. | 2025 | arXiv |
| 12 | ReTool: Integrated Tool-Calling in PPO | Feng et al. | 2025 | arXiv |
| 13 | ExCoT: Executable Chain of Thought for NL2SQL | Zhai et al. | 2025 | arXiv |
| 14 | Think2SQL: Reasoning for Text-to-SQL | Papicchio et al. | 2025 | arXiv |
| 15 | FTRL: Stable Tool-Use Training Data Synthesis | Ye et al. | 2025 | arXiv |
| 16 | Tool-Zero: Training Tool-Augmented LLMs via Pure RL | Zeng et al. | 2025 | EMNLP 2025 |
| 17 | MatchTIR: Fine-Grained Supervision via Bipartite Matching | Ye et al. | 2025 | arXiv |
| 18 | AWPO: Enhancing Tool-Use through Reasoning Rewards | Wang et al. | 2025 | arXiv |
| 19 | RAG-R1: Multi-query Parallelism for Search and Reasoning | - | 2025 | arXiv |
| 20 | The Landscape of Agentic RL for LLMs: A Survey | - | 2025 | arXiv |

---

*报告生成时间: 2025年*
*覆盖论文: 20+ 篇核心论文*
*搜索范围: NeurIPS 2025, ICML 2024, ICLR, ACL, EMNLP, arXiv 2024-2025*
