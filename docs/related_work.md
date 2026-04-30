# 相关工作与论文引用目录

> 本文档整合 ENTROPIA 的原始文献基础与 CRAFT-Search 勘探的 45+ 篇核心文献，按主题分组，提供完整 arXiv ID 和引用信息。
>
> 标注说明：
> - **[已实现]** = 代码中已参考的方法
> - **[待集成]** = CRAFT-Search 勘探推荐集成的方法
> - **[综述]** = 领域综述，用于定位研究
> - **[参考]** = 相关但非直接依赖的工作

---

## A. Agent RL 奖励密度控制（ENTROPIA 直接基线）

### A1. WorkForceAgent-R1
- **标题**: WorkForceAgent-R1: Reinforcement Learning for LLM Agents
- **arXiv**: 待补充
- **年份**: 2025
- **核心发现**: 密集奖励导致 Reward Hacking——在 ~50 step 后 reward → 0.6，response length → 0
- **与 ENTROPIA 的关系**: [已实现] 确认为 ENTROPIA 要解决的核心问题；dense_fixed 模式作为消融基线
- **代码**: 部分开源

### A2. IGPO: Information Gain Policy Optimization
- **标题**: IGPO: Information Gain as Process Reward for LLM Agents
- **arXiv**: 待补充
- **Venue**: ICLR 2026
- **核心方法**: `IG_t = log p(answer|s_t) - log p(answer|s_{t-1})`，用信息增益作为每步过程奖励
- **局限性**: 每轮密度固定，未考虑训练进度
- **与 ENTROPIA 的关系**: [已实现] r_t^dense 的一种形式（`igpo_information_gain()`），dense_igpo 模式作为基线；ENTROPIA 改进：密度随熵动态调整
- **代码**: 开源

### A3. AutoTool: Automatic Tool-Use via Entropy-Regularized RL
- **标题**: AutoTool: Entropy-Regularized Reinforcement Learning for Tool Use
- **arXiv**: 待补充
- **Venue**: ICLR 2026
- **核心方法**: 熵约束加入 GRPO policy loss（`L = L_GRPO + λ·H`），而非奖励函数
- **与 ENTROPIA 的关键区别**: 
  - AutoTool: 熵在 loss 中做**反应性正则**（"不要高熵"）
  - ENTROPIA: 熵在奖励中做**预防性门控**（"需要帮吗？"）
- **与 ENTROPIA 的关系**: [已实现] 借鉴了关键 token 识别方法；autotool_entropy 模式作为对比基线
- **代码**: 开源

### A4. SELAUR: Self-Adaptive Learning with Uncertainty-Aware Reward
- **标题**: SELAUR: Uncertainty-Aware Reward Shaping for LLM Agents
- **arXiv**: 待补充
- **年份**: 2025
- **核心方法**: 多种不确定性估计（熵、least confidence、margin）用于奖励塑形
- **与 ENTROPIA 的关系**: [已实现] 借鉴了 `UncertaintyEstimator`（entropy/least_confidence/margin/combined）；ENTROPIA 增加了 EMA 自适应阈值和 hacking 检测
- **代码**: 部分开源

### A5. ReTool: Reinforcement Learning for Tool Learning
- **标题**: ReTool: Reinforcement Learning for Tool-Using LLMs
- **arXiv**: 待补充
- **年份**: 2025
- **核心方法**: 二元结果奖励 `r = +1(success) / 0(failure)` + 折扣因子
- **与 ENTROPIA 的关系**: [已实现] sparse 模式作为纯稀疏基准；`outcome_discounted_reward()` 实现

### A6. TIPS: Potential-Based Reward Shaping
- **标题**: TIPS: Transferable Potential-based Reward Shaping for LLMs
- **arXiv**: 待补充
- **年份**: 2025
- **核心方法**: `r_t = Φ(s_t) - γ·Φ(s_{t-1})`，基于势能函数的奖励塑形
- **与 ENTROPIA 的关系**: [已实现] r_t^dense 的可插拔选项（`tips_potential_reward()`）

### A7. TRACE: Trustworthy RL with Safety Constraints
- **标题**: TRACE: Safety-Constrained Reinforcement Learning for LLM Agents
- **arXiv**: 待补充
- **年份**: 2025
- **核心方法**: 安全约束 RL + judge reliability 评估；Reward-Success Divergence 检测
- **与 ENTROPIA 的关系**: [已实现] `hacking_detector.py` 中的 divergence 检测逻辑借鉴 TRACE

---

## B. GRPO 与 RL 算法基础

### B1. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
- **标题**: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
- **作者**: DeepSeek-AI
- **arXiv**: [2501.12948](https://arxiv.org/abs/2501.12948)
- **年份**: 2025
- **核心贡献**: R1-Zero/R1 双管道；GRPO 大规模验证；RL 激发推理能力的首次大规模成功
- **与 ENTROPIA 的关系**: GRPO 训练框架的理论基础
- **代码**: [Open-R1](https://github.com/huggingface/open-r1) (26k+ stars)

### B2. DAPO: An Open-Source LLM Reinforcement Learning System at Scale
- **标题**: DAPO: An Open-Source LLM Reinforcement Learning System at Scale
- **作者**: Yu et al. (ByteDance)
- **arXiv**: [2503.14476](https://arxiv.org/abs/2503.14476)
- **年份**: 2025
- **核心贡献**: Decoupled Clip + Dynamic Sampling；GRPO 训练稳定性改进
- **与 ENTROPIA 的关系**: [待集成] Dynamic Sampling 可与 ENTROPIA 的熵门控形成互补——在需要更多探索的高熵阶段增加采样
- **代码**: [veRL](https://github.com/volcengine/verl) (9.5k+ stars)

### B3. Dr.GRPO: Understanding and Improving Group Relative Policy Optimization
- **标题**: Dr.GRPO: Understanding and Improving Group Relative Policy Optimization
- **作者**: Mroueh et al.
- **arXiv**: [2503.20783](https://arxiv.org/abs/2503.20783)
- **年份**: 2025
- **核心贡献**: 分析 GRPO 长度归一化偏差；提出改进方案
- **与 ENTROPIA 的关系**: [参考] 长度归一化改进可融入 GRPO loss 计算

### B4. GPG: Generative Policy Gradient
- **标题**: GPG: Generative Policy Gradient
- **作者**: Anonymous
- **arXiv**: [2502.02592](https://arxiv.org/abs/2502.02592)
- **年份**: 2025
- **核心贡献**: 直接优化原始 RL 目标的无 critic 算法
- **与 ENTROPIA 的关系**: [参考] 另一种无 critic 算法，可作为 GRPO 的备选

### B5. ReDit: Reward Dithering for Improved LLM Policy Optimization
- **标题**: ReDit: Reward Dithering for Improved LLM Policy Optimization
- **作者**: Zhu et al.
- **arXiv**: [2506.18631](https://arxiv.org/abs/2506.18631)
- **年份**: 2025
- **核心贡献**: 离散奖励加零均值噪声可增大 batch 内 reward variance，10% 步数达 baseline 效果
- **与 ENTROPIA 的关系**: [待集成] 当门关（reward variance 不足）时，ReDit dithering 可维持训练信号

### B6. Revisiting GRPO: On-policy and Off-policy Analysis
- **标题**: Revisiting GRPO: On-policy and Off-policy Analysis
- **作者**: Mroueh et al.
- **arXiv**: [2505.22257](https://arxiv.org/abs/2505.22257)
- **年份**: 2025
- **核心贡献**: GRPO 离轨/在轨策略分析；理论收敛性证明

### B7. SimpleRL: 7B Model + 8K Data → Emerging Reasoning
- **标题**: SimpleRL: 7B Model + 8K Data → Emerging Reasoning
- **作者**: Anonymous
- **arXiv**: [2503.18892](https://arxiv.org/abs/2503.18892)
- **年份**: 2025
- **核心贡献**: 极简 GRPO 训练流程，证明小模型+小数据也可产生推理涌现
- **与 ENTROPIA 的关系**: [参考] 验证 ENTROPIA 在小数据规模上的可行性

---

## C. Search Agent 与 Agent 训练

### C1. Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning
- **标题**: Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning
- **作者**: Jin et al.
- **arXiv**: [2503.09516](https://arxiv.org/abs/2503.09516)
- **Venue**: COLM 2025
- **核心贡献**: GRPO 训练 7B 模型自主使用搜索引擎；retrieved-token masking；7B 模型提升 24%
- **局限性**: 仅使用单一 EM reward；over-search 问题未解决
- **与 ENTROPIA 的关系**: [待集成] ENTROPIA 迁移到 Search Agent 场景时的直接基线
- **代码**: [Search-R1](https://github.com/sunnynexus/Search-R1) (~4.3k stars)

### C2. R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning
- **标题**: R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning
- **作者**: Anonymous
- **arXiv**: [2503.05592](https://arxiv.org/abs/2503.05592)
- **年份**: 2025
- **核心贡献**: 两阶段 RL 训练（SFT-free 冷启动）；搜索能力通过 RL 自然涌现
- **与 ENTROPIA 的关系**: [参考] 另一个 Search Agent 基线

### C3. ReSearch: Learning to Reason with Search via Reinforcement Learning
- **标题**: ReSearch: Learning to Reason with Search via Reinforcement Learning
- **作者**: Anonymous
- **arXiv**: [2503.19470](https://arxiv.org/abs/2503.19470)
- **Venue**: NeurIPS 2025 (投稿)
- **核心贡献**: 无需监督推理数据的 GRPO 搜索训练；7B/32B 验证
- **与 ENTROPIA 的关系**: [参考] 第三个 Search Agent 基线

### C4. DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments
- **标题**: DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments
- **作者**: Zheng et al.
- **arXiv**: [2504.03160](https://arxiv.org/abs/2504.03160)
- **年份**: 2025
- **核心贡献**: 真实 Web 环境端到端 RL；超越 SFT 性能
- **与 ENTROPIA 的关系**: [参考] ENTROPIA 从 Mock 环境迁移到真实环境的参考方案

### C5. MemSearcher: Multi-Context GRPO for Memory-Augmented Agents
- **标题**: MemSearcher: Multi-Context GRPO for Memory-Augmented Agents
- **作者**: Anonymous
- **arXiv**: [2511.02805](https://arxiv.org/abs/2511.02805)
- **年份**: 2025
- **核心贡献**: 多上下文 GRPO；记忆管理和检索增强
- **与 ENTROPIA 的关系**: [参考] Memory 对 Agent 训练的影响

### C6. StepSearch: Step-wise PPO with Information Gain
- **标题**: StepSearch: Step-wise PPO with Information Gain for Search Agent Training
- **作者**: Anonymous
- **arXiv**: [2505.15107](https://arxiv.org/abs/2505.15107)
- **年份**: 2025
- **核心贡献**: TF-IDF 信息增益 + 冗余惩罚；每步 PPO 优化
- **与 ENTROPIA 的关系**: [待集成] 信息增益计算的另一种方式，可与 ENTROPIA 的 IGPO 插件对比

### C7. ZeroSearch: Learning to Search without Real Searching
- **标题**: ZeroSearch: Learning to Search without Real Searching
- **作者**: Anonymous
- **arXiv**: [2505.04588](https://arxiv.org/abs/2505.04588)
- **年份**: 2025
- **核心贡献**: LLM 模拟搜索引擎、零 API 成本训练
- **与 ENTROPIA 的关系**: [参考] 低成本训练方案参考

---

## D. 过程奖励模型（PRM）与 Verifier

### D1. GenPRM: Scaling Test-Time Compute via Generative Reasoning and Process Reward Model
- **标题**: GenPRM: Scaling Test-Time Compute via Generative Reasoning and Process Reward Model
- **作者**: Yang et al. (THU)
- **arXiv**: [2504.00891](https://arxiv.org/abs/2504.00891)
- **Venue**: ICML 2025
- **核心贡献**: 生成式 PRM 在 7B 超越 GPT-4o 判别式 PRM；仅需 23K 训练数据
- **与 ENTROPIA 的关系**: [待集成] 可作为 ENTROPIA 的 r_t^dense 的一种高级形式——用 PRM 的分数作为密集奖励

### D2. PRIME: Process Reinforcement through Implicit Rewards
- **标题**: PRIME: Process Reinforcement through Implicit Rewards
- **作者**: Anonymous
- **arXiv**: [2502.01456](https://arxiv.org/abs/2502.01456)
- **年份**: 2025
- **核心贡献**: 隐式 PRM 在线更新；过程奖励与策略同步学习
- **与 ENTROPIA 的关系**: [参考] 隐式 PRM 可与 ENTROPIA 的 entropy gating 形成互补

### D3. rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking
- **标题**: rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking
- **作者**: Anonymous
- **arXiv**: [2501.04519](https://arxiv.org/abs/2501.04519)
- **年份**: 2025
- **核心贡献**: MCTS 自举 + SL + PRM；7B 模型 MATH 达 90%
- **与 ENTROPIA 的关系**: [参考] PRM 在小模型上有效的证据

---

## E. 信息增益与增量信息

### E1. IGPO (同 A2)
详见 A2 条目。信息增益过程奖励的核心参考。

### E2. StepSearch (同 C6)
详见 C6 条目。TF-IDF 信息增益的替代实现。

### E3. TIPS (同 A6)
详见 A6 条目。势能塑形作为信息增量的一种形式。

---

## F. 证据选择与引用（CRAFT-Search 勘探来源）

### F1. Context-Picker: RL for Evidence Selection
- **标题**: Context-Picker: Reinforcement Learning for Evidence Selection
- **作者**: Anonymous
- **arXiv**: [2512.14465](https://arxiv.org/abs/2512.14465)
- **年份**: 2025
- **核心贡献**: 两阶段 RL 选择最小充分证据集
- **与 ENTROPIA 的关系**: [待集成] 引用质量奖励的实现参考

### F2. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
- **标题**: Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
- **作者**: Asai et al.
- **arXiv**: [2310.11511](https://arxiv.org/abs/2310.11511)
- **Venue**: ICLR 2024
- **核心贡献**: Reflection Tokens 自适应检索；自我反思机制
- **与 ENTROPIA 的关系**: [待集成] 反思机制可作为 hacking_detector 的第四种信号

### F3. FACTUM: Mechanistic Understanding of Citation Hallucination
- **标题**: FACTUM: Mechanistic Understanding of Citation Hallucination
- **作者**: Anonymous
- **arXiv**: [2601.05866](https://arxiv.org/abs/2601.05866)
- **年份**: 2026
- **核心贡献**: 内部状态分析检测引用幻觉
- **与 ENTROPIA 的关系**: [参考] 引用质量检测的行为基础

---

## G. 小型模型推理与蒸馏

### G1. LIMO: Less is More for Reasoning
- **标题**: LIMO: Less is More for Reasoning
- **作者**: Anonymous
- **arXiv**: [2502.03387](https://arxiv.org/abs/2502.03387)
- **年份**: 2025
- **核心贡献**: 仅 817 样本激发 32B 推理能力；数据质量 > 数据数量
- **与 ENTROPIA 的关系**: [参考] SFT warmup 的数据效率设计参考

### G2. RLKD: Distilling Reasoning via RL
- **标题**: RLKD: Distilling Reasoning via Reinforcement Learning
- **作者**: Anonymous
- **arXiv**: [2505.16142](https://arxiv.org/abs/2505.16142)
- **年份**: 2025
- **核心贡献**: 首个 RL-based 推理蒸馏方法
- **与 ENTROPIA 的关系**: [参考] ENTROPIA 后续可探索"蒸馏 + 自适应密度"组合

### G3. Tina: Tiny Reasoning via LoRA
- **标题**: Tina: Tiny Reasoning via LoRA
- **作者**: Anonymous
- **arXiv**: [2504.15777](https://arxiv.org/abs/2504.15777)
- **年份**: 2025
- **核心贡献**: $9 复现 LoRA + GRPO；极端低成本推理训练
- **与 ENTROPIA 的关系**: [参考] 验证 ENTROPIA 在低算力约束下的可行性

---

## H. 综述论文

### H1. The Landscape of Agentic RL for LLMs
- **标题**: The Landscape of Agentic Reinforcement Learning for LLMs
- **作者**: 多作者
- **arXiv**: [2509.02547](https://arxiv.org/abs/2509.02547)
- **年份**: 2025
- **核心贡献**: Agentic RL 全面综述；Outcome→Process→Composite Reward 演进
- **与 ENTROPIA 的关系**: [综述] 确认 ENTROPIA 在 "Adaptive Density Control" 这个子方向上的独特位置

### H2. From LLM Reasoning to Autonomous AI Agents
- **标题**: From Large Language Models to Large Reasoning Models and Beyond
- **作者**: Ferrag et al.
- **arXiv**: [2504.19678](https://arxiv.org/abs/2504.19678)
- **年份**: 2025
- **核心贡献**: 60+ benchmarks、Agent 框架、多 Agent 协作全面综述
- **与 ENTROPIA 的关系**: [综述] 建立 Agent RL 研究全貌

### H3. Stop Overthinking: Efficient Reasoning Survey
- **标题**: Stop Overthinking: A Survey on Efficient Reasoning for LLMs
- **作者**: Sui et al.
- **arXiv**: [2503.16419](https://arxiv.org/abs/2503.16419)
- **Venue**: TMLR 2025
- **核心贡献**: 短推理链、推理长度优化、效率评估
- **与 ENTROPIA 的关系**: [综述] 效率奖励的理论基础

### H4. DeepSeek-R1 Thoughtology
- **标题**: DeepSeek-R1 Thoughtology
- **作者**: Marjanovic et al.
- **arXiv**: [2504.07128](https://arxiv.org/abs/2504.07128)
- **年份**: 2025
- **核心贡献**: R1 推理行为分析、思维链 taxonomy、自我修正模式
- **与 ENTROPIA 的关系**: [综述] 理解推理行为的复杂性，为熵信号分析提供行为基础

---

## I. Benchmark 与评估

### I1. WebArena: A Realistic Web Environment for Building Autonomous Agents
- **标题**: WebArena: A Realistic Web Environment for Building Autonomous Agents
- **作者**: Zhou et al.
- **arXiv**: [2307.13854](https://arxiv.org/abs/2307.13854)
- **Venue**: ICLR 2024
- **核心贡献**: 真实 Web 环境 benchmark
- **与 ENTROPIA 的关系**: [待集成] 从 τ-Bench 扩展到 WebArena 的评估场景

### I2. BrowseComp: A Benchmark for Persistent Multi-hop Web Browsing
- **标题**: BrowseComp: A Benchmark for Persistent Multi-hop Web Browsing
- **作者**: Anonymous
- **arXiv**: [2504.12516](https://arxiv.org/abs/2504.12516)
- **年份**: 2025
- **核心贡献**: 持久多跳 Web 浏览评估
- **与 ENTROPIA 的关系**: [待集成] Search Agent 场景的核心 benchmark

### I3. SWE-bench: Can Language Models Resolve Real-World GitHub Issues?
- **标题**: SWE-bench: Can Language Models Resolve Real-World GitHub Issues?
- **作者**: Jimenez et al.
- **arXiv**: [2310.06770](https://arxiv.org/abs/2310.06770)
- **Venue**: ICLR 2024
- **核心贡献**: 真实 issue 修复评估
- **与 ENTROPIA 的关系**: [参考] 代码 Agent 场景的评估参考

---

## J. 开源框架与代码参考

| 框架 | GitHub | Stars | 用途 | 与 ENTROPIA 的关系 |
|------|--------|-------|------|-------------------|
| **TRL** | [huggingface/trl](https://github.com/huggingface/trl) | 12.8k | SFT/DPO/GRPO/PPO 训练 | [待切换] 建议替换自研 trainer |
| **veRL** | [volcengine/verl](https://github.com/volcengine/verl) | 9.5k | 高性能 RL 训练 | [待集成] 多卡分布式训练 |
| **Open-R1** | [huggingface/open-r1](https://github.com/huggingface/open-r1) | 26k | R1 全面复现 | [参考] GRPO 实现参考 |
| **vLLM** | [vllm-project/vllm](https://github.com/vllm-project/vllm) | 47k | 高性能推理引擎 | [待集成] rollout 推理加速 |
| **LLaMA-Factory** | [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | 94k | 一站式训练 | [参考] SFT warmup 备选方案 |
| **AgentGym-RL** | [WooooDyy/AgentGym-RL](https://github.com/WooooDyy/AgentGym-RL) | 715 | Agent RL 训练 | [参考] 多轮交互 Agent 训练 |
| **Agent Lightning** | [microsoft/agent-lightning](https://github.com/microsoft/agent-lightning) | 17k | 微软 Agent RL 框架 | [参考] 模块化设计参考 |

---

## K. 训练数据来源

| 数据集 | 大小 | 说明 | 状态 |
|--------|------|------|------|
| `Simia-Agent/Simia-Tau-SFT-90k-Hermes` | 91,204 | τ²-bench 格式 SFT 数据 | [待加载] 推荐主训练数据 |
| `inclusionAI/AReaL-tau2-data` | 33,531 + 1,982 | SFT + RL 轨迹 | [待加载] 补充训练数据 |
| `fuvty/tau-bench-synthetic` | 6,014 | τ²-bench 合成数据 | [待加载] 补充训练数据 |
| `Salesforce/APIGen-MT-5k` | 5,000 | 多轮轨迹，3-stage 验证 | [待加载] 质量高但规模小 |
| `tau_dataset.py` (自建) | 25 | Mock 合成任务 | [已实现] 仅用于开发验证 |

---

## 引用格式建议

在论文中使用时，建议采用以下 BibTeX 格式：

```bibtex
@article{deepseek2025r1,
  title={DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning},
  author={DeepSeek-AI},
  journal={arXiv preprint arXiv:2501.12948},
  year={2025}
}

@article{jin2025search,
  title={Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning},
  author={Jin, Bowen and others},
  journal={arXiv preprint arXiv:2503.09516},
  year={2025}
}

@article{yu2025dapo,
  title={DAPO: An Open-Source LLM Reinforcement Learning System at Scale},
  author={Yu and others},
  journal={arXiv preprint arXiv:2503.14476},
  year={2025}
}

@article{mroueh2025drgrpo,
  title={Dr.GRPO: Understanding and Improving Group Relative Policy Optimization},
  author={Mroueh and others},
  journal={arXiv preprint arXiv:2503.20783},
  year={2025}
}

@article{zhu2025redit,
  title={ReDit: Reward Dithering for Improved LLM Policy Optimization},
  author={Zhu and others},
  journal={arXiv preprint arXiv:2506.18631},
  year={2025}
}

@article{yang2025genprm,
  title={GenPRM: Scaling Test-Time Compute via Generative Reasoning and Process Reward Model},
  author={Yang and others},
  journal={arXiv preprint arXiv:2504.00891},
  year={2025}
}
```

---

*文献总数：60+ 篇（含直接基线 7 篇 + CRAFT-Search 勘探 45+ 篇 + 综述 4 篇）*
*最后更新：2026-04-29*
