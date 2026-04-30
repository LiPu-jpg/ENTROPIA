# ENTROPIA v2: State-Conditional Reward Routing

> 目标：把 ENTROPIA 从“一个熵门控公式”升级为一篇完整论文可承载的方法框架。
>
> 核心改写：ENTROPIA 不再被叙述为 entropy-gated reward density，而是 **state-conditional reward routing**：在长程 LLM Agent RL 中，根据当前状态决定 **何时给、给哪种、给多强、给到哪个粒度** 的过程监督。
>
> 公式定稿见 `docs/paper_formula_methodology.md`。本文档保留设计推理、扩展路线和调研整合；论文 Method 公式以公式定稿文档为准。

---

## 1. 先诊断：为什么 v1 撑不起论文

当前 v1 的核心是：

```text
r_t = r_sparse + alpha * sigmoid(H_t - H_threshold) * r_dense
```

这个想法有直觉，但论文体量不够，原因不是公式短，而是它只回答了一个问题：

> 模型不确定时，要不要多给一点过程奖励？

但 reviewer 会继续追问：

1. **高熵一定意味着需要帮助吗？** 不一定。模型可能高熵但正在探索正确路径。
2. **低熵一定意味着不用帮助吗？** 不一定。模型可能自信地错。
3. **密集奖励一定是好信号吗？** 不一定。格式奖励、局部相似度、冗余信息增益都可能被 hack。
4. **过程奖励应该给哪个粒度？** token / step / turn / trajectory 的边界没有解释。
5. **如果 reward hacking 已经发生，熵门控怎么闭环修正？** v1 的 hacking detector 只是安全网，不是方法本身的一部分。

所以 v1 的问题是：**只有 need signal，没有 utility signal 和 reliability signal。**

更强的论文命题应该是：

> Dense process rewards are useful only when the agent needs guidance, the step provides task-relevant information, and the reward signal is reliable. ENTROPIA routes process supervision by estimating these three conditions online.

中文讲法：

> 密集过程奖励不是天然有益的“额外监督”，而是一种有成本、有偏差、可被 hack 的训练资源。ENTROPIA 的目标不是简单调节稀疏/密集比例，而是学习一个状态条件的监督分配策略。

---

## 2. 新故事线：从 Reward Density 到 Reward Routing

### 2.1 论文问题

长程 Agent RL 的核心矛盾不是“稀疏奖励 vs 密集奖励”二选一，而是：

> 在什么状态下，哪一种过程信号值得被注入策略梯度？

稀疏奖励安全但容易出现 advantage collapse；固定密集奖励学习快但容易 reward hacking。现有方法通常选择一个固定点：

| 路线 | 做法 | 缺口 |
|---|---|---|
| Sparse RL / RLVR | 只用终态可验证奖励 | 长程任务中优势信号弱 |
| IGPO / StepSearch | 每步给信息增益 | 密度固定，不判断当前步是否需要或可靠 |
| AutoTool | 把熵放进 loss 做约束 | 控制探索强度，但不决定奖励信号是否可信 |
| SELAUR | 用不确定性塑形奖励值 | 不确定性直接变奖励，缺少过程信号可靠性判断 |
| Static Hybrid Reward | 固定混合 outcome/process | 无法随训练阶段和任务状态调整 |

ENTROPIA v2 的位置：

```text
Outcome reward anchors the objective.
Process rewards are routed conditionally.
Uncertainty says whether help is needed.
Information/progress says whether the step is useful.
Reliability says whether the signal is safe to trust.
```

### 2.2 方法一句话

**ENTROPIA is a state-conditional reward router that allocates a limited dense-supervision budget across steps, reward components, and credit granularities using Need, Utility, and Reliability gates.**

---

## 3. 总体框架

### 3.1 三因子门控 + 风险控制

对每个 trajectory 的每个 step/turn \(t\)，对每个候选过程奖励组件 \(k\)，ENTROPIA 计算：

```text
g_{t,k} = Need_t * Utility_{t,k} * Reliability_{t,k} * (1 - Risk_t)
```

更具体地：

```text
Need_t        = sigma((UQ_t - tau_need) / T_need)
Utility_{t,k} = sigma((DeltaInfo_{t,k} - tau_util) / T_util)
Reliability_{t,k} = sigma((Align_{t,k} - tau_rel) / T_rel)
Risk_t        = hacking_risk(state_t, action_t, history_t)
```

最终奖励：

```text
r_t = r_outcome_t + B_s * sum_k w_{s,k} * g_{t,k} * normalize(r^k_process_t) + eps_t
```

其中：

- `r_outcome_t` 是折扣后的终态稀疏奖励，始终作为 anchor。
- `B_s` 是训练 step `s` 的 dense supervision budget。
- `w_{s,k}` 是不同过程奖励组件的阶段性权重。
- `g_{t,k}` 是每个组件自己的路由门，而不是一个所有信号共用的总门。
- `eps_t` 是可选的 reward dithering，只在 reward variance 不足时启用。

更准确地说，ENTROPIA 的主门控是 Need-Utility-Reliability 三因子；Risk 是闭环安全控制器，负责抑制或关闭过程监督。

这比 v1 多了三个关键能力：

1. **不只判断“不确定不确定”，还判断这步是否真的有用。**
2. **不只给一个 dense reward，而是路由多个过程信号。**
3. **不只防 hacking，而是把 hacking risk 作为门控的一部分。**

---

## 4. 三个核心信号 + Risk 控制

### 4.1 Need Gate：什么时候模型需要帮助

v1 的熵门控保留，但从唯一贡献降级为 Need Gate 的一个特征。

候选特征：

| 特征 | 含义 | 实现 |
|---|---|---|
| Key-token entropy | 工具名、参数、stop token 的决策不确定性 | 当前 `core/entropy.py` 已有 |
| Least confidence | `1 - max p(token)` | 当前 `UncertaintyEstimator` 已有 |
| Margin uncertainty | `1 - (p_top1 - p_top2)` | 当前 `UncertaintyEstimator` 已有 |
| Group outcome collapse | 同一 query 的 rollouts outcome 全同 | GRPO batch 里可直接统计 |
| Progress stagnation | 多步后信息/状态无变化 | 搜索或工具状态差分 |

推荐 v2 默认：

```text
Need_t = zscore(
    0.5 * entropy_key_t
  + 0.2 * least_conf_t
  + 0.2 * margin_t
  + 0.1 * collapse_indicator
)
```

论文里不要把它说成“熵等于不确定性”，而说成：

> Key-token entropy is a lightweight online proxy for supervision need.

也就是说，熵是 proxy，不是理论本体。

### 4.2 Utility Gate：这步是否产生了有价值的信息

这是 v1 最缺的一层。高熵只说明“可能需要帮助”，但密集奖励是否该给，要看这步有没有实际贡献。

候选 Utility：

| 场景 | Utility 信号 | 说明 |
|---|---|---|
| Search QA | Information gain | `log p(answer|s_t) - log p(answer|s_{t-1})` |
| Search QA | Evidence novelty | 新文档/新实体/新关键词覆盖 |
| Tool use | State progress | 订单状态、任务 slot、数据库状态是否推进 |
| Web / ALFWorld | Subgoal completion | 是否完成一个可验证子目标 |
| Open task | Judge belief shift | 强 judge 对当前答案正确性的置信度变化 |

推荐默认以 IG 为主：

```text
Utility_t = max(0, IG_t)
IG_t = log p_theta(y* | s_t) - log p_theta(y* | s_{t-1})
```

如果任务没有 gold answer，则用轻量替代：

```text
Utility_t = judge_conf(answer_correct | state_t) - judge_conf(answer_correct | state_{t-1})
```

设计原则：

> ENTROPIA should reward information-bearing steps, not verbose steps.

这也顺手修复当前实现里把 token 增量或 sparse reward 当 dense proxy 的问题。

### 4.3 Reliability Gate：这个过程信号是否可信

这是把 ENTROPIA 讲成“robust reward routing”的关键。

密集奖励有三种常见坏法：

1. **不区分好坏轨迹**：好轨迹和坏轨迹过程分差不多。
2. **与 outcome 方向冲突**：过程分上涨但成功率下降。
3. **可被表面模式解释**：格式、重复、长度、模板被刷分。

Reliability Gate 估计的是：

```text
Reliability_{t,k} ~= P(process signal k is aligned with final outcome | state_t)
```

可实现的在线指标：

| 指标 | 计算 | 作用 |
|---|---|---|
| Batch rank correlation | `corr(r_process^k, r_outcome)` | 过程信号是否预测最终成功 |
| Advantage agreement | `sign(A_process^k) == sign(A_outcome)` | 梯度方向是否一致 |
| Cross-rollout discriminativeness | 同一 query 内成功/失败 rollout 的过程分差 | 是否区分好坏轨迹 |
| Format-valid mask | 工具调用 JSON/动作格式是否合法 | 格式作为 mask，不作为主奖励 |
| Judge confidence | judge 自身置信度或多 judge 一致性 | 开放任务中的评分可靠性 |

推荐默认：

```text
Reliability_{batch,k}
  = sigmoid(a * corr_k + b * agreement_k + c * discriminative_k - d * proxy_risk_k)
```

其中 reliability 可以按 batch 更新，再广播到 step/component。

关键叙事点：

> Format should mostly be a reliability constraint, not a positive dense reward.

原因：把格式当正奖励，最容易变成 format gaming；把格式当门控约束，则表示“格式不合规时，这个过程信号不可信”。

---

## 5. Risk Gate：把 Hacking Detector 变成闭环控制

v1 的 `hacking_detector.py` 是训练失败后的 safety net。v2 应把它变成 reward router 的输入。

### 5.1 Step-level risk

```text
Risk_t =
    short_response_risk
  + repetition_risk
  + over_search_risk
  + token_bloat_risk
  + format_gaming_risk
  + reward_success_divergence_risk
```

### 5.2 控制动作

| Risk 水平 | 动作 |
|---|---|
| 低 | 正常路由 |
| 中 | 降低 dense budget `B_s`，只保留 high-reliability 组件 |
| 高 | fallback 到 outcome-only |
| 持续高 | skip batch 或回滚 checkpoint |

推荐控制公式：

```text
B_{s+1} = clip(
    B_s
  + eta_collapse * collapse_rate_s
  - eta_hack * hacking_rate_s
  - eta_conflict * reward_conflict_s,
  B_min,
  B_max
)
```

解释：

- advantage collapse 多：说明稀疏信号不够，增加 dense budget。
- hacking 多：说明密集信号危险，降低 dense budget。
- 过程/结果冲突多：降低对应组件权重，而不是关闭所有 dense reward。

这样 ENTROPIA 就从“门控公式”变成了“反馈控制系统”。

---

## 6. 层次化 Credit Routing

论文体量不够的另一个原因是 v1 只谈 reward density，没有谈 credit granularity。

v2 应该加一个轻量但可讲清楚的 credit router：

```text
turn-level by default
step-level when sub-decisions have separable uncertainty
hindsight correction when forward signals conflict with outcome
```

### 6.1 默认 turn-level

Agent 任务的自然单元通常是：

```text
observe -> think -> act/tool -> observe
```

默认把一个完整 turn 当作 credit 单元，避免 token-level 高方差。

### 6.2 触发 step-level

当一个 turn 内有多个可分离子决策时，拆成 step-level：

```text
tool selection
argument filling
confirmation / stop decision
```

触发条件：

```text
entropy(tool_name) high OR entropy(tool_args) high OR invalid_action_risk high
```

### 6.3 低频 hindsight correction

不建议一开始实现昂贵反事实 RL。MVP 可以做低频 post-hoc 校准：

1. 对失败轨迹和成功轨迹按 query 匹配。
2. 找到二者第一次状态/动作分叉点。
3. 如果某个 step 的 forward utility 高，但出现在大量失败轨迹分叉后，则降低该类 process reward 的 reliability。

伪公式：

```text
credit_t = eta * forward_IG_t + (1 - eta) * hindsight_alignment_t
```

这能把专题二的信用分配自然接进 ENTROPIA，而不是另起炉灶。

---

## 7. 最终方法命名与模块

建议论文方法写成四个模块：

### Module A: State Need Estimator

输入 key-token uncertainty、group collapse、progress stagnation，输出 `Need_t`。

### Module B: Process Signal Bank

维护多个候选过程奖励：

```text
R_info_gain
R_progress
R_efficiency_cost
R_tool_validity_mask
R_evidence_quality
```

注意：`tool_validity` 和 `format` 默认作为 mask/reliability，不作为强正奖励。

### Module C: Reliability-Calibrated Reward Router

对每个 `R_k` 计算 `Utility_{t,k}` 与 `Reliability_{t,k}`，输出 `g_{t,k}`。

### Module D: Adaptive Budget and Risk Controller

根据 advantage collapse、reward-success divergence、hacking risk 动态调节 dense budget。

---

## 8. 论文贡献可以这样写

### Contribution 1: Formulation

提出长程 LLM Agent RL 中的 **state-conditional reward routing** 问题，把奖励密度从固定超参数转化为状态条件的监督分配策略。

### Contribution 2: Method

提出 ENTROPIA，一个 Need-Utility-Reliability 三因子奖励路由器。它用不确定性估计监督需求，用信息增益/进度估计步骤效用，用 outcome alignment 和 hacking risk 估计过程信号可靠性。

### Contribution 3: Credit Granularity

提出轻量层次化 credit routing：默认 turn-level，在子决策不确定或动作无效风险高时切到 step-level，并用低频 hindsight alignment 校准前向过程信号。

### Contribution 4: Empirical Evidence

在搜索和工具调用 Agent RL 上证明：相比纯稀疏、固定密集、静态混合和 entropy-only gating，ENTROPIA 同时提升成功率、样本效率，并降低 reward hacking 频率。

---

## 9. 实验设计

### 9.1 最小可发表实验矩阵

优先做两个环境即可，不要一开始铺太大。

| 环境 | 目的 | 推荐数据 |
|---|---|---|
| Search QA | 验证 IG utility + reward routing | HotpotQA / NQ + Search-R1 风格环境 |
| Tool use | 验证 tool-state progress + hacking robustness | tau-bench / tau2-bench / SIMIA tau 数据 |

如果资源不足，先做：

1. Search QA 主实验。
2. tau-bench 小规模真实/半真实实验作为迁移验证。

### 9.2 Baselines

必须有：

| Baseline | 解释 |
|---|---|
| Sparse GRPO | outcome-only 下界 |
| Fixed Dense IG | IGPO-style 固定密集 |
| Static Hybrid | outcome + 固定权重 process reward |
| AutoTool-style entropy loss | 熵在 loss 里，而非 reward routing |
| SELAUR-style uncertainty reward | 不确定性直接塑形奖励 |
| ENTROPIA-v1 | entropy-only gate |
| Random Gate | 排除“只是少给 dense reward 就有效” |
| Oracle / Post-hoc Gate | 上界分析，不一定作为正式 baseline |

### 9.3 Ablations

核心消融要围绕故事：

| Ablation | 证明什么 |
|---|---|
| Need only | 熵门控本身不够 |
| Need + Utility | 有用性信号的增益 |
| Need + Reliability | 可靠性过滤的增益 |
| Full NUR | 三因子协同 |
| No risk controller | hacking 闭环是否必要 |
| Fixed dense budget | adaptive budget 是否必要 |
| Turn-only vs step-switch | credit granularity 是否必要 |
| Format as reward vs format as mask | 证明 mask 设计更抗 gaming |

### 9.4 Metrics

不要只报 success rate。需要让方法机制可见。

| 指标 | 意义 |
|---|---|
| Task success / EM / pass rate | 主性能 |
| Steps-to-threshold | 样本效率 |
| Reward hacking rate | robustness |
| Reward-success divergence | 过程奖励是否被刷 |
| Non-zero advantage ratio | GRPO 是否有学习信号 |
| Dense budget curve | 方法是否按训练阶段自适应 |
| Gate precision | 被打开的门是否真对应有贡献 step |
| Process-outcome correlation | reliability gate 是否有效 |
| Token/step cost | 是否过搜或膨胀 |

Gate precision 可以 post-hoc 定义：

```text
useful step = positive IG/progress and appears more in successful than failed rollouts
gate_precision = P(useful step | gate open)
```

这个分析很重要，因为它能证明 ENTROPIA 不是黑箱调参。

---

## 10. 理论/分析角度

论文不需要硬凑大定理，但可以有一个清晰分析命题。

### 10.1 Process reward 的风险

设真实但不可见的 step advantage 为 `A*_t`，过程奖励诱导的 advantage 为 `A^p_t`。如果：

```text
E[A^p_t * A*_t] <= 0
```

则注入该过程奖励会引入有害梯度。

固定密集奖励默认所有 step 都可信，而 ENTROPIA 估计：

```text
P(A^p_t * A*_t > 0 | state_t, signal_k)
```

并只在高概率区域注入该过程信号。

### 10.2 可写成命题

在 process reward reliability estimator 排序一致的假设下，对高 reliability 区域进行门控选择，会提高被注入过程奖励与 outcome advantage 的期望对齐度：

```text
E[A^p_t * A^out_t | g_t > tau] >= E[A^p_t * A^out_t]
```

这不是要证明“ENTROPIA 最优”，而是给 reviewer 一个原则化理由：

> The gate is not an arbitrary heuristic; it is a selection mechanism for advantage-aligned process signals.

---

## 11. 实现路线

### Phase 0: 修正 v1 的实现偏差

当前 `training/trainer.py` 的 adaptive 分支实际上做的是：

```text
r_adaptive = r_sparse + alpha * gate * r_sparse
```

这不是文档里的：

```text
r_adaptive = r_sparse + alpha * gate * r_dense
```

第一步必须把 `r_dense` 作为显式输入接进 reward matrix。

### Phase 1: 实现 Process Signal Bank

新增：

```text
core/process_signals.py
```

包含：

```text
InfoGainSignal
ProgressSignal
EfficiencyCostSignal
ToolValidityMask
EvidenceNoveltySignal
```

每个 signal 返回：

```text
value_t
utility_t
metadata
```

### Phase 2: 实现 Reward Router

新增：

```text
core/reward_router.py
```

核心接口：

```python
router.route(
    outcome_reward,
    process_signals,
    uncertainty_features,
    reliability_stats,
    risk_features,
)
```

输出：

```text
reward_t
gate_{t,k}
dense_budget
diagnostics
```

### Phase 3: 把 HackingDetector 改成 RiskController

保留原 detector，但新增：

```text
risk_score_t
batch_hacking_rate
budget_update
component_downweighting
```

### Phase 4: 跑最小论文实验

先不要做所有 benchmark。建议顺序：

1. `Sparse / Fixed Dense / ENTROPIA-v1 / ENTROPIA-v2` 在现有 mock 或小数据上 smoke test。
2. Search QA 上跑完整 ablation。
3. tau-bench/tau2-bench 上跑主对比和 hacking 分析。

---

## 12. Paper Outline

### Title

```text
ENTROPIA: State-Conditional Reward Routing for Robust LLM Agent Reinforcement Learning
```

### Abstract 主线

1. Long-horizon agent RL needs process supervision, but dense rewards are brittle and hackable.
2. Existing methods use fixed reward density or uncertainty as a direct reward/loss signal.
3. We formulate dense supervision as a routing problem.
4. ENTROPIA routes process rewards using Need, Utility, and Reliability gates under an adaptive dense budget.
5. Experiments show higher sample efficiency and lower reward hacking.

### Section 1: Introduction

讲“过程奖励不是免费的午餐”：它既提供学习信号，也提供捷径。

### Section 2: Problem Formulation

定义：

```text
state-conditioned reward routing
dense supervision budget
process signal reliability
```

### Section 3: Method

四模块：

1. Need estimator
2. Process signal bank
3. Reliability-calibrated router
4. Adaptive risk/budget controller

### Section 4: Experiments

主表 + 消融 + hacking robustness + gate analysis。

### Section 5: Analysis

密度曲线、gate precision、process-outcome correlation、失败案例。

---

## 13. 最小可执行版本

如果只想 1-2 周内把故事和实验跑起来，做这个版本：

```text
Need = key-token entropy + group outcome collapse
Utility = max(0, information gain)
Reliability = batch corr(process reward, outcome reward) + format-valid mask
Risk = repetition + short response + reward-success divergence
Budget = EMA adaptive scalar
Credit = turn-level only
```

奖励：

```text
g_t = Need_t * Utility_t * Reliability_batch * (1 - Risk_t)
r_t = r_outcome_t + B_s * g_t * IG_t
```

这个版本已经比 v1 强很多，因为它能回答：

- 为什么高熵不一定开门？
- 为什么有些 dense reward 被拒绝？
- 为什么格式不作为主奖励？
- 为什么 reward hacking 会反向影响后续密度？
- 为什么这是一个通用调度层，而不是单个奖励 trick？

---

## 14. 什么不要放进主贡献

以下内容可以作为工程增强或消融，但不建议当主贡献：

1. **四阶段 scheduler**：故事好讲，但像课程学习调参，不是核心创新。
2. **ReDit reward dithering**：可作为低方差补丁，不应喧宾夺主。
3. **更多 uncertainty metrics**：entropy/least-confidence/margin 是特征，不是论文贡献。
4. **更多 benchmark**：没有机制分析时，benchmark 多也救不了故事。
5. **hacking detector 单独做安全网**：必须接进 router，变成闭环控制。

---

## 15. 与现有材料的关系

ENTROPIA v2 吸收但重排了现有文档中的模块：

| v1/已有想法 | v2 位置 |
|---|---|
| 熵门控 | Need Gate |
| IGPO 信息增益 | Utility Gate / Process Signal Bank |
| 一致性门控 | Reliability Gate 的特例 |
| Hacking detector | Risk Controller |
| 多维 reward | Process Signal Bank，但格式更多作为 mask |
| 四阶段调度 | Adaptive Budget 的可选初始化 |
| 状态条件贡献估计 | Credit Routing |
| ReDit dithering | reward variance fallback |

最重要的叙事变化：

```text
v1: Entropy decides reward density.
v2: State decides supervision routing.
```

这个变化足以把论文从一个公式扩成一个方法框架。

---

## 16. Reference Anchors

需要在 related work 中重点对比：

- AutoTool: entropy is used as an optimization/loss-side exploration signal, while ENTROPIA uses uncertainty as one input to reward routing.
- IGPO / StepSearch: information gain is a process reward, while ENTROPIA decides when that process reward is useful and reliable.
- SELAUR: uncertainty shapes rewards directly, while ENTROPIA separates need estimation from process-signal utility and reliability.
- Sparse RL / Search-R1 / ReTool: outcome-only rewards are safe anchors, but long-horizon tasks need conditional process supervision.
- Reward hacking / TRACE-style analyses: dense reward failures motivate reliability and risk gates.

---

## 17. 更大工作区调研后的修正

综合前期 gap analysis、实验路线和多条候选方向调研后，ENTROPIA 的定位需要再收紧一次。

### 17.1 关键外部结论

工作区的反复结论是：

1. **单独做信息增益不够新。** IGPO、StepSearch、IG-Search 已经把“增量信息作为过程奖励”占得很深。ENTROPIA 不能把 IG 当主贡献，只能把它当 Utility signal。
2. **单独做自适应搜索深度风险高。** AutoSearch、TSTV 路线很自然，但更适合作为 test-time extension，而不是 ENTROPIA 第一篇主线。
3. **最稳主线是搜索/工具 Agent 的系统性奖励设计。** CRAFT-Search 被多轮调研反复选为主线，原因是它风险低、实验可控、和 Search-R1/veRL 生态对齐。
4. **但纯 CRAFT 也有故事风险。** 如果只是五维 reward 加权，容易被 reviewer 看成“大消融实验”。它需要一个原则化机制来说明“为什么这个 reward 此时该启用”。
5. **Reward hacking 分析是差异化关键。** 现有单维奖励工作多关注效果，较少系统处理格式刷分、无效重复检索、citation gaming、reward-success divergence。

所以最优融合不是：

```text
ENTROPIA vs CRAFT-Search
```

而是：

```text
CRAFT-Search defines the reward signal bank.
ENTROPIA decides when each signal should be trusted and routed.
```

### 17.2 新定位：ENTROPIA as Routing Layer over CRAFT

建议论文方法名可以保留 ENTROPIA，但副标题改成：

```text
ENTROPIA-Search: Reliability-Calibrated Reward Routing for Search Agent RL
```

对应关系：

| CRAFT 组件 | ENTROPIA v2 中的位置 | 设计修正 |
|---|---|---|
| `R_answer` | outcome anchor | 始终保留，不参与门控关闭 |
| `R_info_gain` | Utility signal | 只在 Need 高且 Reliability 高时注入 |
| `R_efficiency` | cost/budget signal | 不应简单惩罚所有长轨迹，只惩罚低 Utility 的长轨迹 |
| `R_citation` | evidence reliability / final quality | 更适合作为终态质量和 citation-risk 信号 |
| `R_format` | validity mask / reliability constraint | 主要作为 parser mask，避免变成 format gaming 正奖励 |
| reward hacking detector | Risk controller | 从安全网升级为动态降权机制 |
| dynamic weight schedule | budget prior | 只是先验，最终由 Need/Utility/Reliability/Risk 在线修正 |

这使 ENTROPIA 的贡献不是“我又加了几个 reward”，而是：

> We turn a composite reward bank into a state-conditioned, reliability-calibrated supervision router.

### 17.3 与 CRAFT-Search 的最小融合公式

设 CRAFT 提供候选过程信号：

```text
S_t = {
  info_gain_t,
  retrieval_quality_t,
  efficiency_cost_t,
  citation_quality_t,
  format_valid_t
}
```

ENTROPIA 不直接线性求和，而是对每个信号单独路由：

```text
r_t =
  r_answer_t
  + B_s * sum_k w_k(s) * g_{t,k} * clip_norm(S_{t,k})

g_{t,k} =
  Need_t
  * Utility_{t,k}
  * Reliability_{batch,k}
  * Validity_{t,k}
  * (1 - Risk_t)
```

其中：

- `Need_t` 来自 key-token entropy、group outcome collapse、progress stagnation。
- `Utility_{t,k}` 来自 IG、evidence novelty、state progress 或 cost saving。
- `Reliability_{batch,k}` 来自 process-outcome correlation、advantage agreement、成功/失败 rollout 区分度。
- `Validity_{t,k}` 主要来自格式/工具调用合法性。
- `Risk_t` 来自重复搜索、短响应、token bloat、format gaming、citation hallucination、reward-success divergence。

关键差异：

```text
CRAFT fixed/dynamic:
  reward = weighted_sum(all reward components)

ENTROPIA routing:
  reward = outcome_anchor + routed subset of reliable process signals
```

### 17.4 论文故事线的最终版

不要把论文讲成“entropy gate 有效”。要讲成三步：

1. **Search Agent RL needs process rewards, but process rewards are unreliable.**
   Search-R1 类方法只有 outcome reward；CRAFT 类复合奖励能补过程信号，但 process rewards 会引入 reward hacking。

2. **Reward reliability is state-conditional.**
   同一个奖励信号在不同状态下含义不同：早期 IG 很有价值，后期重复检索带来的表面 novelty 可能是负价值；格式早期是可解析性保障，后期可能变成刷分捷径。

3. **ENTROPIA routes process supervision through Need, Utility, Reliability and Risk.**
   熵只回答 Need；信息增益回答 Utility；过程/结果一致性回答 Reliability；hacking detector 回答 Risk。

最终摘要里的核心句可以是：

> ENTROPIA separates dense supervision into four questions: whether the agent needs guidance, whether a step contains useful information, whether the reward signal is outcome-aligned, and whether the behavior shows signs of reward hacking.

### 17.5 从其他候选方向吸收什么

#### State Conditional Contribution Estimation

吸收为 **Credit Routing Extension**，不是第一阶段必做主模块。

可加入一组分析实验：

```text
Static contribution weighting vs state-conditional contribution weighting
```

最小实验：

1. 在 HotpotQA 500 条上做反事实干预。
2. 移除第 t 次检索，重跑后续推理。
3. 用 EM/F1 变化作为 ground-truth contribution。
4. 比较 `IG_t`、静态 LLM judge、状态条件估计与 ground-truth 的 Spearman 相关。

如果有效，作为 ENTROPIA 的 credit granularity 贡献；如果无效，作为负面分析也有价值。

#### TSTV / Adaptive Termination

吸收为 **test-time transfer**，不要和主贡献抢位置。

可写成讨论或扩展：

```text
The same Need/Utility/Risk features used for train-time reward routing can be reused for test-time search termination.
```

但第一篇不建议主打 TSTV，因为 AutoSearch 已经覆盖了自适应深度主问题，且推理时终止会引入另一套实验矩阵。

#### Pure RL from Base

作为训练初始化消融即可：

```text
Instruct+RL vs SFT+RL vs Base+RL
```

不要把它放进主贡献。纯 RL 搜索涌现是独立大问题，失败概率高，会拖垮 ENTROPIA 的主线。

#### TriPRM / Search PRM

作为高成本 upper bound 或 optional baseline：

```text
learned PRM signal vs rule/proxy signal bank
```

ENTROPIA 第一版不应依赖外部 PRM 成功，否则数据标注和算力风险都升高。

### 17.6 实验矩阵重排

最小可发表矩阵应从“大而全”改为“能证明路由层必要性”。

#### Tier 1: Sanity and Implementation Correctness

| 方法 | 目的 |
|---|---|
| Search-R1 / sparse GRPO | outcome-only 基线 |
| CRAFT-fixed | 复合奖励但固定权重 |
| CRAFT-dynamic | 阶段式动态权重 |
| ENTROPIA-v1 | entropy-only gate |
| ENTROPIA-v2 | Need-Utility-Reliability-Risk routing |

成功标准：

- v2 不一定要大幅超过 CRAFT-dynamic，但必须在 hacking rate、reward-success correlation、search efficiency 上明显更稳。

#### Tier 2: Mechanism Ablations

| 消融 | 证明 |
|---|---|
| Need only | 仅熵门控不足 |
| Need + Utility | IG/novelty 有必要 |
| Need + Reliability | outcome alignment 有必要 |
| Need + Utility + Reliability | 三因子主效果 |
| Full minus Risk | hacking 闭环的必要性 |
| Format as reward vs format as mask | 格式奖励的 gaming 风险 |
| Fixed budget vs adaptive budget | 密集监督预算自适应是否有效 |

#### Tier 3: Analysis, not just accuracy

必须有机制图表：

| 图表 | 说明 |
|---|---|
| dense budget over training | 是否从高密度自然退火 |
| gate precision/recall | 开门是否对准有用步骤 |
| process-outcome correlation | reliability 是否提升过程信号质量 |
| reward hacking timeline | 风险控制是否提前响应 |
| search turn distribution | 是否减少无效过搜 |
| component downweight events | 哪些 reward 被判定不可靠 |

主结果表只报 EM/F1 不够，ENTROPIA 的说服力来自这些诊断。

### 17.7 对比基线的真实优先级

必做：

1. Search-R1 / outcome-only GRPO
2. Fixed dense IG / StepSearch-style process reward
3. CRAFT fixed composite reward
4. Entropy-only ENTROPIA-v1
5. Random gate / random budget

尽量做：

1. CalibAdv：代表搜索场景下的优势校准。
2. CW-GRPO：代表贡献加权。
3. ReDit：代表 reward variance 处理。

可选：

1. AutoTool-style entropy loss：若从工具调用场景切入则更必要。
2. TSTV/AutoSearch：只在 test-time extension 中做。
3. TriPRM：作为 learned process reward upper bound。

### 17.8 最小实现建议

按工程风险，第一版只实现：

```text
Need:
  key-token entropy
  group outcome collapse

Utility:
  positive information gain
  evidence novelty / retrieval relevance

Reliability:
  batch corr(process_reward, outcome_reward)
  advantage sign agreement
  format-valid mask

Risk:
  repetition
  over-search
  format gaming
  reward-success divergence

Budget:
  scalar dense budget B_s updated by collapse/hacking/conflict rates
```

暂不实现：

- 学习式 PRM
- 完整反事实 credit model
- test-time adaptive beam
- pure base-model RL
- citation NLI 复杂训练闭环

这些作为第二篇或扩展实验。

### 17.9 需要修正的宣传口径

避免说：

```text
ENTROPIA solves reward design.
Entropy tells us when to give dense reward.
Information gain is our main novelty.
Format reward improves training.
```

改成：

```text
ENTROPIA routes process rewards.
Uncertainty estimates supervision need, not correctness.
Information gain is one utility signal among several.
Format is primarily a validity and reliability constraint.
Reward hacking signals close the loop by downweighting unreliable process rewards.
```

这是更抗审稿的说法。

### 17.10 与整个工作区的最终对齐

前期调研最终推荐 CRAFT-Search 作为系统性奖励设计主线；动态 reward density 提供训练时密集监督控制；状态条件贡献提供 credit routing；TSTV 提供 test-time transfer。

ENTROPIA v2 正好可以统一它们：

```text
CRAFT-Search:
  What reward signals exist?

ENTROPIA:
  When should each signal be trusted and routed?

State-Conditional Contribution:
  Which step should receive the routed signal?

TSTV:
  Can the same state features control inference-time budget?
```

这给 ENTROPIA 一个更清楚的位置：

> ENTROPIA is the control layer that turns composite reward design into adaptive, reliable, state-conditioned supervision.

---

## 18. 新文献吸收与设计扩展（2026-04 更新）

综合前沿文献扫描（研究1、研究2报告），ENTROPIA v2 设计在以下方面得到扩展。核心方法（NUR + Risk 路由）不变，新增的是具体实现变体和对比 baseline。

### 18.1 从 GiGPO（NeurIPS 2025）吸收：双层 Credit Granularity

GiGPO 提出两层 GRPO 结构（轨迹级 + 步骤级子组），在不引入辅助奖励模型的情况下同时捕获全局轨迹质量和局部动作效果。

ENTROPIA 的吸收方式：将 GiGPO 的双层结构和 NUR 门控结合。外层用标准 GRPO 标准化，内层在 NUR 门控打开的步骤上再做子组标准化。这比原设计中"手动切换 turn/step 粒度"更优雅——粒度切换由门控值自动决定，无需人为定义切换条件。

对应公式变体：C2（见 `paper_formula_methodology.md` Section 9.1）。

### 18.2 从 CoREN（arXiv 2411.17135）吸收：跨信号一致性

CoREN 的核心思路：多个奖励信号的一致性本身可以作为可靠性的指标。

ENTROPIA 的吸收方式：在 Reliability Gate 中新增第四个统计量 $\kappa$（跨信号一致性），计算多个过程信号对同一步评价的 sign agreement。成本几乎为零（复用已有计算），但能捕获信号矛盾的情况。

对应公式：$\kappa^k_s = \frac{1}{K-1} \sum_{k'\neq k} \mathbb{E}[\mathbb{1}[\operatorname{sign}(S^k) = \operatorname{sign}(S^{k'})]]$

### 18.3 从 TIPS（arXiv 2603.22293）吸收：势函数形式化

TIPS 用势函数理论形式化了 turn-level 奖励设计。

ENTROPIA 的吸收方式：将 Utility Gate 统一写成势函数差分 $\max(0, \Phi_k(s_t) - \Phi_k(s_{t-1}))$。方法上和原设计等价，但论文中可以引用势函数理论（policy-invariance guarantee）作为理论背书。

### 18.4 新增 Baseline（来自研究2报告）

以下 baseline 在原有列表基础上新增，均为 2026 年发表的新工作：

| Baseline | 来源 | 与 ENTROPIA 的关系 |
|---|---|---|
| CW-GRPO | arXiv 2604.14267，ACL 2026 | 有 LLM judge 的贡献加权（上界），ENTROPIA 是无 judge 的轻量替代 |
| CalibAdv | arXiv 2604.18235 | 启发式 advantage 校准，ENTROPIA 的 Reliability Gate 是自适应版本 |
| GiGPO | NeurIPS 2025 | 双层 GRPO 但无路由门控，验证 NUR 门控的增益 |
| StepPO | arXiv 2604.18401 | 步骤对齐优化，步骤级优化的另一路径 |

### 18.5 Reliability Gate 三种变体

基于审稿风险分析（"参数怎么来的？为什么是加法？为什么是 sigmoid？"），Reliability Gate 设计为三种实验变体：

- **R1（加法）**：线性补偿，表达力强但超参多
- **R2（乘法）**：一票否决，和路由门乘法逻辑一致，zscore 归一化 + 共享温度
- **R3（Softmax 竞争）**：信号竞争有限 budget，零阈值，最简洁

三种变体对应不同设计哲学，实验中对比选择。详见 `paper_formula_methodology.md` Section 6.6。

### 18.6 实验矩阵总结

主干实验（R × C 变体组合）：6 组
消融实验（含新增 $\kappa$、Budget、C2 门控）：9 组
外部 baseline（含新增 CW-GRPO/CalibAdv/GiGPO/StepPO）：12 组

总计约 20 组实验，在 HotpotQA 上用 Qwen2.5-7B GRPO 跑 300-400 步。

### 18.7 不吸收的内容及理由

| 内容 | 理由 |
|---|---|
| R*/FORGE 自动奖励设计 | 论文定位是"路由已有信号"，不是"自动生成信号" |
| 能量模型奖励 | 理论太重，实验成本高，和轻量路线冲突 |
| HRDL 层次化语言驱动 | 偏奖励函数自动生成，和路由层定位不同 |
| iStar 隐式步骤奖励 | 实现复杂，NUR 已覆盖类似功能 |
| 跨域统一框架 | 太长期（1-2年），会拖垮当前论文 |
| 小模型蒸馏 | 独立方向，和方法创新不搭 |
