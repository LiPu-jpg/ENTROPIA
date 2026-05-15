# ENTROPIA-Search 组会汇报大纲

日期：2026-05-15  
项目：ENTROPIA-Search: State-Conditional Reward Routing for Robust LLM Agent RL

---

## 1. 研究背景：为什么做 LLM Agent RL 的奖励路由

近两年 LLM+RL 的重点正在从传统 RLHF / DPO 偏好对齐，转向通过可验证奖励激发推理、搜索和工具使用能力。DeepSeek-R1 / GRPO 证明了 outcome reward 可以驱动复杂推理；Search-R1 进一步把这种范式扩展到搜索增强推理和多轮工具交互。

但在长程 Agent RL 中，只用最终结果奖励会遇到两个问题：

- 奖励稀疏，长轨迹上信用分配困难；
- GRPO 依赖组内 reward 差异，若 rollout 结果相近，advantage 容易塌缩。

直接加入密集过程奖励也有风险：

- 局部过程信号不一定和最终成功一致；
- 格式、长度、重复搜索等 proxy reward 容易被 hack；
- 固定加权的复合奖励缺少状态条件判断。

本工作的核心问题：

> 过程奖励不是免费监督。我们需要判断：什么时候需要它，哪个信号有用，当前信号是否可靠，是否存在 reward hacking 风险。

---

## 2. 研究定位：从 Reward Density 到 Reward Routing

最初版本是熵门控奖励密度：

```text
r_adaptive = r_sparse + alpha * gate(entropy) * r_dense
```

但这个定位太窄，只回答“模型不确定时是否多给过程奖励”。

当前定位是：

```text
Composite reward design gives candidate process signals.
ENTROPIA decides which signals are needed, useful, reliable, and safe to inject.
```

一句话：

> ENTROPIA-Search 是一个面向长程 LLM Agent RL 的可靠性校准奖励路由层。

它位于以下工作之间：

- Search-R1 / ReTool：只用 outcome reward；
- IGPO / StepSearch：固定使用信息增益或 step reward；
- CRAFT-style composite reward：固定混合多个过程信号；
- ENTROPIA：动态判断哪些过程信号在当前状态下可信、可用、安全。

---

## 3. 方法总览：NUR + Risk 奖励路由

对每个 rollout 的每个 step，以及每个候选过程信号 `k`，ENTROPIA 计算：

```text
g_t^k = N_t * U_t^k * L_s^k * M_t^k * (1 - H_t^risk)
```

最终奖励：

```text
r_t^ENT = r_t^out + B_s * sum_k w_k * g_t^k * S_t^k
```

各模块含义：

- `Need N_t`：当前状态是否需要过程监督；
- `Utility U_t^k`：信号 k 在当前 step 是否代表真实进展；
- `Reliability L_s^k`：信号 k 在近期 batch 中是否与最终 outcome 对齐；
- `Mask M_t^k`：格式/工具调用是否合法；
- `Risk H_t^risk`：是否出现重复、过搜、格式作弊、reward-success divergence；
- `B_s`：可动态调整的 dense supervision budget。

PPT 图建议：

- 左侧：rollout 轨迹；
- 中间：process signal bank；
- 右侧：NUR + Risk router；
- 输出：routed reward + GRPO update。

---

## 4. 当前过程信号设计

当前代码支持两类场景：tool-use / customer-service 和 Search QA。

### Tool-use 场景中的信号

`info_gain`

```text
新匹配到一个 GT 工具步骤时，给任务进度增量。
```

含义：这个 step 是否推进了正确工具链。

`relevance`

```text
当前工具调用与尚未匹配 GT action 的 name/kwargs 相似度 * novelty
```

含义：工具名和参数是否和任务相关，重复调用不会持续加分。

`efficiency_cost`

```text
超过任务先验工具深度后，如果无进展且弱相关，则给 -1 惩罚。
```

含义：抑制无效长轨迹和过度工具调用。

`format_valid`

```text
能被解析成合法工具调用则为 1，否则为 0。
```

含义：作为 mask，而不是主奖励，避免 format gaming。

### Search QA 场景规划中的信号

`info_gain`

```text
log p(answer | state_t) - log p(answer | state_{t-1})
```

`relevance`

```text
supporting fact / retrieved context relevance * novelty
```

`efficiency_cost`

```text
无新增证据的额外搜索惩罚
```

---

## 5. Reliability Gate：为什么不是固定加权

固定 dense reward 的问题是：默认所有过程信号都可信。

ENTROPIA 的 Reliability Gate 估计：

```text
这个过程信号近期是否真的和 outcome 改善一致？
```

当前实现支持三种变体：

### R1：加法可靠性

```text
L_s^k = sigmoid(a*rho + b*xi + c*delta - d*chi)
```

特点：多个可靠性统计量可以互相补偿。

### R2：乘法可靠性

```text
L_s^k = sigmoid(rho) * sigmoid(xi) * sigmoid(delta) * (1 - sigmoid(chi))
```

特点：更像一票否决，任何维度很差都会压低信号。

### R3：Softmax 竞争

```text
过程信号竞争有限 dense budget。
```

特点：低可靠信号会被其他信号挤掉。

PPT 表建议：

| 变体 | 设计直觉 | 优点 | 风险 |
|---|---|---|---|
| R1 | 线性补偿 | 稳定、宽松 | 可能放过弱信号 |
| R2 | 一票否决 | 更保守 | 可能过度关闭 |
| R3 | 预算竞争 | 超参少 | 早期统计不稳 |

---

## 6. 当前实验表现

已有 80-step MiniMax RL 小规模实验：

| 方法 | 说明 | MiniMax 测试分 |
|---|---|---:|
| sparse | 纯 outcome reward | 0.125 |
| adaptive | v1 熵门控 | 0.085 |
| router R1 | v2 加法路由 | 0.134 |
| router R2 | v2 乘法路由 | 0.079 |
| router R3 | v2 Softmax | 0.128 |

初步观察：

- `router R1` 当前最好，略高于 sparse；
- `router R3` 接近 sparse，说明 softmax 路由可能有潜力；
- `adaptive` 和 `R2` 表现较差，可能因为门控过强或过程信号质量不足；
- 当前差距较小，还不能作为正式结论。

重要说明：

> 这些结果来自早期实现。当时 `info_gain` 和 `relevance` 实际上没有真正接入，主要生效的是 outcome reward + gated efficiency shaping。因此这组实验只能说明 reward routing 方向有初步迹象，不能声称完整 ENTROPIA 已验证。

---

## 7. 最近完成的关键修复

之前的问题：

- `info_gain` 没有传入真实 logprob / trajectory 信息，基本为 0；
- `relevance` 没有实际计算，基本为 0；
- `adaptive` 实际写成了 `r_sparse + gate * r_sparse`，不是 `r_sparse + gate * r_dense`；
- `efficiency_cost` 方向不清晰，可能把低效步骤当成正奖励；
- `format_valid` 没有真正作为 mask 发挥作用。

当前修复：

- Tool-use 轨迹中现在会保存解析后的工具名、参数和格式合法性；
- `SignalBank` 可以基于 `trajectory + task GT actions` 计算真实过程信号；
- `info_gain` 表示工具链进度增量；
- `relevance` 表示工具名/参数匹配度并带 novelty；
- `efficiency_cost` 改为负向信号；
- `format_valid` 作为 gate mask；
- `adaptive / dense_igpo / dense_fixed / random_gate / router` 都改为使用显式过程信号。

修复后 smoke test：

```text
info_gain  [0.5, 0.5, 0.0, 0.0]
relevance  [1.0, 1.0, 0.0, 0.0]
efficiency [0.0, 0.0, 0.0, -1.0]
format     [1.0, 1.0, 0.0, 1.0]
```

---

## 8. 为什么要转向公开 Benchmark

原来的 synthetic tau / SIMIA 数据适合 debug，但不适合证明“比别人方法有提升”。

原因：

- 不是社区通用 benchmark；
- 别人的方法没有在该数据上报告；
- reviewer 难以判断任务难度；
- LLM judge 方差较大，单一分数说服力有限。

下一阶段应使用公开 benchmark：

### 主线：Search QA / HotpotQA

目标：验证 `info_gain + relevance + efficiency` 在搜索任务中的路由价值。

优势：

- 与 Search-R1 路线对齐；
- HotpotQA 支持多跳推理；
- distractor context 可以先作为 per-question corpus，不需要完整 Wikipedia dump；
- 便于计算 EM/F1、supporting fact relevance、搜索步数。

当前已加入仓库：

- `hotpot_train_1k.json`
- `hotpot_dev_distractor_500.json`
- full train compressed shards
- full distractor dev gzip

### 副线：tau / tau2 / tau3 tool-use

目标：验证工具链进度、格式 mask、risk controller。

### 鲁棒性扩展：AgentDojo / Prompt Injection

目标：验证在检索文档或工具返回中存在恶意指令时，Reliability / Risk 是否能降低不可信过程信号。

---

## 9. 下一阶段实验计划

### Phase 1：重新跑修复后的 tool-use 小实验

目的：确认真实过程信号接入后是否改善。

最小矩阵：

| 方法 | 目的 |
|---|---|
| sparse | outcome-only 下界 |
| dense_igpo | 只用 info_gain |
| dense_fixed | 固定复合过程奖励 |
| adaptive | entropy gate + dense reward |
| random_gate | 排除“随机少给奖励也有效” |
| router R1 | 主方法 |
| router R3 | 竞争式路由候选 |

需要报告：

- success / MiniMax score；
- reward-success correlation；
- non-zero advantage ratio；
- average turns；
- gate activation rate；
- process signal statistics。

### Phase 2：接入 HotpotQA Search QA

先用 distractor context 做 per-question corpus：

```text
query -> search context -> answer -> EM/F1
```

优先验证：

- fixed dense IG 是否优于 sparse；
- ENTROPIA 是否优于 fixed dense；
- router 是否降低无效搜索次数；
- gate 是否更常打开在 supporting fact 相关步骤。

### Phase 3：和更强 baseline 对齐

必须对比：

- Search-R1-style sparse GRPO；
- Fixed dense IG / StepSearch-style reward；
- Static composite reward；
- ENTROPIA-v1 entropy-only；
- Random gate。

建议加入：

- GiGPO：双层 credit granularity；
- CalibAdv：advantage 校准；
- CW-GRPO：LLM judge contribution weighting，上界/强 baseline。

---

## 10. 预期论文主张

论文标题候选：

```text
ENTROPIA: State-Conditional Reward Routing for Robust LLM Agent Reinforcement Learning
```

核心贡献：

1. 提出 state-conditional reward routing，把固定过程奖励设计改写为状态条件的监督分配问题。
2. 提出 NUR + Risk 框架，用 Need、Utility、Reliability 和 Risk 控制过程奖励注入。
3. 在 search / tool-use agent RL 中验证：相比 outcome-only、fixed dense、entropy-only，ENTROPIA 能提升训练稳定性和样本效率，并降低 reward hacking / reward-success divergence。

需要谨慎表述：

- 目前已有结果只是早期 smoke test；
- 旧结果不能代表完整 ENTROPIA；
- 修复后需要重跑正式实验；
- 当前可解释性来自门控和信号诊断，不是 mechanistic interpretability。

---

## 11. 组会可以强调的核心 takeaway

一页总结：

> 我们不是单纯提出一个新的过程奖励，而是提出一个判断过程奖励何时可信、何时有用、何时应该关闭的路由层。

当前进展：

- 完成 LLM+RL 综述和方向定位；
- 完成 ENTROPIA v2 公式设计；
- 实现 NUR + Risk reward router；
- 完成初步 MiniMax tool-use RL 实验；
- 发现并修复过程信号未真实接入的问题；
- 准备 HotpotQA 全量/小样本数据，开始向公开 Search QA benchmark 迁移。

下一步最关键：

```text
用修复后的真实过程信号，在公开 HotpotQA / Search-R1-style 设置下重跑：
sparse vs fixed dense vs entropy-only vs random gate vs ENTROPIA。
```

判断标准：

- 不只看最终 EM/F1 或 success；
- 还要看 gate 是否开在正确步骤、过程信号是否与 outcome 对齐、是否减少无效搜索和 reward hacking。

