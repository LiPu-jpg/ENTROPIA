# ENTROPIA-Search: 论文公式与方法论定稿

> 这份文档用于替代散乱的“公式草图”。目标是让 Method section 可以直接按这里写：符号统一、模块边界清楚、公式与实验消融一一对应。

---

## 0. 一句话定位

**ENTROPIA-Search is a reliability-calibrated reward routing layer over a composite reward bank.**

中文：

> CRAFT-Search 回答“有哪些奖励信号”；ENTROPIA-Search 回答“这些过程奖励在当前状态下是否需要、是否有用、是否可靠、是否安全”。

不要再把论文讲成“熵门控奖励密度”。最终故事是：

```text
Outcome reward 是锚点。
Process rewards 是候选监督信号。
ENTROPIA 根据状态、效用、可靠性和风险，选择性注入过程监督。
```

---

## 1. 问题定义

给定一个搜索/工具 Agent，对每个问题 \(q\) 采样 \(G\) 条 rollout。第 \(g\) 条轨迹写作：

\[
\tau_g = (s_{g,1}, a_{g,1}, o_{g,1}, \ldots, s_{g,T_g}, a_{g,T_g}, o_{g,T_g}, y_g)
\]

其中：

- \(s_{g,t}\)：第 \(t\) 轮前的状态，包含原问题、历史动作、已检索文档、已填 slot 等。
- \(a_{g,t}\)：第 \(t\) 轮动作，如搜索 query、工具调用、参数填充、停止决策。
- \(o_{g,t}\)：环境观察，如检索结果、工具返回值。
- \(y_g\)：最终答案或最终任务状态。
- \(R^{out}(\tau_g)\)：最终 outcome reward，如 EM/F1、任务成功率。

现有 Search-R1 风格训练通常只用最终 outcome。主文建议写成终态奖励：

\[
r_{g,t}^{out}
= \mathbb{1}[t=T_g]R^{out}(\tau_g)
\]

随后由 return 计算自然把终态信号折扣回前面的步骤。实现时也可以把 \(R^{out}\) 预先折扣广播到每步，但这两种写法只能选一种，不能先广播再在 return 中二次折扣。

问题是：长轨迹下 outcome-only reward 稀疏，GRPO 容易出现 advantage collapse；但固定密集过程奖励又容易 reward hacking。

ENTROPIA 研究的问题是：

> 给定一组候选过程奖励信号，如何在每个状态下决定哪些信号值得注入策略优化？

---

## 2. 候选过程信号库

令 \(\mathcal{K}\) 表示过程信号集合。第一篇建议只用轻量信号，不依赖训练 PRM：

| 记号 | 信号 | 类型 | 作用 |
|---|---|---|---|
| \(S^{ig}_{t}\) | 信息增益 | 正向过程信号 | 奖励有新增信息的检索/工具调用 |
| \(S^{rel}_{t}\) | 检索/证据相关性 | 正向过程信号 | 奖励命中支持事实或相关文档 |
| \(S^{eff}_{t}\) | 效率成本 | 负向过程信号 | 惩罚低收益的额外搜索/工具调用 |
| \(S^{cit}_{t}\) | 引用/证据忠实性 | 终态或稀疏过程信号 | 约束答案和证据一致 |
| \(M^{valid}_{t}\) | 格式/工具合法性 | mask | 决定过程信号是否可信，不作为强正奖励 |

注意：这里故意把 format 从 reward 降级为 mask。论文里应明确说：

> Format is a validity constraint, not a main optimization target.

否则很容易被质疑 format gaming。

### 2.1 信息增益信号

有 gold answer 时：

\[
S^{ig}_{t}
= \max\left(0,\log p_\theta(y^\star \mid s_t) - \log p_\theta(y^\star \mid s_{t-1})\right)
\]

无 gold answer 或开放任务时，可用候选答案置信度变化：

\[
S^{ig}_{t}
= \max(0, C_\theta(s_t) - C_\theta(s_{t-1}))
\]

其中 \(C_\theta(s)\) 是模型对当前 best answer 的置信度、verifier 分数或 judge 置信度。

核心表述：

> IG is not the novelty of ENTROPIA; IG is one utility signal routed by ENTROPIA.

### 2.2 检索相关性/证据新颖性

搜索场景中可以用：

\[
S^{rel}_{t}
= \operatorname{Rel}(d_t, q, y^\star)
\]

其中 \(d_t\) 是第 \(t\) 轮新增文档。实现上可以用 BM25 overlap、supporting fact hit、reranker score 或 NLI entailment proxy。

为避免重复搜索，需要新颖性折扣：

\[
\operatorname{Novel}(d_t)
= 1 - \max_{j<t}\operatorname{sim}(d_t,d_j)
\]

组合：

\[
S^{rel}_{t}
= \operatorname{Rel}(d_t,q,y^\star)\cdot \operatorname{Novel}(d_t)
\]

### 2.3 效率成本

效率不是简单地“越短越好”。只惩罚**低收益的额外步骤**：

\[
S^{eff}_{t}
= -\mathbb{1}[t>T_{prior}] \cdot (1-U_t)
\]

其中 \(U_t\) 是当前步的整体 utility，\(T_{prior}\) 是任务先验最小搜索深度，或者由数据集 hop 数近似。

直觉：

```text
有新信息的长搜索不惩罚。
没有新信息的长搜索才惩罚。
```

### 2.4 信号归一化

所有过程信号进入路由器前先做稳定化：

\[
\tilde S^k_t
= \operatorname{clip}\left(
\frac{S^k_t-\mu^k_s}{\sigma^k_s+\epsilon},
-c,c
\right)
\]

其中 \(\mu^k_s,\sigma^k_s\) 是训练 step \(s\) 的滑动均值和标准差。这样不同 reward component 不会因为尺度不同主导训练。

---

## 3. ENTROPIA 的核心：NUR + Risk

ENTROPIA 的主门控是三因子：

```text
Need × Utility × Reliability
```

Risk 不是第四个并列贡献，而是一个安全控制器，负责抑制或关闭过程监督。因此论文中建议写成：

> ENTROPIA uses Need-Utility-Reliability routing under a Risk controller.

最终路由门：

\[
g^k_t
= N_t \cdot U^k_t \cdot L^k_s \cdot M^k_t \cdot (1-H_t^{risk})
\]

其中：

- \(N_t\)：Need，当前状态是否需要过程监督。
- \(U^k_t\)：Utility，信号 \(k\) 在当前步是否有任务效用。
- \(L^k_s\)：Reliability，信号 \(k\) 在近期 batch 中是否与 outcome 对齐。
- \(M^k_t\)：Validity mask，动作/格式/证据是否合法。
- \(H_t^{risk}\)：hacking risk，当前行为是否显示刷奖励风险。

对不依赖格式或工具合法性的信号，令 \(M^k_t=1\)。format 本身不作为主正奖励，而是主要进入 \(M^k_t\) 或风险项。

---

## 4. Need Gate：当前状态是否需要帮助

Need 不是“熵”。熵只是 Need 的一个特征。

\[
N_t = \sigma\left(\frac{\phi_N(s_t)-\tau_N}{T_N}\right)
\]

\[
\phi_N(s_t)
= \lambda_H \hat H_t
+ \lambda_C C_s
+ \lambda_P P_t
\]

其中：

- \(\hat H_t\)：关键 action token 熵，如 search query、tool name、tool args、stop token。
- \(C_s\)：group outcome collapse 指标。如果同一 query 的 \(G\) 条 rollout outcome 几乎相同，则 \(C_s=1\)。
- \(P_t\)：progress stagnation 指标。如果多步没有新增信息或状态推进，则 \(P_t=1\)。
- \(\tau_N\)：Need 阈值。
- \(T_N\)：温度。

关键 token 熵：

\[
\hat H_t =
\frac{1}{|\mathcal{A}_t|}
\sum_{j\in \mathcal{A}_t}
-\sum_{v\in V}p_\theta(v\mid h_{t,j})\log p_\theta(v\mid h_{t,j})
\]

\(\mathcal{A}_t\) 是 action-critical token 集合。

解释口径：

```text
高 Need 表示“当前状态需要额外监督”，不表示“当前动作一定错”。
```

---

## 5. Utility Gate：当前信号是否有用

每个过程信号 \(k\) 有自己的 utility：

\[
U^k_t = \sigma\left(\frac{\phi^k_U(s_t,a_t,o_t)-\tau^k_U}{T^k_U}\right)
\]

推荐第一版：

| 信号 | Utility 特征 \(\phi^k_U\) |
|---|---|
| info gain | \(\max(0,\Delta \log p(y^\star))\) |
| retrieval relevance | relevance × novelty |
| efficiency cost | \(\mathbb{1}[\text{no new info}] \cdot \mathbb{1}[t>T_{prior}]\) |
| citation | entailment/confidence at final answer |

注意 efficiency 是负向信号，但它也需要 utility gate。也就是说：

```text
只有当一步确实低效时，效率惩罚才启用。
```

### 5.1 势函数形式化（Potential-Based Formulation）

上述 Utility 定义可以统一写成势函数差分形式：

\[
U^k_t = \max(0, \Phi_k(s_t) - \Phi_k(s_{t-1}))
\]

其中每个信号的势函数 $\Phi_k$ 定义为：

| 信号 | $\Phi_k(s)$ | 直觉 |
|---|---|---|
| info gain | $\Phi = \log p(y^\star \mid s)$ | 答案概率 |
| retrieval relevance | $\Phi = \operatorname{Rel}(d,q,y^\star) \cdot \operatorname{Novel}(d)$ | 证据价值 |
| efficiency cost | $\Phi = -\mathbb{1}[t>T_{prior}] \cdot (1-U_t)$ | 效率代价 |
| citation | $\Phi = \operatorname{entailment}(\text{answer}, \text{evidence} \mid s)$ | 忠实性 |

这一形式化遵循 potential-based reward shaping（Ng et al., 1999）的理论框架，保证势函数奖励不改变最优策略，只改变学习速度。论文中可以引用 TIPS（arXiv 2603.22293）的势函数理论作为理论锚点。

关键表述：

> Utility Gate is not a heuristic; it follows the potential-based reward shaping framework, ensuring policy-invariance while providing denser learning signals at information-bearing steps.

---

## 6. Reliability Gate：这个奖励信号近期可信吗

Reliability 建议按 batch 或滑动窗口估计，而不是每个 step 重新学一个模型。这样实现简单、可解释、风险低。

\[
L^k_s
= \sigma\left(
a\rho^k_s
+ b\xi^k_s
+ c\delta^k_s
- d\chi^k_s
\right)
\]

四个统计量：

### 6.1 Process-outcome correlation

\[
\rho^k_s
= \operatorname{corr}\left(
\sum_t \tilde S^k_{g,t},
R^{out}(\tau_g)
\right)_{g=1}^{G}
\]

如果某个过程信号高，但最终 outcome 不高，说明这个信号不可靠。

### 6.2 Advantage sign agreement

\[
\xi^k_s
= \mathbb{E}_{g,t}
\left[
\mathbb{1}
\left(
\operatorname{sign}(A^{k}_{g,t})
= \operatorname{sign}(A^{out}_{g,t})
\right)
\right]
\]

表示 process advantage 和 outcome advantage 的梯度方向是否一致。

### 6.3 Discriminativeness

\[
\delta^k_s
=
\mathbb{E}[\tilde S^k \mid R^{out}>m]
-
\mathbb{E}[\tilde S^k \mid R^{out}\le m]
\]

其中 \(m\) 是当前 group 或 batch 的 outcome 中位数。好轨迹和坏轨迹过程分差越大，信号越可靠。

### 6.4 Proxy gaming risk

\[
\chi^k_s
= \operatorname{GamingRisk}(S^k)
\]

例如：

- format 高但 answer 低。
- IG 高但检索文档与答案无关。
- citation 高但 answer 错。
- efficiency 高但任务失败率升高。

这一步是 ENTROPIA 和普通动态权重调度的关键区别：**权重不是只按训练阶段调，而是按信号可靠性调。**

### 6.5 跨信号一致性（Cross-Signal Consistency）

\[
\kappa^k_s
= \frac{1}{K-1}
\sum_{k' \neq k}
\mathbb{E}_{g,t}\left[
\mathbb{1}\left[
\operatorname{sign}(\tilde S^k_{g,t})
= \operatorname{sign}(\tilde S^{k'}_{g,t})
\right]
\right]
\]

如果多个过程信号对同一步的评价一致（都认为好或都认为差），说明该步的可靠性更高。
如果信号之间矛盾（IG 高但检索不相关），则可靠性降低。

这一指标受 CoREN（arXiv 2411.17135）的一致性集成思路启发，但无需额外模型，直接复用已有过程信号的计算结果，额外计算成本几乎为零。

### 6.6 Reliability Gate 三种变体设计

Reliability Gate 的聚合方式对训练动态有显著影响。建议设计三种变体，实验中对比选择最优：

**变体 R1（加法版）：**

\[
L^k_s
= \sigma\left(
a\rho^k_s + b\xi^k_s + c\delta^k_s + e\kappa^k_s - d\chi^k_s
\right)
\]

各维度线性补偿，一个维度弱可以靠另一个拉回来。超参 5 个系数。表达力强但需调参。

**变体 R2（乘法版）：**

\[
L^k_s
= \sigma_\rho(\rho^k_s) \cdot \sigma_\xi(\xi^k_s) \cdot \sigma_\delta(\delta^k_s) \cdot \sigma_\kappa(\kappa^k_s) \cdot (1 - \sigma_\chi(\chi^k_s))
\]

其中 $\sigma_x(x) = \operatorname{sigmoid}\left(\frac{x - \mu_x}{\sigma_x \cdot T_L}\right)$，zscore 归一化 + 共享温度 $T_L$。

乘法结构实现一票否决：任何维度差直接压到 0。和路由门 $g = N \times U \times L \times M \times (1-R)$ 的乘法逻辑一致。只有 1 个超参 $T_L$。

**变体 R3（Softmax 竞争版）：**

去掉 $L^k_s$ 作为独立门控。过程信号之间竞争有限的 dense budget：

\[
\alpha^k = \frac{\exp(N_t \cdot U^k_t \cdot \rho^k_s \cdot \tilde S^k_t)}
{\sum_{k'} \exp(N_t \cdot U^{k'}_t \cdot \rho^{k'}_s \cdot \tilde S^{k'}_t)}
\]

\[
r^{ENT}_{g,t} = r^{out}_{g,t} + B_s \sum_{k} \alpha^k \cdot \tilde S^k_{g,t}
\]

信号可靠性高的拿到更大 budget 份额，可靠性低的自动被挤掉。不需要手动设阈值判断"可不可靠"。最简洁，零阈值超参。

**变体选择原则**：三种变体对应不同的设计哲学（补偿/否决/竞争）。实验中对比（见 Section 12），选择在目标场景上最优的版本作为主方法，其余作为消融分析。

---

## 7. Risk Controller：刷奖励风险如何闭环

定义 step-level risk：

\[
H_t^{risk}
= \operatorname{clip}\left(
\sum_j \beta_j h^j_t, 0, 1
\right)
\]

推荐第一版风险信号：

| 风险 | 检测 |
|---|---|
| repetition | 连续相似 query 或相同工具参数 |
| over-search | 多步无新增信息还继续搜索 |
| short-circuit | 异常短响应或直接答题 |
| format gaming | format valid 但 answer/retrieval 质量低 |
| reward-success divergence | 过程奖励上升但 success 下降 |

风险有两个作用：

1. **step-level 抑制**

\[
g^k_t \leftarrow g^k_t (1-H_t^{risk})
\]

2. **batch-level budget 更新**

\[
B_{s+1}
= \operatorname{clip}
\left(
B_s
+ \eta_c \operatorname{Collapse}_s
- \eta_h \operatorname{Hack}_s
- \eta_d \operatorname{Divergence}_s,
B_{\min}, B_{\max}
\right)
\]

解释：

- collapse 多，说明 outcome-only 不够，增加 dense budget。
- hacking 多，说明 process reward 危险，降低 dense budget。
- divergence 多，说明过程信号与最终目标冲突，降低 dense budget。

---

## 8. 最终奖励函数

最终用于 GRPO 的 per-turn reward：

\[
r^{ENT}_{g,t}
= r^{out}_{g,t}
+ B_s
\sum_{k\in \mathcal{K}}
w^k_s
g^k_{g,t}
\tilde S^k_{g,t}
+ \epsilon_{g,t}
\]

其中：

- \(r^{out}_{g,t}=\mathbb{1}[t=T_g]R^{out}(\tau_g)\)。如果实现采用折扣广播版 outcome reward，则后续 return 计算中不要再次对该项折扣。
- \(B_s\in[0,1]\) 是当前 dense supervision budget。
- \(w^k_s\) 是 CRAFT/课程调度提供的先验权重。
- \(g^k_{g,t}\) 是 ENTROPIA 路由门。
- \(\epsilon_{g,t}\) 是可选 reward dithering，仅当 group reward 方差过低时启用。

重要：\(w^k_s\) 和 \(g^k_{g,t}\) 分工不同。

```text
w_s^k: 这个训练阶段大体重视哪个奖励维度。
g_t^k: 当前状态下这个奖励信号是否可信、是否应注入。
```

如果 risk 极高：

\[
H^{risk}_s > \tau_H \Rightarrow B_s=0
\]

即 fallback 到 outcome-only。

其中 \(H^{risk}_s=\mathbb{E}_{g,t}[H^{risk}_{g,t}]\) 是当前 batch 的平均风险。

---

## 9. 与 GRPO 的结合

对每个 query \(q_i\)，采样 \(G\) 条 rollout。得到 per-turn rewards 后计算 return：

\[
\mathcal{R}_{i,g,t}
= \sum_{u=t}^{T_g}
\gamma^{u-t} r^{ENT}_{i,g,u}
\]

组内标准化 advantage：

\[
A_{i,g,t}
=
\frac{
\mathcal{R}_{i,g,t}
- \frac{1}{G}\sum_{g'}\mathcal{R}_{i,g',t}
}{
\operatorname{Std}_{g'}(\mathcal{R}_{i,g',t})+\epsilon
}
\]

GRPO/PPO-style clipped objective：

\[
\mathcal{L}_{GRPO}
=
-\mathbb{E}_{i,g,t}
\left[
\min
\left(
\rho_{i,g,t} A_{i,g,t},
\operatorname{clip}(\rho_{i,g,t},1-\epsilon_{clip},1+\epsilon_{clip})A_{i,g,t}
\right)
\right]
+ \beta_{KL}D_{KL}(\pi_\theta||\pi_{ref})
\]

其中：

\[
\rho_{i,g,t}
=
\frac{\pi_\theta(a_{i,g,t}\mid s_{i,g,t})}
{\pi_{old}(a_{i,g,t}\mid s_{i,g,t})}
\]

如果实现上沿用 trajectory-level GRPO，也可以把 \(r^{ENT}_{g,t}\) 聚合成：

\[
R^{ENT}(\tau_g)=\sum_t \gamma^{t-1}r^{ENT}_{g,t}
\]

再做 trajectory-level group normalization。论文主文建议写 per-turn 版本，附录说明实现可退化为 trajectory-level。

### 9.1 Credit Granularity 变体设计

**变体 C1（单层 GRPO，原版）：**

\[
A_{i,g,t}
= \frac{\mathcal{R}_{i,g,t} - \frac{1}{G}\sum_{g'}\mathcal{R}_{i,g',t}}
{\operatorname{Std}_{g'}(\mathcal{R}_{i,g',t})+\epsilon}
\]

所有步骤统一标准化，粒度固定为 trajectory-level。

**变体 C2（双层 GiGPO + NUR 门控筛选）：**

受 GiGPO（NeurIPS 2025）的双层组结构启发，在标准 GRPO advantage 之上增加步骤级子组标准化：

外层（轨迹级，不变）：

\[
A^{traj}_{i,g,t}
= \frac{\mathcal{R}_{i,g,t} - \frac{1}{G}\sum_{g'}\mathcal{R}_{i,g',t}}
{\operatorname{Std}_{g'}(\mathcal{R}_{i,g',t})+\epsilon}
\]

内层（步骤级，仅在相似状态的子组内）：

\[
G_{i,t} = \{g' : \operatorname{sim}(s_{g',t}, s_{g,t}) > \theta_{sim}\}
\]

\[
A^{step}_{i,g,t}
= \frac{\mathcal{R}_{i,g,t} - \frac{1}{|G_{i,t}|}\sum_{g' \in G_{i,t}}\mathcal{R}_{i,g',t}}
{\operatorname{Std}_{g' \in G_{i,t}}(\mathcal{R}_{i,g',t})+\epsilon}
\]

最终融合：

\[
A_{i,g,t}
= A^{traj}_{i,g,t} + \lambda_{credit} \cdot N_t \cdot U^k_t \cdot A^{step}_{i,g,t}
\]

关键设计：内层 advantage 被乘以 $N_t \times U^k_t$，因此：
- Need 高 + Utility 高的步骤 → step-level advantage 被注入，精细区分
- Need 低或 Utility 低的步骤 → step-level 项趋零，退回纯 trajectory-level

这实现了"自动切换信用粒度"，无需手动定义切换条件。额外超参仅 $\lambda_{credit}$ 和相似度阈值 $\theta_{sim}$。

---

## 10. 训练算法

```text
Algorithm 1: ENTROPIA-Search

Input:
  policy pi_theta, reference pi_ref,
  training queries D,
  process signal bank K,
  dense budget B_0,
  group size G

For training step s = 1...S:
  1. Sample a batch of queries.
  2. For each query, sample G rollouts using pi_theta.
  3. Compute outcome reward R_out for each rollout.
  4. Compute process signals S^k_t for each turn and component.
  5. Estimate Need N_t from key-token uncertainty and group collapse.
  6. Estimate Utility U^k_t for each process signal.
  7. Estimate Reliability L^k_s from process-outcome alignment statistics.
  8. Estimate Risk H^risk_t from hacking detectors.
  9. Compute routing gate:
        g^k_t = N_t * U^k_t * L^k_s * M^k_t * (1 - H^risk_t)
 10. Build routed reward:
        r_ENT_t = r_out_t + B_s * sum_k w^k_s * g^k_t * normalized(S^k_t)
 11. Compute GRPO advantages within each query group.
 12. Update policy using clipped GRPO objective.
 13. Update dense budget B_s using collapse/hacking/divergence statistics.

Return pi_theta.
```

---

## 11. 方法模块写法

Method section 可以分成 5 个小节。

### 11.1 Composite Process Signal Bank

不是贡献点本身，只是 ENTROPIA 的输入。

写法：

> We assume access to a bank of lightweight process signals, including information gain, retrieval relevance, efficiency cost and validity masks. Unlike prior work that directly sums these signals, ENTROPIA treats them as candidates whose reliability must be estimated online.

### 11.2 State Need Estimator

贡献点之一。强调熵只是 Need proxy。

写法：

> Need captures when outcome-only supervision is insufficient.

### 11.3 Utility-Aware Signal Routing

贡献点之一。强调不是所有高熵步骤都给 reward，而是必须有 positive utility。

写法：

> Utility prevents the model from receiving dense rewards for verbose or redundant actions.

### 11.4 Reliability Calibration

最重要的贡献点。把 process reward 是否可信显式建模。

写法：

> Reliability estimates whether a process signal is aligned with outcome improvements in the current training window.

### 11.5 Risk-Controlled Dense Budget

把 hacking detector 变成闭环控制，而不是离线监控。

写法：

> The dense budget increases under advantage collapse and decreases under reward hacking or process-outcome divergence.

---

## 12. 消融实验与公式对应关系

| 实验 | 对应公式组件 | 要证明什么 |
|---|---|---|
| Sparse GRPO | 去掉整个 process term | outcome-only 不够 |
| CRAFT-fixed | \(g^k_t=1\)，固定 \(w^k\) | 复合奖励但不路由的效果 |
| CRAFT-dynamic | \(g^k_t=1\)，阶段性 \(w^k_s\) | 课程式权重是否足够 |
| ENTROPIA-v1 | \(g_t=N_t\) | 仅熵/Need 不够 |
| Need+Utility | \(g^k_t=N_tU^k_t\) | utility 是否必要 |
| Need+Reliability | \(g^k_t=N_tL^k_s\) | reliability 是否必要 |
| Full no Risk | 去掉 \((1-H^{risk}_t)\) 和 budget update | risk controller 是否必要 |
| Full ENTROPIA | 全公式 | 主方法 |
| Random gate | \(g^k_t\sim Bernoulli(p)\) | 排除“少给 dense reward 就行” |
| Format-as-reward | 把 \(M^{valid}_t\) 改成正奖励 | 证明 mask 比正奖励抗 gaming |

这个表很重要。它让公式不是装饰，而是直接决定实验矩阵。

### 12.1 Reliability / Credit 变体对比实验

主干实验（6 组），在最优 NUR+Risk 配置下对比 Reliability 和 Credit 的变体：

| 编号 | 配置 | 目的 |
|---|---|---|
| E1 | R1 + C1 | 加法 R + 单层 C（原版基线） |
| E2 | R2 + C1 | 乘法 R 是否优于加法 |
| E3 | R3 + C1 | Softmax 竞争是否优于独立门控 |
| E4 | R1 + C2 | 双层 Credit 是否优于单层 |
| E5 | R2 + C2 | 乘法 R + 双层 C 的组合 |
| E6 | R3 + C2 | Softmax R + 双层 C 的组合 |

### 12.2 扩展消融实验

| 编号 | 配置 | 要证明什么 |
|---|---|---|
| A5 | 完整但 $\kappa=0$ | 跨信号一致性 $\kappa$ 的增益 |
| A6 | 完整但固定 $B_s$ | 自适应 Budget 的增益 |
| A7 | C2 但无 $N \times U$ 门控（即 $\lambda_{credit}$ 直接乘以 $A^{step}$） | NUR 门控筛选 step-level advantage 的必要性 |

### 12.3 扩展外部 Baseline

在原有 baseline 基础上，增加以下对比：

| Baseline | 说明 | 为什么必要 |
|---|---|---|
| CalibAdv（arXiv 2604.18235） | 启发式 advantage 校准，下调中间步骤的过大负 advantage | ENTROPIA 的 Reliability Gate 是自适应版本，需对比启发式方案 |
| CW-GRPO（arXiv 2604.14267，ACL 2026） | LLM judge 贡献加权 GRPO | 有 LLM judge 的上界，ENTROPIA 是无 LLM judge 的轻量替代 |
| GiGPO（NeurIPS 2025） | 双层 GRPO 但无路由门控 | 验证双层结构 + NUR 门控是否优于纯双层 |
| StepPO（arXiv 2604.18401） | 步骤对齐策略优化 | 步骤级优化的另一路径 |

---

## 13. 评价指标与方法论闭环

主结果只看 EM/F1 不够。ENTROPIA 必须报告机制指标：

| 指标 | 定义 | 证明 |
|---|---|---|
| EM/F1 / success | 最终任务指标 | 方法没有牺牲效果 |
| avg search turns | 平均搜索轮数 | 是否减少无效搜索 |
| dense budget curve | \(B_s\) 随训练变化 | 是否自适应调度 |
| gate precision | \(P(\text{useful step}\mid g_t>\tau)\) | 门是否开对地方 |
| process-outcome corr | \(\rho^k_s\) | reliability 是否提升 |
| reward-success divergence | process reward 上升但 success 下降的频率 | 是否减少 hacking |
| format gaming rate | format valid 但 answer wrong 的比例 | mask 设计是否必要 |
| non-zero advantage ratio | group 内 advantage 非零比例 | 是否缓解 collapse |

“useful step” 可以 post-hoc 定义：

\[
\text{useful}_t
= \mathbb{1}[S^{ig}_t>0]
\land
\mathbb{1}[\text{step appears more in successful rollouts}]
\]

更强版本可用反事实移除该步后的 EM 变化定义。

---

## 14. 论文主贡献建议

### Contribution 1: Problem formulation

提出 **state-conditional reward routing**，把复合过程奖励从固定加权问题改写为状态条件的监督路由问题。

### Contribution 2: Method

提出 ENTROPIA-Search，一个 NUR + Risk 的奖励路由层，用 Need 判断是否需要过程监督，用 Utility 判断当前信号是否有用，用 Reliability 判断过程信号是否与 outcome 对齐，并用 Risk Controller 动态调节 dense budget。

### Contribution 3: Robustness analysis

系统分析搜索/工具 Agent 中的 reward hacking 模式，并证明把 format、citation、efficiency 等信号作为 reliability/risk 约束比直接作为正奖励更稳。

### Contribution 4: Empirical validation

在 Search-R1 风格环境和工具调用环境上，对比 outcome-only、固定密集、CRAFT fixed/dynamic、entropy-only gating，证明 ENTROPIA 提升过程信号质量、训练稳定性和搜索效率。

---

## 15. 最小可执行版本

第一版不要做太复杂。

实现：

```text
Signals:
  S_ig       = positive answer log-prob shift (势函数形式化)
  S_rel      = retrieval relevance × novelty
  S_eff      = no-new-info over-search penalty
  M_valid    = parse/tool validity mask

Need:
  key-token entropy + group outcome collapse

Utility:
  max(0, Φ_k(s_t) - Φ_k(s_{t-1}))  势函数差分统一形式

Reliability（三种变体实验对比）:
  R1: σ(aρ + bξ + cδ + eκ - dχ)    加法版
  R2: σ_ρ(ρ)·σ_ξ(ξ)·σ_δ(δ)·σ_κ(κ)·(1-σ_χ(χ))  乘法版
  R3: softmax 竞争版

  统计量：ρ(process-outcome corr), ξ(advantage agreement),
         δ(discriminativeness), κ(跨信号一致性), χ(gaming risk)

Credit（两种变体实验对比）:
  C1: 单层 GRPO 标准化
  C2: 双层 A^traj + λ·N·U·A^step (GiGPO + NUR 门控)

Risk:
  repetition, over-search, format gaming, reward-success divergence

Budget:
  scalar B_s updated by collapse/hacking/divergence
```

暂不实现：

- learned PRM
- full hindsight credit model
- adaptive beam / TSTV
- pure Base RL
- complicated citation NLI training

这些都是第二阶段扩展。

---

## 16. 最终摘要口径

可以用这段作为 abstract/method 的核心：

> Long-horizon agent RL benefits from process rewards, but dense process signals are often unreliable and hackable. We formulate process supervision as state-conditional reward routing. ENTROPIA-Search routes a bank of lightweight process signals through Need, Utility and Reliability gates under a Risk controller. Need estimates when outcome-only supervision is insufficient, Utility detects whether the current step provides task-relevant information, Reliability measures whether a process signal is aligned with outcome improvements in the current training window, and Risk reduces dense supervision under reward hacking patterns. This turns composite reward design from fixed weighting into adaptive, reliable supervision allocation.

### 16.1 实验变体总结

ENTROPIA-Search 的实验设计覆盖三个设计维度：

1. **Reliability 聚合方式**（R1 加法 / R2 乘法 / R3 Softmax 竞争）
2. **Credit Granularity**（C1 单层 / C2 双层 GiGPO+NUR 门控）
3. **Utility 形式化**（势函数差分，与 TIPS 理论框架对齐）

每个维度都有独立的消融实验验证其必要性。最终推荐配置由实验结果决定，论文中报告最优配置及其与各变体的对比分析。
