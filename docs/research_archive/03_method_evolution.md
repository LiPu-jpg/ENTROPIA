# 03 方法演进

本文件保留了从旧 v1 算法笔记到当前 v2 论文设计的过渡记录。

## V1：基于熵门控的奖励密度

旧核心公式：

```text
r^adaptive_t = r^sparse_t + alpha * sigmoid(H_t - H_threshold) * r^dense_t
```

旧解释：

```text
high entropy -> model uncertain -> open dense reward gate
low entropy -> model confident -> rely on sparse outcome reward
```

熵信号在关键 token 上计算，如工具名称、工具参数、停止 token 和动作类型，而非填充 token。

旧阈值更新：

```text
H_threshold <- beta * mean(H_batch) + (1 - beta) * H_threshold
```

该 v1 设计作为基线有价值，但作为主论文贡献过于狭窄。

## V1 失败模式

| 失败模式 | v1 无法处理的原因 |
| --- | --- |
| 高置信度的错误动作 | 低熵会关闭门控，即使模型需要纠正。 |
| 不确定但正确的探索 | 高熵可能过度注入稠密奖励。 |
| 重复检索的新颖性 | 信息增益代理可能奖励表面新颖性。 |
| 仅格式优化 | 智能体可能学会可解析但无用的工具调用。 |
| 奖励-成功发散 | 稠密奖励可能上升而结果成功率下降。 |
| 训练阶段不匹配 | 早期有用的信号可能在后期变得有害。 |

结论是熵仅回答了一个问题：

```text
Does the state need guidance?
```

它无法回答：

```text
Is this process reward useful?
Is it reliable?
Is it safe?
```

## V2：奖励路由

v2 公式将过程信号库与路由决策分离。

候选过程信号：

```text
S^k_t,  k in K
```

路由门控：

```text
g^k_t = N_t * U^k_t * L^k_s * M^k_t * (1 - H^risk_t)
```

最终奖励：

```text
r^ENT_{g,t}
  = r^out_{g,t}
    + B_s * sum_k w^k_s g^k_{g,t} S^k_{g,t}
    + epsilon_{g,t}
```

其中：

| 项 | 含义 |
| --- | --- |
| `r^out` | 稀疏结果锚点。 |
| `B_s` | 阶段 `s` 的稠密奖励预算。 |
| `w^k_s` | 信号 `k` 的先验或课程权重。 |
| `S^k_t` | 来自信号库的候选过程信号。 |
| `N_t` | Need 门控，通常使用熵/不确定性。 |
| `U^k_t` | 信号特定进展的 Utility 门控。 |
| `L^k_s` | 从近期结果对齐校准的 Reliability 门控。 |
| `M^k_t` | 信号适用性的 Validity 掩码。 |
| `H^risk_t` | 作弊/发散行为的 Risk 分数。 |
| `epsilon` | 可选的零均值探索噪声。 |

## 折扣修正

旧 v1 文档在每个步骤使用了折扣稀疏奖励：

```text
r^sparse_t = R_outcome * gamma^(T - 1 - t)
```

论文设计应避免双重折扣。干净的公式为：

```text
r^out_{g,t} = 1[t = T_g] * R^out(tau_g)
```

然后 RL 回报计算应用一次折扣。

## GRPO 集成

ENTROPIA 不替代 GRPO。它改变 GRPO 使用的奖励向量：

```text
rollout -> process signals -> routing gates -> ENT reward -> GRPO advantage -> policy update
```

所需诊断指标：

1. 稠密预算使用量。
2. 每信号门控激活。
3. 过程-结果一致性。
4. 奖励-成功相关性。
5. 作弊率。

## 论文方法声明

方法声明应为：

```text
Dense process rewards are useful but state-dependent and risky.
ENTROPIA routes them only when they are needed, useful, reliable, and safe.
```
