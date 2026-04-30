# 01 方向选择

## 核心决策

项目不应被定位为一种基于熵门控的奖励密度技巧。

最终定位：

```text
ENTROPIA-Search is a reliability-calibrated reward routing layer
over a composite process-reward bank.
```

论文层面的分解如下：

```text
CRAFT-Search:
  What reward signals exist?

ENTROPIA:
  When should each signal be trusted and routed?

State-conditional contribution:
  Which step should receive the routed signal?

TSTV / adaptive termination:
  Can the same state features control inference-time budget?
```

## 选择标准

| 标准 | 含义 |
| --- | --- |
| 新颖性 | 贡献不应被 IGPO、StepSearch、AutoTool、SELAUR 或 Search-R1 风格的稀疏 RL 所覆盖。 |
| 可行性 | 第一篇论文应能基于当前 ENTROPIA 代码库实现。 |
| 叙事强度 | 方法应能支撑一篇完整论文，而非单一启发式技巧。 |
| 基线清晰度 | 每个组件应能对应一个干净的基线或消融实验。 |
| 审稿人风险 | 工作不应看起来像任意的奖励工程。 |
| 实现路径 | 设计应是对 v1 的升级，而非推倒重来。 |

## 候选方向矩阵

| 方向 | 决策 | 原因 |
| --- | --- | --- |
| 基于熵门控的奖励密度 | 降级为 v1 基线 | 作为唯一贡献太单薄；熵仅估计不确定性，不反映奖励的有用性或可靠性。 |
| 信息增益作为过程奖励 | 用作 Utility 信号 | IGPO/StepSearch/IG-Search 已占据直接的"信息增益奖励"空间。 |
| CRAFT 风格的复合奖励 | 用作信号库 | 有利于系统化的奖励设计，但固定的复合奖励容易被视为一张大型消融表。 |
| 奖励路由 | 主要贡献 | 将稠密奖励转化为基于状态条件的、可靠性校准的监督信号。 |
| 奖励作弊检测 | 核心安全组件 | 为 Reliability 和 Risk 门控提供具体的存在理由。 |
| 状态条件贡献 | 扩展 | 对归因粒度有用，但不是第一个方法声明所必需的。 |
| 自适应搜索终止 | 测试时扩展 | 自然迁移，但会增加一个独立的推理时实验矩阵。 |
| 纯 RL 搜索涌现 | 仅作为初始化消融 | 作为主论文太宽泛且风险太高。 |
| 学习型 PRM / TriPRM | 可选上界 | 有价值的对比，但如果作为必需项会增加数据和计算风险。 |

## 最终叙事

结果奖励仍然是锚点，但长 horizon 的搜索/工具任务通常需要过程监督。问题在于过程奖励并非总是有帮助：它们可能是过时的、冗余的、与最终成功不对齐的，或可被作弊的。

因此，ENTROPIA 将稠密监督分解为四个问题：

1. **Need（需求）**：该状态是否需要额外的过程引导？
2. **Utility（效用）**：该候选过程信号是否表明了有意义的进展？
3. **Reliability（可靠性）**：该信号近期是否与结果改进保持一致？
4. **Risk（风险）**：行为是否表现出奖励作弊症状？

这给出了完整的方法框架：

```text
outcome anchor + process signal bank + state-conditioned reward router + risk controller
```

## 写作边界

避免：

```text
Entropy decides reward density.
```

应使用：

```text
Uncertainty estimates supervision need.
Process signals estimate utility.
Reliability calibrates outcome alignment.
Risk suppresses hackable dense rewards.
```
