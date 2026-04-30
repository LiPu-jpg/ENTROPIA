# ENTROPIA-Search 研究背景

## 最终定位

ENTROPIA-Search 不应被框定为一种熵门控技巧。

最终定位为：

```text
Composite reward design gives us candidate process signals.
ENTROPIA decides which signals are needed, useful, reliable, and safe to inject.
```

## 为何这是正确的范围

整个工作区收敛于三个结论：

1. 单一信息增益奖励不足以作为论文贡献，因为 IGPO/StepSearch 类工作已覆盖了该空间的大部分。
2. 固定复合奖励框架有用，但有被读成大规模消融实验的风险。
3. 更锐利的贡献在于自适应奖励路由：过程奖励仅在状态相关且与结果对齐时才有帮助。

ENTROPIA 位于 CRAFT-Search 与信用分配之间：

```text
CRAFT-Search:
  what reward signals exist?

ENTROPIA:
  when should each signal be trusted and routed?

State-conditional contribution:
  which step should receive the routed signal?

TSTV:
  can the same state features control inference-time search budget?
```

## 论文主张

稠密过程奖励并非免费监督。它们可以加速学习，但也可能注入误导性梯度或创造奖励作弊捷径。

ENTROPIA 将稠密监督分解为四个问题：

1. **Need**：当前状态是否需要过程监督？
2. **Utility**：候选信号是否表明真实的任务进展？
3. **Reliability**：该信号近期是否与结果改善保持对齐？
4. **Risk**：行为是否表现出奖励作弊迹象？

## 规范文档

- `docs/paper_formula_methodology.md` — 公式与方法节的权威来源
- `docs/design_v2_reward_routing.md` — 设计原理与研究整合
- `docs/research_archive/` — 已清理的方向选择与研究归档
- `docs/related_work.md` — 相关工作地图

## 基线优先级

必须运行：

1. 仅结果 GRPO / Search-R1 风格稀疏基线
2. 固定稠密 IG / StepSearch 风格过程奖励
3. CRAFT 固定复合奖励
4. ENTROPIA-v1 仅熵门控
5. ENTROPIA-v2 完整路由
6. 随机门控健全性检查

建议运行：

1. CalibAdv
2. CW-GRPO
3. ReDit

可选：

1. TSTV / AutoSearch 作为测试时迁移
2. TriPRM 作为学习式过程奖励上界

## 实现注意事项

当前训练器在部分地方仍反映旧的 v1 公式。特别是，自适应奖励构建必须停止使用稀疏奖励作为稠密代理，而应消费显式过程信号。

## 新增 Baseline（2026 年前沿）

在原有 baseline 基础上，增加以下对比：

1. **CW-GRPO**（ACL 2026）：LLM judge 贡献加权，有 judge 上界
2. **CalibAdv**（arXiv 2604.18235）：启发式 advantage 校准
3. **GiGPO**（NeurIPS 2025）：双层 GRPO，无路由
4. **StepPO**（arXiv 2604.18401）：步骤对齐优化

## 设计变体

ENTROPIA v2 的三个设计维度各有实验变体：

1. **Reliability 聚合**：R1（加法）/ R2（乘法）/ R3（Softmax 竞争）
2. **Credit Granularity**：C1（单层 GRPO）/ C2（双层 GiGPO + NUR 门控）
3. **Utility 形式化**：势函数差分（与 TIPS 理论框架对齐）

最终推荐配置由实验结果决定。

## 新增文献吸收

- **GiGPO**：双层 Credit Granularity（C2 变体）
- **CoREN**：Reliability Gate 跨信号一致性 $\kappa$
- **TIPS**：Utility 势函数形式化理论锚点
- **CW-GRPO / CalibAdv**：新 baseline 对比
