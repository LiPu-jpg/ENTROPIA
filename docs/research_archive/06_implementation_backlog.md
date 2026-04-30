# 06 实现待办

本文件将清理后的研究决策转化为具体的工程任务。

## 当前差距

当前代码在部分位置仍反映旧的 v1 公式。

已知问题：

```text
training/trainer.py currently builds adaptive reward roughly as:

r_adaptive = r_sparse + alpha * gate * r_sparse
```

论文设计要求：

```text
r_adaptive = r_outcome + routed process signals
```

因此下一步实现不是又一次熵调整，而是引入显式的过程信号和路由器。

## 目标模块

| 模块 | 所需变更 |
| --- | --- |
| `core/entropy.py` | 保留为 Need 特征提供器，而非完整路由逻辑。 |
| `core/adaptive_reward.py` | 从标量门控重构为多信号奖励路由器。 |
| `core/hacking_detector.py` | 暴露可在路由内使用的风险分数。 |
| `training/trainer.py` | 消费路由后的奖励矩阵而非稀疏代理。 |
| `configs/config.py` | 添加信号权重、稠密预算、门控开关、诊断标志。 |
| `scripts/run.py` | 添加命名基线和消融预设。 |

## 新组件

### 过程信号库

建议文件：

```text
core/process_signals.py
```

职责：

1. 计算或接收候选 `S^k_t`；
2. 在 batch/任务内归一化每个信号；
3. 暴露不可用信号的掩码；
4. 记录每信号的统计信息。

### 奖励路由器

建议文件：

```text
core/reward_router.py
```

职责：

1. 计算 `N_t`、`U^k_t`、`L^k_s`、`M^k_t` 和 `H^risk_t`；
2. 构建 `g^k_t`；
3. 应用稠密预算 `B_s`；
4. 返回最终奖励矩阵和诊断信息。

### 诊断工具

建议文件：

```text
utils/reward_diagnostics.py
```

职责：

1. 稠密预算使用量；
2. 门控激活图；
3. 信号-结果相关性；
4. 奖励-成功发散；
5. 作弊检测器比率。

## 实现阶段

### 阶段 1：修正奖励构建

目标：

```text
stop using sparse reward as a dense proxy
```

任务：

1. 添加过程信号接口。
2. 实现简单信号：格式有效性、步骤成本、重复搜索、占位 IG。
3. 重构自适应奖励构建器，将结果奖励与过程信号组合。
4. 添加单元测试或冒烟测试，验证奖励形状和门控范围。

### 阶段 2：ENTROPIA-v2 路由器

任务：

1. 基于熵/不确定性的 Need 门控。
2. 每个过程信号的 Utility 门控。
3. 基于滚动过程-结果一致性的 Reliability。
4. 基于作弊检测器的 Risk。
5. 稠密预算调度。

### 阶段 3：基线自动化

任务：

1. 添加纯结果、固定 IG、固定 CRAFT、动态 CRAFT、仅熵、随机门控、完整 ENTROPIA 的配置。
2. 添加一条命令运行基线矩阵。
3. 导出包含成功率、效率、相关性、门控和作弊指标的汇总表。

### 阶段 4：真实任务集成

任务：

1. 从 Search QA 开始，因为奖励信号最容易检查。
2. 路由稳定后迁移到 tau-bench 风格的工具任务。
3. 在奖励路径正确后才添加真实数据加载器。

## 暂不执行

1. 不要将纯 RL 搜索涌现作为主要目标。
2. 不要将学习型 PRM 作为首次结果的必需项。
3. 在奖励管线正确之前不要优化分布式训练。
4. 在分离 Need、Utility、Reliability 和 Risk 之前不要添加更多熵变体。
