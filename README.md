# ENTROPIA-Search

ENTROPIA-Search 是一个面向长时序 LLM Agent RL 的可靠性校准奖励路由框架。

该项目最初起步于基于熵门控的自适应奖励密度控制。当前面向论文的设计范围更广：

```text
CRAFT-Search: what reward signals exist?
ENTROPIA: when should each signal be trusted and routed?
```

## 核心方法

ENTROPIA 通过 Need、Utility 和 Reliability 门控，在 Risk 控制器下路由一组轻量级过程信号。

\[
g^k_t = N_t \cdot U^k_t \cdot L^k_s \cdot M^k_t \cdot (1-H^{risk}_t)
\]

\[
r^{ENT}_{g,t}
= r^{out}_{g,t}
+ B_s \sum_{k\in\mathcal{K}} w^k_s g^k_{g,t}\tilde S^k_{g,t}
+ \epsilon_{g,t}
\]

其中：

- `Need` 估计仅结果监督是否充分。
- `Utility` 估计当前步骤是否提供任务相关信息。
- `Reliability` 估计过程信号是否与结果改善对齐。
- `Risk` 在出现奖励作弊模式时抑制稠密监督。
- `B_s` 是自适应稠密监督预算。

## 当前代码状态

已实现的 v1 组件：

- `core/entropy.py` — 关键 token 熵与不确定性估计
- `core/adaptive_reward.py` — v1 基于熵门控的自适应奖励密度
- `core/hacking_detector.py` — 奖励作弊监控器
- `training/trainer.py` — 带 sparse/dense/adaptive 基线的 GRPO 训练器

已知的实现差距：

```text
training/trainer.py currently uses:
  r_adaptive = r_sparse + alpha * gate * r_sparse

The paper design requires:
  r_adaptive = r_outcome + routed process signals
```

## 文档

按以下顺序阅读：

1. `docs/paper_formula_methodology.md`
   - 最终符号表示、奖励函数、GRPO 集成与消融映射。

2. `docs/design_v2_reward_routing.md`
   - 设计原理及项目如何从熵门控演进到当前方案。

3. `docs/related_work.md`
   - 文献定位与基线地图。

4. `docs/research_archive/`
   - 已清理的旧研究轮次与方向选择材料归档。

5. `docs/README.md`
   - 文档索引。

## 仓库结构

```text
ENTROPIA/
├── configs/          # training configs
├── core/             # entropy, reward, hacking/risk modules
├── data/             # tau-style data utilities
├── docs/             # paper-facing docs
├── envs/             # mock environment
├── scripts/          # run/download helpers
├── training/         # GRPO trainer
└── utils/
```

## 下一步工程计划

1. 添加 `core/process_signals.py`。
2. 添加 `core/reward_router.py`。
3. 重构 `training/trainer.py`，从显式过程信号构建路由奖励。
4. 添加诊断指标：
   - 稠密预算曲线
   - 门控精度
   - 过程-结果相关性
   - 奖励-成功率散度
   - 格式博弈率
   - 非零优势比例
