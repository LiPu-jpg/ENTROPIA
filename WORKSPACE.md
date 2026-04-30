# ENTROPIA-Search 工作区

## 活跃论点

ENTROPIA-Search 将复合奖励设计转化为自适应、可靠、状态条件化的监督。

当前活跃方法为：

```text
Need-Utility-Reliability reward routing under a Risk controller.
```

## 保留材料

- `docs/paper_formula_methodology.md` — 规范公式与方法结构
- `docs/design_v2_reward_routing.md` — 设计原理与研究故事线
- `docs/research_archive/` — 已清理的方向选择与旧研究轮次结论归档
- `docs/related_work.md` — 文献定位
- `core/`、`training/`、`configs/` — 当前实现基础

## 已清理材料

旧的探索工作区已完成整合：

- 逐轮调查输出已合并至 `docs/research_archive/`
- 侧项目工作区已总结为候选方向
- Direction-A 的重复实现镜像
- 缓存和 pyc 文件
- 已被当前公式/方法文档取代的过时 v1 文档

原始论文文件已移至 `../references/papers/`。

## 近期工程目标

将代码库对齐至论文设计 v2 + 新文献吸收：

1. 构建显式过程信号（Process Signal Bank）。
2. 实现 Reliability Gate 三种变体（R1 加法 / R2 乘法 / R3 Softmax 竞争）。
3. 实现 Credit Granularity 两种变体（C1 单层 / C2 双层 GiGPO+NUR 门控）。
4. Utility Gate 势函数形式化。
5. Reliability Gate 新增跨信号一致性 $\kappa$ 维度。
6. 通过 Need、Utility、Reliability、Validity 和 Risk 进行路由。
7. 从路由后的过程信号更新 GRPO 奖励。
8. 记录路由诊断信息以供论文分析。
9. 新增 Baseline 对比：CW-GRPO、CalibAdv、GiGPO、StepPO。

### 实验矩阵

- 主干实验：R(3) × C(2) = 6 组变体对比
- 消融实验：9 组（含 NUR 各因子、$\kappa$、Budget、C2 门控）
- 外部 baseline：12 组（含新 baseline）
- 总计约 20 组，HotpotQA + Qwen2.5-7B GRPO
