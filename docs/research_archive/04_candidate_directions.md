# 04 候选方向

本文件以归一化格式保留了旧的扩展和候选工作区结论。

## 保留在主线中

### 1. 过程信号库

状态：主线组件。

保留内容：

- 信息增益；
- 证据覆盖；
- 格式/工具调用有效性；
- 子目标进展；
- 效率/成本；
- 作弊或发散特征。

原因：

信号库提供了足够的方法广度来支撑一篇完整论文，而 ENTROPIA 提供了有原则的路由机制。

### 2. 可靠性与风险控制

状态：主线组件。

保留内容：

- 过程-结果一致性；
- 奖励-成功相关性；
- 稠密奖励崩溃检测；
- 重复/引用/格式操纵检测器。

原因：

这是对审稿人关于稠密奖励是任意代理工程的质疑的最强辩护。

## 保留为基线或消融

### 3. 仅熵门控

状态：v1 基线。

原因：

可从当前代码恢复，为仅 Need 路由提供干净的消融实验。

### 4. 固定稠密信息增益

状态：基线。

原因：

测试改进是否仅仅来自 IGPO 风格的过程奖励。

### 5. 固定/动态 CRAFT

状态：基线。

原因：

测试复合奖励库在没有状态条件可靠性路由的情况下是否足够。

## 延后至扩展

### 6. 状态条件贡献

状态：信用路由扩展。

可能实验：

```text
Remove or perturb step t, rerun downstream reasoning,
and compare outcome delta against IG, static judge score,
and state-conditioned contribution estimate.
```

延后原因：

有价值但不是证明主要奖励路由声明所必需的。

### 7. TSTV / 自适应终止

状态：测试时迁移扩展。

可能声明：

```text
The same Need/Utility/Risk features used for train-time reward routing
can control inference-time search termination.
```

延后原因：

引入了一个独立的测试时预算控制论文轴。

### 8. 学习型 PRM / TriPRM / Search PRM

状态：可选上界。

延后原因：

可能需要额外标注或验证器训练，增加计算和数据风险。

## 作为主要贡献拒绝

### 9. 纯 RL 搜索涌现

状态：仅初始化消融。

推荐对比：

```text
Instruct+RL vs SFT+RL vs Base+RL
```

拒绝原因：

搜索行为的纯涌现是一个独立的大问题，方差高且失败风险高。

### 10. 数据集或基础设施作为贡献

状态：仅工程支持。

示例：

- tau-bench 集成；
- SIMIA/AReaL/fuvty 数据加载器；
- 多 GPU 训练；
- 基线自动化。

拒绝原因：

对执行重要，但不是论文的知识贡献。

## 旧扩展笔记保留

| 旧扩展 | v2 中的新位置 |
| --- | --- |
| 一致性门控 | Utility + Reliability 门控 |
| IG 奖励替换 | 过程信号库 |
| 多维奖励 | CRAFT 风格信号库 |
| 真实 tau-bench 集成 | 实验基础设施 |
| 真实训练数据 | 实现待办 |
| SFT 预热 | 训练协议 |
| 分布式训练 | 工程待办 |
| 基线自动化 | 实验规范 |
