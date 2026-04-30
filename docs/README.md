# ENTROPIA 文档

本目录保存面向论文的文档以及经过整理的研究归档。

## 阅读顺序

1. `paper_formula_methodology.md`
   - 整理好的论文级符号表示。
   - 最终奖励函数。
   - GRPO 集成方式。
   - 消融实验与公式的映射关系。

2. `design_v2_reward_routing.md`
   - 设计原理。
   - ENTROPIA 从熵门控到奖励路由的演进过程。
   - 如何吸收 CRAFT-Search、状态条件贡献和风险控制。

3. `research_archive/`
   - 经过规范化整理的历史研究轮次和方向选择文件归档。
   - 清理后保留决策链，去除重复草稿。

4. `related_work.md`
   - 文献图谱与基线定位。
   - 作为引用工作区使用。

## 当前定位

ENTROPIA-Search 是构建在复合奖励库之上的可靠性校准奖励路由层。

```text
CRAFT-Search: what reward signals exist?
ENTROPIA: when should each signal be trusted and routed?
```

当前活跃方法是在风险控制器下的 Need-Utility-Reliability 路由。
