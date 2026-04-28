# 相关论文

以下是与 ENTROPIA 相关的论文，按主题分组。

## 核心参考（已读，代码已实现）

| 论文 | 核心贡献 | 我们的借鉴 |
|------|----------|------------|
| **01_WorkForceAgent-R1** | 发现密集奖励导致 Reward Hacking | 确认问题存在，确定实验基线 |
| **02_IGPO** | 信息增益作为过程奖励 | IGPO 风格的 r_t^dense 实现 |
| **03_AutoTool** | 熵约束加入 policy loss | 熵估计方法、key token 识别 |
| **04_SELAUR** | 不确定性感知奖励 | 多种不确定性估计方法 |
| **05_TIPS** | 基于势能的奖励塑形 | TIPS 风格的 r_t^dense 变体 |
| **06_GiGPO** | 基于梯度的内隐奖励 | 另一种 credit assignment 思路 |
| **07_HERO** | 层次化强化学习 | 多层 Agent 架构参考 |
| **08_TRACE** | 安全约束 RL | reward hacking 检测信号 |
| **09_ReTool** | 工具学习 RL | sparse reward 基线 |
| **10_Environment_Tuning** | 环境自动设计 | 未来合成环境方向 |

---

## 论文详细笔记

### 01_WorkForceAgent-R1
**关键发现**：密集奖励在 ~50 step 后崩溃，reward → 0.6，response length → 0。

**我们的应对**：自适应密度控制，在模型不确定时提供密集信号，在模型自信时关闭。

---

### 02_IGPO (ICLR 2026)
**核心方法**：
```
IG_t = log p(answer|s_t) - log p(answer|s_{t-1})
```

**局限性**：每轮密度固定，未考虑训练进度。

**我们的改进**：密度随熵动态调整，更适应训练过程。

---

### 03_AutoTool (ICLR 2026)
**核心方法**：熵约束加入 GRPO loss，而不是奖励函数。

```python
L = L_GRPO + λ · H_entropy
```

**我们的区别**：我们用熵**门控奖励密度**，不是约束 loss。熵在我们的设计中是**主动控制器**。

---

### 05_TIPS
**核心方法**：势能函数塑形奖励。
```
r_t = Φ(s_t) - γ · Φ(s_{t-1})
```

**我们的借鉴**：可以复用 TIPS 的势能函数作为 r_t^dense 的另一种形式。

---

### 08_TRACE
**核心方法**：安全约束 + judge reliability。提出 reward-success divergence 检测。

**我们的借鉴**：`hacking_detector.py` 中的 divergence 检测逻辑来自 TRACE。

---

## 待读论文（建议跟进）

- **Reinforcement Learning for LLMs**: 综合survey
- **ToolBench**: 工具学习数据构建
- **WebArena**: 真实网页环境
- **ALFWorld**: 多步骤推理环境
- **Agent-R1**: DeepSeek 的 Agent RL 方案

---

## 训练数据来源

| 数据集 | 大小 | 说明 |
|--------|------|------|
| `Simia-Agent/Simia-Tau-SFT-90k-Hermes` | 91,204 | τ²-bench 格式，**推荐** |
| `inclusionAI/AReaL-tau2-data` | 33,531 + 1,982 | SFT + RL 轨迹 |
| `fuvty/tau-bench-synthetic` | 6,014 | τ²-bench 合成数据 |
| `Salesforce/APIGen-MT-5k` | 5,000 | 多轮轨迹，3-stage 验证 |

**注意**：τ-Bench 是**测试基准**，不内置训练数据。需要上述外部数据集。