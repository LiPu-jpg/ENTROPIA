# ENTROPIA — Entropy-Gated Adaptive Reward Density for LLM Agent RL

## 项目概述

**ENTROPIA** 是一个 LLM Agent 强化学习研究框架，核心算法基于自研的"熵门控自适应奖励密度控制"机制。

```
r_t^adaptive = r_t^sparse + α · σ(H_t - H_threshold) · r_t^dense
```

### 核心问题

在 Agent RL 中，密集奖励（dense reward）能加速收敛，但会导致 **Reward Hacking**：
- 模型学会"刷奖励"而不是真正完成任务
- 响应长度趋向零、重复模式、死循环

稀疏奖励（sparse reward）稳定但不提供过程信号，收敛慢。

### 核心思路

用 **token 级熵** 作为"不确定性信号"来动态控制奖励密度：
- 高熵（不确定）→ 门开 → 注入密集奖励 → 引导探索
- 低熵（自信）→ 门关 → 抑制密集奖励 → 防止 Hacking

### 训练环境与硬件

- **验证环境**：MacBook M5（逻辑验证，small model = gpt2）
- **训练环境**：4×NVIDIA L20 GPU 服务器（可 2+2 分配或 4 卡一起）
- **基模**：Qwen2.5-7B-Instruct + LoRA (r=64, α=128)
- **训练框架**：GRPO（Group Relative Policy Optimization）

---

## 技术架构

```
src/
├── core/
│   ├── entropy.py          # Token 级熵估计（仅在关键 token 上计算）
│   ├── adaptive_reward.py  # 熵门控自适应密度函数
│   └── hacking_detector.py # 三信号 Reward Hacking 监控器
├── configs/
│   └── config.py           # 完整训练配置 + 基线预设 + 消融矩阵
├── training/
│   └── trainer.py         # GRPO trainer，支持 5 种奖励模式
├── data/
│   └── tau_dataset.py     # τ-Bench 格式合成数据（25 个任务）
├── envs/
│   └── mock_env.py        # Mock τ-Bench 环境（快速训练）
└── scripts/
    └── run.py             # 主入口，支持 CLI
```

### 奖励模式

| 模式 | 说明 |
|------|------|
| `adaptive` | 完整熵门控自适应密度（Direction A） |
| `sparse` | ReTool 风格二进制结果奖励 |
| `dense_igpo` | IGPO 固定信息增益过程奖励 |
| `dense_fixed` | WorkForceAgent-R1 固定密集奖励（消融） |
| `autotool_entropy` | AutoTool 熵约束（在 loss 中，不在奖励中） |

---

## 快速开始

```bash
# 完整自适应奖励密度
python scripts/run.py --mode adaptive

# 基线对比
python scripts/run.py --mode sparse
python scripts/run.py --mode dense_igpo
python scripts/run.py --mode dense_fixed
python scripts/run.py --mode autotool_entropy

# 消融实验
python scripts/run.py --mode adaptive --ablation threshold   # 固定 vs EMA 阈值
python scripts/run.py --mode adaptive --ablation granularity   # Step vs Traj 熵
python scripts/run.py --mode adaptive --ablation random_gate # 随机门控

# 使用自定义模型
python scripts/run.py --mode adaptive --model Qwen/Qwen2.5-7B-Instruct
```

---

## 核心算法细节

### 熵门控机制

```python
gate_t = sigmoid(temp * (H_t - H_threshold))
```

- `H_t`：关键 token 上的 token 级熵（工具名、参数、停止符）
- `H_threshold`：EMA 平滑更新的阈值（课程式适应）
- `temp`：sigmoid 温度参数（越高门控切换越陡峭）

### EMA 阈值更新

```python
H_threshold ← β · mean(H_batch) + (1-β) · H_threshold
```

创造课程式适应：
- 训练早期：高熵 → 低阈值 → 门开 → 密集奖励 → 快收敛
- 训练后期：低熵 → 高阈值 → 门关 → 稀疏奖励 → 防 Hacking

### GRPO Loss

```python
ratio = π_θ / π_ref
clipped = clip(ratio, 1-ε, 1+ε)
loss = -min(ratio · advantage, clipped · advantage)
```

---

## 已知问题与局限

### Critical

1. **Loss = -0.0**：MacBook 测试时使用 gpt2（base model，无 instruction tuning），生成乱码 → logprobs 极端 → loss 崩溃。在 L20 上使用 Qwen2.5-7B-Instruct 可解决。

2. **Mock 环境**：当前使用预脚本化的观察结果，非真实 τ-Bench 环境的 LLM 用户模拟器。真实环境需接入 `sierra-research/tau2-bench`。

3. **`train()` 调用未初始化 optimizer**：`setup_model()` 不初始化优化器，外部调用前需手动 `trainer.optimizer = torch.optim.AdamW(...)`。

### 诚实评估

Direction A 的创新是"奖励密度调度"，而非根本性的奖励设计创新。实际收益主要是**收敛速度提升**，不是能力边界的突破。

---

## 后续研究方向

详见 `docs/extensions.md`

---

## 相关论文

详见 `docs/papers.md`

---

## 环境要求

```txt
torch>=2.0
transformers>=4.40
peft>=0.10
wandb
```

GPU：8×A100 推荐（完整训练），7B 模型可用 4×A100 + LoRA。