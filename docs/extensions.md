# 后续拓展方向

基于对 8 个研究专题的分析，识别出以下改进方向，按优先级排序。

---

## 方向 1：一致性门控过滤（Consistency Gating）[高优先级]

**来源**：专题 6（CoREN）+ 专题 8（多维 reward）

**核心思想**：
当前的门控只用熵一个信号。但熵不可靠的情况：
- 模型**自信地错了**：低熵但 action 错误
- 模型**不确定地对了**：高熵但 action 正确

**改进**：
只有当 IG reward、格式 reward、和 H_t **三个信号一致**时才开门。

```
gate_open = (IG_reward > 0) AND (format_reward > 0) AND (H_t > threshold)
```

**解决的问题**：
- 熵不可靠时不会被误导
- 只在真正需要引导时才开

**工作量**：小（主要修改 `adaptive_reward.py` 中的 `compute_gate()`）

---

## 方向 2：IG Reward 替代 Token 增量 [高优先级]

**来源**：专题 2（IGPO）

**当前问题**：
`adaptive_reward.py` 中的 `_compute_fixed_dense` 使用 token 长度增量作为 r_t^dense。这不够精确。

**改进**：
用真正的信息增益替代 token 长度增量：
```
r_t^dense = log p(answer|s_t) - log p(answer|s_{t-1})
```

**挑战**：
需要 ground truth answer 或 LLM judge 来估计信息增益。

**工作量**：中

---

## 方向 3：多维 Reward 分解 [长期方向]

**来源**：专题 8（三分量框架）

**核心公式**：
$$R = \lambda_1 \cdot IG - \lambda_2 \cdot efficiency - \lambda_3 \cdot safety$$

- `IG`：信息增益（正确性）
- `efficiency`：效率惩罚（token 数量、step 数量）
- `safety`：安全约束惩罚（Hacking 检测触发）

**工作量**：大，需要重新设计 reward 体系

---

## 方向 4：真实 τ-Bench 环境集成 [高优先级]

**来源**：当前 mock 环境太弱

**改进**：
```python
# 替换 mock_env 为真实环境
from tau_bench import TauBenchEnv

env = TauBenchEnv(
    data_path="sierra-research/tau2-bench",
    agent_model="Qwen2.5-7B-Instruct"
)
```

**真实环境优势**：
- LLM 用户模拟器（非脚本化）
- 真实的工具调用执行
- 更接近实际的客服场景

**工作量**：中

---

## 方向 5：真实训练数据 [高优先级]

**来源**：当前使用 25 个合成任务，太小

**数据源**（已找到）：
1. `Simia-Agent/Simia-Tau-SFT-90k-Hermes` — 91,204 条，τ²-bench 格式
2. `inclusionAI/AReaL-tau2-data` — 33,531 SFT + 1,982 RL 轨迹
3. `fuvty/tau-bench-synthetic` — 6,014 条

**工作量**：中（需要数据加载器适配）

---

## 方向 6：SFT Warmup 完善 [中优先级]

**来源**：当前 SFT warmup 存在但未充分测试

**改进**：
1. 适配真实数据集格式（SIMIA 格式）
2. 添加验证集评估
3. 保存 warmup checkpoint

**工作量**：中

---

## 方向 7：多 GPU 分布式训练 [中优先级]

**来源**：4×L20 服务器可用

**当前问题**：
训练器未实现分布式，single GPU。

**改进**：
```python
# 使用 Accelerate 或 DeepSpeed
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer = accelerator.prepare(model, optimizer)
```

**工作量**：大

---

## 方向 8：Baseline 自动化对比实验 [中优先级]

**来源**：当前需要手动跑 5 个模式

**改进**：
```bash
# 一键跑所有基线
python scripts/run_all_baselines.py --model Qwen2.5-7B-Instruct --output results/
```

自动生成对比表格和图表。

**工作量**：小