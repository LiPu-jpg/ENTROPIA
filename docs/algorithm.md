# 核心算法详解

## 完整公式

$$r_t^{adaptive} = r_t^{sparse} + \alpha \cdot \sigma(H_t - H_{threshold}) \cdot r_t^{dense}$$

## 各组件定义

### H_t — Token 级熵

$$H(p) = -\sum_{v \in V} p(v) \log p(v)$$

**关键设计**：只在关键 token 上计算，不在 filler token（"the", "a"）上计算。

关键 token 类型：
- `tool_name`：工具调用名称（search_airport, cancel_order...）
- `tool_params`：工具参数 JSON
- `stop_token`：终止符（<|im_end|>, </action>）
- `action_type`：动作类型（click, fill, select）

**为什么**：工具名、参数、停止符反映决策质量，"the" 不反映。

---

### H_threshold — EMA 平滑阈值

$$H_{threshold} \leftarrow \beta \cdot \bar{H}_{batch} + (1-\beta) \cdot H_{threshold}$$

- β = 0.02（EMA 衰减率）
- 每 batch 更新一次

**课程式适应效果**：
- 训练早期：模型不确定 → 高 H_t → 低 H_threshold → **门开** → 密集奖励
- 训练后期：模型自信 → 低 H_t → 高 H_threshold → **门关** → 稀疏奖励

---

### σ — Sigmoid 门控

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

门控公式：$$g_t = \sigma(temp \cdot (H_t - H_{threshold}))$$

- temp = 10.0（温度参数，可调）
- gate_min = 0.0（最小门值，防止完全关闭）

**为什么用 sigmoid**：
- 平滑连续，梯度流动好
- 不是硬阈值，不会震荡
- 介于 0-1 之间，可解释

---

### r_t^sparse — 稀疏奖励

$$r_t^{sparse} = R_{outcome} \cdot \gamma^{T-1-t}$$

- outcome：任务成功 = 1.0，失败 = 0.0
- γ = 0.9（折扣因子）
- 仅在终止时非零（类似 ReTool）

---

### r_t^dense — 密集奖励（可插拔）

默认为 0（纯稀疏模式）。

可替换为：
- **IGPO**：`log p(answer|s_t) - log p(answer|s_{t-1})`
- **TIPS**：`Φ(s_t) - γ·Φ(s_{t-1})`
- **Progress**：`progress_t - progress_{t-1}`

---

## 完整训练流程

```
1. rollout(query, task):
   - 交互 max_turns 步
   - 记录每步的 token_ids, entropy H_t, outcome

2. build_reward_matrix(outcomes, entropies):
   - 计算折扣稀疏奖励
   - 对 adaptive 模式：r = r_sparse + α · gate · r_sparse

3. _compute_grad_and_ref_logprobs(token_ids):
   - 重新跑 model forward 计算 logprobs（含梯度）
   - 跑 ref_model forward 计算 logprobs（无梯度）
   - 保证 grad 和 ref 用同一 forward pass

4. compute_grpo_loss(grad_lp, ref_lp, rewards):
   - 计算 advantage（折扣 reward 标准化）
   - GRPO ratio + PPO clip
   - backward + clip grad norm

5. update_threshold(mean_entropy):
   - EMA 更新 H_threshold
```

---

## GRPO 算法

### 目标函数

$$\mathcal{L}^{GRPO}(\theta) = -\mathbb{E}_{q \sim P, g \sim \pi_\theta(\cdot|q)} \left[ \min \frac{\pi_\theta(g|q)}{\pi_{ref}(g|q)} \hat{A}, \quad \text{clip}(\frac{\pi_\theta(g|q)}{\pi_{ref}(g|q)}, 1-\epsilon, 1+\epsilon) \hat{A} \right]$$

### Advantage 估计

$$\hat{A}_t = \frac{R_t - \mu_R}{\sigma_R}$$

- 在每个 query 内跨 rollout 标准化
- 使用折扣因子 γ

### PPO Clip

$$\text{clip}(r, 1-\epsilon, 1+\epsilon)$$

- 防止策略更新过大
- ε = 0.2

---

## 与其他方法的关键区别

| | AutoTool | IGPO | SELAUR | **ENTROPIA** |
|---|---|---|---|---|
| 熵的用法 | Loss 正则项 | 无 | Shape 奖励值 | **门控密度** |
| 密度 | 固定 | 固定 | 固定 | **动态** |
| 课程适应 | 无 | 无 | 无 | **EMA 阈值** |
| 安全网 | 无 | 无 | TRACE judge | **Hacking 检测** |

**核心差异**：ENTROPIA 用熵作为"门"（开关），其他方法用熵作为"正则项"（惩罚项）或"塑形"（加减）。