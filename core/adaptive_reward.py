"""
自适应奖励密度 —— 方向 A 的核心算法。

r_t^adaptive = r_t^sparse + α · σ(H_t - H_threshold) · r_t^dense

其中：
  H_t          = 第 t 步的 token 级别熵（仅计算关键决策 token）
  H_threshold  = EMA 跟踪的平均熵（课程学习式自适应）
  σ            = sigmoid 门控函数（平滑、连续）
  α            = 密度系数（控制最大稠密奖励注入量）
  r_t^sparse   = 折扣结果奖励
  r_t^dense    = 过程奖励（IGPO 风格信息增益，或可插拔）
"""

import torch
import torch.nn.functional as F
from typing import Optional, Callable
from dataclasses import dataclass


@dataclass
class AdaptiveRewardState:
    """自适应奖励计算的运行状态。"""

    H_threshold: float  # 当前 EMA 阈值
    batch_entropies: list = None  # 用于日志记录
    gate_history: list = None  # 用于分析
    hacking_events: int = 0  # 作弊检测计数

    def __post_init__(self):
        self.batch_entropies = []
        self.gate_history = []


class AdaptiveRewardDensity:
    """
    核心算法：熵驱动的动态奖励密度控制。

    相较于先前工作的关键创新：
    1. 熵用作预防性门控，而非反应性正则化（对比 AutoTool）
    2. 密度动态调度，而非固定（对比 IGPO）
    3. 通过 EMA 阈值实现课程学习式的训练进度自适应
    4. 可插拔的 r_t^dense —— 兼容任何过程奖励方法
    """

    def __init__(
        self,
        alpha: float = 1.0,
        H_threshold_init: float = 0.5,
        beta: float = 0.02,
        sigmoid_temp: float = 1.0,
        gate_min: float = 0.0,
        eps: float = 1e-8,
    ):
        """
        参数：
            alpha: 密度控制系数。越高 = 注入越多稠密奖励。
            H_threshold_init: 初始熵阈值。训练期间通过 EMA 更新。
            beta: 阈值更新的 EMA 衰减率（推荐 0.01-0.05）。
            sigmoid_temp: sigmoid 门控的温度。temp=1 为标准 sigmoid。
            gate_min: 门控最小值（0 = 可完全关闭，>0 = 始终有稠密奖励）。
            eps: 数值稳定性常数。
        """
        self.alpha = alpha
        self.beta = beta
        self.sigmoid_temp = sigmoid_temp
        self.gate_min = gate_min
        self.eps = eps
        self.state = AdaptiveRewardState(H_threshold=H_threshold_init)

    def compute_gate(self, H_t: torch.Tensor) -> torch.Tensor:
        """
        计算单步的密度门控 g_t = σ(H_t - H_threshold)。

        高 H_t（不确定）→ 门控 ≈ 1（注入稠密奖励）
        低 H_t（确定）→ 门控 ≈ 0（抑制稠密奖励）

        参数：
            H_t: 第 t 步的熵，标量或 [batch]
        返回：
            g_t: 范围在 [gate_min, 1] 的门控值
        """
        gate = torch.sigmoid(self.sigmoid_temp * (H_t - self.state.H_threshold))
        gate = torch.clamp(gate, min=self.gate_min, max=1.0)
        return gate

    def compute_adaptive_reward(
        self,
        r_sparse: torch.Tensor,  # [num_steps] 折扣结果奖励
        r_dense: torch.Tensor,  # [num_steps] 过程奖励（IG、TIPS 等）
        H_t: torch.Tensor,  # [num_steps] 每步熵
    ) -> torch.Tensor:
        """
        计算完整轨迹的自适应奖励。

        r_t = r_t^sparse + α · σ(H_t - H_threshold) · r_t^dense

        参数：
            r_sparse: 每步的折扣结果奖励
            r_dense: 每步的过程奖励（来自任何功劳分配方法）
            H_t: 每步的熵
        返回：
            r_adaptive: [num_steps] 组合自适应奖励
        """
        gate = self.compute_gate(H_t)  # [num_steps]
        self.state.gate_history.append(gate.mean().item())
        r_adaptive = r_sparse + self.alpha * gate * r_dense
        return r_adaptive

    def update_threshold(self, batch_mean_entropy: float):
        """
        EMA 更新：H_threshold ← β · mean(H_batch) + (1-β) · H_threshold

        这实现了课程学习式自适应：
        - 训练早期：高熵 → 低阈值 → 稠密奖励 → 快速学习
        - 训练后期：低熵 → 高阈值 → 稀疏奖励 → 防止作弊
        """
        self.state.H_threshold = (
            self.beta * batch_mean_entropy + (1 - self.beta) * self.state.H_threshold
        )
        self.state.batch_entropies.append(batch_mean_entropy)

    def get_stats(self) -> dict:
        """获取训练统计数据用于日志记录。"""
        return {
            "H_threshold": self.state.H_threshold,
            "mean_gate": (
                sum(self.state.gate_history[-100:])
                / max(1, len(self.state.gate_history[-100:]))
            )
            if self.state.gate_history
            else 0,
            "hacking_events": self.state.hacking_events,
            "n_updates": len(self.state.batch_entropies),
        }


# ──────────────────────────────────────────────────
# 可插拔的过程奖励函数
# 这些函数用作自适应奖励公式中的 r_t^dense。
# 可以插入 IGPO、TIPS 或任何其他过程奖励。
# ──────────────────────────────────────────────────


def igpo_information_gain(
    model_prob_current: torch.Tensor,  # p(answer | s_t)，每个推理的标量
    model_prob_prev: torch.Tensor,  # p(answer | s_{t-1})，每个推理的标量
    ground_truth_answer_id: int,  # 正确答案的 token ID
) -> torch.Tensor:
    """
    IGPO 风格信息增益过程奖励。
    IG_t = log p(answer|s_t) - log p(answer|s_{t-1})
    = 模型对正确答案信念的边际增长。
    """
    ig = torch.log(model_prob_current.clamp(min=1e-8)) - torch.log(
        model_prob_prev.clamp(min=1e-8)
    )
    return ig


def tips_potential_reward(
    potential_current: torch.Tensor,  # Φ(s_t)，当前状态的势能
    potential_prev: torch.Tensor,  # Φ(s_{t-1})，前一步状态的势能
    gamma: float = 0.9,
) -> torch.Tensor:
    """
    TIPS 风格基于势能的奖励塑形。
    r_t^dense = Φ(s_t) - γ · Φ(s_{t-1})
    """
    return potential_current - gamma * potential_prev


def progress_reward(
    task_progress_current: float,  # 当前步骤的 0-1 进度
    task_progress_prev: float,  # 前一步骤的 0-1 进度
) -> float:
    """
    简单的基于进度的过程奖励。
    r_t^dense = progress_current - progress_prev
    """
    return task_progress_current - task_progress_prev


def outcome_discounted_reward(
    outcome: float,  # +1（成功）、-1（失败）或 0（部分完成）
    step_idx: int,
    total_steps: int,
    gamma: float = 0.9,
) -> float:
    """
    折扣结果奖励：r_t^sparse = outcome * γ^{total_steps - step_idx - 1}
    """
    return outcome * (gamma ** (total_steps - step_idx - 1))
