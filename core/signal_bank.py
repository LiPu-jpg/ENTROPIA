"""
过程信号库 — ENTROPIA v2 的候选过程奖励信号。

对每个 step t 计算:
  S^ig_t:   信息增益（answer likelihood 变化）
  S^rel_t:  检索/证据相关性
  S^eff_t:  效率成本（低收益步骤惩罚）
  M^valid_t: 格式/工具合法性 mask

所有信号进入 RewardRouter 前归一化。
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque


@dataclass
class SignalBatch:
    """一批步骤的过程信号。"""

    info_gain: List[float] = field(default_factory=list)  # [n_steps]
    relevance: List[float] = field(default_factory=list)
    efficiency_cost: List[float] = field(default_factory=list)
    format_valid: List[float] = field(default_factory=list)  # 0/1 mask
    step_entropies: List[float] = field(default_factory=list)
    outcomes: List[float] = field(default_factory=list)  # rollout outcome


class SignalBank:
    """
    维护和计算过程信号。

    当前轻量版（不用 PRM）:
    - info_gain: answer likelihood delta（需要 gold answer 或 judge）
    - efficiency: 步骤计数惩罚
    - format_valid: 工具调用格式检查
    - relevance: 占位（需真实搜索环境）

    所有信号可以逐步添加，Router 会自适应。
    """

    def __init__(
        self,
        window_size: int = 100,
        max_turns: int = 10,
        enable_relevance: bool = False,
    ):
        self.window_size = window_size
        self.max_turns = max_turns
        self.enable_relevance = enable_relevance

        # 滑动窗口统计（用于归一化）
        self._running_stats: Dict[str, dict] = {
            "info_gain": {"mu": 0.0, "sigma": 1.0, "n": 0},
            "efficiency": {"mu": 0.0, "sigma": 1.0, "n": 0},
        }
        self._history: Dict[str, deque] = {
            k: deque(maxlen=window_size) for k in self._running_stats
        }

    def compute_signals(
        self,
        rollout_logprobs: Optional[List[torch.Tensor]] = None,
        outcomes: Optional[List[float]] = None,
        step_count: int = 0,
        format_issues: int = 0,
        task_type: str = "default",
    ) -> SignalBatch:
        signals = SignalBatch()

        # 信息增益（简化实现：用 logprob 变化近似，带 gold answer 的版本需要别处注入）
        if rollout_logprobs:
            for i in range(len(rollout_logprobs)):
                if i > 0 and rollout_logprobs[i].numel() > 0 and rollout_logprobs[i - 1].numel() > 0:
                    ig = max(0.0, rollout_logprobs[i].mean().item() - rollout_logprobs[i - 1].mean().item())
                else:
                    ig = 0.0
                signals.info_gain.append(ig)

        # 效率成本：如果 step 超过任务先验深度且无信息增益
        prior_depth = max(2, self.max_turns // 3)
        for t in range(max(step_count, len(signals.info_gain))):
            if t < len(signals.info_gain):
                has_new_info = signals.info_gain[t] > 0.01
            else:
                has_new_info = False
            if t > prior_depth and not has_new_info:
                signals.efficiency_cost.append(1.0)
            else:
                signals.efficiency_cost.append(0.0)

        # 格式合法性
        n_steps = max(step_count, len(signals.info_gain))
        for t in range(n_steps):
            signals.format_valid.append(1.0 if format_issues == 0 else 0.0)

        # 更新运行统计
        for key in ["info_gain", "efficiency_cost"]:
            vals = getattr(signals, key, [])
            if not vals:
                continue
            if key not in self._history:
                self._history[key] = deque(maxlen=self.window_size)
            for v in vals:
                self._history[key].append(v)
            if len(self._history[key]) >= 2:
                vs = list(self._history[key])
                if key not in self._running_stats:
                    self._running_stats[key] = {"mu": 0.0, "sigma": 1.0}
                self._running_stats[key]["mu"] = sum(vs) / len(vs)
                self._running_stats[key]["sigma"] = max(
                    (sum((x - self._running_stats[key]["mu"]) ** 2 for x in vs) / len(vs)) ** 0.5,
                    1e-4,
                )

        return signals

    def normalize(self, signals: SignalBatch, signal_name: str) -> List[float]:
        """将信号归一化到零均值单位方差。"""
        raw = getattr(signals, signal_name)
        stats = self._running_stats.get(signal_name, {"mu": 0.0, "sigma": 1.0})
        if not raw:
            return []
        mu, sigma = stats["mu"], max(stats["sigma"], 1e-4)
        return [(v - mu) / sigma for v in raw]

    def get_stats(self) -> dict:
        return {k: dict(v) for k, v in self._running_stats.items()}


def compute_ig_from_answer_logprob(
    model_logp_current: float,
    model_logp_previous: float,
) -> float:
    """计算信息增益：log p(answer|s_t) - log p(answer|s_{t-1})。"""
    return max(0.0, model_logp_current - model_logp_previous)


def compute_efficiency_penalty(
    step_idx: int,
    max_steps: int,
    has_new_info: bool,
) -> float:
    """低收益步骤的效率惩罚。"""
    prior_depth = max(2, max_steps // 3)
    if step_idx > prior_depth and not has_new_info:
        return 1.0
    return 0.0
