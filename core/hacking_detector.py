"""
奖励作弊检测模块。

在训练期间监控三种作弊信号：
1. 异常短响应（步数 < 正常步数的 1/3）
2. 重复模式（连续相同的 token 序列）
3. 奖励-成功率背离（过程奖励 ↑ 但任务成功率 ↓）

如果检测到作弊，安全网将强制回退到纯稀疏奖励。
"""

import torch
from typing import List, Tuple, Optional
from collections import deque


class HackingDetector:
    """
    实时奖励作弊监控器。

    作为安全网运行：如果自适应门控工作正常，
    本检测器应该很少触发。其存在本身就是对
    门控机制有效性的验证。
    """

    def __init__(
        self,
        short_response_ratio: float = 0.33,
        repeat_window: int = 3,
        divergence_window: int = 10,
        action: str = "fallback_to_sparse",
    ):
        """
        参数：
            short_response_ratio: 步数低于此比例 * 正常基线即触发警报。
            repeat_window: 连续重复次数超过此值则标记为模式。
            divergence_window: 奖励-成功率背离检测的窗口大小。
            action: "fallback_to_sparse" | "skip_batch" | "log_only"
        """
        self.short_ratio = short_response_ratio
        self.repeat_window = repeat_window
        self.divergence_window = divergence_window
        self.action = action

        # 运行时统计量
        self.normal_step_count = None  # 正常步数的 EMA
        self.recent_process_rewards = deque(maxlen=divergence_window)
        self.recent_success_rates = deque(maxlen=divergence_window)
        self.event_count = 0

    def detect_short_response(
        self,
        step_count: int,
        ema_beta: float = 0.1,
    ) -> Tuple[bool, str]:
        """
        检查响应是否异常短。

        参数：
            step_count: 本次推理的步数
            ema_beta: 正常基线的更新率
        返回：
            (是否作弊, 原因)
        """
        if self.normal_step_count is None:
            self.normal_step_count = step_count
            return False, ""

        # 更新正常基线
        self.normal_step_count = (
            ema_beta * step_count + (1 - ema_beta) * self.normal_step_count
        )

        if step_count < self.short_ratio * self.normal_step_count:
            self.event_count += 1
            return (
                True,
                f"Short response: {step_count} steps vs baseline {self.normal_step_count:.1f}",
            )

        return False, ""

    def detect_repetition(
        self,
        token_ids: List[List[int]],  # [num_steps, seq_len]
    ) -> Tuple[bool, str]:
        """
        检查跨步的重复 token 序列。

        参数：
            token_ids: 推理过程中每步的 token ID
        返回：
            (是否作弊, 原因)
        """
        if len(token_ids) < self.repeat_window:
            return False, ""

        # 比较连续步的 token 序列
        for i in range(len(token_ids) - self.repeat_window + 1):
            window = token_ids[i : i + self.repeat_window]
            # 检查窗口内所有序列是否相同（或近似相同）
            first = window[0]
            all_same = all(
                len(first) == len(w) and all(a == b for a, b in zip(first, w))
                for w in window[1:]
            )
            if all_same:
                # 同时检查非平凡性（不是全部填充 token）
                if len(set(first)) > 1:  # 序列内不全是同一个 token
                    self.event_count += 1
                    return (
                        True,
                        f"Repetition: {self.repeat_window} consecutive identical outputs at step {i}",
                    )

        return False, ""

    def detect_divergence(
        self,
        process_reward: float,
        success: bool,
    ) -> Tuple[bool, str]:
        """
        检查过程奖励上升但任务成功率下降的情况。
        典型的奖励作弊信号：模型在钻过程奖励的空子，
        而非真正提升任务表现。

        参数：
            process_reward: 该批次的平均过程奖励
            success: 该批次成功率是否高于中位数
        返回：
            (是否作弊, 原因)
        """
        self.recent_process_rewards.append(process_reward)
        self.recent_success_rates.append(float(success))

        if len(self.recent_process_rewards) < self.divergence_window:
            return False, ""

        # 检查趋势：过程奖励上升，成功率下降
        pr_list = list(self.recent_process_rewards)
        sr_list = list(self.recent_success_rates)

        pr_trend = pr_list[-1] - pr_list[0]  # 正值 = 上升
        sr_trend = sr_list[-1] - sr_list[0]  # 负值 = 下降

        if pr_trend > 0.05 and sr_trend < -0.05:  # 检测到背离
            self.event_count += 1
            return True, (
                f"Divergence: process_reward Δ={pr_trend:.3f} (↑), "
                f"success_rate Δ={sr_trend:.3f} (↓)"
            )

        return False, ""

    def check(
        self,
        rollout_steps: int,
        rollout_tokens: List[List[int]],
        batch_process_reward: float,
        batch_success: bool,
    ) -> Tuple[bool, List[str]]:
        """
        运行所有作弊检测。返回 (是否存在作弊, 原因列表)。

        如果任何检测触发且 action != "log_only"，调用者应
        将此批次回退到纯稀疏奖励。
        """
        reasons = []

        short, reason = self.detect_short_response(rollout_steps)
        if short:
            reasons.append(reason)

        repeat, reason = self.detect_repetition(rollout_tokens)
        if repeat:
            reasons.append(reason)

        diverge, reason = self.detect_divergence(batch_process_reward, batch_success)
        if diverge:
            reasons.append(reason)

        return len(reasons) > 0, reasons

    def should_fallback(self) -> bool:
        """判断是否应回退到稀疏奖励。"""
        return self.action == "fallback_to_sparse"

    def should_skip_batch(self) -> bool:
        """判断是否应完全跳过该批次。"""
        return self.action == "skip_batch"
