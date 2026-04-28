"""
Reward hacking detection module.

Monitors three hacking signals during training:
1. Abnormally short responses (step count < 1/3 of normal)
2. Repetition patterns (consecutive identical token sequences)
3. Reward-success divergence (process reward ↑ while task success ↓)

If hacking is detected, the safety net forces fallback to sparse-only rewards.
"""

import torch
from typing import List, Tuple, Optional
from collections import deque


class HackingDetector:
    """
    Real-time reward hacking monitor.

    Operates as a safety net: if adaptive gating is working correctly,
    this should rarely trigger. Its presence is itself a validation of
    the gating mechanism's effectiveness.
    """

    def __init__(
        self,
        short_response_ratio: float = 0.33,
        repeat_window: int = 3,
        divergence_window: int = 10,
        action: str = "fallback_to_sparse",
    ):
        """
        Args:
            short_response_ratio: Step count below this * normal_baseline triggers alarm.
            repeat_window: Consecutive repeats to flag as pattern.
            divergence_window: Window size for reward-success divergence check.
            action: "fallback_to_sparse" | "skip_batch" | "log_only"
        """
        self.short_ratio = short_response_ratio
        self.repeat_window = repeat_window
        self.divergence_window = divergence_window
        self.action = action

        # Running statistics
        self.normal_step_count = None  # EMA of normal step counts
        self.recent_process_rewards = deque(maxlen=divergence_window)
        self.recent_success_rates = deque(maxlen=divergence_window)
        self.event_count = 0

    def detect_short_response(
        self,
        step_count: int,
        ema_beta: float = 0.1,
    ) -> Tuple[bool, str]:
        """
        Check if response is abnormally short.

        Args:
            step_count: Number of steps in this rollout
            ema_beta: Update rate for normal baseline
        Returns:
            (is_hacking, reason)
        """
        if self.normal_step_count is None:
            self.normal_step_count = step_count
            return False, ""

        # Update normal baseline
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
        Check for repeated token sequences across steps.

        Args:
            token_ids: Token IDs for each step in the rollout
        Returns:
            (is_hacking, reason)
        """
        if len(token_ids) < self.repeat_window:
            return False, ""

        # Compare consecutive steps' token sequences
        for i in range(len(token_ids) - self.repeat_window + 1):
            window = token_ids[i : i + self.repeat_window]
            # Check if all sequences in window are identical (or nearly)
            first = window[0]
            all_same = all(
                len(first) == len(w) and all(a == b for a, b in zip(first, w))
                for w in window[1:]
            )
            if all_same:
                # Also check they're non-trivial (not all padding tokens)
                if len(set(first)) > 1:  # Not all the same token within sequence
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
        Check if process reward is rising while task success is falling.
        Classic reward hacking signal: model is gaming the process reward
        without actually improving at the task.

        Args:
            process_reward: Mean process reward for this batch
            success: Whether the batch had above-median success rate
        Returns:
            (is_hacking, reason)
        """
        self.recent_process_rewards.append(process_reward)
        self.recent_success_rates.append(float(success))

        if len(self.recent_process_rewards) < self.divergence_window:
            return False, ""

        # Check trend: process reward rising, success rate falling
        pr_list = list(self.recent_process_rewards)
        sr_list = list(self.recent_success_rates)

        pr_trend = pr_list[-1] - pr_list[0]  # Positive = rising
        sr_trend = sr_list[-1] - sr_list[0]  # Negative = falling

        if pr_trend > 0.05 and sr_trend < -0.05:  # Divergence detected
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
        Run all hacking checks. Returns (any_hacking, reasons).

        If any check triggers and action != "log_only", the caller should
        fall back to sparse-only rewards for this batch.
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
        """Determine if we should fall back to sparse rewards."""
        return self.action == "fallback_to_sparse"

    def should_skip_batch(self) -> bool:
        """Determine if we should skip this batch entirely."""
        return self.action == "skip_batch"
