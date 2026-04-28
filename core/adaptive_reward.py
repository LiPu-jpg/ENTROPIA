"""
Adaptive Reward Density — the core algorithm of Direction A.

r_t^adaptive = r_t^sparse + α · σ(H_t - H_threshold) · r_t^dense

where:
  H_t          = token-level entropy at step t (on key decision tokens only)
  H_threshold  = EMA-tracked mean entropy (curriculum-style adaptation)
  σ            = sigmoid gating function (smooth, continuous)
  α            = density coefficient (controls max dense injection)
  r_t^sparse   = discounted outcome reward
  r_t^dense    = process reward (IGPO-style info gain, or pluggable)
"""

import torch
import torch.nn.functional as F
from typing import Optional, Callable
from dataclasses import dataclass


@dataclass
class AdaptiveRewardState:
    """Running state for adaptive reward computation."""

    H_threshold: float  # Current EMA threshold
    batch_entropies: list = None  # For logging
    gate_history: list = None  # For analysis
    hacking_events: int = 0  # Count of hacking detections

    def __post_init__(self):
        self.batch_entropies = []
        self.gate_history = []


class AdaptiveRewardDensity:
    """
    Core algorithm: entropy-driven dynamic reward density control.

    Key innovations vs prior work:
    1. Entropy is used as a PREVENTIVE gate, not a reactive regularizer (vs AutoTool)
    2. Density is dynamically scheduled, not fixed (vs IGPO)
    3. Adaptive to training progress via EMA threshold (curriculum-like)
    4. Pluggable r_t^dense — works with any process reward method
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
        Args:
            alpha: Density control coefficient. Higher = more dense reward injected.
            H_threshold_init: Initial entropy threshold. Updated via EMA during training.
            beta: EMA decay rate for threshold update (0.01-0.05 recommended).
            sigmoid_temp: Temperature for sigmoid gating. temp=1 is standard sigmoid.
            gate_min: Minimum gate value (0 = can fully close, >0 = always some dense).
            eps: Numerical stability.
        """
        self.alpha = alpha
        self.beta = beta
        self.sigmoid_temp = sigmoid_temp
        self.gate_min = gate_min
        self.eps = eps
        self.state = AdaptiveRewardState(H_threshold=H_threshold_init)

    def compute_gate(self, H_t: torch.Tensor) -> torch.Tensor:
        """
        Compute density gate g_t = σ(H_t - H_threshold) for a single step.

        High H_t (uncertain) → gate ≈ 1 (inject dense reward)
        Low H_t (confident) → gate ≈ 0 (suppress dense reward)

        Args:
            H_t: Entropy at step t, scalar or [batch]
        Returns:
            g_t: Gate value in [gate_min, 1]
        """
        gate = torch.sigmoid(self.sigmoid_temp * (H_t - self.state.H_threshold))
        gate = torch.clamp(gate, min=self.gate_min, max=1.0)
        return gate

    def compute_adaptive_reward(
        self,
        r_sparse: torch.Tensor,  # [num_steps] discounted outcome reward
        r_dense: torch.Tensor,  # [num_steps] process reward (IG, TIPS, etc.)
        H_t: torch.Tensor,  # [num_steps] per-step entropy
    ) -> torch.Tensor:
        """
        Compute adaptive reward for a full trajectory.

        r_t = r_t^sparse + α · σ(H_t - H_threshold) · r_t^dense

        Args:
            r_sparse: Discounted outcome rewards per step
            r_dense: Process rewards per step (from any credit assignment method)
            H_t: Per-step entropies
        Returns:
            r_adaptive: [num_steps] combined adaptive rewards
        """
        gate = self.compute_gate(H_t)  # [num_steps]
        self.state.gate_history.append(gate.mean().item())
        r_adaptive = r_sparse + self.alpha * gate * r_dense
        return r_adaptive

    def update_threshold(self, batch_mean_entropy: float):
        """
        EMA update: H_threshold ← β · mean(H_batch) + (1-β) · H_threshold

        This creates curriculum-style adaptation:
        - Early training: high entropy → low threshold → dense rewards → fast learning
        - Late training: low entropy → high threshold → sparse rewards → prevent hacking
        """
        self.state.H_threshold = (
            self.beta * batch_mean_entropy + (1 - self.beta) * self.state.H_threshold
        )
        self.state.batch_entropies.append(batch_mean_entropy)

    def get_stats(self) -> dict:
        """Get training statistics for logging."""
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
# Pluggable process reward functions
# These serve as r_t^dense in the adaptive reward formula.
# You can plug in IGPO, TIPS, or any other process reward.
# ──────────────────────────────────────────────────


def igpo_information_gain(
    model_prob_current: torch.Tensor,  # p(answer | s_t), scalar per rollout
    model_prob_prev: torch.Tensor,  # p(answer | s_{t-1}), scalar per rollout
    ground_truth_answer_id: int,  # token ID of ground truth answer
) -> torch.Tensor:
    """
    IGPO-style information gain process reward.
    IG_t = log p(answer|s_t) - log p(answer|s_{t-1})
    = marginal increase in model's belief in the correct answer.
    """
    ig = torch.log(model_prob_current.clamp(min=1e-8)) - torch.log(
        model_prob_prev.clamp(min=1e-8)
    )
    return ig


def tips_potential_reward(
    potential_current: torch.Tensor,  # Φ(s_t), potential at current state
    potential_prev: torch.Tensor,  # Φ(s_{t-1}), potential at previous state
    gamma: float = 0.9,
) -> torch.Tensor:
    """
    TIPS-style potential-based reward shaping.
    r_t^dense = Φ(s_t) - γ · Φ(s_{t-1})
    """
    return potential_current - gamma * potential_prev


def progress_reward(
    task_progress_current: float,  # 0-1 progress at current step
    task_progress_prev: float,  # 0-1 progress at previous step
) -> float:
    """
    Simple progress-based process reward.
    r_t^dense = progress_current - progress_prev
    """
    return task_progress_current - task_progress_prev


def outcome_discounted_reward(
    outcome: float,  # +1 (success) or -1 (failure) or 0 (partial)
    step_idx: int,
    total_steps: int,
    gamma: float = 0.9,
) -> float:
    """
    Discounted outcome reward: r_t^sparse = outcome * γ^{total_steps - step_idx - 1}
    """
    return outcome * (gamma ** (total_steps - step_idx - 1))
