"""
ENTROPIA v2: Reliability-Calibrated Reward Router。

核心公式:
  g^k_t = N_t × U^k_t × L^k_s × M^k_t × (1 - H^{risk}_t)

最终奖励:
  r_t = r_outcome_t + B_s × Σ_k w_k × g^k_t × S̃^k_t

四模块:
  A. Need Gate       — 状态是否需要过程监督
  B. Utility Gate    — 信号在当前步是否有任务效用
  C. Reliability Gate — 信号近期是否与 outcome 对齐
  D. Risk Controller  — 安全控制器，抑制/关闭过程监督
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import math


# ──────────────────────────────────────────────────
# Module A: Need Gate
# ──────────────────────────────────────────────────


@dataclass
class NeedGate:
    """
    Need_t = σ((φ_N - τ_N) / T_N)

    特征:
      - key-token entropy (λ_H)
      - group outcome collapse (λ_C)
      - progress stagnation (λ_P)
    """

    tau_need: float = 0.0
    temp_need: float = 1.0
    lambda_h: float = 0.5
    lambda_c: float = 0.3
    lambda_p: float = 0.2
    collapse_threshold: float = 0.9  # 同 query rollout 结果全同视为 collapse
    stagnation_window: int = 3

    def __call__(
        self,
        step_entropy: float,
        group_collapse: float,
        stagnation: float,
    ) -> float:
        phi = (
            self.lambda_h * step_entropy
            + self.lambda_c * group_collapse
            + self.lambda_p * stagnation
        )
        return float(torch.sigmoid(
            torch.tensor((phi - self.tau_need) / self.temp_need)
        ).item())

    def compute_group_collapse(self, outcomes: List[float]) -> float:
        """同一 query 的 rollout 结果几乎相同时返回 1.0。"""
        if len(outcomes) < 2:
            return 0.0
        unique = len(set(o > 0 for o in outcomes))
        return 1.0 if unique <= 1 else 0.0

    def compute_stagnation(
        self, recent_entropies: List[float]
    ) -> float:
        """如果近期多步熵无显著变化 → stagnation = 1。"""
        if len(recent_entropies) < self.stagnation_window:
            return 0.0
        recent = recent_entropies[-self.stagnation_window :]
        if max(recent) - min(recent) < 0.1:
            return 1.0
        return 0.0


# ──────────────────────────────────────────────────
# Module B: Utility Gate
# ──────────────────────────────────────────────────


@dataclass
class UtilityGate:
    """
    U^k_t = σ((φ^k_U - τ_U) / T_U)

    每个过程信号 k 独立计算 utility。
    默认: U_t = max(0, Δ log p(answer))
    """

    tau_util: float = 0.0
    temp_util: float = 1.0

    def __call__(self, signal_value: float) -> float:
        phi = signal_value
        return float(torch.sigmoid(
            torch.tensor((phi - self.tau_util) / self.temp_util)
        ).item())


# ──────────────────────────────────────────────────
# Module C: Reliability Gate
# ──────────────────────────────────────────────────


@dataclass
class ReliabilityGate:
    """
    L^k_s = 对每个信号 k 估计近期是否与 outcome 对齐

    三种变体:
      R1 (additive): σ(a·ρ + b·ξ + c·δ − d·χ)         ← 各维度互相补偿
      R2 (multiplicative): σρ(ρ) × σξ(ξ) × σδ(δ)       ← 一票否决
      R3 (softmax): 信号间竞争 dense budget              ← 自动淘汰不可靠信号
    """

    variant: str = "R1"  # "R1" | "R2" | "R3"
    window_size: int = 50
    temp_L: float = 1.0  # shared temperature for R2/R3 sigmoid
    a: float = 0.4
    b: float = 0.3
    c: float = 0.2
    d: float = 0.1

    _process_history: Dict[str, deque] = field(default_factory=dict)
    _outcome_history: deque = field(default_factory=lambda: deque(maxlen=50))

    def __post_init__(self):
        self._process_history = {
            "info_gain": deque(maxlen=self.window_size),
            "efficiency_cost": deque(maxlen=self.window_size),
        }

    def update(self, signal_name: str, process_sum: float, outcome: float):
        if signal_name not in self._process_history:
            self._process_history[signal_name] = deque(maxlen=self.window_size)
        self._process_history[signal_name].append(process_sum)
        self._outcome_history.append(outcome)

    def _compute_stats(self, signal_name: str) -> Tuple[float, float, float, float]:
        """计算四个统计量: (rho, xi, delta, chi)"""
        if signal_name not in self._process_history:
            return 0.0, 0.5, 0.0, 0.0
        proc = list(self._process_history[signal_name])
        outc = list(self._outcome_history)
        n = min(len(proc), len(outc))
        if n < 4:
            return 0.0, 0.5, 0.0, 0.0
        proc = proc[-n:]
        outc = outc[-n:]

        rho = 0.0
        if max(proc) - min(proc) > 1e-6 and max(outc) - min(outc) > 1e-6:
            p_mean = sum(proc) / n; o_mean = sum(outc) / n
            cov = sum((p - p_mean) * (o - o_mean) for p, o in zip(proc, outc)) / n
            p_std = (sum((p - p_mean) ** 2 for p in proc) / n) ** 0.5
            o_std = (sum((o - o_mean) ** 2 for o in outc) / n) ** 0.5
            if p_std > 1e-6 and o_std > 1e-6:
                rho = cov / (p_std * o_std)
            rho = max(-1.0, min(1.0, rho))

        median_out = sorted(outc)[n // 2]
        good_sig = [proc[i] for i in range(n) if outc[i] > median_out]
        bad_sig = [proc[i] for i in range(n) if outc[i] <= median_out]
        delta = 0.0
        if good_sig and bad_sig:
            delta = sum(good_sig) / len(good_sig) - sum(bad_sig) / len(bad_sig)
            delta = max(0.0, delta)

        sig_pos = sum(1 for i in range(n) if proc[i] > 0 and outc[i] > 0)
        sig_neg = sum(1 for i in range(n) if proc[i] <= 0 and outc[i] <= 0)
        xi = (sig_pos + sig_neg) / n if n > 0 else 0.5

        chi = 0.0
        return rho, xi, delta, chi

    def _sigmoid_zscore(self, x: float) -> float:
        return float(torch.sigmoid(torch.tensor(x / self.temp_L)).item())

    def compute(self, signal_name: str) -> float:
        rho, xi, delta, chi = self._compute_stats(signal_name)

        if self.variant == "R1":
            # 加法: 互相补偿
            L = float(torch.sigmoid(
                torch.tensor(self.a * rho + self.b * xi + self.c * delta - self.d * chi)
            ).item())

        elif self.variant == "R2":
            # 乘法: 一票否决
            L = (self._sigmoid_zscore(rho) * self._sigmoid_zscore(xi) *
                 self._sigmoid_zscore(delta))
            L = max(0.01, min(1.0, L))

        elif self.variant == "R3":
            # Softmax: 信号间竞争，compute 返回 ρ 用于后续 softmax
            L = float(torch.sigmoid(torch.tensor(self.a * rho + self.b * xi)).item())

        else:
            L = 0.5

        return L

    def compute_softmax_weights(
        self, need: float, signals: Dict[str, float], utilities: Dict[str, float]
    ) -> Dict[str, float]:
        """R3 Softmax: 信号间竞争 dense budget。返回每个信号的 softmax 权重。"""
        scores = {}
        for name, val in signals.items():
            rho, _, _, _ = self._compute_stats(name)
            scores[name] = need * utilities.get(name, 0.5) * max(0.0, rho) * val
        if not scores:
            return {}
        max_s = max(scores.values()) if scores else 1.0
        if max_s <= 0:
            return {k: 1.0 / len(scores) for k in scores}
        exp_sum = sum(math.exp(s / max_s) for s in scores.values())
        return {k: math.exp(v / max_s) / exp_sum for k, v in scores.items()}


# ──────────────────────────────────────────────────
# Module D: Risk Controller
# ──────────────────────────────────────────────────


@dataclass
class RiskController:
    """
    根据 hacking risk 和 advantage collapse 动态调整 dense budget。

    B_{s+1} = clip(
        B_s + η_collapse × collapse_rate
            - η_hack × hacking_rate
            - η_conflict × conflict_rate,
        B_min, B_max
    )
    """

    B_init: float = 1.0
    B_min: float = 0.0
    B_max: float = 2.0
    eta_collapse: float = 0.1
    eta_hack: float = 0.2
    eta_conflict: float = 0.1

    B_s: float = field(default=1.0)

    _hacking_events: int = 0
    _collapse_count: int = 0
    _conflict_count: int = 0
    _total_steps: int = 0

    def __post_init__(self):
        self.B_s = self.B_init

    def update(
        self,
        hacking_detected: bool,
        group_collapse: float,
        success_divergence: float,
    ):
        self._total_steps += 1
        if hacking_detected:
            self._hacking_events += 1
        if group_collapse > 0.5:
            self._collapse_count += 1
        if success_divergence > 0:
            self._conflict_count += 1

        n = max(1, self._total_steps)

        self.B_s = max(
            self.B_min,
            min(
                self.B_max,
                self.B_s
                + self.eta_collapse * (1 - self._collapse_count / n)
                - self.eta_hack * (self._hacking_events / n)
                - self.eta_conflict * (self._conflict_count / n),
            ),
        )

    @property
    def budget(self) -> float:
        return self.B_s

    @property
    def risk_level(self) -> str:
        if self._hacking_events > 5:
            return "high"
        if self._hacking_events > 2:
            return "medium"
        return "low"


# ──────────────────────────────────────────────────
# Main Router: NUR + Risk
# ──────────────────────────────────────────────────


@dataclass
class RewardRouter:
    """
    ENTROPIA v2: 可靠性校准的奖励路由器。

    对每个 trajectory step t 的每个过程信号 k:
      g_t^k = Need_t × Utility_t^k × Reliability_s^k × Mask_t^k × (1 - Risk_t)

    最终奖励:
      r_t = r_outcome_t + B_s × Σ_k w_k × g_t^k × S̃_t^k
    """

    need: NeedGate = field(default_factory=NeedGate)
    utility: UtilityGate = field(default_factory=UtilityGate)
    reliability: ReliabilityGate = field(default_factory=ReliabilityGate)
    risk_ctrl: RiskController = field(default_factory=RiskController)

    # 信号权重（可随训练阶段调整）
    signal_weights: Dict[str, float] = field(default_factory=lambda: {
        "info_gain": 0.6,
        "efficiency": 0.2,
        "relevance": 0.2,
    })

    # EMA 平滑（避免单步波动）
    ema_beta: float = 0.1
    _ema_signals: Dict[str, float] = field(default_factory=dict)

    def route(
        self,
        step_entropy: float,
        group_collapse: float,
        stagnation: float,
        signal_values: Dict[str, float],
        format_valid: float,
        hacking_detected: bool,
        success_divergence: float,
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算路由门控和最终过程奖励。

        Returns:
            process_reward: 加权过程奖励标量
            gate_info: 各门控值（用于日志）
        """
        # Need gate
        N_t = self.need(step_entropy, group_collapse, stagnation)

        # Risk
        self.risk_ctrl.update(hacking_detected, group_collapse, success_divergence)
        risk = 0.1 if hacking_detected else 0.0

        # Per-signal routing
        total_reward = 0.0
        gates = {"need": N_t, "risk": risk, "reliability_variant": self.reliability.variant}

        if self.reliability.variant == "R3":
            # Softmax competition: signals compete for dense budget
            utilities = {}
            for signal_name in signal_values:
                utilities[signal_name] = self.utility(signal_values[signal_name])
            softmax_w = self.reliability.compute_softmax_weights(N_t, signal_values, utilities)
            for signal_name, alpha in softmax_w.items():
                value = signal_values.get(signal_name, 0.0)
                M_t = format_valid
                contribution = self.risk_ctrl.budget * alpha * value / max(len(softmax_w), 1)
                total_reward += contribution
                gates[f"w_{signal_name}"] = alpha
                gates[f"g_{signal_name}"] = alpha
        else:
            # R1/R2: per-signal gating
            for signal_name, weight in self.signal_weights.items():
                if signal_name not in signal_values:
                    continue

                value = signal_values[signal_name]
                U_t = self.utility(value)
                L_s = self.reliability.compute(signal_name)
                M_t = format_valid
                g_t = N_t * U_t * L_s * M_t * (1.0 - risk)

                if signal_name not in self._ema_signals:
                    self._ema_signals[signal_name] = g_t
                else:
                    self._ema_signals[signal_name] = (
                        self.ema_beta * g_t
                        + (1 - self.ema_beta) * self._ema_signals[signal_name]
                    )
                g_t_smooth = self._ema_signals[signal_name]

                contribution = weight * self.risk_ctrl.budget * g_t_smooth * value
                total_reward += contribution
                gates[f"g_{signal_name}"] = g_t
                gates[f"u_{signal_name}"] = U_t
                gates[f"l_{signal_name}"] = L_s

        gates["budget"] = self.risk_ctrl.budget
        return total_reward, gates

    def get_stats(self) -> dict:
        return {
            "budget": self.risk_ctrl.budget,
            "risk_level": self.risk_ctrl.risk_level,
            "signal_weights": dict(self.signal_weights),
        }
