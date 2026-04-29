# 方向A — 不确定性感知的自适应奖励密度控制
# 完整实现

from dataclasses import dataclass, field
from typing import Optional, List, Literal


@dataclass
class EntropyConfig:
    """Token-level entropy estimation configuration."""

    # Which token positions to compute entropy on
    key_token_types: List[str] = field(
        default_factory=lambda: [
            "tool_name",  # e.g., "search_airport"
            "tool_params",  # e.g., '{"city": "NYC"}'
            "stop_token",  # e.g., <|im_end|>, </action>
            "action_type",  # e.g., click, fill, select
        ]
    )
    # Minimum probability clip to prevent log(0) = NaN
    eps: float = 1e-8
    # Use log-probability or raw probability for entropy
    use_log_prob: bool = True


@dataclass
class AdaptiveRewardConfig:
    """Adaptive reward density configuration."""

    # Density control coefficient (0 = pure sparse, large = nearly full dense)
    alpha: float = 1.0
    # Initial entropy threshold (will be updated via EMA during training)
    H_threshold_init: float = 0.5
    # EMA decay rate for threshold update: H_t = beta * mean(H_batch) + (1-beta) * H_{t-1}
    beta: float = 0.02
    # Sigmoid temperature for smooth gating (higher = sharper transition)
    sigmoid_temp: float = 1.0
    # Minimum gate value (prevent complete shutdown)
    gate_min: float = 0.0


@dataclass
class HackingDetectorConfig:
    """Reward hacking monitoring configuration."""

    # Enable hacking detection safety net
    enabled: bool = True
    # Short response threshold (fraction of normal step count)
    short_response_ratio: float = 0.33
    # Repeat pattern detection: consecutive identical token sequences
    repeat_window: int = 3
    # Reward-success divergence threshold
    divergence_window: int = 10
    # Action on detection: "fallback_to_sparse" | "skip_batch" | "log_only"
    action: Literal["fallback_to_sparse", "skip_batch", "log_only"] = (
        "fallback_to_sparse"
    )


@dataclass
class TrainingConfig:
    """Full training configuration."""

    # Model
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    model_dtype: str = "bfloat16"
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128

    # GRPO
    num_rollouts_per_query: int = 8
    grpo_mu: int = 1  # GRPO inner iterations
    grpo_epsilon: float = 0.2
    kl_beta: float = 0.04
    gamma: float = 0.9  # Discount factor
    max_turns: int = 20  # Max turns per rollout

    # SFT warmup
    sft_warmup_epochs: int = 2
    sft_learning_rate: float = 2e-5
    sft_batch_size: int = 8

    # RL training
    rl_learning_rate: float = 1e-6
    rl_batch_size: int = 32  # Number of queries per RL batch
    total_rl_steps: int = 400
    grad_accum_steps: int = 4

    # Reward mode: "adaptive" | "sparse" | "dense_igpo" | "dense_fixed" | "autotool_entropy"
    reward_mode: Literal[
        "adaptive", "sparse", "dense_igpo", "dense_fixed", "autotool_entropy"
    ] = "adaptive"

    # Adaptive reward config (used when reward_mode == "adaptive")
    adaptive: AdaptiveRewardConfig = field(default_factory=AdaptiveRewardConfig)
    # Entropy config
    entropy: EntropyConfig = field(default_factory=EntropyConfig)
    # Hacking detector config
    hacking: HackingDetectorConfig = field(default_factory=HackingDetectorConfig)

    # Environment
    env_name: str = "tau_bench"  # tau_bench | alfworld | webarena
    env_kwargs: dict = field(default_factory=dict)

    # Logging
    wandb_project: str = "adaptive-reward-density"
    log_interval: int = 10
    save_interval: int = 50
    eval_interval: int = 50

    # Hardware
    num_gpus: int = 4
    per_device_train_batch_size: int = 1

    # Output
    output_dir: str = "./outputs"
    seed: int = 42


# ──────────────────────────────────────────────────
# Pre-built experiment configs
# ──────────────────────────────────────────────────


def get_sparse_baseline_config(**overrides) -> TrainingConfig:
    """Pure sparse reward baseline (ReTool-style)."""
    cfg = TrainingConfig(reward_mode="sparse")
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def get_dense_igpo_baseline_config(**overrides) -> TrainingConfig:
    """Fixed-dense IGPO process reward baseline."""
    cfg = TrainingConfig(reward_mode="dense_igpo")
    cfg.adaptive.alpha = 0.0  # Disable gating, always use dense
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def get_dense_fixed_baseline_config(**overrides) -> TrainingConfig:
    """Fixed dense reward (WorkForceAgent-R1 style ablation)."""
    cfg = TrainingConfig(reward_mode="dense_fixed")
    cfg.adaptive.alpha = 0.0
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def get_autotool_entropy_baseline_config(**overrides) -> TrainingConfig:
    """AutoTool-style entropy constraint baseline."""
    cfg = TrainingConfig(reward_mode="autotool_entropy")
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def get_adaptive_config(**overrides) -> TrainingConfig:
    """Full adaptive reward density (direction A)."""
    cfg = TrainingConfig(reward_mode="adaptive")
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


# ──────────────────────────────────────────────────
# Ablation presets
# ──────────────────────────────────────────────────

ABLATION_PRESETS = {
    # Ablation 1: Density extremes vs adaptive
    "sparse": get_sparse_baseline_config,
    "dense_igpo": get_dense_igpo_baseline_config,
    "dense_fixed": get_dense_fixed_baseline_config,
    "adaptive": get_adaptive_config,
    # Ablation 2: Fixed threshold vs dynamic EMA
    "adaptive_fixed_threshold": lambda **kw: get_adaptive_config(
        **{**kw, "adaptive": AdaptiveRewardConfig(beta=0.0)}  # beta=0 → no EMA
    ),
    # Ablation 3: Entropy granularity
    "adaptive_step_only": lambda **kw: get_adaptive_config(
        **{**kw, "entropy": EntropyConfig(key_token_types=["tool_name", "tool_params"])}
    ),
    "adaptive_traj_only": lambda **kw: get_adaptive_config(
        **{**kw, "entropy": EntropyConfig(key_token_types=[])}  # disable step entropy
    ),
    # Sanity check: random gating
    "adaptive_random_gate": lambda **kw: get_adaptive_config(
        **{**kw, "adaptive": AdaptiveRewardConfig(alpha=1.0, H_threshold_init=0.0)}
        # H_threshold=0 → gate always open, but we add random gating in code
    ),
}
