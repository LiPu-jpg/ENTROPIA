# ENTROPIA

**ENTROPIA** — Entropy-Gated Adaptive Reward Density for LLM Agent RL.

A research framework for studying adaptive reward density control in LLM Agent reinforcement learning.

## Core Algorithm

```
r_t^adaptive = r_t^sparse + α · σ(H_t - H_threshold) · r_t^dense
```

Where:
- `H_t`: token-level entropy on key decision tokens (tool names, parameters, stop tokens)
- `H_threshold`: EMA-tracked mean entropy (curriculum-style adaptation)
- `σ`: sigmoid gating function (smooth, continuous)
- `α`: density coefficient
- `r_t^sparse`: discounted outcome reward
- `r_t^dense`: pluggable process reward (IGPO, TIPS, etc.)

## Key Innovation

Uses entropy as a **preventive gate**, not a reactive regularizer. Density is dynamically scheduled, not fixed. Adaptive to training progress via EMA threshold.

## Quick Start

```bash
# Full adaptive reward density
python scripts/run.py --mode adaptive

# Baselines
python scripts/run.py --mode sparse           # ReTool-style binary reward
python scripts/run.py --mode dense_igpo       # IGPO fixed information gain
python scripts/run.py --mode dense_fixed      # WorkForceAgent-R1 fixed dense
python scripts/run.py --mode autotool_entropy # AutoTool entropy constraint

# Ablation studies
python scripts/run.py --mode adaptive --ablation threshold    # Fixed vs EMA threshold
python scripts/run.py --mode adaptive --ablation granularity  # Step vs traj entropy
python scripts/run.py --mode adaptive --ablation random_gate  # Sanity: random gating

# Custom model
python scripts/run.py --mode adaptive --model Qwen/Qwen2.5-7B-Instruct
```

## Architecture

```
src/
├── core/
│   ├── entropy.py          # Token-level entropy estimation (key tokens only)
│   ├── adaptive_reward.py   # Entropy-gated adaptive density function
│   └── hacking_detector.py # Three-signal reward hacking monitor
├── configs/
│   └── config.py           # Full training config + baselines + ablations
├── training/
│   └── trainer.py         # GRPO trainer, 5 reward modes
├── data/
│   └── tau_dataset.py     # τ-Bench format synthetic data (25 tasks)
├── envs/
│   └── mock_env.py        # Mock τ-Bench environment
└── scripts/
    └── run.py             # Main entry point with CLI
```

## Reward Modes

| Mode | Description |
|------|-------------|
| `adaptive` | Full entropy-gated adaptive density (Direction A) |
| `sparse` | ReTool-style binary outcome reward |
| `dense_igpo` | IGPO-style fixed information gain process reward |
| `dense_fixed` | WorkForceAgent-R1 fixed dense reward (ablation) |
| `autotool_entropy` | AutoTool-style entropy constraint in loss |

## Hardware

- **Validation**: MacBook M5 (logic verification, gpt2)
- **Training**: 4×NVIDIA L20 GPU server (2+2 or 4-together)
- **Base model**: Qwen2.5-7B-Instruct + LoRA (r=64, α=128)

## Requirements

```
torch>=2.0
transformers>=4.40
peft>=0.10
wandb
```

## Papers

See `docs/papers.md` for full paper references and notes.

## Documentation

- `WORKSPACE.md` — Full project overview (Chinese)
- `docs/algorithm.md` — Core algorithm details
- `docs/extensions.md` — Future research directions
- `docs/papers.md` — Related papers and training data sources