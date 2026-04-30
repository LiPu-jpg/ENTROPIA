"""
方向 A 实验的主入口。

用法：
    python run.py --mode adaptive          # 完整自适应奖励密度
    python run.py --mode sparse            # ReTool 风格稀疏基线
    python run.py --mode dense_igpo        # IGPO 固定稠密基线
    python run.py --mode dense_fixed       # WorkForceAgent-R1 固定稠密基线
    python run.py --mode autotool_entropy  # AutoTool 熵约束基线
    python run.py --ablation threshold     # 消融实验：固定阈值 vs EMA 阈值
    python run.py --ablation granularity   # 消融实验：步骤级 vs 轨迹级熵
    python run.py --ablation random_gate   # 合理性检查：随机门控
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    TrainingConfig,
    AdaptiveRewardConfig,
    EntropyConfig,
    HackingDetectorConfig,
    ABLATION_PRESETS,
)
from core.entropy import EntropyEstimator
from core.adaptive_reward import AdaptiveRewardDensity
from core.hacking_detector import HackingDetector
from training.trainer import AdaptiveRewardTrainer
from data.tau_dataset import load_tau_bench_dataset


def build_key_token_ids(tokenizer):
    """从分词器词汇表中构建关键 token ID 映射。"""
    key_patterns = {
        "tool_name": [
            "search",
            "click",
            "fill",
            "select",
            "scroll",
            "type",
            "goto",
            "go_back",
            "refresh",
            "submit",
            "press",
            "hover",
            "find",
            "get",
            "post",
            "put",
            "delete",
        ],
        "action_type": ["click", "fill", "select", "scroll", "type", "press", "hover"],
        "stop_token": [
            "<|im_end|>",
            "</action>",
            "",
            "[STOP]",
            "Observation:",
            "\n\n",
        ],
    }

    key_token_ids = {}
    for token_type, patterns in key_patterns.items():
        ids = set()
        for pattern in patterns:
            encoded = tokenizer.encode(pattern, add_special_tokens=False)
            ids.update(encoded)
        key_token_ids[token_type] = list(ids)

    return key_token_ids


def get_config_for_mode(mode, ablation=None):
    config_class = ABLATION_PRESETS.get(mode)
    if config_class is None:
        valid = list(ABLATION_PRESETS.keys())
        raise ValueError(f"Unknown mode '{mode}'. Valid: {valid}")

    config = config_class()

    if ablation == "threshold":
        config.adaptive.beta = 0.0
    elif ablation == "granularity":
        config.entropy.key_token_types = ["tool_name", "tool_params"]
    elif ablation == "random_gate":
        config.adaptive.H_threshold_init = 0.0
        config.random_gate = True

    return config


def run_experiment(config):
    """使用给定配置运行完整实验。"""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    key_token_ids = build_key_token_ids(tokenizer)

    entropy_estimator = EntropyEstimator(
        key_token_ids=key_token_ids,
        eps=config.entropy.eps,
    )

    adaptive_reward = AdaptiveRewardDensity(
        alpha=config.adaptive.alpha,
        H_threshold_init=config.adaptive.H_threshold_init,
        beta=config.adaptive.beta,
        sigmoid_temp=config.adaptive.sigmoid_temp,
        gate_min=config.adaptive.gate_min,
    )

    hacking_detector = HackingDetector(
        short_response_ratio=config.hacking.short_response_ratio,
        repeat_window=config.hacking.repeat_window,
        divergence_window=config.hacking.divergence_window,
        action=config.hacking.action,
    )

    trainer = AdaptiveRewardTrainer(
        config=config,
        entropy_estimator=entropy_estimator,
        adaptive_reward=adaptive_reward,
        hacking_detector=hacking_detector,
    )

    train_dataset = load_tau_bench_dataset(
        n_samples=config.rl_batch_size * 10,
        split="train",
        seed=config.seed,
    )
    eval_tasks = load_tau_bench_dataset(
        n_samples=5,
        split="eval",
        seed=config.seed,
    ).tasks

    best_metric = trainer.train(
        train_dataset=train_dataset,
        eval_tasks=eval_tasks,
    )

    return best_metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direction A: Adaptive Reward Density")
    parser.add_argument(
        "--mode",
        type=str,
        default="adaptive",
        choices=["adaptive", "sparse", "dense_igpo", "dense_fixed", "autotool_entropy"],
        help="Reward mode",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default=None,
        choices=["threshold", "granularity", "random_gate", None],
        help="Ablation study type",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--env", type=str, default="tau_bench")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", type=str, default="adaptive-reward-density")
    args = parser.parse_args()

    config = get_config_for_mode(args.mode, args.ablation)
    config.model_name = args.model
    config.env_name = args.env
    config.output_dir = args.output_dir
    config.seed = args.seed
    config.wandb_project = args.wandb

    print(f"Running: mode={args.mode}, ablation={args.ablation or 'none'}")
    print(
        f"Adaptive config: alpha={config.adaptive.alpha}, "
        f"H_threshold={config.adaptive.H_threshold_init}, beta={config.adaptive.beta}"
    )
    print(f"Reward mode: {config.reward_mode}")

    best = run_experiment(config)
    print(f"Best success rate: {best:.4f}")
