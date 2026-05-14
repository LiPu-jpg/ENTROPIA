"""
快速对比：Qwen2.5-1.5B-Instruct 上的 sparse vs adaptive。
最小化设置，在约 1-2 小时内验证核心假设。
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    TrainingConfig,
    ABLATION_PRESETS,
)
from core.entropy import EntropyEstimator
from core.adaptive_reward import AdaptiveRewardDensity
from core.hacking_detector import HackingDetector
from core.signal_bank import SignalBank
from core.reward_router import RewardRouter, NeedGate, UtilityGate, ReliabilityGate, RiskController
from training.trainer import AdaptiveRewardTrainer
from data.tau_dataset import load_tau_bench_dataset, SFTDataset


def build_key_token_ids(tokenizer):
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
            "cancel",
            "modify",
            "return",
            "exchange",
            "transfer",
            "list",
        ],
        "action_type": ["click", "fill", "select", "scroll", "type", "press", "hover"],
        "stop_token": [
            "<|im_end|>",
            "</action>",
            "<|endoftext|>",
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


def run_single(mode: str, model_name: str, seed: int, n_steps: int = 200) -> dict:
    """运行一次实验并返回结果字典。"""
    config = ABLATION_PRESETS[mode]()
    config.model_name = model_name
    config.seed = seed
    config.total_rl_steps = n_steps
    config.sft_warmup_epochs = 1
    config.sft_learning_rate = 5e-5
    config.sft_batch_size = 2
    config.rl_batch_size = 2
    config.num_rollouts_per_query = 4
    config.num_gpus = 2
    config.max_turns = 10
    config.log_interval = 5
    config.eval_interval = 10
    config.save_interval = 50
    config.wandb_project = ""
    config.output_dir = f"./outputs/quick_{mode}_{datetime.now().strftime('%H%M')}"

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

    reward_router = None
    signal_bank = None
    if config.use_router:
        signal_bank = SignalBank(max_turns=config.max_turns)
        reward_router = RewardRouter(
            need=NeedGate(**config.router_need),
            utility=UtilityGate(**config.router_utility),
            reliability=ReliabilityGate(
                variant=config.router_reliability_variant,
                **config.router_reliability,
            ),
            risk_ctrl=RiskController(**config.router_risk),
            signal_weights=config.router_signal_weights,
        )

    trainer = AdaptiveRewardTrainer(
        config=config,
        entropy_estimator=entropy_estimator,
        adaptive_reward=adaptive_reward,
        hacking_detector=hacking_detector,
        reward_router=reward_router,
        signal_bank=signal_bank,
    )

    train_dataset = load_tau_bench_dataset(
        n_samples=None,
        split="train",
        seed=config.seed,
        use_simia=True,
    )
    eval_tasks = load_tau_bench_dataset(
        n_samples=50,
        split="eval",
        seed=config.seed + 999,  # 完全不同种子，减少同分布 overlap
        use_simia=True,
    ).tasks

    sft_dataset = SFTDataset(train_dataset.tasks, tokenizer)

    best_metric = trainer.train(
        train_dataset=train_dataset,
        eval_tasks=eval_tasks,
        sft_dataset=sft_dataset,
        sft_checkpoint=args.sft_checkpoint if args.sft_checkpoint else None,
    )

    return {
        "mode": mode,
        "model": model_name,
        "seed": seed,
        "n_steps": n_steps,
        "best_success_rate": best_metric,
        "final_loss": trainer.global_step,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="sparse", choices=[
            "sparse", "dense_igpo", "adaptive",
            "router", "router_r2", "router_r3",
            "router_need_only", "router_no_reliability", "router_no_risk",
            "random_gate",
        ]
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/mnt/home/user41/downloaded_models/Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--sft_checkpoint", type=str, default="")
    args = parser.parse_args()

    print(
        f"=== Quick CMP: mode={args.mode}, model={args.model}, steps={args.n_steps} ==="
    )
    print(f"Start: {datetime.now().isoformat()}")

    result = run_single(args.mode, args.model, args.seed, args.n_steps)

    print(f"\n=== RESULT ===")
    print(json.dumps(result, indent=2))

    os.makedirs("./outputs/quick_results", exist_ok=True)
    with open(f"./outputs/quick_results/{args.mode}_{args.seed}.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Done: {datetime.now().isoformat()}")
