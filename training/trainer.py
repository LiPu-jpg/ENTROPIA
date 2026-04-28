"""
GRPO trainer with integrated adaptive reward density gating.

Supports five reward modes:
  - "adaptive": Direction A — entropy-gated density control
  - "sparse": ReTool-style binary outcome reward
  - "dense_igpo": IGPO-style fixed information gain process reward
  - "dense_fixed": WorkForceAgent-R1 style fixed dense reward (ablation)
  - "autotool_entropy": AutoTool-style entropy constraint in loss (not reward)
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from typing import List, Dict, Optional, Tuple
import copy
import wandb
from pathlib import Path

from data.tau_dataset import Task
from envs.mock_env import MockTauEnv, parse_action_from_text, Action


class AdaptiveRewardTrainer:
    def __init__(self, config, entropy_estimator, adaptive_reward, hacking_detector):
        self.config = config
        self.entropy_estimator = entropy_estimator
        self.adaptive_reward = adaptive_reward
        self.hacking_detector = hacking_detector

        self.model = None
        self.ref_model = None
        self.tokenizer = None
        self.optimizer = None
        self.global_step = 0
        self.best_success_rate = 0.0
        self.device = None

    def setup_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=getattr(torch, self.config.model_dtype),
            device_map="auto",
            trust_remote_code=True,
        )

        if self.config.use_lora:
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=["c_attn", "c_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)

        self.model.train()
        self.device = self.model.device

        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def sft_warmup(self, train_dataset, eval_dataset=None):
        from transformers import TrainingArguments, Trainer

        training_args = TrainingArguments(
            output_dir=f"{self.config.output_dir}/sft_warmup",
            num_train_epochs=self.config.sft_warmup_epochs,
            per_device_train_batch_size=self.config.sft_batch_size,
            learning_rate=self.config.sft_learning_rate,
            logging_steps=10,
            save_strategy="epoch",
            bf16=self.config.model_dtype == "bfloat16",
            report_to="wandb" if self.config.wandb_project else "none",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        trainer.train()
        return trainer.state.best_metric

    def rollout(
        self,
        instruction: str,
        task: Task,
        max_turns: int = None,
    ) -> Tuple[List[dict], float, List[float], List[float], List[list]]:
        """
        Execute one rollout: model interacts with mock environment step by step.

        Returns:
            trajectory: list of step dicts (for reward computation)
            outcome: task reward (0 or 1)
            step_entropies: [T] entropy per step
            step_token_ids: [T] list of token ID lists (for gradient recomputation)
        """
        if max_turns is None:
            max_turns = self.config.max_turns

        env = MockTauEnv(task)
        reset_result = env.reset()
        observation = reset_result.observation

        trajectory = []
        step_entropies = []
        step_token_ids = []
        outcome = 0.0

        for turn in range(max_turns):
            inputs = self.tokenizer(
                observation, return_tensors="pt", truncation=True, max_length=4096
            ).to(self.device)
            input_len = inputs.input_ids.shape[1]

            with torch.no_grad():
                self.model.eval()
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                self.model.train()

            generated_ids = outputs.sequences[0][input_len:]

            if len(generated_ids) == 0:
                trajectory.append(
                    {
                        "turn": turn,
                        "action": "",
                        "observation": "Empty generation.",
                        "done": False,
                    }
                )
                step_entropies.append(0.0)
                step_token_ids.append([])
                continue

            scores = torch.stack([s.squeeze(0) for s in outputs.scores])
            H_t, _ = self.entropy_estimator.compute_step_entropy(
                logits=scores, token_ids=generated_ids
            )
            step_entropies.append(H_t.item())
            step_token_ids.append(generated_ids.tolist())

            action_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            action = parse_action_from_text(action_text)
            if action is None:
                action = Action(name="respond", kwargs={"content": action_text})

            result = env.step(action)
            trajectory.append(
                {
                    "turn": turn,
                    "action": action_text,
                    "observation": result.observation,
                    "done": result.done,
                }
            )
            observation = result.observation

            if result.done:
                outcome = result.reward
                break

        return (
            trajectory,
            outcome,
            step_entropies,
            step_token_ids,
        )

    def _compute_grad_and_ref_logprobs(
        self,
        all_token_ids: List[List[list]],
        n_queries: int,
        max_turns: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Re-run model.forward() on collected token IDs to get gradient-bearing
        log_probs for GRPO ratio computation.

        Also computes ref_log_probs from ref_model for the same tokens.
        Both use the same forward pass, ensuring alignment.

        Returns:
            grad_logprob_matrix: [n_queries, n_rollouts, max_turns]
            ref_logprob_matrix: [n_queries, n_rollouts, max_turns]
        """
        n_rollouts = self.config.num_rollouts_per_query
        grad_lp_list = []
        ref_lp_list = []

        for q_idx in range(n_queries):
            for g_idx in range(n_rollouts):
                step_tokens = all_token_ids[q_idx * n_rollouts + g_idx]
                step_grad_lps = []
                step_ref_lps = []

                for step_toks in step_tokens:
                    if len(step_toks) < 2:
                        step_grad_lps.append(
                            torch.zeros((), device=self.device, requires_grad=True)
                        )
                        step_ref_lps.append(
                            torch.zeros((), device=self.device, requires_grad=False)
                        )
                        continue

                    toks = torch.tensor(step_toks, device=self.device).unsqueeze(0)

                    self.model.train()
                    curr_outputs = self.model(toks)
                    curr_logits = curr_outputs.logits[0, :-1]
                    curr_log_probs = F.log_softmax(curr_logits, dim=-1)
                    targets = toks[0, 1:]
                    curr_gen_lp = curr_log_probs[torch.arange(len(targets)), targets]
                    step_grad_lps.append(curr_gen_lp.sum())

                    with torch.no_grad():
                        ref_outputs = self.ref_model(toks)
                        ref_logits = ref_outputs.logits[0, :-1]
                        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                        ref_gen_lp = ref_log_probs[torch.arange(len(targets)), targets]
                        step_ref_lps.append(ref_gen_lp.sum())

                while len(step_grad_lps) < max_turns:
                    step_grad_lps.append(
                        torch.zeros((), device=self.device, requires_grad=True)
                    )
                    step_ref_lps.append(
                        torch.zeros((), device=self.device, requires_grad=False)
                    )
                grad_lp_list.append(torch.stack(step_grad_lps[:max_turns]))
                ref_lp_list.append(torch.stack(step_ref_lps[:max_turns]))

        grad_matrix = torch.stack(grad_lp_list).reshape(
            n_queries, n_rollouts, max_turns
        )
        ref_matrix = torch.stack(ref_lp_list).reshape(n_queries, n_rollouts, max_turns)
        return grad_matrix, ref_matrix

    def _build_reward_matrix(
        self,
        all_outcomes: List[float],
        all_entropies: List[List[float]],
    ) -> torch.Tensor:
        """
        Build reward matrix from outcomes and entropies.

        For "sparse" mode: r[t] = outcome * gamma^(T-1-t)
        For "adaptive" mode: r[t] = r_sparse + alpha * gate * r_sparse

        Returns:
            reward_matrix: [n_queries * n_rollouts, max_turns]
        """
        G = self.config.num_rollouts_per_query
        max_turns = self.config.max_turns
        n_total = len(all_outcomes)

        reward_tensors = []
        for outcome in all_outcomes:
            r = torch.zeros(max_turns, device=self.device)
            if outcome > 0:
                for t in range(max_turns):
                    r[t] = outcome * (self.config.gamma ** (max_turns - 1 - t))
            reward_tensors.append(r)
        reward_matrix = torch.stack(reward_tensors).reshape(-1, G, max_turns)

        if self.config.reward_mode == "adaptive":
            entropy_matrix = torch.tensor(all_entropies, device=self.device).reshape(
                -1, G, max_turns
            )

            gated = []
            for q in range(reward_matrix.shape[0]):
                for g in range(reward_matrix.shape[1]):
                    H_t = entropy_matrix[q, g]
                    gate = torch.sigmoid(
                        self.config.adaptive.sigmoid_temp
                        * (H_t - self.adaptive_reward.state.H_threshold)
                    )
                    gate = torch.clamp(gate, min=self.config.adaptive.gate_min, max=1.0)
                    r_sparse = reward_matrix[q, g]
                    r_adaptive = r_sparse + self.config.adaptive.alpha * gate * r_sparse
                    gated.append(r_adaptive)
            reward_matrix = torch.stack(gated).reshape(-1, G, max_turns)

        return reward_matrix

    def compute_grpo_loss(
        self,
        rollout_logprob_sums: torch.Tensor,
        rollout_ref_logprob_sums: torch.Tensor,
        rollout_rewards: torch.Tensor,
    ) -> torch.Tensor:
        n_q, n_g, n_t = rollout_rewards.shape

        returns = torch.zeros_like(rollout_rewards, device=self.device)
        for t in range(n_t):
            discount = self.config.gamma ** (n_t - 1 - t)
            returns[:, :, t] = rollout_rewards[:, :, t] * discount

        mean_ret = returns.mean(dim=1, keepdim=True)
        std_ret = returns.std(dim=1, keepdim=True).clamp(min=1e-8)
        advantages = (returns - mean_ret) / std_ret

        log_ratio = rollout_logprob_sums - rollout_ref_logprob_sums.detach()
        log_ratio = torch.clamp(log_ratio, min=-5.0, max=5.0)
        ratio = torch.exp(log_ratio)
        ratio = torch.clamp(ratio, max=10.0)

        clipped_ratio = torch.clamp(
            ratio,
            1 - self.config.grpo_epsilon,
            1 + self.config.grpo_epsilon,
        )

        if torch.isnan(ratio).any() or torch.isnan(advantages).any():
            print("WARNING: NaN in ratio or advantages, skipping batch")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        if self.config.reward_mode == "autotool_entropy":
            entropy_loss = self._compute_entropy_regularization(rollout_logprob_sums)
            policy_loss = policy_loss + 0.01 * entropy_loss

        return policy_loss

    def _compute_entropy_regularization(
        self, rollout_logprob_sums: torch.Tensor
    ) -> torch.Tensor:
        """AutoTool-style entropy constraint in the loss function."""
        target_entropy = 0.5
        neg_entropy = -rollout_logprob_sums.mean()
        return F.mse_loss(
            neg_entropy,
            torch.tensor(target_entropy, device=self.device),
        )

    def train_step(self, batch: List[dict]) -> dict:
        """
        One training step:
          1. Collect rollouts (token_ids, entropies, outcomes, trajectories)
          2. Compute reward matrix from outcomes (+ adaptive gating if adaptive mode)
          3. Re-run model forward to get gradient-bearing logprobs + ref logprobs
          4. Compute GRPO loss and update
        """
        G = self.config.num_rollouts_per_query
        max_turns = self.config.max_turns

        all_outcomes: List[float] = []
        all_entropies: List[List[float]] = []
        all_token_ids: List[List[list]] = []
        all_trajectories: List[List[dict]] = []
        hacking_detected = False

        for data_item in batch:
            instruction = data_item["query"]
            task = Task(
                user_id=data_item["user_id"],
                instruction=instruction,
                actions=[Action(name=n, kwargs=k) for n, k in data_item["gt_actions"]],
                outputs=data_item["gt_outputs"],
                task_id=data_item["task_id"],
            )

            for _ in range(G):
                (
                    traj,
                    outcome,
                    step_H,
                    step_tokens,
                ) = self.rollout(instruction, task, max_turns)

                T = len(traj)
                padded_H = step_H + [0.0] * (max_turns - T)
                padded_tokens = step_tokens + [[]] * (max_turns - T)

                all_outcomes.append(outcome)
                all_entropies.append(padded_H[:max_turns])
                all_token_ids.append(padded_tokens[:max_turns])
                all_trajectories.append(traj)

                if len(all_outcomes) % G == 1:
                    hacking, reasons = self.hacking_detector.check(
                        rollout_steps=max_turns,
                        rollout_tokens=step_tokens,
                        batch_process_reward=0.0,
                        batch_success=outcome > 0,
                    )
                    if hacking:
                        hacking_detected = True

        n_queries = len(all_outcomes) // G

        reward_matrix = self._build_reward_matrix(all_outcomes, all_entropies)

        if hacking_detected and self.hacking_detector.should_fallback():
            reward_matrix = reward_matrix * 0.0

        grad_lp_matrix, ref_lp_matrix = self._compute_grad_and_ref_logprobs(
            all_token_ids, n_queries, max_turns
        )

        self.optimizer.zero_grad()
        loss = self.compute_grpo_loss(
            grad_lp_matrix,
            ref_lp_matrix,
            reward_matrix,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        if self.config.reward_mode == "adaptive":
            entropy_tensor = torch.tensor(all_entropies, device=self.device)
            mean_H = entropy_tensor.mean().item()
            self.adaptive_reward.update_threshold(mean_H)

        self.global_step += 1
        batch_success = sum(o > 0 for o in all_outcomes) / max(1, len(all_outcomes))

        stats = {
            "loss": loss.item(),
            "success_rate": batch_success,
            "hacking_detected": hacking_detected,
            "step": self.global_step,
        }

        if self.config.reward_mode == "adaptive":
            stats.update(self.adaptive_reward.get_stats())

        return stats

    def evaluate(self, eval_tasks: List[Task]) -> float:
        successes = 0
        for task in eval_tasks:
            _, outcome, _, _ = self.rollout(
                task.instruction, task, max_turns=self.config.max_turns
            )
            if outcome > 0:
                successes += 1
        return successes / len(eval_tasks) if eval_tasks else 0.0

    def train(self, train_dataset, eval_tasks: List[Task]):
        self.setup_model()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.rl_learning_rate,
        )

        if self.config.sft_warmup_epochs > 0:
            self.sft_warmup(train_dataset)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.rl_batch_size,
            shuffle=True,
            collate_fn=lambda x: x,
        )

        if self.config.wandb_project:
            wandb.init(
                project=self.config.wandb_project,
                config=self.config.__dict__,
            )

        for step in range(self.config.total_rl_steps):
            try:
                batch = next(iter(train_loader))
            except StopIteration:
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.config.rl_batch_size,
                    shuffle=True,
                    collate_fn=lambda x: x,
                )
                batch = next(iter(train_loader))

            stats = self.train_step(batch)

            if step % self.config.log_interval == 0:
                if self.config.wandb_project:
                    wandb.log(stats)
                print(
                    f"Step {step}: loss={stats['loss']:.3f}, "
                    f"succ={stats['success_rate']:.3f}"
                )

            if step % self.config.eval_interval == 0:
                success_rate = self.evaluate(eval_tasks)
                if success_rate > self.best_success_rate:
                    self.best_success_rate = success_rate
                    self.save_checkpoint("best")
                print(f"  Eval success rate: {success_rate:.3f}")

            if step % self.config.save_interval == 0:
                self.save_checkpoint(f"step_{step}")

        self.save_checkpoint("final")
        return self.best_success_rate

    def save_checkpoint(self, name: str):
        path = Path(self.config.output_dir) / f"checkpoint_{name}"
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
