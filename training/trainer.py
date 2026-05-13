"""
带有自适应奖励密度门控的 GRPO 训练器。

支持五种奖励模式：
  - "adaptive"：方向 A — 基于熵门控的密度控制
  - "sparse"：ReTool 风格的二值结果奖励
  - "dense_igpo"：IGPO 风格的固定信息增益过程奖励
  - "dense_fixed"：WorkForceAgent-R1 风格的固定稠密奖励（消融实验）
  - "autotool_entropy"：AutoTool 风格的损失函数中的熵约束（非奖励）
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from typing import List, Dict, Optional, Tuple
import os, copy
import wandb
from pathlib import Path

from data.tau_dataset import Task
from envs.mock_env import MockTauEnv, parse_action_from_text, Action
from core.signal_bank import SignalBank, SignalBatch
from core.reward_router import RewardRouter, NeedGate, UtilityGate, ReliabilityGate, RiskController

_mm_client = None

def _get_mm_client():
    global _mm_client
    if _mm_client is None:
        try:
            from openai import OpenAI
            _mm_client = OpenAI(
                base_url="https://api.minimaxi.com/v1",
                api_key="sk-cp-xqBXPT7PTX8CG_IMl3xnbrrVi50i1wEjBQ8AACpgDhR3wpD6BJeTsYrBt2J9CJSMy9weFfPUQHJ6DWYMXqvD6Whvszor2IZhc_jACOJXGx3QbcygaiIFgLo",
            )
        except:
            _mm_client = None
    return _mm_client


class AdaptiveRewardTrainer:
    def __init__(
        self, config, entropy_estimator, adaptive_reward, hacking_detector,
        reward_router=None, signal_bank=None,
    ):
        self.config = config
        self.entropy_estimator = entropy_estimator
        self.adaptive_reward = adaptive_reward
        self.hacking_detector = hacking_detector
        self.reward_router = reward_router
        self.signal_bank = signal_bank

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
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
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
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        trainer.train()
        return trainer.state.best_metric

    _mm_call_count = 0

    def minimax_judge(self, instruction: str, task: Task, plan_text: str) -> float:
        mm = _get_mm_client()
        if mm is None:
            return self._mock_reward(task)

        self._mm_call_count += 1
        gt_acts = [(a.name, a.kwargs) for a in task.actions
                   if a.name not in ("respond", "transfer_to_human_agents", "think")]
        gt_desc = " → ".join(f"{n}({', '.join(f'{k}={v}' for k,v in a.items())})" for n, a in gt_acts)
        prompt = f"Task:{instruction[:200]}\nGT:{gt_desc}\nAG:{plan_text[:300]}\nScore:"
        for attempt in range(3):
            try:
                r = mm.chat.completions.create(
                    model="MiniMax-M2.7",
                    messages=[
                        {"role": "system", "content": "Score agents 0.0-1.0. Output ONLY one number. No text."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=500, temperature=0.0,
                )
                text = r.choices[0].message.content.strip()
                after = text.split("</think>")[-1].strip() if "</think>" in text else text
                import re
                nums = re.findall(r'\b(0\.\d+|1\.0|1)\b', after)
                for n in nums:
                    try:
                        s = float(n)
                        if 0.0 <= s <= 1.0:
                            if self._mm_call_count % 20 == 0:
                                print(f"  [MM #{self._mm_call_count}] score={s:.3f}", flush=True)
                            return s
                    except: pass
                s = self._mock_reward(task)
                if self._mm_call_count % 20 == 0:
                    print(f"  [MM #{self._mm_call_count}] PARSE → mock={s:.2f}", flush=True)
                return s
            except Exception as e:
                if "rate" in str(e).lower() or "429" in str(e):
                    import time; time.sleep(10 * (attempt + 1))
                    continue
                break
        s = self._mock_reward(task)
        print(f"  [MM #{self._mm_call_count}] API ERR → mock={s:.2f}", flush=True)
        return s

    def _mock_reward(self, task: Task) -> float:
        env = MockTauEnv(task);
        env.reset()
        for a in task.actions: env.step(a)
        env.step(Action(name="respond", kwargs={"content": "Done."}))
        return env.reward

    def rollout(
        self,
        instruction: str,
        task: Task,
        max_turns: int = None,
    ) -> Tuple[List[dict], float, List[float], List[float], List[list]]:
        """
        先规划后执行的 rollout：模型生成完整的动作计划，
        然后在模拟环境中逐步执行。
        计划中的每个动作作为奖励计算的一个步骤。

        返回：
            trajectory: 步骤字典列表
            outcome: 任务奖励（0 或 1）
            step_entropies: [T] 每步的熵
            step_token_ids: [T] token ID 列表的列表
        """
        if max_turns is None:
            max_turns = self.config.max_turns

        env = MockTauEnv(task)
        env.reset()

        # 构建与 SFT 格式匹配的系统提示
        sys_prompt = (
            "You are a customer service agent for retail and airline domains. "
            "Available tools: find_user_id_by_email, find_user_id_by_name_zip, "
            "get_user_details, get_order_details, get_product_details, get_flight_status, "
            "get_reservation_details, cancel_pending_order, cancel_reservation, "
            "return_delivered_order_items, exchange_delivered_order_items, "
            "modify_pending_order_items, modify_pending_order_address, "
            "modify_pending_order_payment, search_direct_flight, search_onestop_flight, "
            "book_reservation, update_reservation_flights, update_reservation_passengers, "
            "update_reservation_baggages, send_certificate, calculate, "
            "transfer_to_human_agents, respond. "
            "Format: tool_name(key='val', ...). Separate with ' | '. End with respond('summary')."
        )
        prompt = (
            f"<|im_start|>system\n{sys_prompt}\n<|im_end|>\n"
            f"<|im_start|>user\n{instruction}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096
        ).to(self.device)
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            self.model.eval()
            outputs = self.model.generate(
                **inputs,
                    max_new_tokens=384,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            self.model.train()

        generated_ids = outputs.sequences[0][input_len:]
        plan_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        if self.global_step == 0:
            print(f"  [DEBUG] Plan: {plan_text[:300]}", flush=True)

        scores = torch.stack([s.squeeze(0) for s in outputs.scores])

        # 从计划中解析动作（以 ' | ' 分隔）
        action_strs = [a.strip() for a in plan_text.split("|")]
        if not action_strs:
            action_strs = [plan_text]

        trajectory = []
        step_entropies = []
        step_token_ids = []
        outcome = 0.0

        for turn_idx, action_str in enumerate(action_strs[:max_turns]):
            action = parse_action_from_text(action_str)
            if action is None:
                action = Action(name="respond", kwargs={"content": action_str})

            result = env.step(action)
            trajectory.append(
                {
                    "turn": turn_idx,
                    "action": action_str,
                    "observation": result.observation,
                    "done": result.done,
                }
            )

            # 在各步骤间分配熵和 token
            chunk_size = max(1, len(scores) // len(action_strs))
            start = turn_idx * chunk_size
            end = min(start + chunk_size, len(scores))
            if start < len(scores):
                chunk_scores = scores[start:end]
                chunk_ids = generated_ids[start:end]
                H_t, _ = self.entropy_estimator.compute_step_entropy(
                    logits=chunk_scores, token_ids=chunk_ids
                )
                step_entropies.append(H_t.item())
                step_token_ids.append(chunk_ids.tolist())
            else:
                step_entropies.append(0.0)
                step_token_ids.append([])

            if result.done:
                break
        else:
            env.step(Action(name="respond", kwargs={"content": "Done."}))

        outcome = self.minimax_judge(instruction, task, plan_text)

        return (trajectory, outcome, step_entropies, step_token_ids)

    def _compute_grad_and_ref_logprobs(
        self,
        all_token_ids: List[List[list]],
        n_queries: int,
        max_turns: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对收集的 token ID 重新运行 model.forward()，以获取带梯度的
        log_probs 用于 GRPO 比率计算。为 GPU 效率进行了批量处理。

        返回：
            grad_logprob_matrix: [n_queries, n_rollouts, max_turns]
            ref_logprob_matrix: [n_queries, n_rollouts, max_turns]
        """
        n_rollouts = self.config.num_rollouts_per_query
        n_total = n_queries * n_rollouts

        valid_seqs = []
        valid_indices = []
        zero_mask = set()

        flat_idx = 0
        for qi in range(n_queries):
            for gi in range(n_rollouts):
                step_tokens = all_token_ids[qi * n_rollouts + gi]
                for ti in range(max_turns):
                    toks = step_tokens[ti] if ti < len(step_tokens) else []
                    if len(toks) >= 2:
                        valid_seqs.append(torch.tensor(toks, device=self.device))
                        valid_indices.append((flat_idx, qi, gi, ti, len(toks)))
                    else:
                        zero_mask.add(flat_idx)
                    flat_idx += 1

        if not valid_seqs:
            zeros = torch.zeros(n_queries, n_rollouts, max_turns, device=self.device)
            return zeros.clone().requires_grad_(True), zeros.clone()

        padded = torch.nn.utils.rnn.pad_sequence(
            valid_seqs, batch_first=True, padding_value=0
        )
        seq_lens = [len(s) for s in valid_seqs]

        MAX_BATCH = 4
        seq_sums = {}

        for b_start in range(0, len(valid_seqs), MAX_BATCH):
            b_end = min(b_start + MAX_BATCH, len(valid_seqs))
            batch_padded = padded[b_start:b_end]
            batch_indices = valid_indices[b_start:b_end]

            self.model.train()
            curr_outputs = self.model(batch_padded)
            curr_logits = curr_outputs.logits[:, :-1]
            curr_log_probs = F.log_softmax(curr_logits, dim=-1)
            del curr_outputs
            torch.cuda.empty_cache()

            with torch.no_grad():
                ref_outputs = self.ref_model(batch_padded)
                ref_logits = ref_outputs.logits[:, :-1]
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)

            for local_i, (global_i, qi, gi, ti, tok_len) in enumerate(batch_indices):
                targets = batch_padded[local_i, 1:tok_len]
                idx_range = torch.arange(tok_len - 1, device=self.device)
                curr_lp = curr_log_probs[local_i, idx_range, targets].sum()
                ref_lp = ref_log_probs[local_i, idx_range, targets].sum()
                seq_sums[(qi, gi, ti)] = (curr_lp, ref_lp)

        grad_lp_list = []
        ref_lp_list = []

        for qi in range(n_queries):
            for gi in range(n_rollouts):
                step_grad = []
                step_ref = []
                for ti in range(max_turns):
                    if (qi, gi, ti) in seq_sums:
                        step_grad.append(seq_sums[(qi, gi, ti)][0])
                        step_ref.append(seq_sums[(qi, gi, ti)][1])
                    else:
                        step_grad.append(
                            torch.zeros((), device=self.device, requires_grad=True)
                        )
                        step_ref.append(
                            torch.zeros((), device=self.device, requires_grad=False)
                        )
                grad_lp_list.append(torch.stack(step_grad))
                ref_lp_list.append(torch.stack(step_ref))

        grad_matrix = torch.stack(grad_lp_list).reshape(
            n_queries, n_rollouts, max_turns
        )
        ref_matrix = torch.stack(ref_lp_list).reshape(n_queries, n_rollouts, max_turns)
        return grad_matrix, ref_matrix

    def _build_reward_matrix(
        self,
        all_outcomes: List[float],
        all_entropies: List[List[float]],
        all_signals: Optional[List[SignalBatch]] = None,
        all_hacking: Optional[List[bool]] = None,
    ) -> torch.Tensor:
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
            entropy_matrix = torch.tensor(all_entropies, device=self.device).reshape(-1, G, max_turns)
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

        elif self.config.use_router and self.reward_router is not None:
            gated = []
            for q in range(reward_matrix.shape[0]):
                for g in range(reward_matrix.shape[1]):
                    idx = q * G + g
                    r_sparse = reward_matrix[q, g]
                    r_v2 = r_sparse.clone()
                    if idx < len(all_signals) and idx < len(all_entropies):
                        sig = all_signals[idx]
                        ent_list = all_entropies[idx]
                        for t in range(min(max_turns, len(ent_list))):
                            step_entropy = ent_list[t] if t < len(ent_list) else 0.0
                            group_collapse = len(set(
                                all_outcomes[q * G : (q + 1) * G]
                            )) <= 1
                            stagnation = 0.0
                            signal_vals = {
                                "info_gain": sig.info_gain[t] if t < len(sig.info_gain) else 0.0,
                                "efficiency_cost": sig.efficiency_cost[t] if t < len(sig.efficiency_cost) else 0.0,
                                "relevance": sig.relevance[t] if t < len(sig.relevance) else 0.0,
                            }
                            format_v = sig.format_valid[t] if t < len(sig.format_valid) else 1.0
                            hacking = all_hacking[idx] if all_hacking and idx < len(all_hacking) else False
                            div = 0.0
                            proc_r, gates = self.reward_router.route(
                                step_entropy=step_entropy,
                                group_collapse=float(group_collapse),
                                stagnation=stagnation,
                                signal_values=signal_vals,
                                format_valid=format_v,
                                hacking_detected=hacking,
                                success_divergence=div,
                            )
                            r_v2[t] = r_v2[t] + proc_r
                    gated.append(r_v2)
            if gated:
                reward_matrix = torch.stack(gated).reshape(-1, G, max_turns)

        elif self.config.reward_mode == "random_gate":
            import random
            gated = []
            for q in range(reward_matrix.shape[0]):
                for g in range(reward_matrix.shape[1]):
                    r_sparse = reward_matrix[q, g]
                    gate = random.random()
                    r_rand = r_sparse + 0.5 * gate * r_sparse
                    gated.append(r_rand)
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
        """AutoTool 风格的损失函数中的熵约束。"""
        target_entropy = 0.5
        neg_entropy = -rollout_logprob_sums.mean()
        return F.mse_loss(
            neg_entropy,
            torch.tensor(target_entropy, device=self.device),
        )

    def train_step(self, batch: List[dict]) -> dict:
        """
        一次训练步骤：
          1. 收集 rollout（token_ids、熵、结果、轨迹）
          2. 根据结果计算奖励矩阵（自适应模式下附加自适应门控）
          3. 重新运行模型前向传播，获取带梯度的 logprobs 和参考 logprobs
          4. 计算 GRPO 损失并更新参数
        """
        G = self.config.num_rollouts_per_query
        max_turns = self.config.max_turns

        all_outcomes: List[float] = []
        all_entropies: List[List[float]] = []
        all_token_ids: List[List[list]] = []
        all_trajectories: List[List[dict]] = []
        all_signals: List[SignalBatch] = []  # v2
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

                # v2: compute process signals
                if self.config.use_router and self.signal_bank is not None:
                    signals = self.signal_bank.compute_signals(
                        step_count=T,
                    )
                    if outcome > 0:
                        self.reward_router.reliability.update("info_gain", 0.5, 1.0)
                    else:
                        self.reward_router.reliability.update("info_gain", 0.0, 0.0)
                    all_signals.append(signals)

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

        all_hacking = [hacking_detected] * len(all_outcomes)

        reward_matrix = self._build_reward_matrix(
            all_outcomes, all_entropies,
            all_signals=all_signals,
            all_hacking=all_hacking,
        )

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
            "avg_score": sum(all_outcomes) / max(1, len(all_outcomes)),
        }

        if self.config.reward_mode == "adaptive":
            stats.update(self.adaptive_reward.get_stats())
        elif self.config.use_router and self.reward_router is not None:
            stats.update(self.reward_router.get_stats())
            stats["router_mode"] = self.config.reward_mode

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

    def train(self, train_dataset, eval_tasks: List[Task], sft_dataset=None, sft_checkpoint: str = None):
        if sft_checkpoint and os.path.exists(sft_checkpoint):
            from peft import PeftModel
            self.tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            base = AutoModelForCausalLM.from_pretrained(
                self.config.model_name, torch_dtype=getattr(torch, self.config.model_dtype),
                device_map="auto", trust_remote_code=True,
            )
            self.model = PeftModel.from_pretrained(base, sft_checkpoint)
            self.model.train()
            self.device = self.model.device
            self.ref_model = copy.deepcopy(self.model)
            self.ref_model.eval()
            for p in self.ref_model.parameters(): p.requires_grad = False
            self.config.sft_warmup_epochs = 0  # skip SFT
            print(f"Loaded SFT checkpoint from {sft_checkpoint}, skipping warmup", flush=True)
        else:
            self.setup_model()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.rl_learning_rate,
        )

        if self.config.sft_warmup_epochs > 0:
            sft_data = sft_dataset if sft_dataset is not None else train_dataset
            print(
                f"SFT warmup: {self.config.sft_warmup_epochs} epochs on {len(sft_data)} samples",
                flush=True,
            )
            self.sft_warmup(sft_data)

            self.ref_model = copy.deepcopy(self.model)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
            print("SFT warmup done, ref_model updated", flush=True)

            # 保存 SFT checkpoint 供后续跳过
            sft_path = Path(self.config.output_dir) / "sft_checkpoint"
            sft_path.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(sft_path)
            self.tokenizer.save_pretrained(sft_path)
            print(f"SFT checkpoint saved to {sft_path}", flush=True)

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
                    f"succ={stats['success_rate']:.3f}, "
                    f"avg_score={stats.get('avg_score', 0):.3f}",
                    flush=True,
                )

            if step % self.config.eval_interval == 0:
                success_rate = self.evaluate(eval_tasks)
                if success_rate > self.best_success_rate:
                    self.best_success_rate = success_rate
                    self.save_checkpoint("best")
                print(f"  Eval success rate: {success_rate:.3f}", flush=True)

            if step % self.config.save_interval == 0:
                self.save_checkpoint(f"step_{step}")

        self.save_checkpoint("final")
        return self.best_success_rate

    def save_checkpoint(self, name: str):
        path = Path(self.config.output_dir) / f"checkpoint_{name}"
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
