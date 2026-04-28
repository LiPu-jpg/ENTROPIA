"""
Token-level entropy estimation for LLM agent rollouts.

Computes entropy H_t = -sum(p * log(p)) on key decision tokens only:
- tool_name tokens (e.g., "search_airport")
- tool_params tokens (e.g., '{"city": "NYC"}')
- stop/termination tokens (e.g., <|im_end|>, </action>)
- action_type tokens (e.g., click, fill)

NOT computed on: filler tokens, formatting tokens, natural language tokens.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class EntropyResult:
    """Per-step entropy estimates."""

    step_entropies: torch.Tensor  # [num_steps] mean entropy per step
    token_entropies: List[torch.Tensor]  # per-token entropies for each step
    key_mask: List[torch.Tensor]  # which tokens were considered "key"
    traj_entropy: float  # trajectory-level mean
    is_valid: torch.Tensor  # [num_steps] whether step had valid key tokens


class EntropyEstimator:
    """
    Computes token-level entropy on key decision tokens.

    The entropy of a token distribution p over vocabulary V is:
        H(p) = -sum_{v in V} p(v) * log(p(v))

    Key insight: ONLY compute on key tokens, not all tokens.
    Computing entropy on a "the" token is meaningless noise.
    """

    def __init__(self, key_token_ids: Dict[str, List[int]], eps: float = 1e-8):
        """
        Args:
            key_token_ids: Mapping of token type -> list of token IDs considered "key".
                           e.g., {"tool_name": [id_search_airport, id_search_flight, ...]}
            eps: Small epsilon to prevent log(0).
        """
        self.key_token_ids = key_token_ids
        self.key_id_set = set()
        for ids in key_token_ids.values():
            self.key_id_set.update(ids)
        self.eps = eps

    def compute_step_entropy(
        self,
        logits: torch.Tensor,  # [seq_len, vocab_size] or [batch, seq_len, vocab_size]
        token_ids: torch.Tensor,  # [seq_len] or [batch, seq_len] - actual generated token IDs
    ) -> torch.Tensor:
        """
        Compute entropy for ALL tokens in a step, then mask to key tokens.
        Returns mean entropy over key tokens for this step.

        Args:
            logits: Model output logits at this step
            token_ids: Actually generated token IDs (to identify key tokens)
        Returns:
            mean_entropy: Scalar mean entropy over key tokens
        """
        if logits.dim() == 3:
            logits = logits.squeeze(0)
        if token_ids.dim() == 2:
            token_ids = token_ids.squeeze(0)

        seq_len = min(logits.shape[0], token_ids.shape[0])
        logits = logits[:seq_len]
        token_ids = token_ids[:seq_len]

        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs.clamp(min=self.eps))
        entropy_per_token = -(probs * log_probs).sum(dim=-1)

        is_key = torch.tensor(
            [tid.item() in self.key_id_set for tid in token_ids], device=logits.device
        ).float()

        if is_key.sum() == 0:
            return torch.tensor(0.0, device=logits.device), False

        key_entropy = (entropy_per_token * is_key).sum() / is_key.sum()
        return key_entropy, True

    def compute_trajectory_entropy(
        self,
        rollout_logits: List[torch.Tensor],  # List of [seq_len, vocab_size] per step
        rollout_token_ids: List[torch.Tensor],  # List of [seq_len] per step
    ) -> EntropyResult:
        """
        Compute per-step and trajectory-level entropy.

        Args:
            rollout_logits: Logits for each step in the rollout
            rollout_token_ids: Generated token IDs for each step
        Returns:
            EntropyResult with per-step and trajectory-level entropy
        """
        step_entropies = []
        token_entropies = []
        key_masks = []
        valids = []

        for step_logits, step_tokens in zip(rollout_logits, rollout_token_ids):
            # Per-token entropy
            probs = F.softmax(step_logits, dim=-1)
            log_probs = torch.log(probs.clamp(min=self.eps))
            per_token_entropy = -(probs * log_probs).sum(dim=-1)

            is_key = torch.tensor(
                [tid.item() in self.key_id_set for tid in step_tokens],
                device=step_logits.device,
            ).float()

            token_entropies.append(per_token_entropy)
            key_masks.append(is_key)

            if is_key.sum() > 0:
                step_entropy = (per_token_entropy * is_key).sum() / is_key.sum()
                step_entropies.append(step_entropy)
                valids.append(True)
            else:
                step_entropies.append(torch.tensor(0.0, device=step_logits.device))
                valids.append(False)

        if step_entropies:
            step_entropies_tensor = torch.stack(step_entropies)
            traj_entropy = step_entropies_tensor[torch.tensor(valids)].mean().item()
        else:
            step_entropies_tensor = torch.tensor([])
            traj_entropy = 0.0

        return EntropyResult(
            step_entropies=step_entropies_tensor,
            token_entropies=token_entropies,
            key_mask=key_masks,
            traj_entropy=traj_entropy,
            is_valid=torch.tensor(valids),
        )


class UncertaintyEstimator:
    """
    Extended uncertainty estimation combining multiple metrics.
    Reference: SELAUR uses entropy + least-confidence + margin.

    This is optional - direction A primarily uses pure entropy.
    The combined approach can be used as a robustness check.
    """

    def __init__(self, method: str = "entropy"):
        """
        Args:
            method: "entropy" | "least_confidence" | "margin" | "combined"
        """
        self.method = method

    def compute(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute uncertainty score from logits."""
        probs = F.softmax(logits, dim=-1)

        if self.method == "entropy":
            log_probs = torch.log(probs.clamp(min=1e-8))
            return -(probs * log_probs).sum(dim=-1)

        elif self.method == "least_confidence":
            # 1 - max_prob
            max_probs = probs.max(dim=-1).values
            return 1.0 - max_probs

        elif self.method == "margin":
            # 1 - (top1_prob - top2_prob)
            top2 = probs.topk(2, dim=-1).values
            return 1.0 - (top2[:, 0] - top2[:, 1])

        elif self.method == "combined":
            # SELAUR-style: average of all three
            log_probs = torch.log(probs.clamp(min=1e-8))
            h = -(probs * log_probs).sum(dim=-1)
            lc = 1.0 - probs.max(dim=-1).values
            top2 = probs.topk(2, dim=-1).values
            m = 1.0 - (top2[:, 0] - top2[:, 1])
            return (h + lc + m) / 3.0

        else:
            raise ValueError(f"Unknown uncertainty method: {self.method}")
