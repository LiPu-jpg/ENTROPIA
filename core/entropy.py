"""
Token 级别熵估计，用于 LLM Agent 推理过程。

仅对关键决策 token 计算熵 H_t = -sum(p * log(p))：
- tool_name token（如 "search_airport"）
- tool_params token（如 '{"city": "NYC"}'）
- stop/终止 token（如 <|im_end|>, </action>）
- action_type token（如 click, fill）

不计算的对象：填充 token、格式化 token、自然语言 token。
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class EntropyResult:
    """每步熵估计结果。"""

    step_entropies: torch.Tensor  # [num_steps] 每步的平均熵
    token_entropies: List[torch.Tensor]  # 每步的逐 token 熵
    key_mask: List[torch.Tensor]  # 哪些 token 被视为"关键"token
    traj_entropy: float  # 轨迹级别的平均熵
    is_valid: torch.Tensor  # [num_steps] 该步是否包含有效的关键 token


class EntropyEstimator:
    """
    在关键决策 token 上计算 token 级别熵。

    token 分布 p 在词表 V 上的熵为：
        H(p) = -sum_{v in V} p(v) * log(p(v))

    核心洞察：仅在关键 token 上计算，而非所有 token。
    在 "the" 这样的 token 上计算熵是没有意义的噪声。
    """

    def __init__(self, key_token_ids: Dict[str, List[int]], eps: float = 1e-8):
        """
        参数：
            key_token_ids: token 类型 -> 被视为"关键"的 token ID 列表的映射。
                           例如 {"tool_name": [id_search_airport, id_search_flight, ...]}
            eps: 防止 log(0) 的小常数。
        """
        self.key_token_ids = key_token_ids
        self.key_id_set = set()
        for ids in key_token_ids.values():
            self.key_id_set.update(ids)
        self.eps = eps

    def compute_step_entropy(
        self,
        logits: torch.Tensor,  # [seq_len, vocab_size] 或 [batch, seq_len, vocab_size]
        token_ids: torch.Tensor,  # [seq_len] 或 [batch, seq_len] - 实际生成的 token ID
    ) -> torch.Tensor:
        """
        计算一步中所有 token 的熵，然后掩码至关键 token。
        返回该步关键 token 的平均熵。

        参数：
            logits: 该步的模型输出 logits
            token_ids: 实际生成的 token ID（用于识别关键 token）
        返回：
            mean_entropy: 关键 token 上的标量平均熵
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
        rollout_logits: List[torch.Tensor],  # 每步的 [seq_len, vocab_size] 列表
        rollout_token_ids: List[torch.Tensor],  # 每步的 [seq_len] 列表
    ) -> EntropyResult:
        """
        计算每步和轨迹级别的熵。

        参数：
            rollout_logits: 推理过程中每步的 logits
            rollout_token_ids: 每步生成的 token ID
        返回：
            包含每步和轨迹级别熵的 EntropyResult
        """
        step_entropies = []
        token_entropies = []
        key_masks = []
        valids = []

        for step_logits, step_tokens in zip(rollout_logits, rollout_token_ids):
            # 逐 token 熵
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
    扩展的不确定性估计，融合多种度量。
    参考：SELAUR 使用 熵 + 最小置信度 + 间距。

    这是可选的 —— 方向 A 主要使用纯熵。
    组合方法可用于鲁棒性验证。
    """

    def __init__(self, method: str = "entropy"):
        """
        参数：
            method: "entropy" | "least_confidence" | "margin" | "combined"
        """
        self.method = method

    def compute(self, logits: torch.Tensor) -> torch.Tensor:
        """从 logits 计算不确定性分数。"""
        probs = F.softmax(logits, dim=-1)

        if self.method == "entropy":
            log_probs = torch.log(probs.clamp(min=1e-8))
            return -(probs * log_probs).sum(dim=-1)

        elif self.method == "least_confidence":
            # 1 - 最大概率
            max_probs = probs.max(dim=-1).values
            return 1.0 - max_probs

        elif self.method == "margin":
            # 1 - (第一大概率 - 第二大概率)
            top2 = probs.topk(2, dim=-1).values
            return 1.0 - (top2[:, 0] - top2[:, 1])

        elif self.method == "combined":
            # SELAUR 风格：三者的平均值
            log_probs = torch.log(probs.clamp(min=1e-8))
            h = -(probs * log_probs).sum(dim=-1)
            lc = 1.0 - probs.max(dim=-1).values
            top2 = probs.topk(2, dim=-1).values
            m = 1.0 - (top2[:, 0] - top2[:, 1])
            return (h + lc + m) / 3.0

        else:
            raise ValueError(f"Unknown uncertainty method: {self.method}")
