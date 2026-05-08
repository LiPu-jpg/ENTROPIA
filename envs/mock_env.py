"""
用于训练 Direction A 的 Mock τ-Bench 环境。

实现了 τ-Bench Env API：reset → step → reward。
不使用语言模型用户模拟器——观察结果是预脚本化的，
以保持训练快速且确定性。
"""

import re
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from data.tau_dataset import Task, Action, TAU_BENCH_TOOLS


@dataclass
class EnvStepResult:
    observation: str
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


@dataclass
class EnvResetResult:
    observation: str
    info: dict = field(default_factory=dict)


def parse_action_from_text(action_text: str) -> Optional[Action]:
    """
    从模型生成的文本中提取 τ-Bench Action。

    尝试多种解析策略：
    1. JSON: {"name": "tool", "kwargs": {...}}
    2. 函数调用: tool_name(key=val, ...)
    3. XML 风格: <action>tool_name</action>
    4. 回退：从文本中提取第一个已知工具名称
    """
    action_text = action_text.strip()
    if action_text.startswith("(") and action_text.endswith(")"):
        action_text = action_text[1:-1].strip()

    # 策略 1：JSON 对象
    if action_text.startswith("{"):
        try:
            obj = json.loads(action_text)
            return Action(name=obj["name"], kwargs=obj.get("kwargs", {}))
        except (json.JSONDecodeError, KeyError):
            pass

    # 策略 2：Python 风格函数调用: tool_name(key=val, ...)
    func_match = re.match(r"^(\w+)\s*\((.+)\)\s*$", action_text, re.DOTALL)
    if func_match:
        name = func_match.group(1)
        if name in TAU_BENCH_TOOLS:
            kwargs = _parse_python_kwargs(func_match.group(2))
            return Action(name=name, kwargs=kwargs)

    # 策略 3：XML 风格 <action>tool_name</action> 或类似格式
    for tool_name in TAU_BENCH_TOOLS:
        if f"<action>{tool_name}</action>" in action_text:
            return Action(name=tool_name, kwargs={})

    # 策略 4：在文本中找到的第一个已知工具名称
    for tool_name in sorted(TAU_BENCH_TOOLS, key=len, reverse=True):
        if tool_name in action_text:
            return Action(name=tool_name, kwargs={})

    return None


def _parse_python_kwargs(kwargs_str: str) -> Dict[str, Any]:
    """解析 Python 风格的 kwargs 字符串，格式如: key1='val1', key2=['a','b']"""
    kwargs = {}

    # 按逗号分隔，但忽略括号/引号内的逗号
    pairs = _smart_split(kwargs_str, ",")

    for pair in pairs:
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue

        key, _, val_str = pair.partition("=")
        key = key.strip()
        val_str = val_str.strip()

        # 解析值
        val = _parse_python_value(val_str)
        kwargs[key] = val

    return kwargs


def _parse_python_value(s: str) -> Any:
    """解析单个 Python 风格的值。"""
    s = s.strip()

    # 字符串
    if (s.startswith('"') and s.endswith('"')) or (
        s.startswith("'") and s.endswith("'")
    ):
        return s[1:-1]

    # 列表
    if s.startswith("[") and s.endswith("]"):
        items = _smart_split(s[1:-1], ",")
        return [_parse_python_value(item.strip()) for item in items if item.strip()]

    # 字典
    if s.startswith("{") and s.endswith("}"):
        return _parse_python_kwargs(s[1:-1])

    # 数字
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass

    # 回退：字符串
    return s


# ─── 参数格式检查 ───

PARAM_PATTERNS = {
    "order_id": r"^#W\d+$",
    "reservation_id": r"^[A-Z0-9]{6}$",
    "email": r"^[a-z.]+@example\.com$",
    "zip": r"^\d{5}$",
    "product_id": r"^\d+$",
    "payment_method_id": r"^(credit_card|paypal|gift_card)_\d+$",
    "user_id": r"^[a-z_]+$",
    "item_ids": r"^\[.*\]$|^\d+$",
    "new_item_ids": r"^\[.*\]$|^\d+$",
    "first_name": r"^[A-Z][a-z]+$",
    "last_name": r"^[A-Z][a-z]+$",
    "cabin": r"^[a-z_]+$",
}


def _check_param_format(param_name: str, value) -> bool:
    if not value:
        return False
    pattern = PARAM_PATTERNS.get(param_name)
    if not pattern:
        return isinstance(value, (str, int, float, bool, list))
    s = str(value)
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        s = "[" + ",".join(value) + "]"
    return bool(re.match(pattern, s))


def _smart_split(text: str, sep: str) -> List[str]:
    parts = []
    current = []
    depth_bracket = 0
    depth_brace = 0
    in_quote = None

    for ch in text:
        if ch == in_quote:
            in_quote = None
        elif in_quote is None and ch in ('"', "'"):
            in_quote = ch
        elif in_quote is None:
            if ch == "[":
                depth_bracket += 1
            elif ch == "]":
                depth_bracket -= 1
            elif ch == "{":
                depth_brace += 1
            elif ch == "}":
                depth_brace -= 1

        if ch == sep and not in_quote and depth_bracket == 0 and depth_brace == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)

    if current:
        parts.append("".join(current))

    return parts


class MockTauEnv:
    """
    用于训练的 Mock τ-Bench 环境。

    API 与 τ-Bench 的 Env 类匹配：
    - reset(task_index) → EnvResetResult
    - step(action: Action) → EnvStepResult

    奖励：二进制的（1.0 = 任务正确完成，0.0 = 其他情况）。
    观察结果是按动作类型预脚本化的。
    """

    OBSERVATIONS = {
        "find_user_id_by_email": "User found. User ID: user_{user_id}",
        "find_user_id_by_name_zip": "User found. User ID: user_{user_id}",
        "find_user_by_email": "User found. User ID: user_{user_id}",
        "find_user_by_name": "User found. User ID: user_{user_id}",
        "find_user_by_contact": "User found. User ID: user_{user_id}",
        "get_user_details": "User details: name, email, zip code, payment methods on file.",
        "get_order_details": "Order details: items, status, shipping address, payment method.",
        "get_product_details": "Product details: name, price, specifications, availability.",
        "get_flight_status": "Flight status: on time, departure 14:30, gate B12.",
        "get_reservation_details": "Reservation details: flight, passengers, baggage, status.",
        "list_all_product_types": "Available product categories: Electronics, Home & Kitchen, Sports, Books, Clothing, Toys.",
        "modify_pending_order_items": "Order items updated successfully.",
        "modify_pending_order_address": "Shipping address updated successfully.",
        "modify_pending_order_payment": "Payment method updated successfully.",
        "modify_user_address": "User address updated successfully.",
        "cancel_pending_order": "Order cancelled successfully.",
        "cancel_order": "Order cancelled successfully.",
        "cancel_reservation": "Reservation cancelled successfully.",
        "return_delivered_order_items": "Return initiated. Refund will be processed within 5-7 business days.",
        "exchange_delivered_order_items": "Exchange initiated. New items will be shipped within 3-5 business days.",
        "search_direct_flight": "Direct flights found: 3 options available.",
        "search_onestop_flight": "One-stop flights found: 5 options available.",
        "book_reservation": "Reservation booked successfully.",
        "update_reservation_baggages": "Baggage updated successfully.",
        "update_reservation_flights": "Flight updated successfully.",
        "update_reservation_passengers": "Passenger info updated successfully.",
        "send_certificate": "Certificate sent to email.",
        "calculate": "Calculation completed.",
        "think": "(Internal reasoning recorded)",
        "transfer_to_human_agents": "Transferring to human agent...",
        "respond": "###STOP###",
}

    def __init__(self, task: Task):
        self.task = task
        self.gt_actions = task.actions
        self.taken_actions: List[Action] = []
        self.current_gt_index = 0
        self.done = False
        self.reward = 0.0

    def reset(self) -> EnvResetResult:
        self.taken_actions = []
        self.current_gt_index = 0
        self.done = False
        self.reward = 0.0

        return EnvResetResult(
            observation=self.task.instruction,
            info={"task": self.task, "source": "user"},
        )

    def step(self, action: Action) -> EnvStepResult:
        self.taken_actions.append(action)

        obs = self.OBSERVATIONS.get(
            action.name,
            f"Action '{action.name}' completed.",
        ).format(user_id=self.task.user_id)

        if action.name == "respond" or action.name == "transfer_to_human_agents":
            self.done = True
            self.reward = self._compute_reward()
            obs += f"\nTask {'succeeded' if self.reward > 0 else 'failed'}."

        return EnvStepResult(
            observation=obs,
            reward=self.reward,
            done=self.done,
            info={"task": self.task, "source": action.name},
        )

    def _compute_reward(self) -> float:
        gt_actions = [a for a in self.gt_actions
                      if a.name not in ("respond", "transfer_to_human_agents", "think")]
        if not gt_actions:
            return 1.0
        taken = [a for a in self.taken_actions
                 if a.name not in ("respond", "transfer_to_human_agents", "think")]

        # L1: 工具名 (0.10) — SFT 已背熟，当保底
        gt_names = {a.name for a in gt_actions}
        taken_names = {a.name for a in taken}
        hit = sum(1 for n in gt_names if n in taken_names)
        l1 = 0.10 * hit / len(gt_names)

        # L2: 参数匹配 (0.50) — 比 GT 参数值
        score_params = 0.0
        for gt in gt_actions:
            for k, gt_val in gt.kwargs.items():
                for t in taken:
                    if t.name != gt.name or k not in t.kwargs:
                        continue
                    val = t.kwargs[k]
                    got = 0.0
                    if isinstance(gt_val, list) and isinstance(val, list):
                        if len(val) == len(gt_val):
                            got = 0.6
                        elif len(val) > 0:
                            got = 0.3
                    elif isinstance(val, str):
                        if val == str(gt_val):
                            got = 1.0
                        elif len(val) >= 3 and _check_param_format(k, val):
                            got = 0.5
                        elif val:
                            got = 0.2
                    elif val:
                        got = 0.3
                    score_params += got
                    break
        max_possible = sum(len(gt.kwargs) for gt in gt_actions)
        l2 = 0.50 * score_params / max(1, max_possible)

        # L3: 无多余/重复 (0.25)
        gt_name_counts = {}
        for gt in gt_actions:
            gt_name_counts[gt.name] = gt_name_counts.get(gt.name, 0) + 1
        taken_counts = {}
        extra_penalty = 0.0
        for t in taken:
            taken_counts[t.name] = taken_counts.get(t.name, 0) + 1
            expected = gt_name_counts.get(t.name, 0)
            if taken_counts[t.name] > expected:
                extra_penalty += 0.05
            if t.name not in gt_name_counts:
                extra_penalty += 0.10
        l3 = max(0.0, 0.25 - min(0.25, extra_penalty))

        # L4: respond 质量 (0.15) — 是否有 respond 结束
        has_respond = any(t.name == "respond" or t.name == "transfer_to_human_agents"
                          for t in self.taken_actions)
        l4 = 0.15 if has_respond else 0.0

        return min(1.0, l1 + l2 + l3 + l4)


def create_mock_env(task: Task) -> MockTauEnv:
    return MockTauEnv(task)
