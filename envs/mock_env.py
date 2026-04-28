"""
Mock τ-Bench environment for training Direction A.

Implements the τ-Bench Env API: reset → step → reward.
Does NOT use a language-model user simulator — observations are
pre-scripted to keep training fast and deterministic.
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
    Extract a τ-Bench Action from model-generated text.

    Tries multiple parsing strategies:
    1. JSON: {"name": "tool", "kwargs": {...}}
    2. Function call: tool_name(key=val, ...)
    3. XML-style: <action>tool_name</action>
    4. Fallback: extract first known tool name from text
    """
    action_text = action_text.strip()

    # Strategy 1: JSON object
    if action_text.startswith("{"):
        try:
            obj = json.loads(action_text)
            return Action(name=obj["name"], kwargs=obj.get("kwargs", {}))
        except (json.JSONDecodeError, KeyError):
            pass

    # Strategy 2: Python-style function call: tool_name(key=val, ...)
    func_match = re.match(r"^(\w+)\s*\((.+)\)\s*$", action_text, re.DOTALL)
    if func_match:
        name = func_match.group(1)
        if name in TAU_BENCH_TOOLS:
            kwargs = _parse_python_kwargs(func_match.group(2))
            return Action(name=name, kwargs=kwargs)

    # Strategy 3: XML-style <action>tool_name</action> or similar
    for tool_name in TAU_BENCH_TOOLS:
        if f"<action>{tool_name}</action>" in action_text:
            return Action(name=tool_name, kwargs={})

    # Strategy 4: First known tool name found in text
    for tool_name in sorted(TAU_BENCH_TOOLS, key=len, reverse=True):
        if tool_name in action_text:
            return Action(name=tool_name, kwargs={})

    return None


def _parse_python_kwargs(kwargs_str: str) -> Dict[str, Any]:
    """Parse Python-style kwargs string like: key1='val1', key2=['a','b']"""
    kwargs = {}

    # Split by commas that aren't inside brackets/quotes
    pairs = _smart_split(kwargs_str, ",")

    for pair in pairs:
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue

        key, _, val_str = pair.partition("=")
        key = key.strip()
        val_str = val_str.strip()

        # Parse the value
        val = _parse_python_value(val_str)
        kwargs[key] = val

    return kwargs


def _parse_python_value(s: str) -> Any:
    """Parse a single Python-style value."""
    s = s.strip()

    # String
    if (s.startswith('"') and s.endswith('"')) or (
        s.startswith("'") and s.endswith("'")
    ):
        return s[1:-1]

    # List
    if s.startswith("[") and s.endswith("]"):
        items = _smart_split(s[1:-1], ",")
        return [_parse_python_value(item.strip()) for item in items if item.strip()]

    # Dict
    if s.startswith("{") and s.endswith("}"):
        return _parse_python_kwargs(s[1:-1])

    # Number
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass

    # Fallback: string
    return s


def _smart_split(text: str, sep: str) -> List[str]:
    """Split by separator, respecting nested brackets and quotes."""
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
    Mock τ-Bench environment for training.

    API matches τ-Bench's Env class:
    - reset(task_index) → EnvResetResult
    - step(action: Action) → EnvStepResult

    Reward: binary (1.0 = task completed correctly, 0.0 = otherwise).
    Observations are pre-scripted per action type.
    """

    OBSERVATIONS = {
        "find_user_id_by_email": "User found. User ID: user_{user_id}",
        "find_user_id_by_name_zip": "User found. User ID: user_{user_id}",
        "get_user_details": "User details: name, email, zip code, payment methods on file.",
        "get_order_details": "Order details: items, status, shipping address, payment method.",
        "get_product_details": "Product details: name, price, specifications, availability.",
        "list_all_product_types": "Available product categories: Electronics, Home & Kitchen, Sports, Books, Clothing, Toys.",
        "modify_pending_order_items": "Order items updated successfully.",
        "modify_pending_order_address": "Shipping address updated successfully.",
        "modify_pending_order_payment": "Payment method updated successfully.",
        "cancel_pending_order": "Order cancelled successfully.",
        "return_delivered_order_items": "Return initiated. Refund will be processed within 5-7 business days.",
        "exchange_delivered_order_items": "Exchange initiated. New items will be shipped within 3-5 business days.",
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
        """
        Binary task success reward.

        Checks:
        1. All ground truth non-respond actions were taken.
        2. If task has outputs, at least one output string appears in responses.
        """
        gt_non_respond = [
            a
            for a in self.gt_actions
            if a.name not in ("respond", "transfer_to_human_agents")
        ]

        if not gt_non_respond:
            return 1.0

        # Check each required action was taken (by name match)
        taken_names = {a.name for a in self.taken_actions}
        all_actions_taken = all(gt.name in taken_names for gt in gt_non_respond)

        if not all_actions_taken:
            return 0.0

        # Check outputs if any
        if self.task.outputs:
            respond_texts = [
                a.kwargs.get("content", "")
                for a in self.taken_actions
                if a.name == "respond"
            ]
            all_responses = " ".join(respond_texts).lower()

            for output in self.task.outputs:
                if output.lower() not in all_responses:
                    return 0.0

        return 1.0


def create_mock_env(task: Task) -> MockTauEnv:
    return MockTauEnv(task)
