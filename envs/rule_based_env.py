"""
Rule-based mock environment with state tracking.
Key innovation: tool calls modify task state, final state compared to expected state.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from data.tau_dataset import Task, Action


@dataclass
class TaskState:
    """Trackable task progress state."""
    user_found: bool = False
    order_checked: bool = False
    order_cancelled: bool = False
    order_modified: bool = False
    address_updated: bool = False
    payment_updated: bool = False
    items_returned: bool = False
    items_exchanged: bool = False
    reservation_checked: bool = False
    reservation_cancelled: bool = False
    reservation_modified: bool = False
    product_checked: bool = False
    correct_order_id: Optional[str] = None
    correct_reservation_id: Optional[str] = None
    correct_user_name: Optional[str] = None
    correct_email: Optional[str] = None


def extract_expected_state(task: Task) -> Tuple[TaskState, float]:
    """Extract expected final state and bonus score from ground truth actions."""
    state = TaskState()
    bonus = 0

    for a in task.actions:
        if a.name in ("respond", "think", "transfer_to_human_agents"):
            continue
        kwargs = a.kwargs

        if "user_id" in kwargs or "email" in kwargs:
            state.user_found = True
            state.correct_email = kwargs.get("email", "")
        if "first_name" in kwargs or "name" in kwargs:
            state.user_found = True
            state.correct_user_name = kwargs.get("first_name", kwargs.get("name", ""))

        if "order_id" in kwargs:
            state.correct_order_id = str(kwargs["order_id"])
        if "reservation_id" in kwargs:
            state.correct_reservation_id = str(kwargs["reservation_id"])

        if a.name == "get_order_details":
            state.order_checked = True
        elif a.name == "get_user_details":
            state.user_found = True
        elif a.name == "get_product_details":
            state.product_checked = True
        elif a.name == "get_reservation_details":
            state.reservation_checked = True
        elif a.name in ("cancel_pending_order", "cancel_order"):
            state.order_cancelled = True
        elif a.name == "cancel_reservation":
            state.reservation_cancelled = True
        elif a.name == "modify_pending_order_items":
            state.order_modified = True
        elif a.name == "modify_pending_order_address":
            state.address_updated = True
        elif a.name == "modify_pending_order_payment":
            state.payment_updated = True
        elif a.name == "return_delivered_order_items":
            state.items_returned = True
        elif a.name == "exchange_delivered_order_items":
            state.items_exchanged = True
        elif a.name == "update_reservation_flights":
            state.reservation_modified = True

    return state, bonus


def apply_action_to_state(state: TaskState, action: Action) -> int:
    """Apply one agent action to task state. Returns how many correct state changes occurred."""
    correct = 0
    name = action.name
    kwargs = action.kwargs

    if name in ("find_user_id_by_email", "find_user_id_by_name_zip",
                "find_user_by_email", "find_user_by_name", "find_user_by_contact",
                "get_user_details"):
        state.user_found = True
        correct += 1 if state.correct_user_name or state.correct_email else 0

    elif name == "get_order_details":
        order_id = str(kwargs.get("order_id", ""))
        if state.correct_order_id and order_id == state.correct_order_id:
            state.order_checked = True
            correct += 1

    elif name == "get_reservation_details":
        resid = str(kwargs.get("reservation_id", ""))
        if state.correct_reservation_id and resid == state.correct_reservation_id:
            state.reservation_checked = True
            correct += 1

    elif name == "get_product_details":
        state.product_checked = True
        pid = str(kwargs.get("product_id", ""))
        correct += 1 if len(pid) >= 3 else 0

    elif name in ("cancel_pending_order", "cancel_order"):
        oid = str(kwargs.get("order_id", ""))
        if state.correct_order_id and oid == state.correct_order_id:
            state.order_cancelled = True
            correct += 1

    elif name == "cancel_reservation":
        rid = str(kwargs.get("reservation_id", ""))
        if state.correct_reservation_id and rid == state.correct_reservation_id:
            state.reservation_cancelled = True
            correct += 1

    elif name == "modify_pending_order_items":
        state.order_modified = True
        correct += 1 if state.order_checked else 0

    elif name == "modify_pending_order_address":
        state.address_updated = True
        correct += 1 if state.order_checked else 0

    elif name == "modify_pending_order_payment":
        state.payment_updated = True
        correct += 1 if state.order_checked else 0

    elif name == "return_delivered_order_items":
        state.items_returned = True
        correct += 1 if state.order_checked else 0

    elif name == "exchange_delivered_order_items":
        state.items_exchanged = True
        correct += 1 if state.order_checked else 0

    elif name in ("update_reservation_flights", "update_reservation_baggages",
                  "update_reservation_passengers"):
        state.reservation_modified = True
        correct += 1 if state.reservation_checked else 0

    elif name == "calculate":
        correct += 1

    return correct


def compute_rule_based_reward(task: Task, agent_actions: List[Action]) -> float:
    """
    Rule-based reward: agent actions modify task state,
    compare final state with expected state from ground truth.

    Returns float 0.0-1.0.
    """
    expected, _ = extract_expected_state(task)
    actual = TaskState()

    correct_steps = 0
    for a in agent_actions:
        correct_steps += apply_action_to_state(actual, a)

    # Compare states: count how many expected state flags match actual
    expected_flags = [
        ("user_found", expected.user_found, actual.user_found),
        ("order_checked", expected.order_checked, actual.order_checked),
        ("order_cancelled", expected.order_cancelled, actual.order_cancelled),
        ("order_modified", expected.order_modified, actual.order_modified),
        ("address_updated", expected.address_updated, actual.address_updated),
        ("payment_updated", expected.payment_updated, actual.payment_updated),
        ("items_returned", expected.items_returned, actual.items_returned),
        ("items_exchanged", expected.items_exchanged, actual.items_exchanged),
        ("reservation_checked", expected.reservation_checked, actual.reservation_checked),
        ("reservation_cancelled", expected.reservation_cancelled, actual.reservation_cancelled),
        ("reservation_modified", expected.reservation_modified, actual.reservation_modified),
        ("product_checked", expected.product_checked, actual.product_checked),
    ]

    total_expected = sum(1 for _, e, _ in expected_flags if e)
    if total_expected == 0:
        return 1.0 if correct_steps > 0 else 0.0

    matched = sum(1 for _, e, a in expected_flags if e and a)
    # Bonus for correct steps
    step_score = min(0.5, 0.05 * correct_steps)
    return (matched / total_expected) * 0.5 + step_score
