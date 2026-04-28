"""
τ-Bench format dataset for training Direction A.

Each task follows τ-Bench's Task schema:
- user_id: str
- instruction: str (user scenario with personality + goals)
- actions: List[Action] (ground truth tool calls)
- outputs: List[str] (expected text in agent response)

Data is synthetic but follows τ-Bench's retail domain conventions:
tools: find_user_id_by_email, find_user_id_by_name_zip, get_user_details,
       get_order_details, get_product_details, list_all_product_types,
       modify_pending_order_items, modify_pending_order_address,
       modify_pending_order_payment, cancel_pending_order,
       return_delivered_order_items, exchange_delivered_order_items,
       think, transfer_to_human_agents, respond
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import random


@dataclass
class Action:
    """τ-Bench action: tool name + kwargs."""

    name: str
    kwargs: Dict[str, Any]


@dataclass
class Task:
    """τ-Bench task definition."""

    user_id: str
    instruction: str
    actions: List[Action] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    task_id: int = 0


# ──────────────────────────────────────────────────
# Synthetic τ-Bench retail tasks (25 tasks)
# ──────────────────────────────────────────────────

SYNTHETIC_TASKS = [
    # === Single-action tasks ===
    Task(
        task_id=1,
        user_id="alice_wong_3203",
        instruction=(
            "Your name is Alice Wong and your zip code is 19031. "
            "You are polite, organized. "
            "Return order #W6067464 via credit_card_4190576: Electric Kettle and Wall Clock."
        ),
        actions=[
            Action("find_user_id_by_name_zip", {"name": "Alice Wong", "zip": "19031"}),
            Action(
                "return_delivered_order_items",
                {
                    "order_id": "#W6067464",
                    "item_ids": ["9624127908", "8917609800"],
                    "payment_method_id": "credit_card_4190576",
                },
            ),
            Action(
                "respond",
                {
                    "content": "I've processed your return for order #W6067464. The refund will go to your credit card."
                },
            ),
        ],
        outputs=["processed your return", "#W6067464"],
    ),
    Task(
        task_id=2,
        user_id="bob_chen_2370",
        instruction=(
            "Your name is Bob Chen and your zip code is 20171. "
            "You are confident, busy. "
            "Return order #W6619432 via paypal_3738584: Dumbbell Set and Yoga Mat."
        ),
        actions=[
            Action("find_user_id_by_name_zip", {"name": "Bob Chen", "zip": "20171"}),
            Action(
                "return_delivered_order_items",
                {
                    "order_id": "#W6619432",
                    "item_ids": ["3735133539", "6195938807"],
                    "payment_method_id": "paypal_3738584",
                },
            ),
            Action(
                "respond",
                {"content": "Return processed for #W6619432. Refund via PayPal."},
            ),
        ],
        outputs=["Return processed", "#W6619432"],
    ),
    Task(
        task_id=3,
        user_id="carol_li_5688",
        instruction=(
            "Your name is Carol Li and your email is carol.li4495@example.com. "
            "You are rigid, curious. "
            "Cancel order #W4435622 because ordered by mistake."
        ),
        actions=[
            Action("find_user_id_by_email", {"email": "carol.li4495@example.com"}),
            Action(
                "cancel_pending_order",
                {
                    "order_id": "#W4435622",
                    "reason": "ordered by mistake",
                },
            ),
            Action("respond", {"content": "Order #W4435622 has been cancelled."}),
        ],
        outputs=["cancelled", "#W4435622"],
    ),
    Task(
        task_id=4,
        user_id="dave_wilson_7075",
        instruction=(
            "Your name is Dave Wilson and your zip code is 19049. "
            "You are patient, organized. "
            "Cancel order #W5765741 because ordered by mistake."
        ),
        actions=[
            Action("find_user_id_by_name_zip", {"name": "Dave Wilson", "zip": "19049"}),
            Action(
                "cancel_pending_order",
                {
                    "order_id": "#W5765741",
                    "reason": "ordered by mistake",
                },
            ),
            Action("respond", {"content": "Cancelled #W5765741."}),
        ],
        outputs=["Cancelled", "#W5765741"],
    ),
    # === Exchange tasks ===
    Task(
        task_id=5,
        user_id="eve_nguyen_2175",
        instruction=(
            "Your name is Eve Nguyen and your email is eve.nguyen3664@example.com. "
            "You are outgoing, flexible. "
            "For #W1504875, exchange Notebook size A6 to size A5 via paypal_6262583."
        ),
        actions=[
            Action("find_user_id_by_email", {"email": "eve.nguyen3664@example.com"}),
            Action("get_order_details", {"order_id": "#W1504875"}),
            Action(
                "exchange_delivered_order_items",
                {
                    "order_id": "#W1504875",
                    "item_ids": ["9421195098"],
                    "new_item_ids": ["9799386954"],
                    "payment_method_id": "paypal_6262583",
                },
            ),
            Action(
                "respond", {"content": "Exchanged Notebook in #W1504875 to A5 size."}
            ),
        ],
        outputs=["Exchanged", "#W1504875"],
    ),
    Task(
        task_id=6,
        user_id="frank_martin_4549",
        instruction=(
            "Your name is Frank Martin and your email is frank.martin5733@example.com. "
            "You are cautious, organized. "
            "For #W9318778, exchange Bicycle medium frame to large frame via credit_card_7862034."
        ),
        actions=[
            Action("find_user_id_by_email", {"email": "frank.martin5733@example.com"}),
            Action("get_order_details", {"order_id": "#W9318778"}),
            Action(
                "exchange_delivered_order_items",
                {
                    "order_id": "#W9318778",
                    "item_ids": ["2143041831"],
                    "new_item_ids": ["5606522780"],
                    "payment_method_id": "credit_card_7862034",
                },
            ),
            Action(
                "respond", {"content": "Bicycle in #W9318778 exchanged to large frame."}
            ),
        ],
        outputs=["exchanged", "#W9318778"],
    ),
    # === Modify order tasks ===
    Task(
        task_id=7,
        user_id="grace_ito_8499",
        instruction=(
            "Your name is Grace Ito and your email is grace.ito7353@example.com. "
            "You are cautious, flexible. "
            "For #W8353027, modify Grill to add rotisserie feature via paypal_1679017."
        ),
        actions=[
            Action("find_user_id_by_email", {"email": "grace.ito7353@example.com"}),
            Action(
                "modify_pending_order_items",
                {
                    "order_id": "#W8353027",
                    "item_ids": ["7717598293"],
                    "new_item_ids": ["7848293342"],
                    "payment_method_id": "paypal_1679017",
                },
            ),
            Action(
                "respond",
                {"content": "Modified Grill in #W8353027 with rotisserie feature."},
            ),
        ],
        outputs=["Modified", "#W8353027"],
    ),
    Task(
        task_id=8,
        user_id="henry_kim_5477",
        instruction=(
            "Your name is Henry Kim and your email is henry.kim5723@example.com. "
            "You are direct, organized. "
            "For #W7109609, change payment method to gift_card_2748512."
        ),
        actions=[
            Action("find_user_id_by_email", {"email": "henry.kim5723@example.com"}),
            Action(
                "modify_pending_order_payment",
                {
                    "order_id": "#W7109609",
                    "payment_method_id": "gift_card_2748512",
                },
            ),
            Action(
                "respond",
                {"content": "Payment method for #W7109609 updated to gift card."},
            ),
        ],
        outputs=["payment", "#W7109609"],
    ),
    Task(
        task_id=9,
        user_id="iris_lee_5820",
        instruction=(
            "Your name is Iris Lee and your zip code is 85060. "
            "You are organized, direct. "
            "For #W3386832, change shipping address to 411 Park Avenue, Suite 987, Phoenix, AZ 85060."
        ),
        actions=[
            Action("find_user_id_by_name_zip", {"name": "Iris Lee", "zip": "85060"}),
            Action(
                "modify_pending_order_address",
                {
                    "order_id": "#W3386832",
                    "address1": "411 Park Avenue",
                    "address2": "Suite 987",
                    "city": "Phoenix",
                    "country": "USA",
                    "state": "AZ",
                    "zip": "85060",
                },
            ),
            Action("respond", {"content": "Shipping address for #W3386832 updated."}),
        ],
        outputs=["address", "#W3386832"],
    ),
    # === Multi-action tasks (harder) ===
    Task(
        task_id=10,
        user_id="jack_brown_1356",
        instruction=(
            "Your name is Jack Brown and your zip code is 91203. "
            "You are outgoing, polite. "
            "First cancel order #W7109609 because ordered by mistake. "
            "Then cancel order #W6554908 because no longer needed."
        ),
        actions=[
            Action("find_user_id_by_name_zip", {"name": "Jack Brown", "zip": "91203"}),
            Action(
                "cancel_pending_order",
                {
                    "order_id": "#W7109609",
                    "reason": "ordered by mistake",
                },
            ),
            Action(
                "cancel_pending_order",
                {
                    "order_id": "#W6554908",
                    "reason": "no longer needed",
                },
            ),
            Action(
                "respond", {"content": "Both orders #W7109609 and #W6554908 cancelled."}
            ),
        ],
        outputs=["Both", "cancelled"],
    ),
    Task(
        task_id=11,
        user_id="kate_jones_4279",
        instruction=(
            "Your name is Kate Jones and your email is kate.jones2789@example.com. "
            "You are patient, pessimistic. "
            "Return #W5285031: Tablet via credit_card_7952624. "
            "Also exchange Hiking Boots size 10 to size 12 in #W8632528 via gift_card_7219486."
        ),
        actions=[
            Action("find_user_id_by_email", {"email": "kate.jones2789@example.com"}),
            Action(
                "return_delivered_order_items",
                {
                    "order_id": "#W5285031",
                    "item_ids": ["2235648106"],
                    "payment_method_id": "credit_card_7952624",
                },
            ),
            Action(
                "exchange_delivered_order_items",
                {
                    "order_id": "#W8632528",
                    "item_ids": ["2185126308"],
                    "new_item_ids": ["4582956489"],
                    "payment_method_id": "gift_card_7219486",
                },
            ),
            Action(
                "respond", {"content": "Returned #W5285031 and exchanged #W8632528."}
            ),
        ],
        outputs=["Returned", "exchanged"],
    ),
    Task(
        task_id=12,
        user_id="leo_smith_2259",
        instruction=(
            "Your name is Leo Smith and your email is leo.smith8320@example.com. "
            "You are insecure, polite. "
            "For #W2575533: change address to 812 Cedar Avenue Suite 500, Houston TX 77129. "
            "Then modify E-Reader to 32GB with Wi-Fi+Cellular via paypal_3024827."
        ),
        actions=[
            Action("find_user_id_by_email", {"email": "leo.smith8320@example.com"}),
            Action(
                "modify_pending_order_address",
                {
                    "order_id": "#W2575533",
                    "address1": "812 Cedar Avenue",
                    "address2": "Suite 500",
                    "city": "Houston",
                    "country": "USA",
                    "state": "TX",
                    "zip": "77129",
                },
            ),
            Action(
                "modify_pending_order_items",
                {
                    "order_id": "#W2575533",
                    "item_ids": ["9494281769"],
                    "new_item_ids": ["4273929280"],
                    "payment_method_id": "paypal_3024827",
                },
            ),
            Action(
                "respond",
                {"content": "Updated address and modified E-Reader in #W2575533."},
            ),
        ],
        outputs=["address", "E-Reader", "#W2575533"],
    ),
    # === More diverse tasks ===
    Task(
        task_id=13,
        user_id="mia_zhang_8533",
        instruction=(
            "Your name is Mia Zhang and your zip code is 85010. "
            "You are relaxing, impatient. "
            "Return order #W1523776: Smart Thermostat via gift_card_2748512."
        ),
        actions=[
            Action("find_user_id_by_name_zip", {"name": "Mia Zhang", "zip": "85010"}),
            Action(
                "return_delivered_order_items",
                {
                    "order_id": "#W1523776",
                    "item_ids": ["8593894906"],
                    "payment_method_id": "gift_card_2748512",
                },
            ),
            Action("respond", {"content": "Return processed for #W1523776."}),
        ],
        outputs=["Return", "#W1523776"],
    ),
    Task(
        task_id=14,
        user_id="nick_park_7021",
        instruction=(
            "Your name is Nick Park and your email is nick.park5925@example.com. "
            "You are happy, independent. "
            "For #W5801125, modify Tea Kettle material to ceramic via paypal_5543657."
        ),
        actions=[
            Action("find_user_id_by_email", {"email": "nick.park5925@example.com"}),
            Action(
                "modify_pending_order_items",
                {
                    "order_id": "#W5801125",
                    "item_ids": ["9647374798"],
                    "new_item_ids": ["3312883418"],
                    "payment_method_id": "paypal_5543657",
                },
            ),
            Action(
                "respond", {"content": "Tea Kettle in #W5801125 updated to ceramic."}
            ),
        ],
        outputs=["ceramic", "#W5801125"],
    ),
    Task(
        task_id=15,
        user_id="olivia_garcia_7119",
        instruction=(
            "Your name is Olivia Garcia and your email is olivia.garcia9875@example.com. "
            "You are pessimistic, outgoing. "
            "Exchange Water Bottle from stainless steel to glass in #W3977493 via credit_card_6748580."
        ),
        actions=[
            Action("find_user_id_by_email", {"email": "olivia.garcia9875@example.com"}),
            Action("get_order_details", {"order_id": "#W3977493"}),
            Action(
                "exchange_delivered_order_items",
                {
                    "order_id": "#W3977493",
                    "item_ids": ["7533802601"],
                    "new_item_ids": ["5758737025"],
                    "payment_method_id": "credit_card_6748580",
                },
            ),
            Action(
                "respond", {"content": "Water Bottle in #W3977493 exchanged to glass."}
            ),
        ],
        outputs=["exchanged", "glass"],
    ),
    # === Three-action tasks (harder) ===
    Task(
        task_id=16,
        user_id="peter_tan_6696",
        instruction=(
            "Your name is Peter Tan and your zip code is 77209. "
            "You are independent, curious. "
            "Cancel order #W1242543. Cancel order #W9232383. Cancel order #W8367380. "
            "All because they are no longer needed."
        ),
        actions=[
            Action("find_user_id_by_name_zip", {"name": "Peter Tan", "zip": "77209"}),
            Action(
                "cancel_pending_order",
                {
                    "order_id": "#W1242543",
                    "reason": "no longer needed",
                },
            ),
            Action(
                "cancel_pending_order",
                {
                    "order_id": "#W9232383",
                    "reason": "no longer needed",
                },
            ),
            Action(
                "cancel_pending_order",
                {
                    "order_id": "#W8367380",
                    "reason": "no longer needed",
                },
            ),
            Action("respond", {"content": "All three orders cancelled."}),
        ],
        outputs=["cancelled", "three"],
    ),
    Task(
        task_id=17,
        user_id="qi_liu_3061",
        instruction=(
            "Your name is Qi Liu and your zip code is 75368. "
            "You are rigid, busy. "
            "For #W9933266: modify Pet Bed to medium size grey, "
            "and Yoga Mat to 6mm green via paypal_4133936."
        ),
        actions=[
            Action("find_user_id_by_name_zip", {"name": "Qi Liu", "zip": "75368"}),
            Action(
                "modify_pending_order_items",
                {
                    "order_id": "#W9933266",
                    "item_ids": ["4537595158", "5586947715"],
                    "new_item_ids": ["6857426243", "7510236436"],
                    "payment_method_id": "paypal_4133936",
                },
            ),
            Action(
                "respond", {"content": "Modified Pet Bed and Yoga Mat in #W9933266."}
            ),
        ],
        outputs=["Modified", "#W9933266"],
    ),
    Task(
        task_id=18,
        user_id="rachel_kim_4909",
        instruction=(
            "Your name is Rachel Kim and your email is rachel.kim4901@example.com. "
            "You are busy, optimistic. "
            "For #W2598324: modify Espresso Machine to capsule type. "
            "Also exchange Tablet to 7-inch 128GB black via credit_card_5902940."
        ),
        actions=[
            Action("find_user_id_by_email", {"email": "rachel.kim4901@example.com"}),
            Action(
                "modify_pending_order_items",
                {
                    "order_id": "#W2598324",
                    "item_ids": ["3379843752"],
                    "new_item_ids": ["6200867091"],
                    "payment_method_id": "credit_card_5902940",
                },
            ),
            Action(
                "exchange_delivered_order_items",
                {
                    "order_id": "#W3239882",
                    "item_ids": ["2106335193"],
                    "new_item_ids": ["4913411651"],
                    "payment_method_id": "credit_card_5902940",
                },
            ),
            Action(
                "respond", {"content": "Modified #W2598324 and exchanged #W3239882."}
            ),
        ],
        outputs=["Modified", "exchanged"],
    ),
    # === Tasks with identity lookup + action ===
    Task(
        task_id=19,
        user_id="sam_wright_8900",
        instruction=(
            "You are a customer. Your name is Sam Wright, zip 91455. "
            "You want to cancel order #W7613749 because you ordered by mistake."
        ),
        actions=[
            Action("find_user_id_by_name_zip", {"name": "Sam Wright", "zip": "91455"}),
            Action(
                "cancel_pending_order",
                {
                    "order_id": "#W7613749",
                    "reason": "ordered by mistake",
                },
            ),
            Action("respond", {"content": "Order #W7613749 has been cancelled."}),
        ],
        outputs=["cancelled"],
    ),
    Task(
        task_id=20,
        user_id="tina_adams_9003",
        instruction=(
            "Your name is Tina Adams and your email is tina.adams4109@example.com. "
            "You are logical, direct. "
            "Return #W6426438: Wristwatch via gift_card_7219486."
        ),
        actions=[
            Action("find_user_id_by_email", {"email": "tina.adams4109@example.com"}),
            Action(
                "return_delivered_order_items",
                {
                    "order_id": "#W6426438",
                    "item_ids": ["8886009523"],
                    "payment_method_id": "gift_card_7219486",
                },
            ),
            Action("respond", {"content": "Return processed for #W6426438."}),
        ],
        outputs=["Return", "#W6426438"],
    ),
    # === Complex multi-step tasks ===
    Task(
        task_id=21,
        user_id="uma_patel_2152",
        instruction=(
            "Your name is Uma Patel and your email is uma.patel9391@example.com. "
            "You are polite, organized. "
            "First return #W5565470: Electric Kettle, Mechanical Keyboard, and Pet Bed via paypal_3024827. "
            "Then exchange Skateboard to plastic in #W3792453 via same paypal."
        ),
        actions=[
            Action("find_user_id_by_email", {"email": "uma.patel9391@example.com"}),
            Action(
                "return_delivered_order_items",
                {
                    "order_id": "#W5565470",
                    "item_ids": ["7602931732", "9570044148", "6857426243"],
                    "payment_method_id": "paypal_3024827",
                },
            ),
            Action(
                "exchange_delivered_order_items",
                {
                    "order_id": "#W3792453",
                    "item_ids": ["4293355847"],
                    "new_item_ids": ["3877188862"],
                    "payment_method_id": "paypal_3024827",
                },
            ),
            Action(
                "respond", {"content": "Returned #W5565470 and exchanged #W3792453."}
            ),
        ],
        outputs=["Returned", "exchanged"],
    ),
    Task(
        task_id=22,
        user_id="viktor_chen_6291",
        instruction=(
            "Your name is Viktor Chen and your email is viktor.chen8943@example.com. "
            "You are patient, confident. "
            "For #W6779827: modify Espresso Machine to 1.5L manual. "
            "For #W6111820: modify Wireless Earbuds battery to 8 hours via credit_card_3816099."
        ),
        actions=[
            Action("find_user_id_by_email", {"email": "viktor.chen8943@example.com"}),
            Action(
                "modify_pending_order_items",
                {
                    "order_id": "#W6779827",
                    "item_ids": ["3379843752"],
                    "new_item_ids": ["2190871011"],
                    "payment_method_id": "credit_card_9789590",
                },
            ),
            Action(
                "modify_pending_order_items",
                {
                    "order_id": "#W6111820",
                    "item_ids": ["2757705742"],
                    "new_item_ids": ["8555936349"],
                    "payment_method_id": "credit_card_3816099",
                },
            ),
            Action(
                "respond", {"content": "Both orders modified: #W6779827 and #W6111820."}
            ),
        ],
        outputs=["modified", "orders"],
    ),
    # === Simpler tasks for curriculum ===
    Task(
        task_id=23,
        user_id="wang_lee_1273",
        instruction=(
            "Your name is Wang Lee and your zip code is 95014. "
            "You are direct, impatient. "
            "Cancel order #W3525030 because no longer needed."
        ),
        actions=[
            Action("find_user_id_by_name_zip", {"name": "Wang Lee", "zip": "95014"}),
            Action(
                "cancel_pending_order",
                {
                    "order_id": "#W3525030",
                    "reason": "no longer needed",
                },
            ),
            Action("respond", {"content": "Cancelled #W3525030."}),
        ],
        outputs=["Cancelled"],
    ),
    Task(
        task_id=24,
        user_id="xenia_rossi_5471",
        instruction=(
            "Your name is Xenia Rossi and your email is xenia.rossi5471@example.com. "
            "You are confident, calm. "
            "Return #W7450915: Bookshelf via gift_card_6892585."
        ),
        actions=[
            Action("find_user_id_by_email", {"email": "xenia.rossi5471@example.com"}),
            Action(
                "return_delivered_order_items",
                {
                    "order_id": "#W7450915",
                    "item_ids": ["6735339143"],
                    "payment_method_id": "gift_card_6892585",
                },
            ),
            Action("respond", {"content": "Return for #W7450915 processed."}),
        ],
        outputs=["Return", "#W7450915"],
    ),
    Task(
        task_id=25,
        user_id="yuri_tanaka_8801",
        instruction=(
            "Your name is Yuri Tanaka and your zip code is 10001. "
            "You are patient, happy. "
            "For #W4840405: exchange Backpack from small green to large black via gift_card_4710915."
        ),
        actions=[
            Action("find_user_id_by_name_zip", {"name": "Yuri Tanaka", "zip": "10001"}),
            Action("get_order_details", {"order_id": "#W4840405"}),
            Action(
                "exchange_delivered_order_items",
                {
                    "order_id": "#W4840405",
                    "item_ids": ["9990204880"],
                    "new_item_ids": ["6886091285"],
                    "payment_method_id": "gift_card_4710915",
                },
            ),
            Action(
                "respond",
                {"content": "Exchanged Backpack in #W4840405 to large black."},
            ),
        ],
        outputs=["Exchanged", "#W4840405"],
    ),
]

# All valid τ-Bench tool names (for action parsing)
TAU_BENCH_TOOLS = {
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
    "get_user_details",
    "get_order_details",
    "get_product_details",
    "list_all_product_types",
    "modify_pending_order_items",
    "modify_pending_order_address",
    "modify_pending_order_payment",
    "cancel_pending_order",
    "return_delivered_order_items",
    "exchange_delivered_order_items",
    "think",
    "transfer_to_human_agents",
    "respond",
}


class TauBenchDataset:
    """
    Dataset class wrapping τ-Bench format tasks.

    Each sample = (task_id, instruction, tools_summary, ground_truth_actions).
    Compatible with torch.utils.data.DataLoader.
    """

    def __init__(self, tasks: List[Task], split: str = "train"):
        self.tasks = tasks
        self.split = split

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]
        return {
            "query": task.instruction,
            "task_id": task.task_id,
            "gt_actions": [(a.name, a.kwargs) for a in task.actions],
            "gt_outputs": task.outputs,
            "user_id": task.user_id,
        }


def load_tau_bench_dataset(
    n_samples: Optional[int] = None,
    split: str = "train",
    seed: int = 42,
) -> TauBenchDataset:
    """
    Load synthetic τ-Bench format training data.

    Args:
        n_samples: Number of samples to load (None = all 25).
        split: "train" or "eval". Currently both use same pool.
        seed: Random seed for shuffling.

    Returns:
        TauBenchDataset instance.
    """
    rng = random.Random(seed)
    tasks = list(SYNTHETIC_TASKS)
    rng.shuffle(tasks)

    if split == "train":
        selected = tasks[: int(len(tasks) * 0.8)]  # 80% for training
    else:
        selected = tasks[int(len(tasks) * 0.8) :]  # 20% for eval

    if n_samples is not None:
        selected = selected[:n_samples]

    return TauBenchDataset(selected, split=split)
