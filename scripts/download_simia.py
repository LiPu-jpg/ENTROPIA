"""
将 Simia-Agent/Simia-Tau-SFT-90k-Hermes 转换为 ENTROPIA Task 格式。
多轮对话 → instruction + action plan。
"""

import json
import re
import sys
import os
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset


def extract_function_calls(text: str) -> List[Tuple[str, dict]]:
    """从 gpt turn 文本中提取函数调用。"""
    calls = []
    pattern = r"FUNCTION_CALL:\s*(\{.*?\})"
    for match in re.finditer(pattern, text):
        try:
            obj = json.loads(match.group(1))
            name = obj.get("name", "")
            args = obj.get("arguments", {})
            if name:
                calls.append((name, args))
        except json.JSONDecodeError:
            continue

    if not calls:
        tool_call_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
        for match in re.finditer(tool_call_pattern, text, re.DOTALL):
            try:
                obj = json.loads(match.group(1))
                name = obj.get("name", "")
                args = obj.get("arguments", {})
                if name:
                    calls.append((name, args))
            except json.JSONDecodeError:
                continue

    return calls


def convert_simia_to_tasks(ds, n: int = 3000) -> list:
    """将 Simia 对话转换为 Task 对象。"""
    from data.tau_dataset import Task, Action

    tasks = []
    for idx, item in enumerate(ds):
        if idx >= n:
            break

        try:
            conv = item.get("conversations", [])
            if not conv:
                continue

            instruction = conv[0]["value"] if conv else ""
            actions = []
            for turn in conv[1:]:  # 跳过第一个人类 turn
                if turn.get("from") == "gpt":
                    calls = extract_function_calls(turn["value"])
                    for name, args in calls:
                        actions.append(Action(name=name, kwargs=args))
                elif turn.get("from") == "tool":
                    pass

            if not actions:
                continue

            tasks.append(
                Task(
                    task_id=idx,
                    user_id=f"simia_{idx}",
                    instruction=instruction,
                    actions=actions,
                    outputs=[],
                )
            )
        except Exception as e:
            continue

    return tasks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--pct", type=str, default="4%")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    pct = args.pct if args.n > 10000 else f"train[:{args.n}]"
    print(f"Loading Simia dataset: {args.n} examples...", flush=True)
    ds = load_dataset("Simia-Agent/Simia-Tau-SFT-90k-Hermes", split="train[:4%]")
    print(f"Loaded {len(ds)} examples", flush=True)

    tasks = convert_simia_to_tasks(ds, n=args.n)
    print(f"Converted {len(tasks)} tasks", flush=True)

    if tasks:
        t = tasks[0]
        print(
            f"  Example: {len(t.actions)} actions, instruction: {t.instruction[:100]}..."
        )

    if args.dry_run:
        print("Dry run done.")
        sys.exit(0)

    import shutil

    dataset_dir = "/mnt/home/user46/ENTROPIA/data"
    os.makedirs(dataset_dir, exist_ok=True)

    output_path = os.path.join(dataset_dir, "simia_tasks.py")
    from data.tau_dataset import Task, Action

    with open(output_path, "w") as f:
        f.write('"""Auto-generated from Simia-Agent/Simia-Tau-SFT-90k-Hermes."""\n')
        f.write("from data.tau_dataset import Task, Action\n\n")
        f.write(f"# {len(tasks)} tasks\nSIMIA_TASKS = [\n")
        for task in tasks:
            actions_repr = []
            for a in task.actions:
                actions_repr.append(f'Action(name="{a.name}", kwargs={a.kwargs!r})')
            safe_inst = (
                task.instruction.replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("'", "\\'")
            )
            f.write(
                f'    Task(task_id={task.task_id}, user_id="{task.user_id}",\n'
                f'         instruction="{safe_inst}",\n'
                f"         actions=[{', '.join(actions_repr)}],\n"
                f"         outputs=[]),\n"
            )
        f.write("]\n")

    tool_names = set()
    for t in tasks:
        for a in t.actions:
            tool_names.add(a.name)
    print(f"Tool names ({len(tool_names)}): {sorted(tool_names)}")
    print(f"Saved {len(tasks)} tasks to {output_path}")
