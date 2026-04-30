"""
将 fuvty/tau-bench-synthetic 转换为 ENTROPIA Task 格式。
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from data.tau_dataset import Task, Action, TauBenchDataset


def convert_fuvty_to_tasks(ds) -> list:
    """将 fuvty/tau-bench-synthetic 的 'tasks' 配置转换为 ENTROPIA Task 格式。"""
    tasks = []
    for idx, item in enumerate(ds):
        try:
            criteria = json.loads(item["evaluation_criteria"])
            scenario = json.loads(item["user_scenario"])
            instructions = scenario.get("instructions", {})

            # 构建 instruction 文本
            task_inst = instructions.get("task_instructions", "")
            reason = instructions.get("reason_for_call", "")
            known = instructions.get("known_info", "")
            instruction_parts = [task_inst]
            if reason:
                instruction_parts.append(reason)
            if known:
                instruction_parts.append(known)
            instruction = " ".join(instruction_parts)

            # 提取真实动作（非 respond）
            gt_actions = []
            for act in criteria.get("actions", []):
                name = act["name"]
                if name in ("respond", "transfer_to_human_agents"):
                    continue
                gt_actions.append(Action(name=name, kwargs=act.get("arguments", {})))

            # 提取输出断言作为期望关键词
            outputs = []
            for assertion in criteria.get("nl_assertions", []):
                if isinstance(assertion, str):
                    outputs.append(assertion)
                elif isinstance(assertion, dict):
                    outputs.append(assertion.get("content", ""))

            # 从 instruction 生成 user_id
            user_id = item.get("id", f"user_{idx}")

            task = Task(
                task_id=idx,
                user_id=user_id,
                instruction=instruction,
                actions=gt_actions,
                outputs=outputs if outputs else [],
            )
            tasks.append(task)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: skipping item {idx}: {e}", file=sys.stderr)
            continue

    return tasks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    print("Downloading fuvty/tau-bench-synthetic 'tasks' config...", flush=True)
    ds = load_dataset("fuvty/tau-bench-synthetic", "tasks", split="train")
    print(f"Downloaded {len(ds)} examples", flush=True)

    tasks = convert_fuvty_to_tasks(ds)
    print(f"Converted to {len(tasks)} ENTROPIA tasks", flush=True)

    if tasks:
        t = tasks[0]
        print(f"  Example: {len(t.actions)} actions, {len(t.outputs)} outputs")
        print(f"  Instruction: {t.instruction[:120]}...")

    if args.dry_run:
        print("Dry run done.")
        sys.exit(0)

    # 保存为模块
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "fuvty_tasks.py",
    )
    n = args.n_samples if args.n_samples else len(tasks)
    selected = tasks[:n]

    with open(output_path, "w") as f:
        f.write('"""Auto-generated from fuvty/tau-bench-synthetic."""\n')
        f.write("from data.tau_dataset import Task, Action\n\n")
        f.write(f"# {len(selected)} tasks from fuvty/tau-bench-synthetic\n")
        f.write("FUVTY_TASKS = [\n")
        for i, task in enumerate(selected):
            actions_repr = []
            for a in task.actions:
                actions_repr.append(f'Action(name="{a.name}", kwargs={a.kwargs!r})')
            outputs_repr = repr(task.outputs) if task.outputs else "[]"
            f.write(
                f'    Task(task_id={task.task_id}, user_id="{task.user_id}",\n'
                f"         instruction={task.instruction!r},\n"
                f"         actions=[{', '.join(actions_repr)}],\n"
                f"         outputs={outputs_repr}),\n"
            )
        f.write("]\n")

    # 同时追踪新的工具名称
    tool_names = set()
    for t in selected:
        for a in t.actions:
            tool_names.add(a.name)
    print(f"New tool names found: {sorted(tool_names)}")
    print(f"Saved {len(selected)} tasks to {output_path}")
