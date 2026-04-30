"""
真实评测: 加载 checkpoint，用 GPT-4o (AIHubMix) 做 judge 评估成功率。
不再用 mock 环境，而是让 GPT-4o 判断 agent 的行为是否完成了任务。
"""
import sys, os, json, torch, openai
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from data.tau_dataset import load_tau_bench_dataset, task_to_sft_string
from envs.mock_env import MockTauEnv, parse_action_from_text, Action

client = openai.OpenAI(
    api_key="sk-7twVpm2srQC45pzJ468233Ab72594715BeDcA7E03525Eb2d",
    base_url="https://aihubmix.com/v1",
)

SP = (
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


def evaluate_with_gpt4(instruction: str, gt_actions: list, agent_actions: list) -> dict:
    """用 GPT-4o 判断 agent 是否正确完成了任务。"""
    gt_desc = "\n".join(f"{i+1}. {a[0]}({json.dumps(a[1])})" for i, a in enumerate(gt_actions))
    agent_desc = "\n".join(f"{i+1}. {a}" for i, a in enumerate(agent_actions))

    prompt = f"""You are an evaluator for a customer service agent.

Task instruction: {instruction}

Ground truth required actions:
{gt_desc}

Agent's actual actions:
{agent_desc}

Judge if the agent correctly completed the task. Consider:
1. Did the agent call all required ground truth tools?
2. Were the actions in a logical order?
3. Did the agent provide appropriate responses?

Respond with a JSON object:
{{"success": true/false, "reason": "brief explanation", "score": 0.0-1.0}}"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.0,
        )
        text = resp.choices[0].message.content
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text)
    except Exception as e:
        return {"success": False, "reason": f"Eval error: {e}", "score": 0.0}


def test_real(model, tok, tasks, name: str, n: int = 50):
    model.eval()
    dev = model.device
    ok, total_score = 0, 0.0

    for i, task in enumerate(tasks[:n]):
        prompt = (
            f"<|im_start|>system\n{SP}\n<|im_end|>\n"
            f"<|im_start|>user\n{task.instruction}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(dev)
        ilen = inp.input_ids.shape[1]
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=512, do_sample=True, temperature=0.6, top_p=0.95, pad_token_id=tok.pad_token_id)
        plan = tok.decode(out[0][ilen:], skip_special_tokens=True)

        # Parse + execute in mock env (just to get action strings)
        env = MockTauEnv(task); env.reset()
        agent_actions = []
        for a_str in plan.split("|"):
            a_str = a_str.strip()
            act = parse_action_from_text(a_str)
            if not act: act = Action(name="respond", kwargs={"content": a_str})
            env.step(act)
            agent_actions.append(a_str)
            if act.name == "respond":
                break
        else:
            agent_actions.append("respond('summary')")

        # GPT-4 judge
        gt_actions = [(a.name, a.kwargs) for a in task.actions]
        result = evaluate_with_gpt4(task.instruction, gt_actions, agent_actions)
        if result.get("success"):
            ok += 1
        total_score += result.get("score", 0.0)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n}] acc={ok/(i+1):.3f} avg_score={total_score/(i+1):.3f}", flush=True)

    acc = ok / n
    avg_s = total_score / n
    print(f"  {name}: success={acc:.4f} avg_score={avg_s:.4f}", flush=True)
    return acc, avg_s


if __name__ == "__main__":
    import argparse, glob
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="sparse", help="训练 mode 名")
    parser.add_argument("--n", type=int, default=50, help="测试样本数")
    parser.add_argument("--all", action="store_true", help="测试所有 checkpoint")
    args = parser.parse_args()

    BASE = "/mnt/home/user41/downloaded_models/Qwen/Qwen2.5-7B-Instruct"

    print("Loading tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    if tok.padding_side == "right": tok.padding_side = "left"

    print(f"Loading test set ({args.n} tasks)...", flush=True)
    tasks = load_tau_bench_dataset(n_samples=args.n, split="eval", seed=123, use_simia=True).tasks
    print(f"Test: {len(tasks)} tasks")

    results = {}

    if args.all:
        ckpt_dirs = sorted(glob.glob("outputs/quick_*/checkpoint_best"))
        for cp in ckpt_dirs:
            mode_name = cp.split("/")[1].replace("quick_", "")
            print(f"\n=== {mode_name} ===", flush=True)
            m = PeftModel.from_pretrained(
                AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True),
                cp,
            ).eval()
            acc, avg_s = test_real(m, tok, tasks, mode_name, args.n)
            results[mode_name] = {"success": round(acc, 4), "avg_score": round(avg_s, 4)}
            del m; torch.cuda.empty_cache()
    else:
        cp = sorted(glob.glob(f"outputs/quick_{args.mode}_*/checkpoint_best"), reverse=True)
        if not cp:
            print(f"No checkpoint found for mode={args.mode}")
            sys.exit(1)
        print(f"\n=== {args.mode} ({cp[0]}) ===", flush=True)
        m = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True),
            cp[0],
        ).eval()
        acc, avg_s = test_real(m, tok, tasks, args.mode, args.n)
        results[args.mode] = {"success": round(acc, 4), "avg_score": round(avg_s, 4)}

    print(f"\n=== REAL EVAL RESULTS (GPT-4o Judge) ===")
    for mode, r in sorted(results.items()):
        print(f"  {mode:<25} success={r['success']:.4f}  avg_score={r['avg_score']:.4f}")
    with open("outputs/real_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to outputs/real_eval_results.json")
