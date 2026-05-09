"""
真实评测: 加载所有 checkpoint，用 MiniMax M2.7 做 judge。
"""
import sys, os, json, torch, glob, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from data.tau_dataset import load_tau_bench_dataset
from envs.mock_env import MockTauEnv, parse_action_from_text, Action

client = OpenAI(
    base_url="https://api.minimaxi.com/v1",
    api_key="sk-cp-xqBXPT7PTX8CG_IMl3xnbrrVi50i1wEjBQ8AACpgDhR3wpD6BJeTsYrBt2J9CJSMy9weFfPUQHJ6DWYMXqvD6Whvszor2IZhc_jACOJXGx3QbcygaiIFgLo",
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
    "Format: tool_name(key='val'). Separate with |. End with respond(summary)."
)


def judge_with_minimax(instruction: str, gt_actions: list, agent_actions: list) -> float:
    gt_desc = " → ".join(f"{n}({', '.join(f'{k}={v}' for k,v in a.items())})" for n, a in gt_actions)
    agent_desc = " → ".join(str(a) for a in agent_actions)

    prompt = f"""Task: {instruction}

Ground truth actions: {gt_desc}

Agent actions: {agent_desc}

Score how well the agent completed the task (0.0 to 1.0):
- 1.0: All correct actions in right order, correct parameters
- 0.7-0.9: Correct actions but some parameters wrong or missing
- 0.4-0.6: Major actions present but incomplete or extra wrong steps
- 0.1-0.3: Only partially started the task
- 0.0: Completely wrong or irrelevant

Reply with ONLY a number like 0.75, no explanation."""

    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model="MiniMax-M2.7",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.0,
            )
            text = r.choices[0].message.content.strip()
            for token in text.replace(",", ".").split():
                try:
                    score = float(token)
                    if 0 <= score <= 1:
                        return score
                except ValueError:
                    pass
            time.sleep(0.5)
            return 0.0
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                wait = 30 * (attempt + 1)
                print(f"  rate limited, waiting {wait}s...", flush=True)
                time.sleep(wait)
            else:
                if attempt < 2:
                    time.sleep(5)  # retry on other errors
                else:
                    print(f"  API error (skipping): {str(e)[:80]}", flush=True)
                    return 0.0


def test_one(model, tok, tasks, name: str, n: int = 50):
    model.eval()
    dev = model.device
    total = 0.0

    for i, task in enumerate(tasks[:n]):
        prompt = (
            f"<|im_start|>system\n{SP}\n<|im_end|>\n"
            f"<|im_start|>user\n{task.instruction}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(dev)
        ilen = inp.input_ids.shape[1]
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=512, do_sample=True,
                                temperature=0.6, top_p=0.95, pad_token_id=tok.pad_token_id)
        plan = tok.decode(out[0][ilen:], skip_special_tokens=True)

        agent_actions = []
        for a_str in plan.split("|"):
            a_str = a_str.strip()
            act = parse_action_from_text(a_str)
            if not act:
                act = Action(name="respond", kwargs={"content": a_str})
            agent_actions.append(a_str)
            if act.name == "respond":
                break

        gt_acts = [(a.name, a.kwargs) for a in task.actions
                   if a.name not in ("respond", "transfer_to_human_agents", "think")]
        score = judge_with_minimax(task.instruction, gt_acts, agent_actions)
        total += score

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n}] avg={total/(i+1):.3f}", flush=True)

    avg = total / n
    print(f"  {name}: {avg:.4f}", flush=True)
    return avg


if __name__ == "__main__":
    import argparse, glob
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="all")
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()

    BASE = "/mnt/home/user41/downloaded_models/Qwen/Qwen2.5-7B-Instruct"

    print("Loading tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    if tok.padding_side == "right": tok.padding_side = "left"

    print(f"Loading test set ({args.n} tasks)...", flush=True)
    tasks = load_tau_bench_dataset(n_samples=args.n, split="eval", seed=9999, use_simia=True).tasks

    # Find latest checkpoint per mode, 7B only (skip old 1.5B)
    cp_dirs = {}
    targets = ["sparse", "adaptive", "router", "router_r2", "router_r3"]
    for mode in targets:
        matches = sorted(glob.glob(f"outputs/quick_{mode}_*/checkpoint_best"))
        if matches:
            cp_dirs[mode] = matches[-1]  # latest one

    results = {}
    for mode, cp in sorted(cp_dirs.items()):
        print(f"\n=== {mode} ({cp}) ===", flush=True)
        try:
            m = PeftModel.from_pretrained(
                AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True),
                cp,
            ).eval()
        except Exception as e:
            print(f"  SKIP: load error ({str(e)[:80]})", flush=True)
            continue
        avg = test_one(m, tok, tasks, mode, args.n)
        results[mode] = round(avg, 4)
        del m; torch.cuda.empty_cache()

    print(f"\n=== REAL EVAL RESULTS (MiniMax M2.7 Judge) ===")
    for mode, r in sorted(results.items()):
        print(f"  {mode:<30} {r:.4f}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/real_eval_minimax.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to outputs/real_eval_minimax.json")
