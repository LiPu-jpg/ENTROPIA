"""
Final test evaluation: load saved checkpoints, evaluate on fresh test set.
"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from data.tau_dataset import load_tau_bench_dataset
from envs.mock_env import MockTauEnv, parse_action_from_text, Action

SP = "You are a customer service agent. Available tools: find_user_id_by_email, find_user_id_by_name_zip, get_user_details, get_order_details, get_product_details, get_flight_status, get_reservation_details, cancel_pending_order, cancel_reservation, return_delivered_order_items, exchange_delivered_order_items, modify_pending_order_items, modify_pending_order_address, modify_pending_order_payment, search_direct_flight, search_onestop_flight, book_reservation, update_reservation_flights, update_reservation_passengers, update_reservation_baggages, send_certificate, calculate, transfer_to_human_agents, respond. Format: tool_name(key='val'). Separate with ' | '. End with respond('summary')."

def test(model, tok, tasks, name):
    model.eval()
    ok, dev = 0, model.device
    for i, t in enumerate(tasks):
        p = f"<|im_start|>system\n{SP}\n<|im_end|>\n<|im_start|>user\n{t.instruction}\n<|im_end|>\n<|im_start|>assistant\n"
        inp = tok(p, return_tensors="pt", truncation=True, max_length=4096).to(dev)
        ilen = inp.input_ids.shape[1]
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=512, do_sample=True, temperature=0.6, top_p=0.95, pad_token_id=tok.pad_token_id)
        plan = tok.decode(out[0][ilen:], skip_special_tokens=True)
        env = MockTauEnv(t); env.reset()
        for a in plan.split("|"):
            act = parse_action_from_text(a.strip())
            if not act: act = Action(name="respond", kwargs={"content": a.strip()})
            r = env.step(act)
            if r.done: ok += int(r.reward > 0); break
        else: env.step(Action(name="respond", kwargs={"content": "Done."})); ok += int(env.reward > 0)
        if (i+1) % 50 == 0: print(f"  [{i+1}/{len(tasks)}] acc={ok/(i+1):.3f}", flush=True)
    acc = ok / len(tasks)
    print(f"  {name}: {ok}/{len(tasks)} = {acc:.4f}", flush=True)
    return acc

if __name__ == "__main__":
    BASE = "/mnt/home/user41/downloaded_models/Qwen/Qwen2.5-7B-Instruct"
    SPATH = "/mnt/home/user46/ENTROPIA/outputs/quick_sparse_1514/checkpoint_best"
    APATH = "/mnt/home/user46/ENTROPIA/outputs/quick_adaptive_1532/checkpoint_best"

    print("Load tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    if tok.padding_side == "right": tok.padding_side = "left"

    print("Load sparse...", flush=True)
    m1 = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True), SPATH).eval()

    print("Load adaptive...", flush=True)
    m2 = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True), APATH).eval()

    print("Load 200 test tasks...", flush=True)
    tasks = load_tau_bench_dataset(n_samples=200, split="eval", seed=123, use_simia=True).tasks
    print(f"Test: {len(tasks)} tasks\n", flush=True)

    r1 = test(m1, tok, tasks, "sparse")
    r2 = test(m2, tok, tasks, "adaptive")
    print(f"\n=== RESULTS ===", flush=True)
    print(f"sparse:   {r1:.4f}", flush=True)
    print(f"adaptive: {r2:.4f}", flush=True)
    print(f"delta:    {r2-r1:+.4f}", flush=True)
