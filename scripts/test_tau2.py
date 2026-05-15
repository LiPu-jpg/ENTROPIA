"""
τ²-Bench 真实评测: 加载 checkpoint，用 τ²-Bench 规则系统打分。
"""
import sys, os, json, torch, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def run_tau2_eval(checkpoint_path: str, domain: str = "retail", n_tasks: int = 50):
    """用 τ²-Bench 评测一个 checkpoint。"""
    import subprocess
    tau2_python = "/mnt/home/user46/.conda/envs/tau2/bin/python"

    script = f"""
import sys, json, torch
sys.path.insert(0, '/mnt/home/user46/ENTROPIA')
sys.path.insert(0, '/mnt/home/user46/tau2-bench/src')
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tau2.run import build_environment, resolve_component_names
from omegaconf import OmegaConf

cp = '{checkpoint_path}'
domain = '{domain}'

base_model = AutoModelForCausalLM.from_pretrained(
    '/mnt/home/user41/downloaded_models/Qwen/Qwen2.5-7B-Instruct',
    torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, cp).eval()
tok = AutoTokenizer.from_pretrained('/mnt/home/user41/downloaded_models/Qwen/Qwen2.5-7B-Instruct')
if tok.pad_token is None: tok.pad_token = tok.eos_token

# Build environment
names = resolve_component_names(domain=domain, task_set_name=domain, user_mode='user_simulator')
env = build_environment(seed=42, **names)
n = min({n_tasks}, len(env.tasks))
results = []

for i in range(n):
    task = env.tasks[i]
    obs = env.reset(task.id)
    done = False
    while not done:
        prompt = obs
        inp = tok(prompt, return_tensors='pt', truncation=True, max_length=4096).to(model.device)
        ilen = inp.input_ids.shape[1]
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=256, do_sample=True, temperature=0.6, top_p=0.95, pad_token_id=tok.pad_token_id)
        action_text = tok.decode(out[0][ilen:], skip_special_tokens=True)
        obs, reward, done, info = env.step(action_text)
    results.append({{'task_id': task.id, 'reward': reward, 'done': done}})
    if (i+1) % 10 == 0:
        succ = sum(1 for r in results if r['reward'] > 0)
        print(f'  [{i+1}/{n}] success={succ}/{i+1}', flush=True)

succ = sum(1 for r in results if r['reward'] > 0)
print(f'RESULT: {{succ}}/{{len(results)}} = {{succ/len(results):.4f}}', flush=True)
print(json.dumps({{'success': succ, 'total': len(results), 'rate': succ/len(results)}}))
"""
    result = subprocess.run(
        [tau2_python, "-c", script],
        capture_output=True, text=True, timeout=3600,
    )
    for line in result.stdout.strip().split("\n"):
        print(f"  {line}", flush=True)
    if result.stderr:
        print(f"  ERR: {result.stderr[-200:]}", flush=True)
    try:
        return float(result.stdout.split("RESULT:")[-1].split("/")[0].split()[-1])
    except:
        return 0.0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="all")
    parser.add_argument("--domain", type=str, default="retail")
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    targets = ["sparse", "router", "router_r3"]
    cp_dirs = {}
    for mode in targets:
        matches = sorted(glob.glob(f"outputs/quick_{mode}_*/checkpoint_best"))
        if matches:
            cp_dirs[mode] = matches[-1]

    results = {}
    for mode, cp in sorted(cp_dirs.items()):
        print(f"\n=== {mode} ({cp}) ===", flush=True)
        rate = run_tau2_eval(cp, args.domain, args.n)
        results[mode] = round(rate, 4)
        torch.cuda.empty_cache()

    print(f"\n=== TAU2-BENCH {args.domain} ===")
    for mode, r in sorted(results.items()):
        print(f"  {mode:<20} {r:.4f}")
    with open(f"outputs/tau2_eval_{args.domain}.json", "w") as f:
        json.dump(results, f, indent=2)
