#!/usr/bin/env python
"""
全自动实验流水线: SFT → RL → Test → 结果汇总

用法:
  python scripts/run_pipeline.py --priority 1     # 只跑必须的baseline
  python scripts/run_pipeline.py --priority 2     # 包含消融
  python scripts/run_pipeline.py --test_only      # 只跑测试（训练已做完）
  python scripts/run_pipeline.py --summary_only   # 只看汇总
"""

import sys, os, json, subprocess, time, argparse
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJ = "/mnt/home/user46/ENTROPIA"
OUTDIR = f"{PROJ}/outputs"
RESULTS = f"{PROJ}/outputs/exp_results.json"

EXP_MATRIX = [
    {"id":"B0_sparse","mode":"sparse","priority":1,"label":"Sparse GRPO (ReTool)"},
    {"id":"B4_v1","mode":"adaptive","priority":1,"label":"ENTROPIA v1 entropy-gate"},
    {"id":"M1_router","mode":"router","priority":1,"label":"v2 NUR-R1 additive"},
    {"id":"M2_router_r2","mode":"router_r2","priority":1,"label":"v2 NUR-R2 multiplicative"},
    {"id":"M3_router_r3","mode":"router_r3","priority":1,"label":"v2 NUR-R3 softmax"},
]

MODEL = "/mnt/home/user41/downloaded_models/Qwen/Qwen2.5-7B-Instruct"
N_STEPS = 80
N_TEST = 200
TEST_SEED = 123


def submit_train(mode: str, exp_id: str, seed: int = 42) -> int:
    """提交训练 job，返回 slurm job ID。"""
    script = f"""#!/bin/bash
#SBATCH --job-name=ent_{exp_id[:10]}
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output={PROJ}/logs/{exp_id}_%j.out
#SBATCH --error={PROJ}/logs/{exp_id}_%j.err

source /mnt/data/hpc/support/soft/anaconda3/bin/activate entropia
cd {PROJ}
mkdir -p logs outputs/exp_{exp_id}

export WANDB_MODE=offline PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_LAUNCH_BLOCKING=1

python -u scripts/quick_cmp.py \\
    --mode {mode} --model {MODEL} --n_steps {N_STEPS} --seed {seed}

# Move checkpoint to experiment dir
LATEST_DIR=$(ls -td outputs/quick_{mode}_* 2>/dev/null | head -1)
if [ -n "$LATEST_DIR" ] && [ -d "$LATEST_DIR/checkpoint_best" ]; then
    cp -r "$LATEST_DIR/checkpoint_best" outputs/exp_{exp_id}/
fi
echo "EXIT_CODE:$?"
"""
    result = subprocess.run(["sbatch"], input=script, capture_output=True, text=True, cwd=PROJ)
    if result.returncode != 0:
        print(f"  submit failed: {result.stderr}")
        return -1
    try:
        job_id = int(result.stdout.strip().split()[-1])
        print(f"  submitted: job {job_id}")
        return job_id
    except:
        return -1


def submit_test(exp_id: str, checkpoint_path: str) -> int:
    """提交测试 job。"""
    script = f"""#!/bin/bash
#SBATCH --job-name=tst_{exp_id[:10]}
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output={PROJ}/logs/test_{exp_id}_%j.out
#SBATCH --error={PROJ}/logs/test_{exp_id}_%j.err

source /mnt/data/hpc/support/soft/anaconda3/bin/activate entropia
cd {PROJ}

python -u -c "
import sys, os, json, torch
sys.path.insert(0, '{PROJ}')
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from training.trainer import AdaptiveRewardTrainer
from core.reward_router import RewardRouter, NeedGate, UtilityGate, ReliabilityGate, RiskController
from core.signal_bank import SignalBank
from data.tau_dataset import load_tau_bench_dataset, SFTDataset
from envs.mock_env import MockTauEnv, parse_action_from_text, Action

SP = 'You are a customer service agent. Available tools: find_user_id_by_email, find_user_id_by_name_zip, get_user_details, get_order_details, get_product_details, get_flight_status, get_reservation_details, cancel_pending_order, cancel_reservation, return_delivered_order_items, exchange_delivered_order_items, modify_pending_order_items, modify_pending_order_address, modify_pending_order_payment, search_direct_flight, search_onestop_flight, book_reservation, update_reservation_flights, update_reservation_passengers, update_reservation_baggages, send_certificate, calculate, transfer_to_human_agents, respond. Format: tool_name(key=val). Separate with |. End with respond.'

def test(model, tok, tasks):
    model.eval()
    ok, dev = 0, model.device
    for i, t in enumerate(tasks):
        p = f\"<|im_start|>system\\n{{SP}}\\n<|im_end|>\\n<|im_start|>user\\n{{t.instruction}}\\n<|im_end|>\\n<|im_start|>assistant\\n\"
        inp = tok(p, return_tensors='pt', truncation=True, max_length=4096).to(dev)
        ilen = inp.input_ids.shape[1]
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=512, do_sample=True, temperature=0.6, top_p=0.95, pad_token_id=tok.pad_token_id)
        plan = tok.decode(out[0][ilen:], skip_special_tokens=True)
        env = MockTauEnv(t); env.reset()
        for a in plan.split('|'):
            act = parse_action_from_text(a.strip())
            if not act: act = Action(name='respond', kwargs={{'content': a.strip()}})
            r = env.step(act)
            if r.done: ok += int(r.reward > 0); break
        else: env.step(Action(name='respond', kwargs={{'content': 'Done.'}})); ok += int(env.reward > 0)
        if (i+1) % 50 == 0: print(f'  [{{i+1}}/{{len(tasks)}}] acc={{ok/(i+1):.3f}}', flush=True)
    return ok / len(tasks)

BASE = '{MODEL}'
CP = '{checkpoint_path}'
tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None: tok.pad_token = tok.eos_token
if tok.padding_side == 'right': tok.padding_side = 'left'

m = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True), CP).eval()
tasks = load_tau_bench_dataset(n_samples={N_TEST}, split='eval', seed={TEST_SEED}, use_simia=True).tasks
print(f'Test: {{len(tasks)}} tasks', flush=True)
acc = test(m, tok, tasks)
print(f'RESULT: {{acc:.4f}}', flush=True)
with open('{RESULTS}', 'r') as f: all_r = json.load(f)
all_r['{exp_id}'] = round(acc, 4)
with open('{RESULTS}', 'w') as f: json.dump(all_r, f, indent=2)
print('DONE', flush=True)
"
"""
    result = subprocess.run(["sbatch"], input=script, capture_output=True, text=True, cwd=PROJ)
    if result.returncode != 0:
        print(f"  test submit failed: {result.stderr}")
        return -1
    try:
        job_id = int(result.stdout.strip().split()[-1])
        print(f"  test submitted: job {job_id}")
        return job_id
    except:
        return -1


def get_job_status(job_id: int) -> str:
    """查询 slurm job 状态。"""
    r = subprocess.run(["sacct", "-j", str(job_id), "--format=State", "--noheader", "--parsable2"],
                       capture_output=True, text=True)
    states = [s.strip() for s in r.stdout.strip().split("\n") if s.strip()]
    if not states:
        return "UNKNOWN"
    if "RUNNING" in states:
        return "RUNNING"
    if "PENDING" in states:
        return "PENDING"
    if all(s in ("COMPLETED",) for s in states):
        return "COMPLETED"
    if any(s in ("FAILED", "TIMEOUT", "CANCELLED") for s in states):
        return "FAILED"
    return states[0]


def wait_jobs(job_ids: list, poll_sec: int = 60):
    """等待所有 job 完成。"""
    pending = dict(job_ids)
    while pending:
        for jid, exp_id in list(pending.items()):
            status = get_job_status(jid)
            if status == "COMPLETED":
                print(f"  [{exp_id}] done")
                del pending[jid]
            elif status in ("FAILED", "TIMEOUT", "CANCELLED"):
                print(f"  [{exp_id}] FAILED ({status})")
                del pending[jid]
        if pending:
            print(f"  waiting for {len(pending)} jobs...")
            time.sleep(poll_sec)


def run_training(priority: int):
    """提交所有训练 job。"""
    os.makedirs(f"{PROJ}/logs", exist_ok=True)
    os.makedirs(f"{PROJ}/outputs", exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump({"_timestamp": datetime.now().isoformat()}, f)

    jobs = []
    for exp in EXP_MATRIX:
        if exp["priority"] > priority:
            continue
        print(f"[{exp['id']}] {exp['label']} (mode={exp['mode']})")
        jid = submit_train(exp["mode"], exp["id"])
        if jid > 0:
            jobs.append((jid, exp["id"], exp["mode"]))
        print()

    # 保存 job 映射
    with open(f"{PROJ}/outputs/_jobs.json", "w") as f:
        json.dump([{"job_id": j, "exp_id": e, "mode": m} for j, e, m in jobs], f)

    print(f"Submitted {len(jobs)} training jobs.")
    return jobs


def run_tests(checkpoint_map: dict):
    """为所有 checkpoint 提交测试 job。"""
    jobs = []
    for exp_id, cp_path in checkpoint_map.items():
        if not os.path.exists(cp_path):
            print(f"  [{exp_id}] checkpoint not found: {cp_path}")
            continue
        jid = submit_test(exp_id, cp_path)
        if jid > 0:
            jobs.append((jid, exp_id))
    return jobs


def find_checkpoints():
    """自动发现 output 目录下的 checkpoint。"""
    import glob
    cp_map = {}
    for exp in EXP_MATRIX:
        # 按模式匹配目录
        pattern = f"{PROJ}/outputs/exp_{exp['id']}/checkpoint_best/adapter_model.safetensors"
        if os.path.exists(pattern):
            cp_map[exp["id"]] = os.path.dirname(pattern)
        else:
            # 回退: 搜索 quick_* 目录
            dirs = sorted(glob.glob(f"{PROJ}/outputs/quick_{exp['mode']}_*/checkpoint_best"), reverse=True)
            if dirs:
                cp_map[exp["id"]] = dirs[0]
    return cp_map


def show_summary():
    """显示汇总结果。"""
    if not os.path.exists(RESULTS):
        print("No results yet.")
        return
    with open(RESULTS) as f:
        data = json.load(f)
    print(f"\n{'='*60}")
    print(f"ENTROPIA v2 Experiment Results")
    print(f"{'='*60}")
    print(f"{'ID':<20} {'Label':<35} {'Test Acc':>10}")
    print(f"{'-'*65}")
    best = ("", 0.0)
    for exp in EXP_MATRIX:
        eid = exp["id"]
        if eid in data:
            acc = data[eid]
            marker = " ← BEST" if acc > best[1] else ""
            print(f"{eid:<20} {exp['label']:<35} {acc:>10.4f}{marker}")
            if acc > best[1]:
                best = (eid, acc)
    print(f"{'='*60}")
    print(f"Best: {best[0]} ({best[1]:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ENTROPIA experiment pipeline")
    parser.add_argument("--priority", type=int, default=1, help="最高 priority 级别 (1=必需, 2=消融, 3=可选)")
    parser.add_argument("--test_only", action="store_true", help="只运行测试（跳过训练）")
    parser.add_argument("--summary_only", action="store_true", help="只显示汇总")
    parser.add_argument("--wait", action="store_true", help="等待所有 job 完成")
    args = parser.parse_args()

    if args.summary_only:
        show_summary()
        sys.exit(0)

    if args.test_only:
        cps = find_checkpoints()
        if not cps:
            print("No checkpoints found. Run training first.")
        else:
            print(f"Found {len(cps)} checkpoints:")
            for eid, cp in cps.items():
                print(f"  {eid}: {cp}")
            jobs = run_tests(cps)
            print(f"Submitted {len(jobs)} test jobs.")
            if args.wait and jobs:
                wait_jobs(jobs, 60)
        show_summary()
        sys.exit(0)

    # 训练阶段
    jobs = run_training(args.priority)
    if args.wait and jobs:
        wait_jobs(jobs, 120)

    if not args.wait:
        print(f"\nAll {len(jobs)} jobs submitted. Use --wait to block until completion.")
    show_summary()
