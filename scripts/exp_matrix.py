"""
实验矩阵配置 — ENTROPIA v2 完整比较。

每个实验: {id, mode, label, ablation, priority}

priority:
  1 = 必须跑（论文必需）
  2 = 消融（核心论据）
  3 = 可选（锦上添花）
"""

EXPERIMENTS = [
    # ── Must-Have Baselines ──
    {"id": "B0_sparse",     "mode": "sparse",     "priority": 1, "label": "Sparse GRPO (ReTool)"},
    {"id": "B1_dense_ig",   "mode": "dense_igpo", "priority": 1, "label": "Fixed Dense IG (IGPO)"},
    {"id": "B2_craft",      "mode": "dense_fixed","priority": 2, "label": "Fixed CRAFT composite"},
    {"id": "B3_autotool",   "mode": "autotool_entropy", "priority": 2, "label": "AutoTool entropy loss"},
    {"id": "B4_v1_adaptive","mode": "adaptive",   "priority": 1, "label": "ENTROPIA v1 (entropy only)"},
    {"id": "B5_random",     "mode": "random_gate","priority": 1, "label": "Random Gate (sanity check)"},

    # ── ENTROPIA v2 Main Method ──
    {"id": "M1_v2_router",  "mode": "router",     "priority": 1, "label": "ENTROPIA v2 (full NUR router)"},

    # ── v2 Ablations ──
    {"id": "A1_need_only",  "mode": "router_need_only", "priority": 2, "label": "v2: Need Gate Only"},
    {"id": "A2_no_rel",     "mode": "router_no_reliability", "priority": 2, "label": "v2: No Reliability Gate"},
    {"id": "A3_no_risk",    "mode": "router_no_risk", "priority": 2, "label": "v2: No Risk Controller"},

    # ── Optional Upper Bounds ──
    # {"id": "U1_oracle",   "mode": "oracle_gate", "priority": 3, "label": "Oracle Gate (upper bound)"},
]

# 跑哪些 priority 级别的实验 (1=全跑, 2=消融, 3=可选)
RUN_PRIORITY = 2

# 训练参数
TRAIN_CONFIG = {
    "model": "/mnt/home/user41/downloaded_models/Qwen/Qwen2.5-7B-Instruct",
    "n_steps": 50,
    "seed": 42,
    "use_simia": True,
}

# 测试参数  
TEST_CONFIG = {
    "n_test_samples": 200,
    "test_seed": 123,
}

# Slurm 配置
SLURM_CONFIG = {
    "partition": "q_intel_share_L20",
    "gpus": 2,
    "cpus": 16,
    "mem": "128G",
    "time": "12:00:00",
}
