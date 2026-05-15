# ENTROPIA 项目状态

## 已完成实验

### 80步 MiniMax RL (5组)
| 实验 | 方法 | MiniMax 测试分 |
|------|------|---------------|
| sparse | 纯稀疏基线 | 0.125 |
| adaptive | v1 熵门控 | 0.085 |
| router R1 | v2 加法路由 | 0.134 |
| router R2 | v2 乘法路由 | 0.079 |
| router R3 | v2 Softmax | 0.128 |

### 200步 MiniMax RL (3组)
- router R1: 已完成，约 2600 次 MiniMax 调用
- router R3: 已完成，约 2600 次 MiniMax 调用
- sparse: 排队中

## 关键修复

1. MiniMax M2.7 judge: system prompt + max_tokens=500 + parse after `</think>`
2. SFT checkpoint 保存: `outputs/quick_XX_XXXX/sft_checkpoint/`
3. 跳过SFT: `--sft_checkpoint PATH`
4. OOM: 2query/2rollout (不要用4/4)
5. OpenAI 懒加载: 避免 CUDA 导入冲突
6. CUDA assert: 部分节点有问题，用 `--sft_checkpoint` 跳过

## 训练命令

```bash
# 跳过 SFT 的训练
python scripts/quick_cmp.py --mode router --n_steps 200 \
  --sft_checkpoint outputs/quick_adaptive_1408/sft_checkpoint
```

## 评测命令

```bash
# MiniMax LLM judge 评测
sbatch run_real_eval.sh

# 独立测试
python scripts/test_minimax.py --n 50
```

## 新增文件

- `envs/rule_based_env.py` — 状态机规则评测器
- `scripts/test_tau2.py` — τ²-Bench 集成
