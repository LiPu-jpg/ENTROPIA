# ENTROPIA 项目交接提示词给 Codex

## 项目位置
```
/mnt/home/user46/ENTROPIA
GitHub: https://github.com/LiPu-jpg/ENTROPIA
Python 环境: source /mnt/data/hpc/support/soft/anaconda3/bin/activate entropia
基础模型: /mnt/home/user41/downloaded_models/Qwen/Qwen2.5-7B-Instruct
```

## 项目目标
ENTROPIA — Entropy-Gated Adaptive Reward Density for LLM Agent RL。核心创新：用 Need × Utility × Reliability 三因子门控，动态路由过程奖励信号到 GRPO 训练中。`docs/paper_formula_methodology.md` 和 `docs/design_v2_reward_routing.md` 有完整公式。

## 当前状态

### 已跑完的实验（Simia 工具调用 + MiniMax judge）
MiniMax judge 在用：system prompt "Score agents 0.0-1.0. Output ONLY one number." + max_tokens=500 + 解析 `</think>` 后的数字。API key 在 `training/trainer.py` 里。

| mode | 方法 | MiniMax 测试分 |
|------|------|:---:|
| sparse | 纯 GRPO baseline | 0.125 |
| adaptive | v1 entropy-gate | 0.085 |
| router (R1 additive) | v2 NUR | 0.134 |
| router_r2 (R2 multiplicative) | v2 NUR | 0.079 |
| router_r3 (R3 softmax) | v2 NUR | 0.128 |

训练命令：
```bash
# 如果有 SFT checkpoint 可以直接跳过 SFT
python scripts/quick_cmp.py --mode router --n_steps 200 \
  --sft_checkpoint outputs/quick_adaptive_1408/sft_checkpoint \
  --model /mnt/home/user41/downloaded_models/Qwen/Qwen2.5-7B-Instruct
```

### 核心代码
- `core/reward_router.py` — Need/Utility/Reliability 三因子门控 + Risk budget 控制器
- `core/signal_bank.py` — 过程信号库（info_gain/efficiency/relevance）
- `training/trainer.py` — GRPO trainer + MiniMax judge 集成 + SFT checkpoint 保存
- `envs/mock_env.py` — mock 环境（action 解析 + 多级评分）
- `envs/rule_based_env.py` — 新写的规则评测器（状态机+确定性打分）
- `scripts/quick_cmp.py` — 训练入口，支持 `--sft_checkpoint` 跳过 SFT
- `scripts/test_minimax.py` — MiniMax LLM judge 评测
- `scripts/run_pipeline.py` — 批量实验提交

### 已知坑
1. **CUDA assert 在 SFT 时部分节点崩溃** — 用 `--sft_checkpoint` 跳过 SFT 解决
2. **2 GPU 配置不敢开到 4rollout/4query** — OOM，当前用 2/2
3. **info_gain 和 relevance 全是 0** — compute_signals 没传真实 logprobs，只有 efficiency_cost 有信号
4. **MiniMax M2.7 有 `<think>` 块** — 需要 max_tokens≥500 且解析 `</think>` 后的内容
5. **L20 集群节点质量不一** — 部分节点 bad，提交后失败就重投直到成功
6. **Simia 任务在 SFT 后被模型背熟** — mock eval 永远是 1.0，区分度靠 MiniMax judge

### 当前 12 个信号里 6 个真在工作
- ✅ outcome_score (MiniMax 或 mock)
- ✅ step_entropy, ✅ group_collapse (Need Gate)
- ✅ efficiency_cost (Utility Gate)
- ✅ ρ correlation, ✅ ξ agreement, ✅ δ discriminative (Reliability Gate)
- ✗ info_gain, ✗ relevance, ✗ stagnation, ✗ format_valid, ✗ χ gaming — 全部未实现或永远 0

## 下一步：转 Search-R1 风格 Search QA

advisor 建议主线换 Search QA，原因：info_gain/relevance/efficiency 在搜索场景最自然，且 Search-R1 是公认 baseline。

### 需要做的
1. **数据集** — HotpotQA 或 NaturalQuestions 训练集
2. **检索环境** — 本地 Wikipedia dump + 稠密检索器（如 E5 或 BGE），替代 mock
3. **真实信号** — info_gain（新文档包含答案关键词→正值）、relevance（文档和答案的 NLI/BM25 匹配）、efficiency（多余搜索→惩罚）
4. **baseline** — 跑 Search-R1 的 outcome-only GRPO 做对比
5. **评测** — HotpotQA/NQ 的 EM/F1

### 可能会用到的资源
```
Search-R1 paper: arxiv 2602.19526 (outcome-only GRPO for search)
R1-Searcher: 更完整的 search RL 实现
Wikipedia 2018 dump: 群里可能有存过的
E5 retriever: intfloat/e5-base-v2
HotpotQA: huggingface datasets load
```
