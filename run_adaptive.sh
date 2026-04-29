#!/bin/bash
#SBATCH --job-name=ent_adapt
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00

echo "=== ENTROPIA 7B: ADAPTIVE entropy-gated ==="
echo "Node: $(hostname)"
nvidia-smi

source /mnt/data/hpc/support/soft/anaconda3/bin/activate entropia
cd /mnt/home/user46/ENTROPIA
mkdir -p logs outputs

export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

python -u scripts/quick_cmp.py --mode adaptive --n_steps 50 --model /mnt/home/user41/downloaded_models/Qwen/Qwen2.5-7B-Instruct

echo "Done."
