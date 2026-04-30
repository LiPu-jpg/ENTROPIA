#!/bin/bash
#SBATCH --job-name=ent_real
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/mnt/home/user46/ENTROPIA/logs/real_eval_%j.out
#SBATCH --error=/mnt/home/user46/ENTROPIA/logs/real_eval_%j.err
#SBATCH --time=4:00:00

echo "=== ENTROPIA Real Evaluation (GPT-4o Judge) ==="
echo "Node: $(hostname)"

source /mnt/data/hpc/support/soft/anaconda3/bin/activate entropia
cd /mnt/home/user46/ENTROPIA
mkdir -p logs

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# 先测已有的 sparse + adaptive checkpoints
python -u scripts/test_real.py --mode sparse --n 50
python -u scripts/test_real.py --mode adaptive --n 50

echo "Done."
