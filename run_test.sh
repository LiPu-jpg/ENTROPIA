#!/bin/bash
#SBATCH --job-name=ent_test
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/mnt/home/user46/ENTROPIA/logs/test_%j.out
#SBATCH --error=/mnt/home/user46/ENTROPIA/logs/test_%j.err
#SBATCH --time=4:00:00

source /mnt/data/hpc/support/soft/anaconda3/bin/activate entropia
cd /mnt/home/user46/ENTROPIA
mkdir -p logs

export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

python -u scripts/test_eval.py

echo "Test done."
