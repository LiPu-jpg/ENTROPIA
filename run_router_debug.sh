#!/bin/bash
#SBATCH --job-name=ent_M1_v2
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/home/user46/ENTROPIA/logs/M1_router_%j.out
#SBATCH --error=/mnt/home/user46/ENTROPIA/logs/M1_router_%j.err

source /mnt/data/hpc/support/soft/anaconda3/bin/activate entropia
cd /mnt/home/user46/ENTROPIA
mkdir -p logs outputs
export WANDB_MODE=offline PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1

python -u scripts/quick_cmp.py --mode router --n_steps 50 --model /mnt/home/user41/downloaded_models/Qwen/Qwen2.5-7B-Instruct
