#!/bin/bash
#SBATCH --job-name=entropia
#SBATCH --partition=q_intel_share_L20
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=/mnt/home/user46/ENTROPIA/logs/%j_%x.out
#SBATCH --error=/mnt/home/user46/ENTROPIA/logs/%j_%x.err
#SBATCH --time=48:00:00

# ENTROPIA training script for 4xL20 GPUs
# Usage: sbatch run_entropia.sh [adaptive|sparse|dense_igpo|dense_fixed|autotool_entropy] [ablation]

set -e

MODE=${1:-adaptive}
ABLATION=${2:-}

echo "Starting ENTROPIA training: mode=$MODE ablation=$ABLATION"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

source /mnt/data/hpc/support/soft/anaconda3/bin/activate entropia

cd /mnt/home/user46/ENTROPIA
mkdir -p logs outputs

export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

if [ -n "$ABLATION" ]; then
    python scripts/run.py --mode "$MODE" --ablation "$ABLATION" --output_dir "./outputs/${MODE}_${ABLATION}"
else
    python scripts/run.py --mode "$MODE" --output_dir "./outputs/${MODE}"
fi

echo "Training completed."
