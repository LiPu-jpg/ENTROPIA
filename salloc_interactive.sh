#!/bin/bash
# Interactive GPU session for ENTROPIA development
# Usage: bash salloc_interactive.sh [num_gpus]

GPUS=${1:-4}
salloc --partition=q_intel_share_L20 --gres=gpu:$GPUS --cpus-per-task=32 --mem=256G --time=12:00:00 --job-name=entropia_dev
