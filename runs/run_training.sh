#!/bin/bash
# General training script
# Usage: ./run_training.sh [CONFIG] [RUN_NAME] [RUN_GROUP] [NUM_GPUS]

set -e
CONFIG="${1:-launch_configs/parcae/pretrain/parcae-small-140m.yaml}"
RUN_NAME="${2:-parcae-small-140m}"
RUN_GROUP="${3:-parcae}"
NUM_GPUS="${4:-8}"

echo "=============================================="
echo "Starting run: ${RUN_NAME}"
echo "Config: ${CONFIG}"
echo "GPUs: ${NUM_GPUS}"
echo "=============================================="

torchrun --nproc_per_node=${NUM_GPUS} \
    scripts/train.py \
    --config ${CONFIG} \
    --run_name ${RUN_NAME} \
    --logger_group ${RUN_GROUP}

echo "=============================================="
echo "Finished: ${RUN_NAME}"
echo "=============================================="

