#!/bin/bash
# General finetuning script
# Usage: ./run_finetuning.sh [CONFIG] [RUN_NAME] [RUN_GROUP] [NUM_GPUS] [extra finetune.py args...]
# Configs in launch_configs/benice/finetune need
# PARCAE_TOKENIZER=SandyResearch/parcae-climbmix-tokenizer

set -e
CONFIG="${1:-launch_configs/parcae/finetune/benice-770m.yaml}"
RUN_NAME="${2:-$(basename "${CONFIG}" .yaml)}"
RUN_GROUP="${3:-benice}"
NUM_GPUS="${4:-8}"
EXTRA_ARGS=("${@:5}")

echo "=============================================="
echo "Starting run: ${RUN_NAME}"
echo "Config: ${CONFIG}"
echo "GPUs: ${NUM_GPUS}"
echo "=============================================="

torchrun --nproc_per_node=${NUM_GPUS} \
    scripts/finetune.py \
    --config ${CONFIG} \
    --run_name ${RUN_NAME} \
    --out_dir outputs/${RUN_NAME} \
    --wandb_group ${RUN_GROUP} \
    "${EXTRA_ARGS[@]}"

echo "=============================================="
echo "Finished: ${RUN_NAME}"
echo "=============================================="

