#!/bin/bash
set -e

MODEL="parcae-small-140m"
CONFIG="launch_configs/parcae/pretrain/parcae-small-140m.yaml"
LABEL="feb26"
GPUS=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=""
MASTER_PORT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --label) LABEL="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --nnodes) NNODES="$2"; shift 2 ;;
        --node_rank) NODE_RANK="$2"; shift 2 ;;
        --master_addr) MASTER_ADDR="$2"; shift 2 ;;
        --master_port) MASTER_PORT="$2"; shift 2 ;;
        *) shift ;;
    esac
done

RECURRENCES=(1 2 4 6 8 10 12)
FLOPS_BUDGETS=(1e18 2e18 4e18 8e18 16e18)

BASE_DIR="outputs/parcae_scaling_laws/${MODEL}_${LABEL}"
export OMP_NUM_THREADS=1

if [[ -n "$MASTER_ADDR" ]]; then
    TORCHRUN="torchrun --nnodes=$NNODES --nproc_per_node=$GPUS --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"
else
    TORCHRUN="torchrun --standalone --nproc_per_node=$GPUS"
fi

for flops in "${FLOPS_BUDGETS[@]}"; do
    for rec in "${RECURRENCES[@]}"; do
        bwd=$(((rec + 1) / 2))
        OUT_DIR="${BASE_DIR}/rec_${rec}/flops_${flops}"
        RUN_NAME="${MODEL}-rec${rec}-${flops}"

        HAS_CKPT=false
        HAS_EVAL=false
        ls "$OUT_DIR"/checkpoints-*/step-* &>/dev/null && HAS_CKPT=true
        [ -f "$OUT_DIR/eval.log" ] && HAS_EVAL=true

        if $HAS_CKPT && $HAS_EVAL; then
            echo "Skipping rec=$rec flops=$flops (complete)"
            continue
        fi

        if ! $HAS_CKPT; then
            echo "Training rec=$rec flops=$flops -> $OUT_DIR"
            mkdir -p "$OUT_DIR"
            $TORCHRUN scripts/train.py \
                --config "$CONFIG" \
                --out_dir "$OUT_DIR" \
                --run_name "$RUN_NAME" \
                --resume false \
                --max_flops "$flops" \
                --model_overwrite.mean_recurrence "$rec" \
                --model_overwrite.mean_backprop_depth "$bwd" \
                --logger_project "parcae-scaling-laws-2" \
                --logger_group "$MODEL" \
                --save_step_interval 999999999 \
                --eval_step_interval 999999999 \
                2>&1 | tee "${OUT_DIR}/train.log"
        fi

        if ! $HAS_EVAL; then
            echo "Evaluating rec=$rec flops=$flops"
            $TORCHRUN scripts/eval.py \
                --out_dir "$OUT_DIR" \
                --config eval_configs/eval-val-loss.yaml \
                2>&1 | tee "${OUT_DIR}/eval.log"
        fi
    done
done

echo "Sweep complete: $BASE_DIR"

