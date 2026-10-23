#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=450000
#SBATCH --partition=oneday
#SBATCH --job-name=parcae-multinode
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --mail-type=ALL
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

# =============================================================================
# User configuration — set these for your environment
# =============================================================================
PROJECT_DIR="${PROJECT_DIR:-/data/$USER/parcae-pr}"     # Host path to this repo
DOCKER_IMAGE="${DOCKER_IMAGE:-ghcr.io/sandyresearch/parcae:sha-49c27cb}"
# =============================================================================

LOG_DIR="slurmlogs/${SLURM_JOB_ID}"
mkdir -p "${LOG_DIR}"
exec > "${LOG_DIR}/slurm.out" 2> "${LOG_DIR}/slurm.err"

set -x

echo "=============================================="
echo "Job started at $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NNODES}"
echo "Node list: ${SLURM_NODELIST}"
echo "=============================================="

CONFIG="${CONFIG:-launch_configs/parcae/pretrain/parcae-large-770m.yaml}"
RUN_NAME="${RUN_NAME:-parcae-multinode-${SLURM_JOB_ID}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
SWEEP_SCRIPT="${SWEEP_SCRIPT:-}"
MODEL="${MODEL:-}"
CONTAINER_NAME="$(whoami)-parcae-${SLURM_JOB_ID}"

if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

cd "${PROJECT_DIR}" || exit 1

MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
MASTER_PORT=${MASTER_PORT:-29501}

echo "Master address: ${MASTER_ADDR}:${MASTER_PORT}"

setup_node() {
    local NODE_RANK=$SLURM_NODEID
    echo "[Node ${NODE_RANK}] Setting up on $(hostname)..."

    nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r sudo kill -9 2>/dev/null || true

    if ! sudo docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^${DOCKER_IMAGE}$"; then
        echo "[Node ${NODE_RANK}] Pulling Docker image..."
        sudo docker pull "${DOCKER_IMAGE}"
    fi

    sudo docker stop "${CONTAINER_NAME}" 2>/dev/null || true
    sudo docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

    echo "[Node ${NODE_RANK}] Starting container ${CONTAINER_NAME}..."
    sudo docker run -d --rm --ipc host --gpus all \
        -v "${PROJECT_DIR}":/resource \
        --workdir /resource \
        --network host \
        --name "${CONTAINER_NAME}" \
        --privileged \
        "${DOCKER_IMAGE}" \
        sleep infinity

    for i in {1..30}; do
        if sudo docker exec "${CONTAINER_NAME}" echo "ready" &>/dev/null; then
            echo "[Node ${NODE_RANK}] Container ready!"
            break
        fi
        sleep 1
    done

    sudo docker exec "${CONTAINER_NAME}" pip install -q -e .
    echo "[Node ${NODE_RANK}] Setup complete!"
}

run_training() {
    local NODE_RANK=$SLURM_NODEID
    echo "[Node ${NODE_RANK}] Starting training on $(hostname)..."

    if [[ -n "${SWEEP_SCRIPT}" ]]; then
        sudo docker exec \
            -e WANDB_API_KEY="${WANDB_API_KEY}" \
            -e NCCL_DEBUG=INFO \
            -e NCCL_SOCKET_IFNAME=^lo,docker \
            -e SLURM_JOB_NUM_NODES=${SLURM_NNODES} \
            "${CONTAINER_NAME}" \
            bash ${SWEEP_SCRIPT} \
                --model ${MODEL} \
                --gpus ${GPUS_PER_NODE} \
                --nnodes ${SLURM_NNODES} \
                --node_rank ${NODE_RANK} \
                --master_addr ${MASTER_ADDR} \
                --master_port ${MASTER_PORT} \
                ${EXTRA_ARGS}
    else
        sudo docker exec \
            -e WANDB_API_KEY="${WANDB_API_KEY}" \
            -e NCCL_DEBUG=INFO \
            -e NCCL_SOCKET_IFNAME=^lo,docker \
            -e SLURM_JOB_NUM_NODES=${SLURM_NNODES} \
            "${CONTAINER_NAME}" \
            torchrun \
                --nnodes=${SLURM_NNODES} \
                --nproc_per_node=${GPUS_PER_NODE} \
                --node_rank=${NODE_RANK} \
                --master_addr=${MASTER_ADDR} \
                --master_port=${MASTER_PORT} \
                scripts/train.py \
                --config ${CONFIG} \
                --run_name="${RUN_NAME}" \
                ${EXTRA_ARGS}
    fi
}

cleanup_node() {
    sudo docker stop "${CONTAINER_NAME}" 2>/dev/null || true
    sudo docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
}

export -f setup_node run_training cleanup_node
export DOCKER_IMAGE CONTAINER_NAME PROJECT_DIR
export MASTER_ADDR MASTER_PORT GPUS_PER_NODE
export CONFIG RUN_NAME EXTRA_ARGS WANDB_API_KEY
export SWEEP_SCRIPT MODEL

trap 'srun --nodes=${SLURM_NNODES} --ntasks-per-node=1 bash -c "$(declare -f cleanup_node); cleanup_node"' EXIT
trap 'echo "Preemption signal received"; srun --nodes=${SLURM_NNODES} --ntasks-per-node=1 bash -c "sudo docker exec ${CONTAINER_NAME} pkill -USR1 -f train.py 2>/dev/null || true"; sleep 90' USR1

echo "Phase 1: Setting up containers on all nodes..."

srun --nodes=${SLURM_NNODES} --ntasks-per-node=1 bash -c "
    export DOCKER_IMAGE='${DOCKER_IMAGE}'
    export CONTAINER_NAME='${CONTAINER_NAME}'
    export PROJECT_DIR='${PROJECT_DIR}'
    $(declare -f setup_node)
    setup_node
"

echo "Phase 2: Starting distributed training..."

srun --nodes=${SLURM_NNODES} --ntasks-per-node=1 bash -c "
    export DOCKER_IMAGE='${DOCKER_IMAGE}'
    export CONTAINER_NAME='${CONTAINER_NAME}'
    export MASTER_ADDR='${MASTER_ADDR}'
    export MASTER_PORT='${MASTER_PORT}'
    export GPUS_PER_NODE='${GPUS_PER_NODE}'
    export CONFIG='${CONFIG}'
    export RUN_NAME='${RUN_NAME}'
    export EXTRA_ARGS='${EXTRA_ARGS}'
    export WANDB_API_KEY='${WANDB_API_KEY}'
    export SWEEP_SCRIPT='${SWEEP_SCRIPT}'
    export MODEL='${MODEL}'
    export SLURM_NNODES='${SLURM_NNODES}'
    $(declare -f run_training)
    run_training
"

TRAIN_EXIT=$?

echo "=============================================="
echo "Training finished with exit code: ${TRAIN_EXIT}"
echo "Job completed at $(date)"
echo "=============================================="

exit $TRAIN_EXIT
