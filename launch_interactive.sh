#!/bin/bash
# Interactive shell that sets up the same Docker environment as launch_job.slurm
set -e

# =============================================================================
# User configuration — set these for your environment
# =============================================================================
PROJECT_DIR="${PROJECT_DIR:-/data/$USER/parcae-pr}"     # Host path to this repo
DOCKER_IMAGE="${DOCKER_IMAGE:-ghcr.io/sandyresearch/parcae:sha-49c27cb}"
# =============================================================================

CONFIG="${CONFIG:-launch_configs/parcae/pretrain/parcae-small-140m.yaml}"
DATA_DIR="/resource/data"

echo "Setting up interactive environment..."
echo "Started at $(date)"
echo "Node: $(hostname)"

cd "${PROJECT_DIR}" || exit 1

RAW_TAG="${1:-}"
TAG=""
if [ -n "$RAW_TAG" ]; then
    TAG="$(echo "$RAW_TAG" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9_.-]+/-/g; s/^-+//; s/-+$//')"
    if [ -z "$TAG" ]; then
        echo "ERROR: container tag '$RAW_TAG' becomes empty after sanitization. Use letters/numbers/._-"
        exit 1
    fi
fi

BASE_CONTAINER_NAME="$(whoami)-parcae"
if [ -n "$TAG" ]; then
    CONTAINER_NAME="${BASE_CONTAINER_NAME}-${TAG}"
else
    CONTAINER_NAME="${BASE_CONTAINER_NAME}"
fi

if ! sudo docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^${DOCKER_IMAGE}$"; then
    echo "Pulling Docker image ${DOCKER_IMAGE}..."
    sudo docker pull "${DOCKER_IMAGE}"
fi

cleanup_container() {
    sudo docker stop --time=5 "$CONTAINER_NAME" 2>/dev/null || true
    sudo docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
}

CONTAINER_CREATED_BY_SCRIPT=false
cleanup_and_exit() {
    local exit_code=${1:-0}
    if [ "$CONTAINER_CREATED_BY_SCRIPT" = "true" ] && [ $exit_code -ne 0 ]; then
        cleanup_container
    fi
    exit $exit_code
}
trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM

if [ -z "$WANDB_API_KEY" ] && [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

ENV_VARS=(
    "DATA_DIR=${DATA_DIR}"
    "WANDB_API_KEY=${WANDB_API_KEY}"
    "CONFIG=${CONFIG}"
)
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    ENV_VARS+=("CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}")
fi

DOCKER_ENV_ARGS=()
for env_var in "${ENV_VARS[@]}"; do
    if [ -n "$env_var" ]; then
        DOCKER_ENV_ARGS+=(-e "$env_var")
    fi
done

exec_into_container() {
    exec sudo docker exec -it "${DOCKER_ENV_ARGS[@]}" "$CONTAINER_NAME" bash -c "source ~/.bashrc 2>/dev/null || true; exec bash"
}

if sudo docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    sudo docker exec "$CONTAINER_NAME" pip install -q -e .
    exec_into_container
fi

if sudo docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    sudo docker start "$CONTAINER_NAME"
    sleep 2
    sudo docker exec "$CONTAINER_NAME" pip install -q -e .
    exec_into_container
fi

cleanup_container

DOCKER_RUN_ENV_ARGS=()
for env_var in "${ENV_VARS[@]}"; do
    if [ -n "$env_var" ]; then
        DOCKER_RUN_ENV_ARGS+=(-e "$env_var")
    fi
done

echo "Starting new container..."
CONTAINER_CREATED_BY_SCRIPT=true
sudo docker run -dit --tty --ipc host --gpus all \
    -v "${PROJECT_DIR}":/resource \
    --workdir /resource \
    --entrypoint /bin/bash \
    --name "$CONTAINER_NAME" \
    --privileged \
    --pull never \
    "${DOCKER_RUN_ENV_ARGS[@]}" \
    "${DOCKER_IMAGE}"

for i in {1..30}; do
    if sudo docker exec "$CONTAINER_NAME" echo "ready" &>/dev/null; then
        break
    fi
    sleep 2
done

sudo docker exec "$CONTAINER_NAME" pip install -q -e .

for env_var in "${ENV_VARS[@]}"; do
    if [ -n "$env_var" ]; then
        key=$(echo "$env_var" | cut -d'=' -f1)
        value=$(echo "$env_var" | cut -d'=' -f2-)
        if ! sudo docker exec "$CONTAINER_NAME" grep -q "export $key=" ~/.bashrc 2>/dev/null; then
            sudo docker exec "$CONTAINER_NAME" bash -c "echo 'export $key=\"$value\"' >> ~/.bashrc" || true
        fi
    fi
done

exec_into_container
