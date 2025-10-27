#!/bin/bash
set -e

# Configuration
IMAGE="ssregpros:latest"
DEV_MODE="${DEV_MODE:-0}"

# Directories for mounting.
CACHE_DIR="${CACHE_DIR:-$(pwd)/cache}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$(pwd)/checkpoints}"
DATASETS_DIR="${DATASETS_DIR:-$(pwd)/datasets}"

# Check for GPU support.
if docker run --rm --gpus all hello-world &> /dev/null 2>&1; then
    GPU_FLAG="--gpus all"
else
    GPU_FLAG=""
fi

# Mount cache, checkpoints, and dataset folders if in dev mode.
VOLUME_MOUNTS=()
if [ "$DEV_MODE" = "1" ]; then
    echo "[DEV MODE] Mounting "
    VOLUME_MOUNTS+=(
        -v "$CACHE_DIR:/workspace/cache"
        -v "$CHECKPOINTS_DIR:/workspace/checkpoints"
        -v "$DATASETS_DIR:/workspace/datasets"
    )
fi

# Run container.
docker run \
    --rm \
    -it \
    $GPU_FLAG \
    "${VOLUME_MOUNTS[@]}" \
    -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
    $IMAGE \
    "$@"