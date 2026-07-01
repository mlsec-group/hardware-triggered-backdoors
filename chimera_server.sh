#!/bin/bash

GENERATOR_BACKEND="$1"
BLIS_BACKEND="$2"
OPENBLAS_BACKEND="$3"
CIFAR_BATCH="${4:-data/cifar10/data_batch_1}"
N_SAMPLES="${CHIMERA_N_SAMPLES:-100}"
GENERATOR_DEVICE="${CHIMERA_GENERATOR_DEVICE:-cpu}"
PORT="${CHIMERA_PORT:-13370}"
HTTP_PORT="${CHIMERA_HTTP_PORT:-9696}"
NO_TTY_ARGS=()

if [[ "${CHIMERA_NO_TTY:-}" == "1" || "${CHIMERA_NO_TTY:-}" == "true" ]]; then
    NO_TTY_ARGS+=(--no-tty)
fi

if [[ -z "$GENERATOR_BACKEND" || -z "$BLIS_BACKEND" || -z "$OPENBLAS_BACKEND" ]]; then
    echo "usage: ./chimera_server.sh <generator_backend> <blis_backend> <openblas_backend> [cifar_batch]"
    exit 1
fi

python3 main.py \
    "${NO_TTY_ARGS[@]}" \
    --backends "$GENERATOR_BACKEND" \
    --backends "$BLIS_BACKEND" \
    --backends "$OPENBLAS_BACKEND" \
    --port "$PORT" \
    --http_port "$HTTP_PORT" \
    --seed 1230 \
    --share_dir output/chimera \
    --readonly_dir models/ \
    chimera \
    --generator_backend "$GENERATOR_BACKEND" \
    --blis_backend "$BLIS_BACKEND" \
    --openblas_backend "$OPENBLAS_BACKEND" \
    --cifar_batch "$CIFAR_BATCH" \
    --sample_index 0 \
    --n_samples "$N_SAMPLES" \
    --model_path models/cifar10/final.pt \
    --generator_device "$GENERATOR_DEVICE"
