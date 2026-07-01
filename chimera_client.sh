#!/bin/bash

HOSTNAME="$1"
BACKEND_NAME="$2"
PYTHON_CMD="${3:-python3}"
PORT="${CHIMERA_PORT:-13370}"
CLIENT_IMAGE="${CHIMERA_CLIENT_IMAGE:-apptainer/chimera-generator.sif}"

if [[ -z "$HOSTNAME" || -z "$BACKEND_NAME" ]]; then
    echo "usage: ./chimera_client.sh <server_hostname> <backend_name> [python_cmd]"
    echo "set CHIMERA_CLIENT_IMAGE to choose the Apptainer image"
    exit 1
fi

if [[ ! -f "$CLIENT_IMAGE" ]]; then
    echo "missing client image: $CLIENT_IMAGE" >&2
    exit 1
fi

apptainer run --nv "$CLIENT_IMAGE" "$PYTHON_CMD" src/client_remote.py \
    --hostname "$HOSTNAME" \
    --port "$PORT" \
    --backend "$BACKEND_NAME"
