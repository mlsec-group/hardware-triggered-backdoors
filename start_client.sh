#!/bin/bash

HOSTNAME=$1
BACKEND_NAME=$2

if [[ -z "$HOSTNAME" ]]; then
    echo "error: did not provide a hostname"
    echo "usage: ./start_client.sh <hostname> <backend_name>"
    exit 1
fi

if [[ -z "$BACKEND_NAME" ]]; then
    echo "error: did not provide a backend_name"
    echo "usage: ./start_client.sh <hostname> <backend_name>"
    exit 1
fi

apptainer run --nv apptainer/gpu.sif python3 src/client_remote.py --hostname ${HOSTNAME} --port 13370 --backend ${BACKEND_NAME}:gpu
