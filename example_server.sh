#!/bin/bash

BACKEND_ONE="$1"
BACKEND_TWO="$2"

if [[ -z "$1" ]]; then
    echo "error: did not provide first backend name"
    echo "usage: ./example_server.sh <backend_name_1> <backend_name_2>"
    exit 1
fi

if [[ -z "$2" ]]; then
    echo "error: did not provide second backend name"
    echo "usage: ./example_server.sh <backend_name_1> <backend_name_2>"
    exit 1
fi

python3 main.py \
    --backends $BACKEND_ONE \
    --backends $BACKEND_TWO \
    --port 13370 \
    --http_port 9696 \
    --seed 1230 \
    --share_dir output/backdoor \
    --readonly_dir models/ \
    example \
    --example_server_arg example-server \
    --example_client_arg example-client \