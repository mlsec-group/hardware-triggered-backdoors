#!/bin/bash

BACKEND_ONE="$1"
BACKEND_TWO="$2"

if [[ -z "$1" ]]; then
    echo "error: did not provide first backend name"
    echo "usage: ./start_server.sh <backend_name_1> <backend_name_2>"
    exit 1
fi

if [[ -z "$2" ]]; then
    echo "error: did not provide second backend name"
    echo "usage: ./start_server.sh <backend_name_1> <backend_name_2>"
    exit 1
fi

DATASET=imagenet

if [ $DATASET = "imagenet" ]; then
    DATASET_ARGS="--model_type cnn --model_path models/imagenet/efficientnet_v2_s.pt --dataset imagenet"
    DATASET_ARGS="--model_type cnn --model_path models/imagenet/resnet18_model.pt --dataset imagenet"
    DATASET_ARGS="--model_type cnn --model_path models/imagenet/vit_b_32.pt --dataset imagenet"
    DATASET_ARGS="--model_type cnn --model_path hf://google/siglip-so400m-patch14-384 --dataset imagenet"
fi

python3 main.py \
    --backends $BACKEND_ONE \
    --backends $BACKEND_TWO \
    --port 13370 \
    --http_port 9696 \
    --seed 1230 \
    --share_dir output/backdoor \
    --readonly_dir models/ \
    backdoor \
    ${DATASET_ARGS} \
    --model_dtype float32 \
    --n_poison_samples 1 \
    --n_iterations 5 \
    --heuristic gradient_sgd \
    --permute_after_gradient false \
    --flip_bits_after_gradient true \
    --n_bits_flipped 5 \
    --n_samples 10 \
    --do_crossover false \
    --do_one_vs_all false \
    --use_full_model false \
    --use_deterministic false
