#!/bin/bash

# Require at least 2 arguments
if [[ $# -lt 2 ]]; then
    echo "error: need at least two backend names"
    echo "usage: ./start_server.sh <backend_name_1> <backend_name_2> [backend_name_3 ...]"
    exit 1
fi

BACKENDS=("$@")

# Build argument array
BACKEND_ARGS=()
for b in "${BACKENDS[@]}"; do
    BACKEND_ARGS+=(--backends "$b")
done

DATASET=imagenet

if [[ "$DATASET" == "imagenet" ]]; then
    DATASET_ARGS="--model_type cnn --model_path models/imagenet/efficientnet_v2_s.pt --dataset imagenet"
    DATASET_ARGS="--model_type cnn --model_path models/imagenet/resnet18_model.pt --dataset imagenet"
    DATASET_ARGS="--model_type cnn --model_path models/imagenet/vit_b_32.pt --dataset imagenet"
    DATASET_ARGS="--model_type cnn --model_path hf://google/siglip-so400m-patch14-384 --dataset imagenet"
fi

python3 main.py \
    "${BACKEND_ARGS[@]}" \
    --port 13370 \
    --http_port 9696 \
    --seed 1230 \
    --share_dir output/backdoor \
    --readonly_dir models/ \
    backdoor \
    ${DATASET_ARGS} \
    --model_dtype float32 \
    --n_poison_samples 1 \
    --n_iterations 6 \
    --heuristic gradient_sgd \
    --permute_after_gradient false \
    --flip_bits_after_gradient true \
    --n_bits_flipped 5 \
    --n_samples 100 \
    --do_crossover false \
    --do_one_vs_all false \
    --use_full_model false \
    --use_deterministic false