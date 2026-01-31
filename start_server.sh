#!/bin/bash

BACKEND_ONE="$1:gpu"
BACKEND_TWO="$2:gpu"

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
    DATASET_ARGS="--model_type cnn --model_path models/imagenet/vit_b_32.pt --dataset imagenet"
    DATASET_ARGS="--model_type cnn --model_path models/imagenet/efficientnet_v2_s.pt --dataset imagenet"
    DATASET_ARGS="--model_type cnn --model_path models/imagenet/resnet18_model.pt --dataset imagenet"
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
    --n_iterations 6 \
    --heuristic gradient_sgd \
    --permute_after_gradient false \
    --flip_bits_after_gradient true \
    --n_bits_flipped 5 \
    --n_samples 100 \
    --do_crossover false \
    --do_one_vs_all false \
    --use_full_model true \
    --use_deterministic false \
    --skip_is_prediction_close_check true