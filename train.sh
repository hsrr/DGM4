#!/usr/bin/env bash
set -euo pipefail

EXPID=$(date +"%Y%m%d_%H%M%S")

HOST=${HOST:-127.0.0.1}
# Keep port strictly numeric; do not mix with GPU id strings.
DIST_PORT=${DIST_PORT:-10031}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
NUM_GPU=$(echo "${GPU_IDS}" | awk -F',' '{print NF}')

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

python train.py \
--config 'configs/train.yaml' \
--output_dir 'results' \
--checkpoint 'ALBEF_4M.pth' \
--launcher pytorch \
--rank 0 \
--log_num "${EXPID}" \
--dist-url "tcp://${HOST}:${DIST_PORT}" \
--token_momentum \
--world_size "${NUM_GPU}" \
--model_save_epoch 100
