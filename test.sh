#!/usr/bin/env bash
set -euo pipefail

EXPID=${EXPID:-your_best_model_dir_name}
HOST=${HOST:-127.0.0.1}
DIST_PORT=${DIST_PORT:-10031}
GPU_IDS=${GPU_IDS:-0}
NUM_GPU=$(echo "${GPU_IDS}" | awk -F',' '{print NF}')

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

python test.py \
--config 'configs/test.yaml' \
--output_dir 'results' \
--launcher pytorch \
--rank 0 \
--log_num "${EXPID}" \
--dist-url "tcp://${HOST}:${DIST_PORT}" \
--token_momentum \
--world_size "${NUM_GPU}" \
--test_epoch best \
--local_files_only \

