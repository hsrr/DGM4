EXPID=your_best_model_dir_name

HOST='127.0.0.1'
PORT='1'

NUM_GPU=1
TEXT_ENCODER_PATH=${TEXT_ENCODER_PATH:-bert-base-uncased}
EXTRA_ARGS=${EXTRA_ARGS:-}

python test.py \
--config 'configs/test.yaml' \
--output_dir 'results' \
--text_encoder "${TEXT_ENCODER_PATH}" \
--launcher pytorch \
--rank 0 \
--log_num ${EXPID} \
--dist-url tcp://${HOST}:1003${PORT} \
--token_momentum \
--world_size $NUM_GPU \
--test_epoch best \
${EXTRA_ARGS}

