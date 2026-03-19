#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   TRAIN_LOG_NUM=20260318_140809 \
#   TEXT_ENCODER_PATH=/map-vepfs/liniuniu/hesirui/bert-base-uncased \
#   bash eval_4sets_with_summary.sh
#
# Optional overrides:
#   OUTPUT_ROOT=results
#   BASE_TEST_CONFIG=configs/test.yaml
#   DATA_ROOT=/path/to/DGM4/metadata_split
#   EXTRA_ARGS="--local_files_only --no_deit_init --token_momentum"

TRAIN_LOG_NUM="${TRAIN_LOG_NUM:-}"
TEXT_ENCODER_PATH="${TEXT_ENCODER_PATH:-}"

OUTPUT_ROOT="${OUTPUT_ROOT:-results}"
BASE_TEST_CONFIG="${BASE_TEST_CONFIG:-configs/test.yaml}"
DATA_ROOT="${DATA_ROOT:-/map-vepfs/liniuniu/hesirui/MultiModal-DeepFake-main/dataset/DGM4/metadata_split}"
EXTRA_ARGS="${EXTRA_ARGS:---local_files_only --no_deit_init --token_momentum}"
HOST="${HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-23031}"

if [[ -z "${TRAIN_LOG_NUM}" ]]; then
  echo "❌ TRAIN_LOG_NUM is required"
  echo "   Example: TRAIN_LOG_NUM=20260318_140809 bash eval_4sets_with_summary.sh"
  exit 1
fi

if [[ -z "${TEXT_ENCODER_PATH}" ]]; then
  echo "❌ TEXT_ENCODER_PATH is required"
  echo "   Example: TEXT_ENCODER_PATH=/path/to/bert-base-uncased bash eval_4sets_with_summary.sh"
  exit 1
fi

if [[ ! -f "${BASE_TEST_CONFIG}" ]]; then
  echo "❌ Base test config not found: ${BASE_TEST_CONFIG}"
  exit 1
fi

if [[ ! -f "${OUTPUT_ROOT}/${TRAIN_LOG_NUM}/checkpoint_best.pth" ]]; then
  echo "❌ Missing checkpoint: ${OUTPUT_ROOT}/${TRAIN_LOG_NUM}/checkpoint_best.pth"
  exit 1
fi

DATASET_NAMES=(guardian bbc usa_today washington_post)
declare -A TEST_FILES=(
  [guardian]="${DATA_ROOT}/guardian/test.json"
  [bbc]="${DATA_ROOT}/bbc/test.json"
  [usa_today]="${DATA_ROOT}/usa_today/test.json"
  [washington_post]="${DATA_ROOT}/washington_post/test.json"
)

for name in "${DATASET_NAMES[@]}"; do
  if [[ ! -f "${TEST_FILES[$name]}" ]]; then
    echo "❌ Missing test file for ${name}: ${TEST_FILES[$name]}"
    exit 1
  fi
done

TMP_CFG_DIR="tmp_eval_cfg_${TRAIN_LOG_NUM}"
mkdir -p "${TMP_CFG_DIR}"

echo "🚀 Start 4-set evaluation"
echo "   train log: ${TRAIN_LOG_NUM}"
echo "   output:    ${OUTPUT_ROOT}"
echo "   data root: ${DATA_ROOT}"
echo "   extra:     ${EXTRA_ARGS}"

idx=0
for name in "${DATASET_NAMES[@]}"; do
  ann="${TEST_FILES[$name]}"
  eval_log="${TRAIN_LOG_NUM}_${name}"
  port=$((BASE_PORT + idx))
  idx=$((idx + 1))

  echo "============================================================"
  echo "▶ Evaluating [${name}]"
  echo "  ann:      ${ann}"
  echo "  log_num:  ${eval_log}"
  echo "  dist-url: tcp://${HOST}:${port}"
  echo "============================================================"

  mkdir -p "${OUTPUT_ROOT}/${eval_log}"
  ln -sfn "../${TRAIN_LOG_NUM}/checkpoint_best.pth" "${OUTPUT_ROOT}/${eval_log}/checkpoint_best.pth"

  cfg_file="${TMP_CFG_DIR}/test_${name}.yaml"
  cp "${BASE_TEST_CONFIG}" "${cfg_file}"
  python3 - <<PY
import re
cfg_path = "${cfg_file}"
ann = "${ann}"
text = open(cfg_path, "r", encoding="utf-8").read()
text = re.sub(r"^val_file:\\s*\\[.*\\]\\s*$", f'val_file: ["{ann}"]', text, flags=re.M)
open(cfg_path, "w", encoding="utf-8").write(text)
PY

  python test.py \
    --config "${cfg_file}" \
    --output_dir "${OUTPUT_ROOT}" \
    --log_num "${eval_log}" \
    --test_epoch best \
    --text_encoder "${TEXT_ENCODER_PATH}" \
    --launcher pytorch \
    --rank 0 \
    --world_size 1 \
    --dist-url "tcp://${HOST}:${port}" \
    ${EXTRA_ARGS}
done

SUMMARY_CSV="${OUTPUT_ROOT}/${TRAIN_LOG_NUM}_4sets_summary.csv"
python3 - <<PY
import csv
import json
import os

output_root = "${OUTPUT_ROOT}"
train_log = "${TRAIN_LOG_NUM}"
dataset_names = ["guardian", "bbc", "usa_today", "washington_post"]
summary_csv = "${SUMMARY_CSV}"

cols = [
    "dataset",
    "val_AUC_cls", "val_EER_cls", "val_ACC_cls", "val_F1_cls", "val_MCC_cls",
    "val_ACC_multicls", "val_Macro_F1_multicls", "val_Weighted_F1_multicls",
    "val_MAP", "val_CF1", "val_OF1",
]

rows = []
for name in dataset_names:
    eval_log = f"{train_log}_{name}"
    result_file = os.path.join(output_root, eval_log, "evaluation", "results_all.txt")
    if not os.path.exists(result_file):
        rows.append({"dataset": name})
        continue

    last = None
    with open(result_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                last = obj
            except Exception:
                pass

    row = {"dataset": name}
    if last is not None:
        row.update({
            "val_AUC_cls": last.get("val_AUC_cls", ""),
            "val_EER_cls": last.get("val_EER_cls", ""),
            "val_ACC_cls": last.get("val_ACC_cls", ""),
            "val_F1_cls": last.get("val_F1_cls", ""),
            "val_MCC_cls": last.get("val_MCC_cls", ""),
            "val_ACC_multicls": last.get("val_ACC_multicls", ""),
            "val_Macro_F1_multicls": last.get("val_Macro_F1_multicls", ""),
            "val_Weighted_F1_multicls": last.get("val_Weighted_F1_multicls", ""),
            "val_MAP": last.get("val_MAP", ""),
            "val_CF1": last.get("val_CF1", ""),
            "val_OF1": last.get("val_OF1", ""),
        })
    rows.append(row)

with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=cols)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print(f"✅ Summary CSV saved: {summary_csv}")
for r in rows:
    print(r)
PY

echo "✅ Done."
echo "   - per-set raw logs: ${OUTPUT_ROOT}/${TRAIN_LOG_NUM}_*/evaluation/results_all.txt"
echo "   - summary csv:      ${SUMMARY_CSV}"
