#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-python}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-outputs/train_data}"
REAL_JSONL="${REAL_JSONL:-outputs/imi_partition/calib_booksum_nolima_babilong.jsonl}"
RULER_ROOT="${RULER_ROOT:-data/RULER}"

config_model_path() {
  local config_path="$1"
  "$PYTHON" - "$config_path" <<'PY'
import json
import os
import sys

config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as f:
    payload = json.load(f)
model_path = payload.get("model_path")
if not model_path:
    raise SystemExit(f"missing model_path in {config_path}")
if not os.path.isabs(str(model_path)):
    model_path = os.path.abspath(os.path.join(os.getcwd(), str(model_path)))
print(model_path)
PY
}

LLAMA3_MODEL="${LLAMA3_MODEL:-$(config_model_path config/Llama-3.1-8B-Instruct.json)}"
QWEN3_MODEL="${QWEN3_MODEL:-$(config_model_path config/Qwen3-8B.json)}"
MISTRAL_MODEL="${MISTRAL_MODEL:-$(config_model_path config/mistral-7b-Instruct-32k.json)}"
MIX_NAME="${MIX_NAME:-real100_ruler100_mixed_128k_heavy}"
MIXED_CALIB_JSONL="${MIXED_CALIB_JSONL:-${TRAIN_DATA_DIR}/calib_${MIX_NAME}.jsonl}"
OUT_DIR="${OUT_DIR:-${TRAIN_DATA_DIR}/partitions/${MIX_NAME}}"
PARTITION_TMP_DIR="${PARTITION_TMP_DIR:-${OUT_DIR}/tmp_layer_partitions}"
REBUILD_CALIB="${REBUILD_CALIB:-0}"

REAL_COUNT="${REAL_COUNT:-100}"
SYNTHETIC_COUNT="${SYNTHETIC_COUNT:-100}"
SYNTHETIC_LENGTH_WEIGHTS="${SYNTHETIC_LENGTH_WEIGHTS:-131072:40,98304:40,65536:10,32768:5,16384:3,8192:2}"
RULER_TASKS="${RULER_TASKS:-all}"
RANDOM_SEED="${RANDOM_SEED:-0}"

MODELS="${MODELS:-llama3,qwen3}"
MAX_LENGTH="${MAX_LENGTH:-131072}"
MAX_TOKENS_PER_PROMPT="${MAX_TOKENS_PER_PROMPT:-131072}"
SAMPLE_TOKENS="${SAMPLE_TOKENS:-131072}"
TARGET_CLUSTER_SIZE="${TARGET_CLUSTER_SIZE:-16}"
CALIB_ORDER="${CALIB_ORDER:-shuffle}"
CALIB_MIN_CHARS="${CALIB_MIN_CHARS:-0}"
CALIB_REQUIRE_MAX_CHARS="${CALIB_REQUIRE_MAX_CHARS:-0}"
AUTO_LONGBENCH_CALIB="${AUTO_LONGBENCH_CALIB:-0}"
ANSWER_PREFIX_FIELD="${ANSWER_PREFIX_FIELD:-answer_prefix}"
COLLECT_BATCH_SIZE="${COLLECT_BATCH_SIZE:-1}"

mkdir -p "$TRAIN_DATA_DIR" "$OUT_DIR"

if [[ "$REBUILD_CALIB" == "1" || ! -s "$MIXED_CALIB_JSONL" ]]; then
  echo "[mixed-run] building calibration data -> $MIXED_CALIB_JSONL"
  "$PYTHON" experiments/imi_partition/build_mixed_real_ruler_calib.py \
    --out "$MIXED_CALIB_JSONL" \
    --real-jsonl "$REAL_JSONL" \
    --ruler-root "$RULER_ROOT" \
    --real-count "$REAL_COUNT" \
    --synthetic-count "$SYNTHETIC_COUNT" \
    --synthetic-length-weights "$SYNTHETIC_LENGTH_WEIGHTS" \
    --ruler-tasks "$RULER_TASKS" \
    --seed "$RANDOM_SEED" \
    --write-components
else
  echo "[mixed-run] reusing calibration data: $MIXED_CALIB_JSONL"
fi

echo "[mixed-run] training models: $MODELS"
echo "[mixed-run] output dir: $OUT_DIR"
echo "[mixed-run] llama3 model: $LLAMA3_MODEL"
echo "[mixed-run] qwen3 model: $QWEN3_MODEL"
echo "[mixed-run] mistral model: $MISTRAL_MODEL"

PYTHON="$PYTHON" \
LLAMA3_MODEL="$LLAMA3_MODEL" \
QWEN3_MODEL="$QWEN3_MODEL" \
MISTRAL_MODEL="$MISTRAL_MODEL" \
CALIB_JSONLS="$MIXED_CALIB_JSONL" \
MERGED_CALIB_NAME="$MIX_NAME" \
OUT_DIR="$OUT_DIR" \
PARTITION_TMP_DIR="$PARTITION_TMP_DIR" \
AUTO_LONGBENCH_CALIB="$AUTO_LONGBENCH_CALIB" \
CALIB_ORDER="$CALIB_ORDER" \
CALIB_MIN_CHARS="$CALIB_MIN_CHARS" \
CALIB_REQUIRE_MAX_CHARS="$CALIB_REQUIRE_MAX_CHARS" \
MODELS="$MODELS" \
MAX_LENGTH="$MAX_LENGTH" \
MAX_TOKENS_PER_PROMPT="$MAX_TOKENS_PER_PROMPT" \
SAMPLE_TOKENS="$SAMPLE_TOKENS" \
TARGET_CLUSTER_SIZE="$TARGET_CLUSTER_SIZE" \
ANSWER_PREFIX_FIELD="$ANSWER_PREFIX_FIELD" \
COLLECT_BATCH_SIZE="$COLLECT_BATCH_SIZE" \
RANDOM_SEED="$RANDOM_SEED" \
bash experiments/imi_partition/run_domain_partitions.sh
