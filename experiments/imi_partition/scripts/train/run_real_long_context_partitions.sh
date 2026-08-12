#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

PYTHON="${PYTHON:-python}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-outputs/train_data}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$("$PYTHON" -c 'import torch; print(",".join(str(i) for i in range(torch.cuda.device_count())))')"
fi
export CUDA_VISIBLE_DEVICES

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
DEEPSEEK32B_MODEL="${DEEPSEEK32B_MODEL:-$(config_model_path config/DeepSeek-R1-Distill-Qwen-32B-FP8-Dynamic.json)}"
MISTRAL_MODEL="${MISTRAL_MODEL:-$(config_model_path config/mistral-7b-Instruct-32k.json)}"
MODELS="${MODELS:-llama3,qwen3}"

MIX_NAME="${MIX_NAME:-booksum_nolima_babilong_long64k_128k_token_balanced}"
CALIB_JSONL="${CALIB_JSONL:-${TRAIN_DATA_DIR}/calib_${MIX_NAME}.jsonl}"
OUT_DIR="${OUT_DIR:-${TRAIN_DATA_DIR}/partitions/${MIX_NAME}}"
PARTITION_TMP_DIR="${PARTITION_TMP_DIR:-${OUT_DIR}/tmp_layer_partitions}"
REBUILD_CALIB="${REBUILD_CALIB:-0}"

BOOKSUM_TRAIN="${BOOKSUM_TRAIN:-experiments/data/booksum/train.csv}"
NOLIMA_HAYSTACK_DIR="${NOLIMA_HAYSTACK_DIR:-experiments/data/NoLiMa/haystack/rand_shuffle}"
BABILONG_ZIP="${BABILONG_ZIP:-experiments/data/babilong/data/tasks_1-20_v1-2.zip}"

DEFAULT_TOKENIZER_PATH="$LLAMA3_MODEL"
case "${MODELS}" in
  deepseek32b|deepseek32b-fp8|deepseek-r1-distill-qwen-32b-fp8-dynamic)
    DEFAULT_TOKENIZER_PATH="$DEEPSEEK32B_MODEL"
    ;;
  qwen3)
    DEFAULT_TOKENIZER_PATH="$QWEN3_MODEL"
    ;;
esac
TOKENIZER_PATH="${TOKENIZER_PATH:-$DEFAULT_TOKENIZER_PATH}"
BOOKSUM_LIMIT="${BOOKSUM_LIMIT:-256}"
NOLIMA_LIMIT="${NOLIMA_LIMIT:-128}"
BABILONG_LIMIT="${BABILONG_LIMIT:-128}"
MIN_TOKENS="${MIN_TOKENS:-65536}"
MAX_TOKENS="${MAX_TOKENS:-131072}"
SOURCE_TOKEN_BUDGET="${SOURCE_TOKEN_BUDGET:-393216}"
CHUNK_CHARS="${CHUNK_CHARS:-524288}"
SELECTION_ORDER="${SELECTION_ORDER:-longest}"
RANDOM_SEED="${RANDOM_SEED:-0}"

DEVICE="${DEVICE:-auto}"
SEARCH_DEVICE="${SEARCH_DEVICE:-$DEVICE}"
MAX_LENGTH="${MAX_LENGTH:-131072}"
MAX_TOKENS_PER_PROMPT="${MAX_TOKENS_PER_PROMPT:-131072}"
SAMPLE_TOKENS="${SAMPLE_TOKENS:-131072}"
TARGET_CLUSTER_SIZE="${TARGET_CLUSTER_SIZE:-16}"
MAX_FULL_CLUSTERS="${MAX_FULL_CLUSTERS:-0}"
KMEANS_ITERS="${KMEANS_ITERS:-10}"
CDIST_CHUNK_SIZE="${CDIST_CHUNK_SIZE:-8192}"
OBJECTIVE_MODE="${OBJECTIVE_MODE:-constrained_mse}"
OVERSIZE_SOFT_CAP="${OVERSIZE_SOFT_CAP:-64}"
OVERSIZE_HARD_CAP="${OVERSIZE_HARD_CAP:-96}"
OVERSIZE_WEIGHT="${OVERSIZE_WEIGHT:-0.1}"
OVERSIZE_HARD_WEIGHT="${OVERSIZE_HARD_WEIGHT:-10.0}"
STARTS="${STARTS:-32}"
STEPS="${STEPS:-256}"
PROPOSAL_BATCH_SIZE="${PROPOSAL_BATCH_SIZE:-64}"
GROUP_KMEANS_BATCH_SIZE="${GROUP_KMEANS_BATCH_SIZE:-64}"
CALIB_ORDER="${CALIB_ORDER:-shuffle}"
CALIB_MIN_CHARS="${CALIB_MIN_CHARS:-0}"
CALIB_REQUIRE_MAX_CHARS="${CALIB_REQUIRE_MAX_CHARS:-0}"
AUTO_LONGBENCH_CALIB="${AUTO_LONGBENCH_CALIB:-0}"
ANSWER_PREFIX_FIELD="${ANSWER_PREFIX_FIELD:-answer_prefix}"
COLLECT_BATCH_SIZE="${COLLECT_BATCH_SIZE:-1}"

mkdir -p "$TRAIN_DATA_DIR" "$OUT_DIR"

if [[ "$REBUILD_CALIB" == "1" || ! -s "$CALIB_JSONL" ]]; then
  echo "[real-long-run] building BookSum/NoLiMa/BABILong calibration -> $CALIB_JSONL"
  "$PYTHON" experiments/imi_partition/calib/build_long_context_calib.py \
    --out "$CALIB_JSONL" \
    --booksum-train "$BOOKSUM_TRAIN" \
    --nolima-haystack-dir "$NOLIMA_HAYSTACK_DIR" \
    --babilong-zip "$BABILONG_ZIP" \
    --booksum-limit "$BOOKSUM_LIMIT" \
    --nolima-limit "$NOLIMA_LIMIT" \
    --babilong-limit "$BABILONG_LIMIT" \
    --min-chars 0 \
    --chunk-chars "$CHUNK_CHARS" \
    --max-chars 0 \
    --tokenizer-path "$TOKENIZER_PATH" \
    --min-tokens "$MIN_TOKENS" \
    --max-tokens "$MAX_TOKENS" \
    --source-token-budget "$SOURCE_TOKEN_BUDGET" \
    --selection-order "$SELECTION_ORDER" \
    --seed "$RANDOM_SEED"
else
  echo "[real-long-run] reusing calibration data: $CALIB_JSONL"
fi

echo "[real-long-run] training models: $MODELS"
echo "[real-long-run] output dir: $OUT_DIR"
echo "[real-long-run] calibration: $CALIB_JSONL"
echo "[real-long-run] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES DEVICE=$DEVICE"

PYTHON="$PYTHON" \
LLAMA3_MODEL="$LLAMA3_MODEL" \
QWEN3_MODEL="$QWEN3_MODEL" \
DEEPSEEK32B_MODEL="$DEEPSEEK32B_MODEL" \
MISTRAL_MODEL="$MISTRAL_MODEL" \
CALIB_JSONLS="$CALIB_JSONL" \
MERGED_CALIB_NAME="$MIX_NAME" \
OUT_DIR="$OUT_DIR" \
PARTITION_TMP_DIR="$PARTITION_TMP_DIR" \
AUTO_LONGBENCH_CALIB="$AUTO_LONGBENCH_CALIB" \
CALIB_ORDER="$CALIB_ORDER" \
CALIB_MIN_CHARS="$CALIB_MIN_CHARS" \
CALIB_REQUIRE_MAX_CHARS="$CALIB_REQUIRE_MAX_CHARS" \
MODELS="$MODELS" \
DEVICE="$DEVICE" \
SEARCH_DEVICE="$SEARCH_DEVICE" \
MAX_LENGTH="$MAX_LENGTH" \
MAX_TOKENS_PER_PROMPT="$MAX_TOKENS_PER_PROMPT" \
SAMPLE_TOKENS="$SAMPLE_TOKENS" \
TARGET_CLUSTER_SIZE="$TARGET_CLUSTER_SIZE" \
MAX_FULL_CLUSTERS="$MAX_FULL_CLUSTERS" \
KMEANS_ITERS="$KMEANS_ITERS" \
CDIST_CHUNK_SIZE="$CDIST_CHUNK_SIZE" \
OBJECTIVE_MODE="$OBJECTIVE_MODE" \
OVERSIZE_SOFT_CAP="$OVERSIZE_SOFT_CAP" \
OVERSIZE_HARD_CAP="$OVERSIZE_HARD_CAP" \
OVERSIZE_WEIGHT="$OVERSIZE_WEIGHT" \
OVERSIZE_HARD_WEIGHT="$OVERSIZE_HARD_WEIGHT" \
STARTS="$STARTS" \
STEPS="$STEPS" \
PROPOSAL_BATCH_SIZE="$PROPOSAL_BATCH_SIZE" \
GROUP_KMEANS_BATCH_SIZE="$GROUP_KMEANS_BATCH_SIZE" \
ANSWER_PREFIX_FIELD="$ANSWER_PREFIX_FIELD" \
COLLECT_BATCH_SIZE="$COLLECT_BATCH_SIZE" \
RANDOM_SEED="$RANDOM_SEED" \
bash experiments/imi_partition/scripts/train/run_domain_partitions.sh
