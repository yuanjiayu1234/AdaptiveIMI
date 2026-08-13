#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/benchmark/AIME26"

ATTN_TYPE="${1:-Full_Flash_Attn}"
PASS_K="${2:-1}"
TEMPERATURE="${3:-0.0}"
RETRIEVAL_BUDGET="${4:-0.1}"
CUDA_DEVICE="${5:-${CUDA_VISIBLE_DEVICES:-0}}"
MODEL_PATH="${6:-Qwen/Qwen3-8B}"

DATA_PATH="${DATA_PATH:-$SCRIPT_DIR/data/aime2026.jsonl}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
START_IDX="${START_IDX:-0}"
DTYPE="${DTYPE:-bf16}"
SEED="${SEED:-2026}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
CONFIG_MODEL_NAME="${CONFIG_MODEL_NAME:-Qwen3-8B}"
BOXED_ANSWER_STOP_INTERVAL="${BOXED_ANSWER_STOP_INTERVAL:-8}"
OUT_PATH="${OUT_PATH:-$SCRIPT_DIR/results/${ATTN_TYPE,,}_pass@${PASS_K}.jsonl}"

if [[ "$PASS_K" -le 0 ]]; then
  echo "PASS_K must be positive" >&2
  exit 2
fi

case "$ATTN_TYPE" in
  Full_Flash_Attn|Full_Flash_Attn_Offload|AdaptiveIMI) ;;
  *)
    echo "Unsupported ATTN_TYPE: $ATTN_TYPE" >&2
    exit 2
    ;;
esac

if [[ "$PASS_K" -gt 1 && "$TEMPERATURE" == "0.0" ]]; then
  echo "Warning: pass@${PASS_K} with temperature=0.0 repeats greedy decoding." >&2
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

cd "$ROOT_DIR"
python -u benchmark/AIME26/aime26_inference.py \
  --data_path "$DATA_PATH" \
  --model_path "$MODEL_PATH" \
  --config_model_name "$CONFIG_MODEL_NAME" \
  --out_path "$OUT_PATH" \
  --max_model_len "$MAX_MODEL_LEN" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --start_idx "$START_IDX" \
  --num_samples "$NUM_SAMPLES" \
  --attn_type "$ATTN_TYPE" \
  --retrieval_budget "$RETRIEVAL_BUDGET" \
  --cache_ratio "${CACHE_RATIO:-0.05}" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --top_k "$TOP_K" \
  --boxed_answer_stop_interval "$BOXED_ANSWER_STOP_INTERVAL" \
  --n_generate "$PASS_K" \
  --pass_k "$PASS_K" \
  --dtype "$DTYPE" \
  --device cuda:0 \
  --parallel_mode single \
  --seed "$SEED"

python -u benchmark/AIME26/evaluate_aime26.py \
  "$OUT_PATH" \
  --pass_k "$PASS_K" \
  --complete_only
