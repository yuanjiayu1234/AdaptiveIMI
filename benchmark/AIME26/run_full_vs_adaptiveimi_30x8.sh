#!/usr/bin/env bash
set -euo pipefail

# Fixed, comparable AIME26 pass@8 experiment:
#   30 problems x 8 attempts for exact Full Flash Attention with CPU KV
#   offload, then AdaptiveIMI. Both modes retain the complete generated KV
#   history; only the storage/retrieval implementation differs.
# Each attempt naturally ends at the model EOS token; the large token cap is a
# safety limit for unusually long Qwen3 reasoning traces.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/benchmark/AIME26"

CUDA_DEVICE="${CUDA_DEVICE:-1}"
MODEL_PATH="${MODEL_PATH:-/home/yjy/models/Qwen3-8B}"
CONFIG_MODEL_NAME="${CONFIG_MODEL_NAME:-Qwen3-8B}"
DATA_PATH="${DATA_PATH:-$SCRIPT_DIR/data/aime2026.jsonl}"

NUM_SAMPLES="${NUM_SAMPLES:-30}"
START_IDX="${START_IDX:-0}"
PASS_K="${PASS_K:-8}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
SEED="${SEED:-2026}"
DTYPE="${DTYPE:-bf16}"
# Qwen3-8B has max_position_embeddings=40960. Keep the default benchmark
# capacity within that native context window, leaving room for the prompt.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-40000}"

# AdaptiveIMI settings requested for this experiment.
RETRIEVAL_BUDGET="${RETRIEVAL_BUDGET:-0.1}"
CACHE_RATIO="${CACHE_RATIO:-0.1}"

OFFLOAD_OUT_PATH="${OFFLOAD_OUT_PATH:-$SCRIPT_DIR/results/full_flash_attn_offload_30x8.jsonl}"
ADAPTIVE_OUT_PATH="${ADAPTIVE_OUT_PATH:-$SCRIPT_DIR/results/adaptiveimi_30x8_budget0.1_cache0.1.jsonl}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export DATA_PATH MODEL_PATH CONFIG_MODEL_NAME
export NUM_SAMPLES START_IDX DTYPE SEED TOP_P TOP_K
export MAX_MODEL_LEN MAX_NEW_TOKENS
export CACHE_RATIO

cd "$ROOT_DIR"

echo "=== Full Flash Attention Offload: 30 problems x pass@8 ==="
OUT_PATH="$OFFLOAD_OUT_PATH" \
  "$SCRIPT_DIR/run_aime26.sh" \
  Full_Flash_Attn_Offload "$PASS_K" "$TEMPERATURE" "$RETRIEVAL_BUDGET" \
  "$CUDA_DEVICE" "$MODEL_PATH"

echo "=== AdaptiveIMI: 30 problems x pass@8 (budget=0.1, cache_ratio=0.1) ==="
OUT_PATH="$ADAPTIVE_OUT_PATH" \
  "$SCRIPT_DIR/run_aime26.sh" \
  AdaptiveIMI "$PASS_K" "$TEMPERATURE" "$RETRIEVAL_BUDGET" \
  "$CUDA_DEVICE" "$MODEL_PATH"

echo "=== Completed ==="
echo "Full-attention offload results: $OFFLOAD_OUT_PATH"
echo "AdaptiveIMI results: $ADAPTIVE_OUT_PATH"
