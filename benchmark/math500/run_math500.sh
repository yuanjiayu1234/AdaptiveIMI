#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATA_PATH="${DATA_PATH:-${SCRIPT_DIR}/test.jsonl}"
MODEL_PATH="${MODEL_PATH:-/home/yjy/models/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659}"
CONFIG_MODEL_NAME="${CONFIG_MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
ATTN_TYPE="${ATTN_TYPE:-AdaptiveIMI}"
DTYPE="${DTYPE:-bf16}"
DEVICE="${DEVICE:-cuda:0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
RETRIEVAL_BUDGET="${RETRIEVAL_BUDGET:-}"
CACHE_RATIO="${CACHE_RATIO:-}"
PARTITION_PATH="${PARTITION_PATH:-}"
GPU_ONLY="${GPU_ONLY:-0}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-1}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
DISABLE_RESUME="${DISABLE_RESUME:-0}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/results}"
OUT_PATH="${OUT_PATH:-${OUT_DIR}/${ATTN_TYPE,,}.jsonl}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUT_PATH%.jsonl}.summary.json}"

mkdir -p "${OUT_DIR}"

ARGS=(
  --data_path "${DATA_PATH}"
  --model_path "${MODEL_PATH}"
  --config_model_name "${CONFIG_MODEL_NAME}"
  --out_path "${OUT_PATH}"
  --summary_path "${SUMMARY_PATH}"
  --max_model_len "${MAX_MODEL_LEN}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --attn_type "${ATTN_TYPE}"
  --dtype "${DTYPE}"
  --device "${DEVICE}"
  --parallel_mode "${PARALLEL_MODE:-single}"
  --weight_loading "${WEIGHT_LOADING:-hf}"
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
)

if [[ -n "${RETRIEVAL_BUDGET}" ]]; then
  ARGS+=(--retrieval_budget "${RETRIEVAL_BUDGET}")
fi
if [[ -n "${CACHE_RATIO}" ]]; then
  ARGS+=(--cache_ratio "${CACHE_RATIO}")
fi
if [[ -n "${PARTITION_PATH}" ]]; then
  ARGS+=(--partition_path "${PARTITION_PATH}")
fi
if [[ -n "${IMI_ENGINE:-}" ]]; then
  ARGS+=(--imi_engine "${IMI_ENGINE}")
fi
if [[ "${GPU_ONLY:-0}" == "1" || "${GPU_ONLY,,}" == "true" || "${GPU_ONLY,,}" == "yes" ]]; then
  ARGS+=(--gpu_only)
fi
if [[ "${CHAT_TEMPLATE}" == "0" ]]; then
  ARGS+=(--no_chat_template)
fi
if [[ "${ENABLE_THINKING}" == "1" || "${ENABLE_THINKING,,}" == "true" ]]; then
  ARGS+=(--enable_thinking)
fi
if [[ "${DISABLE_RESUME}" == "1" || "${DISABLE_RESUME,,}" == "true" ]]; then
  ARGS+=(--disable_resume)
fi
if [[ -n "${START:-}" ]]; then
  ARGS+=(--start "${START}")
fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  ARGS+=(--max_samples "${MAX_SAMPLES}")
fi
if [[ -n "${SAMPLE_INDICES:-}" ]]; then
  ARGS+=(--sample_indices "${SAMPLE_INDICES}")
fi

cd "${ROOT_DIR}"
python -u benchmark/math500/math500_inference.py "${ARGS[@]}" "$@"
python -u benchmark/math500/evaluate_math500.py "${OUT_PATH}" --summary_path "${SUMMARY_PATH}"

echo "Prediction file: ${OUT_PATH}"
echo "Summary file: ${SUMMARY_PATH}"
