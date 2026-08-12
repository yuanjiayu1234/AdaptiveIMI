#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
DATA_PATH="${DATA_PATH:-/home/yjy/LongBench/data.json}"
if [[ $# -gt 0 && "${1}" != -* ]]; then
  OUT_DIR="${1}"
  EXTRA_ARGS=("${@:2}")
else
  OUT_DIR="${ROOT_DIR}/benchmark/Longbenchv2/results"
  EXTRA_ARGS=("$@")
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$(dirname "${ROOT_DIR}"):${ROOT_DIR}:${PYTHONPATH:-}"

LLAMA_MODEL="${LLAMA_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-8B}"
DEEPSEEK32B_MODEL="${DEEPSEEK32B_MODEL:-DeepSeek-R1-Distill-Qwen-32B-FP8-Dynamic}"
MODEL_FAMILY="${MODEL_FAMILY:-llama3}"
IMI_ENGINE="${IMI_ENGINE:-}"
OPTIONAL_ARGS=()

if [[ -n "${IMI_ENGINE}" ]]; then
  OPTIONAL_ARGS+=(--imi_engine "${IMI_ENGINE}")
fi
if [[ -n "${HYBRID_POLICY:-}" ]]; then
  OPTIONAL_ARGS+=(--hybrid_policy "${HYBRID_POLICY}")
fi
if [[ -n "${HYBRID_START:-}" ]]; then
  OPTIONAL_ARGS+=(--hybrid_start "${HYBRID_START}")
fi
if [[ -n "${HYBRID_GPU_EVERY:-}" ]]; then
  OPTIONAL_ARGS+=(--hybrid_gpu_every "${HYBRID_GPU_EVERY}")
fi
if [[ -n "${DECODE_SEARCH_MODE:-}" ]]; then
  OPTIONAL_ARGS+=(--decode_search_mode "${DECODE_SEARCH_MODE}")
fi
if [[ -n "${CANDIDATE_CLUSTERS:-}" ]]; then
  OPTIONAL_ARGS+=(--candidate_clusters "${CANDIDATE_CLUSTERS}")
fi
if [[ -n "${TOKEN_RERANK_BUDGET_RATIO:-}" ]]; then
  OPTIONAL_ARGS+=(--token_rerank_budget_ratio "${TOKEN_RERANK_BUDGET_RATIO}")
fi
if [[ "${GPU_ONLY:-0}" == "1" ]]; then
  OPTIONAL_ARGS+=(--gpu_only)
fi

cd "${ROOT_DIR}"
mkdir -p "${OUT_DIR}"

add_optional_arg() {
  local env_name="$1"
  local flag_name="$2"
  local value="${!env_name:-}"
  if [[ -n "${value}" ]]; then
    OPTIONAL_ARGS+=("${flag_name}" "${value}")
  fi
}

add_optional_arg RETRIEVAL_BUDGET --retrieval_budget
add_optional_arg CACHE_RATIO --cache_ratio
add_optional_arg SUBSPACE_PARTS --subspace_parts
add_optional_arg WEIGHT_LOADING --weight_loading
add_optional_arg PARALLEL_MODE --parallel_mode
add_optional_arg GPU_MEMORY_UTILIZATION --gpu_memory_utilization

run_one() {
  local model_name="$1"
  local config_model_name="$2"
  local out_path="$3"

  local run_args=("${OPTIONAL_ARGS[@]}")
  if [[ -n "${PARTITION_PATH:-}" ]]; then
    run_args+=(--partition_path "${PARTITION_PATH}")
  fi

  python -u benchmark/Longbenchv2/longbenchv2_adaptiveimi.py \
    --data_path "${DATA_PATH}" \
    --model_name "${model_name}" \
    --config_model_name "${config_model_name}" \
    --out_path "${out_path}" \
    --max_model_len "${MAX_MODEL_LEN:-131072}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-128}" \
    --attn_type "${ATTN_TYPE:-AdaptiveIMI}" \
    --dtype "${DTYPE:-bf16}" \
    --device "${DEVICE:-auto}" \
    "${run_args[@]}" \
    "${EXTRA_ARGS[@]}"
}

if [[ -n "${MODEL_NAME:-}" ]]; then
  run_one "${MODEL_NAME}" "${CONFIG_MODEL_NAME:-${MODEL_NAME}}" "${OUT_PATH:-${OUT_DIR}/adaptiveimi.jsonl}"
else
  case "${MODEL_FAMILY}" in
    llama|llama3|llama3.1)
      run_one "${LLAMA_MODEL}" "${CONFIG_MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}" "${OUT_PATH:-${OUT_DIR}/llama3_adaptiveimi.jsonl}"
      ;;
    qwen|qwen3)
      run_one "${QWEN_MODEL}" "${CONFIG_MODEL_NAME:-Qwen/Qwen3-8B}" "${OUT_PATH:-${OUT_DIR}/qwen3_adaptiveimi.jsonl}"
      ;;
    deepseek|deepseek32b|deepseek32b-fp8)
      run_one "${DEEPSEEK32B_MODEL}" "${CONFIG_MODEL_NAME:-DeepSeek-R1-Distill-Qwen-32B-FP8-Dynamic}" "${OUT_PATH:-${OUT_DIR}/deepseek32b_fp8_adaptiveimi.jsonl}"
      ;;
    both|all)
      run_one "${LLAMA_MODEL}" "meta-llama/Llama-3.1-8B-Instruct" "${OUT_DIR}/llama3_adaptiveimi.jsonl"
      run_one "${QWEN_MODEL}" "Qwen/Qwen3-8B" "${OUT_DIR}/qwen3_adaptiveimi.jsonl"
      ;;
    *)
      echo "Unsupported MODEL_FAMILY=${MODEL_FAMILY}. Use llama, qwen, or both."
      exit 1
      ;;
  esac
fi
