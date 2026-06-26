#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/AdaptiveIMI"
DATA_PATH="${DATA_PATH:-/root/LongBench/data.json}"
OUT_DIR="${1:-${ROOT_DIR}/benchmark/Longbenchv2/results}"
EXTRA_ARGS=("${@:2}")

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LLAMA_MODEL="${LLAMA_MODEL:-/root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659}"
QWEN_MODEL="${QWEN_MODEL:-/root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
LLAMA_PARTITION_PATH="${LLAMA_PARTITION_PATH:-/root/AdaptiveIMI/outputs/new/llama31_multidomain_recall256_per_layer_partition.json}"
QWEN_PARTITION_PATH="${QWEN_PARTITION_PATH:-/root/AdaptiveIMI/outputs/imi_partition/qwen3_domains_mean_learned_partition_balanced.json}"
MODEL_FAMILY="${MODEL_FAMILY:-qwen3}"
IMI_ENGINE="${IMI_ENGINE:-hybrid_8way_v2}"
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

run_one() {
  local model_name="$1"
  local config_model_name="$2"
  local out_path="$3"
  local partition_path="$4"

  local run_args=("${OPTIONAL_ARGS[@]}")
  if [[ -n "${partition_path}" ]]; then
    run_args+=(--partition_path "${partition_path}")
  fi

  python -u benchmark/Longbenchv2/longbenchv2_adaptiveimi.py \
    --data_path "${DATA_PATH}" \
    --model_name "${model_name}" \
    --config_model_name "${config_model_name}" \
    --out_path "${out_path}" \
    --max_model_len "${MAX_MODEL_LEN:-131072}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-128}" \
    --attn_type "${ATTN_TYPE:-AdaptiveIMI}" \
    --retrieval_budget "${RETRIEVAL_BUDGET:-0.1}" \
    --cache_ratio "${CACHE_RATIO:-0.0}" \
    --subspace_parts "${SUBSPACE_PARTS:-2}" \
    --dtype "${DTYPE:-bf16}" \
    --device "${DEVICE:-auto}" \
    "${run_args[@]}" \
    "${EXTRA_ARGS[@]}"
}

if [[ -n "${MODEL_NAME:-}" ]]; then
  run_one "${MODEL_NAME}" "${CONFIG_MODEL_NAME:-${MODEL_NAME}}" "${OUT_PATH:-${OUT_DIR}/adaptiveimi.jsonl}" "${PARTITION_PATH:-}"
else
  case "${MODEL_FAMILY}" in
    llama|llama3|llama3.1)
      run_one "${LLAMA_MODEL}" "${CONFIG_MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}" "${OUT_PATH:-${OUT_DIR}/llama3_adaptiveimi.jsonl}" "${PARTITION_PATH:-${LLAMA_PARTITION_PATH}}"
      ;;
    qwen|qwen3)
      run_one "${QWEN_MODEL}" "${CONFIG_MODEL_NAME:-Qwen/Qwen3-8B}" "${OUT_PATH:-${OUT_DIR}/qwen3_adaptiveimi.jsonl}" "${PARTITION_PATH:-${QWEN_PARTITION_PATH}}"
      ;;
    both|all)
      run_one "${LLAMA_MODEL}" "meta-llama/Llama-3.1-8B-Instruct" "${OUT_DIR}/llama3_adaptiveimi.jsonl" "${LLAMA_PARTITION_PATH}"
      run_one "${QWEN_MODEL}" "Qwen/Qwen3-8B" "${OUT_DIR}/qwen3_adaptiveimi.jsonl" "${QWEN_PARTITION_PATH}"
      ;;
    *)
      echo "Unsupported MODEL_FAMILY=${MODEL_FAMILY}. Use llama, qwen, or both."
      exit 1
      ;;
  esac
fi
