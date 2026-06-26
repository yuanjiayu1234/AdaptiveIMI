#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DEFAULT_MODEL_NAME="${DEFAULT_MODEL_NAME:-llama-3.1-8b}"
DEFAULT_ATTN_TYPE="${DEFAULT_ATTN_TYPE:-AdaptiveIMI}"
DEFAULT_CONTEXT_LENGTH="${DEFAULT_CONTEXT_LENGTH:-98304}"
DEFAULT_DTYPE="${DEFAULT_DTYPE:-bf16}"
DEFAULT_BUDGET_RATIO="${DEFAULT_BUDGET_RATIO:-0.1}"

if [[ $# -eq 0 ]]; then
    MODEL_ARG="${DEFAULT_MODEL_NAME}"
    TASK_ARG="all"
    ATTN_TYPE_ARG="${DEFAULT_ATTN_TYPE}"
    CONTEXT_LENGTH_ARG="${DEFAULT_CONTEXT_LENGTH}"
    DTYPE_ARG="${DEFAULT_DTYPE}"
    BUDGET_RATIO_ARG="${DEFAULT_BUDGET_RATIO}"
elif [[ $# -eq 1 ]]; then
    MODEL_ARG="${DEFAULT_MODEL_NAME}"
    TASK_ARG="${1}"
    ATTN_TYPE_ARG="${DEFAULT_ATTN_TYPE}"
    CONTEXT_LENGTH_ARG="${DEFAULT_CONTEXT_LENGTH}"
    DTYPE_ARG="${DEFAULT_DTYPE}"
    BUDGET_RATIO_ARG="${DEFAULT_BUDGET_RATIO}"
elif [[ $# -eq 6 ]]; then
    MODEL_ARG="${1}"
    TASK_ARG="${2}"
    ATTN_TYPE_ARG="${3}"
    CONTEXT_LENGTH_ARG="${4}"
    DTYPE_ARG="${5}"
    BUDGET_RATIO_ARG="${6}"
else
    echo "Usage: $0 [task|category|all]"
    echo "   or: $0 <model_name> <task|category|all> <attn_type> <context_length> <dtype> <budget_ratio>"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$(dirname "${PROJECT_ROOT}"):${PROJECT_ROOT}:${PYTHONPATH:-}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${PROJECT_ROOT}/outputs/flashinfer}"
export MODEL_PATH_OVERRIDE="${MODEL_PATH_OVERRIDE:-/home/yjy/model/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659}"
mkdir -p "${FLASHINFER_WORKSPACE_BASE}"

ROOT_DIR="${ROOT_DIR:-${PROJECT_ROOT}/longbench_eval_result}"
DATA_DIR="${LONG_BENCH_DATA_DIR:-/home/yjy/LongBench/data}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
DEVICE="${DEVICE:-cuda:0}"
SUBSPACE_PARTS="${SUBSPACE_PARTS:-8}"
IMI_ENGINE="${IMI_ENGINE:-hybrid_8way_v2}"
PARTITION_PATH="${PARTITION_PATH:-${PROJECT_ROOT}/outputs/new/llama31_online_flat_partition.json}"
DECODE_SEARCH_MODE="${DECODE_SEARCH_MODE:-flat}"
GPU_ONLY="${GPU_ONLY:-1}"
CHAT_TEMPLATE_MODE="${CHAT_TEMPLATE_MODE:-longbench}"
LONG_BENCH_E="${LONG_BENCH_E:-0}"

if [[ "${GPU_ONLY}" == "1" || "${GPU_ONLY,,}" == "true" || "${GPU_ONLY,,}" == "yes" || "${GPU_ONLY,,}" == "on" ]]; then
    export IMI_HYBRID_INDEX_POLICY="${IMI_HYBRID_INDEX_POLICY:-gpu_only}"
fi

if [[ "${NUM_SAMPLES}" -gt 0 ]]; then
    SAMPLE_TAG="n${NUM_SAMPLES}"
else
    SAMPLE_TAG="nall"
fi
RUN_TAG="${ATTN_TYPE_ARG}-br${BUDGET_RATIO_ARG}-sp${SUBSPACE_PARTS}-${SAMPLE_TAG}"
if [[ -n "${IMI_ENGINE}" ]]; then
    RUN_TAG="${RUN_TAG}-${IMI_ENGINE}"
fi
OUTPUT_DIR="${ROOT_DIR}/${MODEL_ARG}/${CONTEXT_LENGTH_ARG}/${RUN_TAG}"
mkdir -p "${OUTPUT_DIR}"

EXTRA_ARGS=()
if [[ -n "${IMI_ENGINE}" ]]; then
    EXTRA_ARGS+=(--imi_engine "${IMI_ENGINE}")
fi
if [[ -n "${PARTITION_PATH}" ]]; then
    EXTRA_ARGS+=(--partition_path "${PARTITION_PATH}")
fi
if [[ -n "${DECODE_SEARCH_MODE}" ]]; then
    EXTRA_ARGS+=(--decode_search_mode "${DECODE_SEARCH_MODE}")
fi
if [[ -n "${CANDIDATE_CLUSTERS:-}" ]]; then
    EXTRA_ARGS+=(--candidate_clusters "${CANDIDATE_CLUSTERS}")
fi
if [[ -n "${TOKEN_RERANK_BUDGET_RATIO:-}" ]]; then
    EXTRA_ARGS+=(--token_rerank_budget_ratio "${TOKEN_RERANK_BUDGET_RATIO}")
fi
if [[ "${GPU_ONLY}" == "1" || "${GPU_ONLY,,}" == "true" || "${GPU_ONLY,,}" == "yes" || "${GPU_ONLY,,}" == "on" ]]; then
    EXTRA_ARGS+=(--gpu_only)
fi
if [[ "${LONG_BENCH_E}" == "1" || "${LONG_BENCH_E,,}" == "true" || "${LONG_BENCH_E,,}" == "yes" || "${LONG_BENCH_E,,}" == "on" ]]; then
    EXTRA_ARGS+=(--e)
fi
if [[ "${LONG_BENCH_RESUME:-1}" != "1" ]]; then
    EXTRA_ARGS+=(--disable_resume)
fi

EVAL_EXTRA_ARGS=()
if [[ "${LONG_BENCH_E}" == "1" || "${LONG_BENCH_E,,}" == "true" || "${LONG_BENCH_E,,}" == "yes" || "${LONG_BENCH_E,,}" == "on" ]]; then
    EVAL_EXTRA_ARGS+=(--e)
fi

cd "${SCRIPT_DIR}"

python -u pred.py \
    --model "${MODEL_ARG}" \
    --model_path "${MODEL_PATH_OVERRIDE}" \
    --config_model_name "${MODEL_ARG}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --task "${TASK_ARG}" \
    --num_examples "${NUM_SAMPLES}" \
    --attn_type "${ATTN_TYPE_ARG}" \
    --max_len "${CONTEXT_LENGTH_ARG}" \
    --dtype "${DTYPE_ARG}" \
    --device "${DEVICE}" \
    --retrieval_budget "${BUDGET_RATIO_ARG}" \
    --subspace_parts "${SUBSPACE_PARTS}" \
    --chat_template_mode "${CHAT_TEMPLATE_MODE}" \
    "${EXTRA_ARGS[@]}"

python -u eval.py \
    --model "${MODEL_ARG}" \
    --attn_type "${ATTN_TYPE_ARG}" \
    --output_dir "${OUTPUT_DIR}" \
    "${EVAL_EXTRA_ARGS[@]}"

RESULT_FILE="${OUTPUT_DIR}/pred/${MODEL_ARG}/${ATTN_TYPE_ARG}/result.json"
if [[ "${LONG_BENCH_E}" == "1" || "${LONG_BENCH_E,,}" == "true" || "${LONG_BENCH_E,,}" == "yes" || "${LONG_BENCH_E,,}" == "on" ]]; then
    RESULT_FILE="${OUTPUT_DIR}/pred_e/${MODEL_ARG}/${ATTN_TYPE_ARG}/result.json"
fi

echo "Results:"
cat "${RESULT_FILE}"
