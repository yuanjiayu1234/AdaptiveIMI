#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e


if [ $# -ne 7 ]; then
    echo "Usage: $0 <model_name> <benchmark_name> <attn_type> <context_length> <task> <dtype> <budget_ratio>"
    exit 1
fi

# Root Directories
ROOT_DIR="./ruler_eval_result" # the path that stores generated task samples and model predictions.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
NUM_SAMPLES=${NUM_SAMPLES:-200}
MAX_SEQ_LENGTH=${4}
ATTN_TYPE=${3}
DEVICE=${DEVICE:-cuda:0}
BUDGET_RATIO=${7}
SUBSPACE_PARTS=${SUBSPACE_PARTS:-2}
IMI_ENGINE=${IMI_ENGINE:-}
PARTITION_PATH=${PARTITION_PATH:-}
DECODE_SEARCH_MODE=${DECODE_SEARCH_MODE:-}
CANDIDATE_CLUSTERS=${CANDIDATE_CLUSTERS:-}
RERANK_CLUSTERS=${RERANK_CLUSTERS:-}
RULER_RESUME=${RULER_RESUME:-0}

# Model and Tokenizer
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="$(dirname "${PROJECT_ROOT}"):${PROJECT_ROOT}:${PYTHONPATH:-}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${PROJECT_ROOT}/outputs/flashinfer}"
mkdir -p "${FLASHINFER_WORKSPACE_BASE}"

MODEL2PATH_FILE="${SCRIPT_DIR}/model2path.json"

resolve_repo_model_path() {
    local model_key="$1"
    python - "${PROJECT_ROOT}" "${model_key}" <<'PY'
import sys

project_root = sys.argv[1]
model_key = sys.argv[2]

sys.path.insert(0, project_root)

try:
    from config import resolve_model_path
    resolved = resolve_model_path(model_key)
except Exception:
    resolved = model_key

print(resolved, end="")
PY
}

get_local_model_path() {
    local model_key="$1"
    local resolved_path=""

    if [[ -f "${MODEL2PATH_FILE}" ]]; then
        resolved_path=$(python - "${MODEL2PATH_FILE}" "${model_key}" <<'PY'
import json
import sys

file_path = sys.argv[1]
model_key = sys.argv[2]

with open(file_path, "r") as fin:
    mapping = json.load(fin)

print(mapping.get(model_key, ""), end="")
PY
)
    fi

    echo "${resolved_path}"
}

MODEL_SELECT() {
    MODEL_NAME=$1

    case $MODEL_NAME in
        qwen2.5-7b)
            REMOTE_MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
            MODEL_TEMPLATE_TYPE="meta-chat"
            MODEL_FRAMEWORK="hf"
            ;;
        qwen3-8b|Qwen/Qwen3-8B)
            REMOTE_MODEL_PATH="Qwen/Qwen3-8B"
            MODEL_TEMPLATE_TYPE="hf-chat"
            MODEL_FRAMEWORK="hf"
            ;;
        llama-3-8b-1048k|gradientai/Llama-3-8B-Instruct-Gradient-1048k|Llama-3-8B-Instruct-Gradient-1048k)
            REMOTE_MODEL_PATH="gradientai/Llama-3-8B-Instruct-Gradient-1048k"
            MODEL_TEMPLATE_TYPE="meta-chat"
            MODEL_FRAMEWORK="hf"
            ;;
        llama-3.1-8b|meta-llama/Llama-3.1-8B-Instruct|Llama-3.1-8B-Instruct)
            REMOTE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
            MODEL_TEMPLATE_TYPE="meta-chat"
            MODEL_FRAMEWORK="hf"
            ;;
        qwen2.5-72b)
            REMOTE_MODEL_PATH="Qwen/Qwen2.5-72B-Instruct"
            MODEL_TEMPLATE_TYPE="meta-chat"
            MODEL_FRAMEWORK="hf"
            ;;
        *)
            REMOTE_MODEL_PATH=""
            ;;
    esac

    MODEL_PATH=$(resolve_repo_model_path "${MODEL_NAME}")

    LOCAL_MODEL_PATH=$(get_local_model_path "${MODEL_NAME}")
    if [[ -z "${LOCAL_MODEL_PATH}" && -n "${REMOTE_MODEL_PATH}" ]]; then
        LOCAL_MODEL_PATH=$(get_local_model_path "${REMOTE_MODEL_PATH}")
    fi
    if [[ -n "${LOCAL_MODEL_PATH}" ]]; then
        MODEL_PATH=${LOCAL_MODEL_PATH}
    fi
    if [[ -z "${MODEL_PATH}" ]]; then
        MODEL_PATH=${REMOTE_MODEL_PATH}
    fi

    TOKENIZER_PATH=${MODEL_PATH}
    TOKENIZER_TYPE="hf"

    echo "$MODEL_PATH:$MODEL_TEMPLATE_TYPE:$MODEL_FRAMEWORK:$TOKENIZER_PATH:$TOKENIZER_TYPE"
}

MODEL_NAME=${1}
MODEL_CONFIG=$(MODEL_SELECT ${MODEL_NAME})
IFS=":" read MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE <<< "$MODEL_CONFIG"
if [[ -n "${MODEL_PATH_OVERRIDE:-}" ]]; then
    MODEL_PATH="${MODEL_PATH_OVERRIDE}"
    TOKENIZER_PATH="${MODEL_PATH_OVERRIDE}"
fi
if [[ -n "${MODEL_TEMPLATE_TYPE_OVERRIDE:-}" ]]; then
    MODEL_TEMPLATE_TYPE="${MODEL_TEMPLATE_TYPE_OVERRIDE}"
fi
if [ -z "${MODEL_PATH}" ]; then
    echo "Model: ${MODEL_NAME} is not supported"
    exit 1
fi

if [[ -e "${MODEL_PATH}" ]]; then
    echo "Using local model path: ${MODEL_PATH}"
else
    echo "Local model not found, fallback to HF: ${MODEL_PATH}"
fi

# Benchmark and Tasks
source "${SCRIPT_DIR}/ruler_config_tasks.sh"
BENCHMARK=${2}
declare -n TASKS=$BENCHMARK
if [ -z "${TASKS}" ]; then
    echo "Benchmark: ${BENCHMARK} is not supported"
    exit 1
fi

# Start client (prepare data / call model API / obtain final metrics)
    
RUN_TAG="${ATTN_TYPE}-br${BUDGET_RATIO}-sp${SUBSPACE_PARTS}"
if [[ -n "${IMI_ENGINE}" ]]; then
    RUN_TAG="${RUN_TAG}-${IMI_ENGINE}"
fi
RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}/${RUN_TAG}"
DATA_DIR="${RESULTS_DIR}/data"
PRED_DIR="${RESULTS_DIR}/pred"
if [[ "${RULER_RESUME}" != "1" ]]; then
    rm -rf "${DATA_DIR}"
    rm -rf "${PRED_DIR}"
fi
mkdir -p ${DATA_DIR}
mkdir -p ${PRED_DIR}

REQUESTED_TASK=${5}
if [ "${REQUESTED_TASK}" = "all" ]; then
    TASK_LIST=("${TASKS[@]}")
else
    TASK_LIST=("${REQUESTED_TASK}")
fi

DTYPE=${6}
SUBSET="validation"
RULER_DATA_ROOT="${RULER_DATA_ROOT:-${PROJECT_ROOT}/data/RULER}"
RULER_USE_SOURCE_CACHE="${RULER_USE_SOURCE_CACHE:-0}"

jsonl_count() {
    local file="$1"
    python - "${file}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print(0)
else:
    with path.open("r", encoding="utf-8") as fin:
        print(sum(1 for line in fin if line.strip()))
PY
}

pred_unique_count() {
    local file="$1"
    python - "${file}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
indices = set()
if path.exists():
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "index" in sample:
                indices.add(sample["index"])
print(len(indices))
PY
}

for TASK in "${TASK_LIST[@]}"; do
    if [[ ! " ${TASKS[*]} " =~ " ${TASK} " ]]; then
        echo "Task: ${TASK} is not supported in benchmark ${BENCHMARK}"
        exit 1
    fi

    echo "Running task ${TASK} ..."
    SOURCE_DATA="${RULER_DATA_ROOT}/${MAX_SEQ_LENGTH}/${TASK}/${SUBSET}.jsonl"
    TARGET_DATA="${DATA_DIR}/${TASK}/${SUBSET}.jsonl"
    PRED_FILE="${PRED_DIR}/${TASK}.jsonl"

    if [[ "${RULER_RESUME}" = "1" && -f "${TARGET_DATA}" ]]; then
        TARGET_COUNT=$(jsonl_count "${TARGET_DATA}")
        if [[ "${TARGET_COUNT}" -ne "${NUM_SAMPLES}" ]]; then
            echo "Existing dataset at ${TARGET_DATA} has ${TARGET_COUNT} samples, but NUM_SAMPLES=${NUM_SAMPLES}."
            echo "Refusing to resume with a different sample count. Use the same NUM_SAMPLES, or remove/use a different results directory."
            exit 1
        fi
        echo "Found existing dataset at ${TARGET_DATA}, reusing for resume..."
    elif [[ "${RULER_USE_SOURCE_CACHE}" = "1" && -f "${SOURCE_DATA}" ]]; then
        echo "Found cached dataset at ${SOURCE_DATA}, copying..."
        mkdir -p "$(dirname "${TARGET_DATA}")"
        cp "${SOURCE_DATA}" "${TARGET_DATA}"
    else
        echo "Generating ${TASK} with template ${MODEL_TEMPLATE_TYPE} ..."
        python -u "${SCRIPT_DIR}/data/prepare.py" \
            --save_dir ${DATA_DIR} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
            --tokenizer_path ${TOKENIZER_PATH} \
            --tokenizer_type ${TOKENIZER_TYPE} \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --model_template_type ${MODEL_TEMPLATE_TYPE} \
            --num_samples ${NUM_SAMPLES} \
            ${REMOVE_NEWLINE_TAB}
    fi

    TARGET_COUNT=$(jsonl_count "${TARGET_DATA}")
    PRED_COUNT=$(pred_unique_count "${PRED_FILE}")
    if [[ "${RULER_RESUME}" = "1" && "${TARGET_COUNT}" -gt 0 && "${PRED_COUNT}" -ge "${TARGET_COUNT}" ]]; then
        echo "Prediction for ${TASK} is complete (${PRED_COUNT}/${TARGET_COUNT}), skipping..."
        continue
    fi

    EXTRA_IMI_ARGS=()
    if [[ -n "${IMI_ENGINE}" ]]; then
        EXTRA_IMI_ARGS+=(--imi_engine "${IMI_ENGINE}")
    fi
    if [[ -n "${PARTITION_PATH}" ]]; then
        EXTRA_IMI_ARGS+=(--partition_path "${PARTITION_PATH}")
    fi
    if [[ -n "${DECODE_SEARCH_MODE}" ]]; then
        EXTRA_IMI_ARGS+=(--decode_search_mode "${DECODE_SEARCH_MODE}")
    fi
    if [[ -n "${CANDIDATE_CLUSTERS}" ]]; then
        EXTRA_IMI_ARGS+=(--candidate_clusters "${CANDIDATE_CLUSTERS}")
    fi
    if [[ -n "${RERANK_CLUSTERS}" ]]; then
        EXTRA_IMI_ARGS+=(--rerank_clusters "${RERANK_CLUSTERS}")
    fi

    python -u "${SCRIPT_DIR}/pred/call_api.py" \
        --model_name ${MODEL_NAME} \
        --model_path ${TOKENIZER_PATH} \
        --attn_type ${ATTN_TYPE} \
        --max_len ${MAX_SEQ_LENGTH} \
        --batch_size 1 \
        --data_dir ${DATA_DIR} \
        --save_dir ${PRED_DIR} \
        --benchmark ${BENCHMARK} \
        --task ${TASK} \
        --dtype ${DTYPE} \
        --server_type ${MODEL_FRAMEWORK} \
        --device ${DEVICE} \
        --budget_ratio ${BUDGET_RATIO} \
        --synthetic_len ${MAX_SEQ_LENGTH} \
        --subspace_parts ${SUBSPACE_PARTS} \
        "${EXTRA_IMI_ARGS[@]}"
done

python -u "${SCRIPT_DIR}/eval/evaluate.py" \
    --data_dir ${PRED_DIR} \
    --benchmark ${BENCHMARK}
