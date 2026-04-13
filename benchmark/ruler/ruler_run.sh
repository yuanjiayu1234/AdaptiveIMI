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


if [ $# -ne 7 ]; then
    echo "Usage: $0 <model_name> <benchmark_name> <attn_type> <context_length> <task> <dtype> <budget_ratio>"
    exit 1
fi

# Root Directories
ROOT_DIR="./ruler_eval_result" # the path that stores generated task samples and model predictions.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
NUM_SAMPLES=200
MAX_SEQ_LENGTH=${4}
ATTN_TYPE=${3}
DEVICE=${DEVICE:-cuda:0}
BUDGET_RATIO=${7}
SUBSPACE_PARTS=${SUBSPACE_PARTS:-2}

# Model and Tokenizer
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="$(dirname "${PROJECT_ROOT}"):${PROJECT_ROOT}:${PYTHONPATH:-}"

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
    
RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}/${ATTN_TYPE}-br${BUDGET_RATIO}-sp${SUBSPACE_PARTS}"
DATA_DIR="${RESULTS_DIR}/data"
PRED_DIR="${RESULTS_DIR}/pred"
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

for TASK in "${TASK_LIST[@]}"; do
    if [[ ! " ${TASKS[*]} " =~ " ${TASK} " ]]; then
        echo "Task: ${TASK} is not supported in benchmark ${BENCHMARK}"
        exit 1
    fi

    echo "Running task ${TASK} ..."
    SOURCE_DATA="${RULER_DATA_ROOT}/${MAX_SEQ_LENGTH}/${TASK}/${SUBSET}.jsonl"
    TARGET_DATA="${DATA_DIR}/${TASK}/${SUBSET}.jsonl"

    if [[ -f "${SOURCE_DATA}" ]]; then
        echo "Found cached dataset at ${SOURCE_DATA}, copying..."
        mkdir -p "$(dirname "${TARGET_DATA}")"
        cp "${SOURCE_DATA}" "${TARGET_DATA}"
    else
        echo "Cached dataset not found, generating ${TASK} ..."
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
        --subspace_parts ${SUBSPACE_PARTS}
done

python -u "${SCRIPT_DIR}/eval/evaluate.py" \
    --data_dir ${PRED_DIR} \
    --benchmark ${BENCHMARK}
