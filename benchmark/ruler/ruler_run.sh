
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DEFAULT_MODEL_NAME="${DEFAULT_MODEL_NAME:-llama-3.1-8b}"
DEFAULT_BENCHMARK="${DEFAULT_BENCHMARK:-synthetic}"
DEFAULT_ATTN_TYPE="${DEFAULT_ATTN_TYPE:-AdaptiveIMI}"
DEFAULT_CONTEXT_LENGTH="${DEFAULT_CONTEXT_LENGTH:-16384}"
DEFAULT_DTYPE="${DEFAULT_DTYPE:-bf16}"

if [ $# -eq 0 ]; then
    MODEL_ARG="${DEFAULT_MODEL_NAME}"
    BENCHMARK_ARG="${DEFAULT_BENCHMARK}"
    ATTN_TYPE_ARG="${DEFAULT_ATTN_TYPE}"
    CONTEXT_LENGTH_ARG="${DEFAULT_CONTEXT_LENGTH}"
    TASK_ARG="all"
    DTYPE_ARG="${DEFAULT_DTYPE}"
elif [ $# -eq 1 ]; then
    MODEL_ARG="${DEFAULT_MODEL_NAME}"
    BENCHMARK_ARG="${DEFAULT_BENCHMARK}"
    ATTN_TYPE_ARG="${DEFAULT_ATTN_TYPE}"
    CONTEXT_LENGTH_ARG="${DEFAULT_CONTEXT_LENGTH}"
    TASK_ARG="${1}"
    DTYPE_ARG="${DEFAULT_DTYPE}"
elif [ $# -eq 5 ]; then
    MODEL_ARG="${1}"
    BENCHMARK_ARG="${2}"
    ATTN_TYPE_ARG="${3}"
    CONTEXT_LENGTH_ARG="${4}"
    TASK_ARG="${5}"
    DTYPE_ARG="${DEFAULT_DTYPE}"
elif [ $# -eq 6 ]; then
    MODEL_ARG="${1}"
    BENCHMARK_ARG="${2}"
    ATTN_TYPE_ARG="${3}"
    CONTEXT_LENGTH_ARG="${4}"
    TASK_ARG="${5}"
    DTYPE_ARG="${6}"
else
    echo "Usage: $0 [task]"
    echo "   or: $0 <model_name> <benchmark_name> <attn_type> <context_length> <task> [dtype]"
    exit 1
fi

# Root directory for generated task samples and model predictions.
ROOT_DIR="${ROOT_DIR:-./ruler_eval_result}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
NUM_SAMPLES=${NUM_SAMPLES:-50}
MAX_SEQ_LENGTH=${CONTEXT_LENGTH_ARG}
ATTN_TYPE=${ATTN_TYPE_ARG}
DEVICE=${DEVICE:-cuda:0}
RULER_RESUME=${RULER_RESUME:-1}
RULER_USE_SOURCE_CACHE="${RULER_USE_SOURCE_CACHE:-1}"
export RULER_DATA_ROOT="${RULER_DATA_ROOT:-${PROJECT_ROOT}/data/RULER}"

export PYTHONPATH="$(dirname "${PROJECT_ROOT}"):${PROJECT_ROOT}:${PYTHONPATH:-}"
WEIGHTED_FLASH_ATTENTION_ROOT="${WEIGHTED_FLASH_ATTENTION_ROOT:-${PROJECT_ROOT}/third_party/weighted_flash_attention}"
if [[ -d "${WEIGHTED_FLASH_ATTENTION_ROOT}" ]]; then
    WEIGHTED_FLASH_ATTENTION_BUILD=""
    if [[ -d "${WEIGHTED_FLASH_ATTENTION_ROOT}/build" ]]; then
        WEIGHTED_FLASH_ATTENTION_BUILD=$(find "${WEIGHTED_FLASH_ATTENTION_ROOT}/build" -maxdepth 1 -type d -name 'lib.*' 2>/dev/null | sort | tail -n 1)
    fi
    if [[ -n "${WEIGHTED_FLASH_ATTENTION_BUILD}" ]]; then
        export PYTHONPATH="${WEIGHTED_FLASH_ATTENTION_BUILD}:${WEIGHTED_FLASH_ATTENTION_ROOT}:${PYTHONPATH}"
    else
        export PYTHONPATH="${WEIGHTED_FLASH_ATTENTION_ROOT}:${PYTHONPATH}"
    fi
fi
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${PROJECT_ROOT}/outputs/flashinfer}"
mkdir -p "${FLASHINFER_WORKSPACE_BASE}"

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

resolve_repo_config_name() {
    local model_key="$1"
    python - "${PROJECT_ROOT}" "${model_key}" <<'PY'
import sys

project_root = sys.argv[1]
model_key = sys.argv[2]

sys.path.insert(0, project_root)

from config import resolve_config_name

print(resolve_config_name(model_key), end="")
PY
}

MODEL_SELECT() {
    MODEL_NAME=$1
    MODEL_PATH=$(resolve_repo_model_path "${MODEL_NAME}")

    case $MODEL_NAME in
        qwen3-8b|Qwen/Qwen3-8B)
            MODEL_TEMPLATE_TYPE="hf-chat"
            ;;
        *)
            MODEL_TEMPLATE_TYPE="meta-chat"
            ;;
    esac
    MODEL_FRAMEWORK="hf"
    TOKENIZER_PATH=${MODEL_PATH}
    TOKENIZER_TYPE="hf"

    echo "$MODEL_PATH:$MODEL_TEMPLATE_TYPE:$MODEL_FRAMEWORK:$TOKENIZER_PATH:$TOKENIZER_TYPE"
}

MODEL_NAME=${MODEL_ARG}
MODEL_CONFIG=$(MODEL_SELECT ${MODEL_NAME})
IFS=":" read MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE <<< "$MODEL_CONFIG"
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
echo "Using model config: $(resolve_repo_config_name "${MODEL_NAME}")"

# Benchmark and Tasks
source "${SCRIPT_DIR}/ruler_config_tasks.sh"
BENCHMARK=${BENCHMARK_ARG}
if ! declare -p "${BENCHMARK}" >/dev/null 2>&1; then
    echo "Benchmark: ${BENCHMARK} is not supported"
    exit 1
fi
declare -n TASKS=$BENCHMARK
if [ "${#TASKS[@]}" -eq 0 ]; then
    echo "Benchmark: ${BENCHMARK} is not supported"
    exit 1
fi

# Start client (prepare data / call model API / obtain final metrics)
    
RUN_TAG="${ATTN_TYPE}-config-n${NUM_SAMPLES}"
RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}/${RUN_TAG}"
DATA_DIR="${RESULTS_DIR}/data"
PRED_DIR="${RESULTS_DIR}/pred"
if [[ "${RULER_RESUME}" != "1" ]]; then
    rm -rf "${DATA_DIR}"
    rm -rf "${PRED_DIR}"
fi
mkdir -p "${DATA_DIR}"
mkdir -p "${PRED_DIR}"

REQUESTED_TASK=${TASK_ARG}
if [ "${REQUESTED_TASK}" = "all" ]; then
    TASK_LIST=("${TASKS[@]}")
else
    TASK_LIST=("${REQUESTED_TASK}")
fi

DTYPE=${DTYPE_ARG}
SUBSET="validation"

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

    NEED_DATA=1
    if [[ "${RULER_RESUME}" = "1" && -f "${TARGET_DATA}" ]]; then
        TARGET_COUNT=$(jsonl_count "${TARGET_DATA}")
        if [[ "${TARGET_COUNT}" -eq "${NUM_SAMPLES}" ]]; then
            echo "Found existing dataset at ${TARGET_DATA}, reusing for resume..."
            NEED_DATA=0
        else
            echo "Existing dataset at ${TARGET_DATA} has ${TARGET_COUNT} samples; rebuilding ${NUM_SAMPLES}-sample target data."
        fi
    fi

    if [[ "${NEED_DATA}" = "1" ]]; then
        if [[ "${RULER_USE_SOURCE_CACHE}" = "1" && -f "${SOURCE_DATA}" ]]; then
            echo "Found cached dataset at ${SOURCE_DATA}, copying first ${NUM_SAMPLES} samples..."
            mkdir -p "$(dirname "${TARGET_DATA}")"
            head -n "${NUM_SAMPLES}" "${SOURCE_DATA}" > "${TARGET_DATA}"
        else
            echo "Generating ${TASK} with template ${MODEL_TEMPLATE_TYPE} ..."
            python -u "${SCRIPT_DIR}/data/prepare.py" \
                --save_dir "${DATA_DIR}" \
                --benchmark "${BENCHMARK}" \
                --task "${TASK}" \
                --tokenizer_path "${TOKENIZER_PATH}" \
                --tokenizer_type "${TOKENIZER_TYPE}" \
                --max_seq_length "${MAX_SEQ_LENGTH}" \
                --model_template_type "${MODEL_TEMPLATE_TYPE}" \
                --num_samples "${NUM_SAMPLES}" \
                ${REMOVE_NEWLINE_TAB}
        fi
    fi

    TARGET_COUNT=$(jsonl_count "${TARGET_DATA}")
    PRED_COUNT=$(pred_unique_count "${PRED_FILE}")
    if [[ "${RULER_RESUME}" = "1" && "${TARGET_COUNT}" -gt 0 && "${PRED_COUNT}" -ge "${TARGET_COUNT}" ]]; then
        echo "Prediction for ${TASK} is complete (${PRED_COUNT}/${TARGET_COUNT}), skipping..."
        continue
    fi

    python -u "${SCRIPT_DIR}/pred/call_api.py" \
        --model_name "${MODEL_NAME}" \
        --model_path "${TOKENIZER_PATH}" \
        --attn_type "${ATTN_TYPE}" \
        --max_len "${MAX_SEQ_LENGTH}" \
        --batch_size 1 \
        --data_dir "${DATA_DIR}" \
        --save_dir "${PRED_DIR}" \
        --benchmark "${BENCHMARK}" \
        --task "${TASK}" \
        --dtype "${DTYPE}" \
        --server_type "${MODEL_FRAMEWORK}" \
        --device "${DEVICE}" \
        --synthetic_len "${MAX_SEQ_LENGTH}"
done

python -u "${SCRIPT_DIR}/eval/evaluate.py" \
    --data_dir "${PRED_DIR}" \
    --benchmark "${BENCHMARK}"
