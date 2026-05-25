#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
CALIB_DIR="${CALIB_DIR:-outputs/imi_partition/calib_longbench_by_domain}"
OUT_DIR="${OUT_DIR:-outputs/imi_partition}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

LIMIT="${LIMIT:-8}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
MAX_TOKENS_PER_PROMPT="${MAX_TOKENS_PER_PROMPT:-2048}"
STATIC_START="${STATIC_START:-32}"
STATIC_END="${STATIC_END:-64}"
DTYPE="${DTYPE:-bf16}"
DEVICE="${DEVICE:-cuda:0}"

LAYER_GROUPS="${LAYER_GROUPS:-3}"
MAX_TOKENS_PER_SCOPE="${MAX_TOKENS_PER_SCOPE:-65536}"
MI_WEIGHT="${MI_WEIGHT:-0.25}"

M_VALUES="${M_VALUES:-4,8,16}"
MIN_KEEP_RATIO="${MIN_KEEP_RATIO:-0.25}"
SEEDS="${SEEDS:-16}"
MAX_ITER="${MAX_ITER:-100}"
SWAP_PASSES="${SWAP_PASSES:-4}"
GAMMA="${GAMMA:-0.01}"

DOMAINS=(
  multi_document_qa
  long_in_context_learning
  code_repository_understanding
  long_dialogue_history_understanding
  long_structured_data_understanding
)

run_step() {
  local output="$1"
  shift
  if [[ -s "$output" ]]; then
    echo "[skip] $output"
    return
  fi
  echo "[run] $*"
  "$@"
}

mkdir -p "$OUT_DIR"

for dataset in "${DOMAINS[@]}"; do
  calib="${CALIB_DIR}/${dataset}.jsonl"
  keys="${OUT_DIR}/qwen3_${dataset}_keys.pt"
  dependency="${OUT_DIR}/qwen3_${dataset}_dependency.pt"
  partitions="${OUT_DIR}/qwen3_${dataset}_partitions_balanced.json"
  summary="${OUT_DIR}/qwen3_${dataset}_summary_balanced.csv"
  learned="${OUT_DIR}/qwen3_${dataset}_learned_partition_balanced.json"

  if [[ ! -s "$calib" ]]; then
    echo "[error] missing calibration file: $calib" >&2
    exit 1
  fi

  echo
  echo "========== ${dataset} =========="

  run_step "$keys" env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python -m experiments.imi_partition.collect_llama3_rope_keys \
    --model "$MODEL" \
    --jsonl "$calib" \
    --text-field input \
    --limit "$LIMIT" \
    --max-length "$MAX_LENGTH" \
    --max-tokens-per-prompt "$MAX_TOKENS_PER_PROMPT" \
    --static-start "$STATIC_START" \
    --static-end "$STATIC_END" \
    --dtype "$DTYPE" \
    --device "$DEVICE" \
    --out "$keys"

  run_step "$dependency" python -m experiments.imi_partition.compute_dependency \
    --keys "$keys" \
    --layout lhtd \
    --scope layer_group \
    --layer-groups "$LAYER_GROUPS" \
    --max-tokens-per-scope "$MAX_TOKENS_PER_SCOPE" \
    --mi-weight "$MI_WEIGHT" \
    --out "$dependency"

  run_step "$partitions" python -m experiments.imi_partition.search_partitions \
    --dependency "$dependency" \
    --m-values "$M_VALUES" \
    --min-pairs 1 \
    --max-pairs 16 \
    --gamma "$GAMMA" \
    --seeds "$SEEDS" \
    --max-iter "$MAX_ITER" \
    --swap-passes "$SWAP_PASSES" \
    --min-keep-ratio "$MIN_KEEP_RATIO" \
    --constraint-metric keep_ratio \
    --select-preference largest \
    --out-json "$partitions" \
    --out-csv "$summary"

  run_step "$learned" python -m experiments.imi_partition.aggregate_partitions \
    --partitions "$partitions" \
    --min-keep-ratio "$MIN_KEEP_RATIO" \
    --constraint-metric keep_ratio \
    --select-preference largest \
    --out "$learned"
done

echo
echo "[done] domain partition experiments finished"
