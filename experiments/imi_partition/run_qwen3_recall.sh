#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES HF_HUB_OFFLINE TRANSFORMERS_OFFLINE PYTORCH_CUDA_ALLOC_CONF

MODEL="${MODEL:-/root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
JSONL="${JSONL:-/root/LongBench/data.json}"
PARTITION="${PARTITION:-outputs/imi_partition/qwen3_partition.json}"
OUTDIR="${OUTDIR:-outputs/imi_partition/recall_qwen3_longbench_r100}"

TEXT_FIELD="${TEXT_FIELD:-auto}"
LIMIT="${LIMIT:-1}"
MIN_PROMPT_CHARS="${MIN_PROMPT_CHARS:-0}"
LONGEST_FIRST="${LONGEST_FIRST:-1}"
MAX_LENGTH="${MAX_LENGTH:-131072}"
LAYERS="${LAYERS:-all}"
STATIC_START="${STATIC_START:-32}"
STATIC_END="${STATIC_END:-64}"
MAX_SEARCH_TOKENS="${MAX_SEARCH_TOKENS:-0}"
DTYPE="${DTYPE:-bf16}"
DEVICE="${DEVICE:-cuda:0}"
ANN_DEVICE="${ANN_DEVICE:-$DEVICE}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"

METHODS="${METHODS:-learned,sequential,full_kmeans}"
DECODE_MODES="${DECODE_MODES:-flat,subspace}"
RETRIEVAL_BUDGETS="${RETRIEVAL_BUDGETS:-0.01,0.025,0.05,0.1}"
TOKEN_RERANK_BUDGET_RATIOS="${TOKEN_RERANK_BUDGET_RATIOS:-0.1}"
RECALL_KS="${RECALL_KS:-100}"
TARGET_CLUSTER_SIZE="${TARGET_CLUSTER_SIZE:-16}"
K_VALUES="${K_VALUES:-}"
LEARNED_K_VALUES="${LEARNED_K_VALUES:-}"
SEQUENTIAL_K_VALUES="${SEQUENTIAL_K_VALUES:-}"
PAIR_SEQUENTIAL_K_VALUES="${PAIR_SEQUENTIAL_K_VALUES:-}"
MAX_FULL_CLUSTERS="${MAX_FULL_CLUSTERS:-2048}"
SUBSPACE_PARTS="${SUBSPACE_PARTS:-8}"
KMEANS_ITERS="${KMEANS_ITERS:-4}"
SEED="${SEED:-0}"
GQA_MODE="${GQA_MODE:-sum}"

mkdir -p "$OUTDIR"

if [[ ! -e "$MODEL" ]]; then
  echo "[error] MODEL does not exist: $MODEL" >&2
  exit 1
fi

if [[ ! -s "$JSONL" ]]; then
  echo "[error] JSONL does not exist or is empty: $JSONL" >&2
  exit 1
fi

if [[ ! -s "$PARTITION" ]]; then
  echo "[error] PARTITION does not exist or is empty: $PARTITION" >&2
  exit 1
fi

records="${RECORDS:-${OUTDIR}/records.pt}"
csv="${OUTDIR}/recall.csv"
summary="${OUTDIR}/recall_summary.json"

args=(
  -m experiments.imi_partition.evaluate_two_stage_recall
  --model "$MODEL"
  --text-field "$TEXT_FIELD"
  --partition "$PARTITION"
  --limit "$LIMIT"
  --min-prompt-chars "$MIN_PROMPT_CHARS"
  --max-length "$MAX_LENGTH"
  --layers "$LAYERS"
  --static-start "$STATIC_START"
  --static-end "$STATIC_END"
  --max-search-tokens "$MAX_SEARCH_TOKENS"
  --methods "$METHODS"
  --decode-modes "$DECODE_MODES"
  --retrieval-budgets "$RETRIEVAL_BUDGETS"
  --token-rerank-budget-ratios "$TOKEN_RERANK_BUDGET_RATIOS"
  --recall-ks "$RECALL_KS"
  --target-cluster-size "$TARGET_CLUSTER_SIZE"
  --max-full-clusters "$MAX_FULL_CLUSTERS"
  --subspace-parts "$SUBSPACE_PARTS"
  --kmeans-iters "$KMEANS_ITERS"
  --seed "$SEED"
  --gqa-mode "$GQA_MODE"
  --dtype "$DTYPE"
  --device "$DEVICE"
  --ann-device "$ANN_DEVICE"
  --attn-implementation "$ATTN_IMPLEMENTATION"
  --out-csv "$csv"
  --out-summary "$summary"
)

if [[ -n "$K_VALUES" ]]; then
  args+=(--k-values "$K_VALUES")
fi
if [[ -n "$LEARNED_K_VALUES" ]]; then
  args+=(--learned-k-values "$LEARNED_K_VALUES")
fi
if [[ -n "$SEQUENTIAL_K_VALUES" ]]; then
  args+=(--sequential-k-values "$SEQUENTIAL_K_VALUES")
fi
if [[ -n "$PAIR_SEQUENTIAL_K_VALUES" ]]; then
  args+=(--pair-sequential-k-values "$PAIR_SEQUENTIAL_K_VALUES")
fi

case "${LONGEST_FIRST,,}" in
  1|true|yes|y|on)
    args+=(--longest-first)
    ;;
esac

if [[ -s "$records" ]]; then
  echo "[records] reuse $records"
  args+=(--records-in "$records")
else
  echo "[records] collect into $records"
  args+=(--jsonl "$JSONL" --records-out "$records")
fi

echo "[model] $MODEL"
echo "[jsonl] $JSONL"
echo "[partition] $PARTITION"
echo "[out] $OUTDIR"
echo "[run] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $PYTHON ${args[*]}"

"$PYTHON" -u "${args[@]}"

echo
echo "[summary] $summary"
"$PYTHON" - "$summary" <<'PY'
import json
import sys
from collections import defaultdict

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = data.get("summary", [])
by_method = defaultdict(list)
for row in rows:
    by_method[row.get("method")].append(row)

print("method,mode,budget,rerank_ratio,recall_k,candidate_recall,rerank_recall,candidate_count")
for method in sorted(by_method):
    for row in sorted(
        by_method[method],
        key=lambda x: (
            str(x.get("decode_mode")),
            float(x.get("retrieval_budget", 0.0)),
            int(x.get("recall_k", 0)),
        ),
    ):
        print(
            f"{method},"
            f"{row.get('decode_mode')},"
            f"{row.get('retrieval_budget')},"
            f"{row.get('token_rerank_budget_ratio')},"
            f"{row.get('recall_k')},"
            f"{float(row.get('candidate_recall_mean', 0.0)):.6f},"
            f"{float(row.get('rerank_recall_mean', 0.0)):.6f},"
            f"{float(row.get('candidate_count_mean', 0.0)):.1f}"
        )
PY
