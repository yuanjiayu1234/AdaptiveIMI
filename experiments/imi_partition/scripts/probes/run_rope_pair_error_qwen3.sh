#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

PYTHON="${PYTHON:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

MODEL="${MODEL:-/root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
JSONL="${JSONL:-outputs/imi_partition/calib_external_mix.jsonl}"
TEXT_FIELD="${TEXT_FIELD:-input}"
OUTDIR="${OUTDIR:-outputs/imi_partition/rope_pair_qwen3}"

LIMIT="${LIMIT:-2}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
LAYERS="${LAYERS:-0,8,16,24,31}"
STATIC_START="${STATIC_START:-32}"
STATIC_END="${STATIC_END:-64}"
MAX_SEARCH_TOKENS="${MAX_SEARCH_TOKENS:-8192}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bf16}"

SUBSPACE_PARTS="${SUBSPACE_PARTS:-8}"
TARGET_CLUSTER_SIZE="${TARGET_CLUSTER_SIZE:-16}"
KMEANS_ITERS="${KMEANS_ITERS:-8}"
FIT_TOKENS="${FIT_TOKENS:-8192}"
ASSIGN_CHUNK_SIZE="${ASSIGN_CHUNK_SIZE:-32768}"
GQA_MODE="${GQA_MODE:-per_q_head}"
MARGIN_SAMPLE_PAIRS="${MARGIN_SAMPLE_PAIRS:-20000}"
ROPE_THETA="${ROPE_THETA:-1000000}"
SEED="${SEED:-0}"

mkdir -p "$OUTDIR"

if [[ ! -e "$MODEL" ]]; then
  echo "[error] MODEL does not exist: $MODEL" >&2
  echo "[hint] override with: MODEL=/path/to/Qwen3-8B/snapshot bash $0" >&2
  exit 1
fi

if [[ ! -s "$JSONL" ]]; then
  echo "[error] JSONL does not exist or is empty: $JSONL" >&2
  echo "[hint] override with: JSONL=/path/to/calib.jsonl bash $0" >&2
  exit 1
fi

records="${OUTDIR}/records.pt"
csv="${OUTDIR}/pair_vs_no_pair.csv"
summary="${OUTDIR}/pair_vs_no_pair.summary.json"

args=(
  -m experiments.imi_partition.probes.rope_pair_dot_error
  --model "$MODEL"
  --text-field "$TEXT_FIELD"
  --limit "$LIMIT"
  --max-length "$MAX_LENGTH"
  --layers "$LAYERS"
  --static-start "$STATIC_START"
  --static-end "$STATIC_END"
  --max-search-tokens "$MAX_SEARCH_TOKENS"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --methods sequential,pair_sequential
  --subspace-parts "$SUBSPACE_PARTS"
  --target-cluster-size "$TARGET_CLUSTER_SIZE"
  --kmeans-iters "$KMEANS_ITERS"
  --fit-tokens "$FIT_TOKENS"
  --assign-chunk-size "$ASSIGN_CHUNK_SIZE"
  --gqa-mode "$GQA_MODE"
  --margin-sample-pairs "$MARGIN_SAMPLE_PAIRS"
  --rope-theta "$ROPE_THETA"
  --seed "$SEED"
  --out-csv "$csv"
  --out-summary "$summary"
)

if [[ -s "$records" ]]; then
  echo "[records] reuse $records"
  args+=(--records "$records")
else
  echo "[records] collect into $records"
  args+=(--jsonl "$JSONL" --records-out "$records")
fi

echo "[model] $MODEL"
echo "[jsonl] $JSONL"
echo "[out] $OUTDIR"
echo "[run] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $PYTHON ${args[*]}"

env \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
  TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE" \
  "$PYTHON" -u "${args[@]}"

echo
echo "[summary] $summary"
"$PYTHON" - "$summary" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

methods = data.get("methods", {})
deltas = data.get("deltas", {})
metrics = [
    "dot_mae_mean",
    "dot_rmse_mean",
    "dot_nrmse_by_std_mean",
    "dot_rel_mae_mean",
    "attn_tv_mean",
    "margin_flip_rate_mean",
    "score_pearson_mean",
    "rope_commutator_rel_mean",
    "rope_commutator_mse_mean",
]

print("method,count," + ",".join(metrics))
for method in ("sequential", "pair_sequential"):
    row = methods.get(method, {})
    values = [method, str(row.get("count", ""))]
    values.extend(f"{float(row[m]):.8g}" if m in row else "" for m in metrics)
    print(",".join(values))

print()
print("delta = sequential - pair_sequential; positive means pair is smaller/better for error metrics.")
for key in (
    "sequential_minus_pair_dot_mae",
    "sequential_minus_pair_dot_rmse",
    "sequential_minus_pair_dot_nrmse_by_std",
    "sequential_minus_pair_dot_rel_mae",
    "sequential_minus_pair_attn_tv",
    "sequential_minus_pair_margin_flip_rate",
    "sequential_minus_pair_rope_commutator_rel",
    "sequential_minus_pair_rope_commutator_mse",
):
    if key in deltas:
        print(f"{key}: {float(deltas[key]):.8g}")
PY

echo
echo "[done] detail csv: $csv"
