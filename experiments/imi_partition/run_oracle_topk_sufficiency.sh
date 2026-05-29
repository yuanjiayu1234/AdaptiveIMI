#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

MODEL="${MODEL:-/root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659}"
JSONL="${JSONL:-outputs/imi_partition/calib_longbench_by_domain/single_document_qa.jsonl}"
OUTDIR="${OUTDIR:-outputs/oracle_topk_sufficiency}"
LIMIT="${LIMIT:-8}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
LAYERS="${LAYERS:-all}"
FIXED_KS="${FIXED_KS:-32,64,128,256,512}"
PCT_BUDGETS="${PCT_BUDGETS:-0.01,0.025,0.05,0.10,0.20}"
MASS_THRESHOLDS="${MASS_THRESHOLDS:-0.8,0.9,0.95}"
STATIC_START="${STATIC_START:-0}"
STATIC_END="${STATIC_END:-0}"
MAX_SEARCH_TOKENS="${MAX_SEARCH_TOKENS:-0}"
DTYPE="${DTYPE:-bf16}"
DEVICE="${DEVICE:-cuda:0}"
EVAL_DEVICE="${EVAL_DEVICE:-${DEVICE}}"

mkdir -p "${OUTDIR}"

python experiments/imi_partition/oracle_topk_attention_sufficiency.py \
  --model "${MODEL}" \
  --jsonl "${JSONL}" \
  --limit "${LIMIT}" \
  --max-length "${MAX_LENGTH}" \
  --layers "${LAYERS}" \
  --static-start "${STATIC_START}" \
  --static-end "${STATIC_END}" \
  --max-search-tokens "${MAX_SEARCH_TOKENS}" \
  --fixed-ks "${FIXED_KS}" \
  --pct-budgets "${PCT_BUDGETS}" \
  --mass-thresholds "${MASS_THRESHOLDS}" \
  --dtype "${DTYPE}" \
  --device "${DEVICE}" \
  --eval-device "${EVAL_DEVICE}" \
  --out-csv "${OUTDIR}/oracle_topk_detail.csv" \
  --out-threshold-csv "${OUTDIR}/oracle_topk_thresholds.csv" \
  --out-summary "${OUTDIR}/oracle_topk_summary.json" \
  --plot-dir "${OUTDIR}/plots"
